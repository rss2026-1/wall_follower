#!/usr/bin/env python3
"""
Open-loop step test node for characterizing wall-following plant dynamics.

PURPOSE:
  Applies a scheduled sequence of steering step inputs at constant speed while
  recording wall distance from the laser scan. The resulting data lets you
  estimate the plant gain (K_plant) and choose PD gains.

PHYSICS:
  Simplified bicycle model (small steering angles):
    d(wall_dist)/dt ≈ -v * steering_angle
  So K_plant ≈ v.  If you measure Δdist/Δt when steering = δ, you should get
  Δdist/Δt ≈ v * δ.

  This means Kp should scale inversely with speed.  Rule-of-thumb starting point
  after running this test:
    K_plant_measured = (Δdist/Δt) / δ        # from the CSV data

  BUT the steering is clamped to ±0.34 rad (max_steering), so Kp is bounded:
    Kp_sat = max_steering / max_error_expected
  e.g. max_steering=0.34, max comfortable error=0.5m → Kp_sat = 0.68

  Start with the smaller of bandwidth-based and saturation-based Kp:
    Kp = min(2.0 / K_plant, max_steering / max_error_expected)
    Kd = Kp / (2 * sqrt(Kp * K_plant))   # critical damping

  At runtime the node prints these numbers for you.

TEST SEQUENCE (one run):
  Phase 0 SETTLE   – drive straight at test speed, establish baseline wall dist
  Phase 1 STEP_POS – step steering to +delta_steering for step_duration seconds
  Phase 2 RECOVER  – steer back straight, let robot re-center
  Phase 3 STEP_NEG – step steering to -delta_steering (opposite direction)
  Phase 4 STOP     – publish speed=0, node exits

Run multiple times at different speeds to verify K_plant ∝ v.

OUTPUT:
  CSV written to /tmp/step_test_<timestamp>.csv with columns:
    time_s, phase, cmd_speed, cmd_steering, wall_dist_m, wall_slope

USAGE:
  ros2 run wall_follower step_test --ros-args \
    -p side:=-1 \
    -p velocity:=1.0 \
    -p delta_steering:=0.15 \
    -p settle_duration:=3.0 \
    -p step_duration:=3.0 \
    -p recover_duration:=4.0

  side: +1=left wall, -1=right wall (same convention as wall_follower)
  delta_steering: step size in radians (keep <=0.2 for safety)
"""

import csv
import os
import time

import numpy as np
import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import LaserScan


PHASES = ["SETTLE", "STEP_POS", "RECOVER", "STEP_NEG", "STOP"]


class StepTest(Node):
    def __init__(self):
        super().__init__("step_test")

        # --- parameters ---
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("drive_topic", "/drive")
        self.declare_parameter("side", -1)               # +1 left, -1 right
        self.declare_parameter("velocity", 1.0)          # m/s during test
        self.declare_parameter("delta_steering", 0.15)   # step size, radians
        self.declare_parameter("max_steering", 0.34)     # hardware limit, rad
        self.declare_parameter("max_error_expected", 0.4) # largest error you expect in operation, m
        self.declare_parameter("settle_duration", 3.0)   # seconds
        self.declare_parameter("step_duration", 3.0)     # seconds per step
        self.declare_parameter("recover_duration", 4.0)  # seconds between steps

        self.SCAN_TOPIC       = self.get_parameter("scan_topic").get_parameter_value().string_value
        self.DRIVE_TOPIC      = self.get_parameter("drive_topic").get_parameter_value().string_value
        self.SIDE             = self.get_parameter("side").get_parameter_value().integer_value
        self.VELOCITY         = self.get_parameter("velocity").get_parameter_value().double_value
        self.DELTA            = self.get_parameter("delta_steering").get_parameter_value().double_value
        self.MAX_STEERING     = self.get_parameter("max_steering").get_parameter_value().double_value
        self.MAX_ERR_EXPECTED = self.get_parameter("max_error_expected").get_parameter_value().double_value

        self.phase_durations = [
            self.get_parameter("settle_duration").get_parameter_value().double_value,
            self.get_parameter("step_duration").get_parameter_value().double_value,
            self.get_parameter("recover_duration").get_parameter_value().double_value,
            self.get_parameter("step_duration").get_parameter_value().double_value,
            0.0,  # STOP — no duration, just publishes stop and exits
        ]

        # Steering command for each phase
        # STEP_POS steers toward the wall (SIDE * +delta so it starts moving away to test response)
        # STEP_NEG steers away
        self.phase_steering = [
            0.0,                  # SETTLE  – straight
            self.SIDE * self.DELTA,   # STEP_POS – steer toward wall
            0.0,                  # RECOVER – straight
            -self.SIDE * self.DELTA,  # STEP_NEG – steer away from wall
            0.0,                  # STOP
        ]

        # Laser filter params (same as wall_follower)
        self.range_min = 0.05
        self.range_max = 10.0
        self.x_min = 0.05
        self.x_max = 4.0

        # State machine
        self.phase_idx = 0
        self.phase_start_time = None
        self.t0 = None  # absolute start time for CSV timestamps

        # CSV
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.csv_path = f"/tmp/step_test_{timestamp}.csv"
        self._csv_file = open(self.csv_path, "w", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow(
            ["time_s", "phase", "cmd_speed", "cmd_steering", "wall_dist_m", "wall_slope"]
        )
        self.get_logger().info(f"Logging to {self.csv_path}")

        # ROS interfaces
        self.drive_pub = self.create_publisher(AckermannDriveStamped, self.DRIVE_TOPIC, 1)
        self.scan_sub  = self.create_subscription(LaserScan, self.SCAN_TOPIC, self.scan_callback, 1)

        self.get_logger().info(
            f"Step test ready  side={self.SIDE}  v={self.VELOCITY} m/s  "
            f"delta={self.DELTA:.3f} rad  phases={self.phase_durations[:4]}"
        )
        self.get_logger().info("Waiting for first scan …")

    # ------------------------------------------------------------------
    def scan_callback(self, msg: LaserScan):
        now = self.get_clock().now().nanoseconds * 1e-9

        # initialise timers on first scan
        if self.phase_start_time is None:
            self.phase_start_time = now
            self.t0 = now
            self.get_logger().info(f"Starting phase 0: {PHASES[0]}")

        # advance phase?
        elapsed_in_phase = now - self.phase_start_time
        if (self.phase_idx < len(PHASES) - 1 and
                elapsed_in_phase >= self.phase_durations[self.phase_idx]):
            self.phase_idx += 1
            self.phase_start_time = now
            self.get_logger().info(f"→ Phase {self.phase_idx}: {PHASES[self.phase_idx]}")
            if PHASES[self.phase_idx] == "STOP":
                self._publish_drive(0.0, 0.0)
                self._csv_file.flush()
                self._csv_file.close()
                self.get_logger().info(f"Done. CSV saved to {self.csv_path}")
                self._print_gain_guide()
                rclpy.shutdown()
                return

        # compute wall distance from laser
        wall_dist, wall_slope = self._compute_wall(msg)

        # command for this phase
        speed   = self.VELOCITY
        steering = self.phase_steering[self.phase_idx]
        self._publish_drive(speed, steering)

        # log
        t_rel = now - self.t0
        self._csv_writer.writerow([
            f"{t_rel:.4f}",
            PHASES[self.phase_idx],
            f"{speed:.3f}",
            f"{steering:.4f}",
            f"{wall_dist:.4f}" if wall_dist is not None else "nan",
            f"{wall_slope:.4f}" if wall_slope is not None else "nan",
        ])

    # ------------------------------------------------------------------
    def _print_gain_guide(self):
        """Print saturation-aware gain recommendations based on test params."""
        v     = self.VELOCITY
        delta = self.DELTA
        max_s = self.MAX_STEERING
        max_e = self.MAX_ERR_EXPECTED

        # Expected slope from bicycle model — compare to what you see in the CSV
        expected_slope = v * delta
        K_plant = v  # bicycle model: K_plant ≈ v

        # Bandwidth-based Kp (unconstrained)
        Kp_bw = 2.0 / K_plant

        # Saturation-based ceiling: Kp * max_error < max_steering
        # (ignoring Kd contribution — conservative)
        Kp_sat = max_s / max_e

        Kp = min(Kp_bw, Kp_sat)

        # Critical damping: closed-loop char eq is s^2 + Kd*K*s + Kp*K = 0
        # ωn = sqrt(Kp*K), critically damped: 2*ζ*ωn = Kd*K, ζ=1 → Kd = 2*ωn/K
        import math
        wn = math.sqrt(Kp * K_plant)
        Kd = 2.0 * wn / K_plant  # = 2*sqrt(Kp/K_plant)

        # Error at which steering saturates with these gains
        # steering = Kp*e + Kd*de/dt; at steady approach de/dt ≈ K*Kp*e (simplification)
        # simple saturation at e_sat = max_s / Kp (ignoring Kd)
        e_sat = max_s / Kp

        sep = "=" * 60
        self.get_logger().info(sep)
        self.get_logger().info("GAIN GUIDE  (read CSV to confirm K_plant)")
        self.get_logger().info(sep)
        self.get_logger().info(
            f"  Expected slope during step:  {expected_slope:.4f} m/s"
        )
        self.get_logger().info(
            f"  (measure actual slope in CSV; K_plant_meas = slope / {delta:.3f})"
        )
        self.get_logger().info(f"  Assumed K_plant = v = {K_plant:.2f}  (update if CSV differs)")
        self.get_logger().info(f"  Bandwidth-based Kp:    {Kp_bw:.3f}")
        self.get_logger().info(
            f"  Saturation ceiling Kp:  {Kp_sat:.3f}  "
            f"(max_steering={max_s} / max_err={max_e})"
        )
        if Kp_bw > Kp_sat:
            self.get_logger().warn(
                f"  *** Bandwidth Kp ({Kp_bw:.3f}) exceeds saturation ceiling "
                f"({Kp_sat:.3f}) — using {Kp_sat:.3f} ***"
            )
        self.get_logger().info(f"  --> Recommended Kp:  {Kp:.3f}")
        self.get_logger().info(f"  --> Recommended Kd:  {Kd:.3f}  (critically damped)")
        self.get_logger().info(f"  --> Steering saturates when |error| > {e_sat:.3f} m")
        self.get_logger().info(
            "  If you see oscillation: increase Kd.  "
            "If sluggish: increase Kp (up to sat ceiling)."
        )
        self.get_logger().info(sep)

    # ------------------------------------------------------------------
    def _compute_wall(self, msg: LaserScan):
        """Same linear-regression wall fit as wall_follower.py."""
        ranges = np.array(msg.ranges, dtype=np.float64)
        n = len(ranges)
        angles = msg.angle_min + np.arange(n, dtype=np.float64) * msg.angle_increment

        x = ranges * np.cos(angles)
        y = ranges * np.sin(angles)

        valid = (
            np.isfinite(ranges)
            & (ranges >= self.range_min)
            & (ranges <= self.range_max)
            & (self.SIDE * y > 0)
            & (x >= self.x_min)
            & (x <= self.x_max)
        )
        xs, ys = x[valid], y[valid]
        if xs.size < 6:
            return None, None

        A = np.vstack([xs, np.ones_like(xs)]).T
        try:
            m, b = np.linalg.lstsq(A, ys, rcond=None)[0]
        except Exception:
            return None, None

        wall_dist = abs(b) / np.sqrt(1.0 + m ** 2)
        return wall_dist, m

    # ------------------------------------------------------------------
    def _publish_drive(self, speed: float, steering: float):
        msg = AckermannDriveStamped()
        msg.drive.speed = float(speed)
        msg.drive.steering_angle = float(steering)
        self.drive_pub.publish(msg)


def main():
    rclpy.init()
    node = StepTest()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node._publish_drive(0.0, 0.0)
        node._csv_file.flush()
        node._csv_file.close()
    finally:
        node.destroy_node()


if __name__ == "__main__":
    main()
