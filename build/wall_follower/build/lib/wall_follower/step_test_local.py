#!/usr/bin/env python3
"""
Open-loop step test node for characterizing wall-following plant dynamics.

PURPOSE:
  Applies a scheduled sequence of steering step inputs at constant speed while
  recording wall distance from the laser scan. The resulting data lets you
  estimate the plant gain (K_plant) and choose PD gains.

PHYSICS:
  Bicycle model (small steering angles, small heading error):
    d²e/dt² = (v²/L) * delta
  So the true plant transfer function is:
    G(s) = (v²/L) / s²
  and K_plant = v²/L  (NOT simply v).

  For a PD controller C(s) = Kp + Kd*s, the closed-loop characteristic eq is:
    s² + (Kd * K_plant)*s + (Kp * K_plant) = 0
  which gives:
    wn   = sqrt(Kp * K_plant)
    zeta = (Kd * K_plant) / (2 * wn)

  Inverting for desired wn and zeta:
    Kp = wn² / K_plant
    Kd = 2 * zeta * wn / K_plant

  Rise time approximation for second-order system:
    t_rise ≈ 1.8 / wn

  Saturation ceiling (conservative, ignoring Kd contribution):
    Kp_sat = max_steering / max_error_expected
  Use Kp = min(wn-based, Kp_sat) and recompute Kd from the chosen Kp.

TEST SEQUENCE (one run):
  Phase 0 SETTLE   – drive straight at test speed, establish baseline wall dist
  Phase 1 STEP_NEG – step steering AWAY from wall (open space side — cleaner response)
  Phase 2 RECOVER  – steer straight, let robot re-center
  Phase 3 STEP_POS – step steering TOWARD wall
  Phase 4 STOP     – publish speed=0, node exits

  NOTE: STEP_NEG (away from wall) runs first deliberately. Steering toward open
  space gives a clean, unconstrained response. Steering toward the wall risks
  oblique LiDAR hits and — if close — collision before the step completes.

Run at multiple speeds to verify K_plant ∝ v².

OUTPUT:
  CSV written to /tmp/step_test_<timestamp>.csv with columns:
    time_s, phase, cmd_speed, cmd_steering, wall_dist_m, wall_slope

USAGE:
  ros2 run wall_follower step_test --ros-args \
    -p side:=-1 \
    -p velocity:=2.0 \
    -p delta_steering:=0.15 \
    -p wheelbase:=0.32 \
    -p desired_wn:=9.0 \
    -p zeta:=0.7 \
    -p settle_duration:=3.0 \
    -p step_duration:=3.0 \
    -p recover_duration:=4.0

  side:          +1=left wall, -1=right wall
  velocity:      test speed in m/s
  delta_steering: step size in radians (keep <=0.2 for safety)
  wheelbase:     vehicle wheelbase in metres (F1Tenth ≈ 0.32 m)
  desired_wn:    target closed-loop natural frequency in rad/s
                 rise time ≈ 1.8/wn  →  wn=9 gives ~0.2 s at 2 m/s
  zeta:          desired damping ratio (0.7 recommended; 1.0 = critical)
"""

import csv
import math
import os
import time

import numpy as np
import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped
from sensor_msgs.msg import LaserScan


PHASES = ["SETTLE", "STEP_NEG", "RECOVER", "STEP_POS", "STOP"]

# Safety: abort if wall closer than this during any phase
WALL_ABORT_DIST_M = 0.18

# Flush CSV every this many rows to avoid data loss on crash
CSV_FLUSH_INTERVAL = 50


class StepTest(Node):
    def __init__(self):
        super().__init__("step_test")

        # --- parameters ---
        self.declare_parameter("scan_topic",           "/scan")
        self.declare_parameter("drive_topic",          "/vesc/low_level/input/navigation")
        self.declare_parameter("side",                 -1)      # +1 left, -1 right
        self.declare_parameter("velocity",              1.0)    # m/s during test
        self.declare_parameter("delta_steering",        0.15)   # step size, radians
        self.declare_parameter("max_steering",          0.34)   # hardware limit, rad
        self.declare_parameter("max_error_expected",    0.4)    # largest expected error, m
        self.declare_parameter("wheelbase",             0.32)   # metres — F1Tenth default
        self.declare_parameter("desired_wn",            9.0)    # rad/s closed-loop nat. freq.
        self.declare_parameter("zeta",                  0.7)    # damping ratio
        self.declare_parameter("settle_duration",       3.0)
        self.declare_parameter("step_duration",         3.0)
        self.declare_parameter("recover_duration",      4.0)

        def gd(name):   return self.get_parameter(name).get_parameter_value().double_value
        def gi(name):   return self.get_parameter(name).get_parameter_value().integer_value
        def gs(name):   return self.get_parameter(name).get_parameter_value().string_value

        self.SCAN_TOPIC       = gs("scan_topic")
        self.DRIVE_TOPIC      = gs("drive_topic")
        self.SIDE             = gi("side")
        self.VELOCITY         = gd("velocity")
        self.DELTA            = gd("delta_steering")
        self.MAX_STEERING     = gd("max_steering")
        self.MAX_ERR_EXPECTED = gd("max_error_expected")
        self.WHEELBASE        = gd("wheelbase")
        self.DESIRED_WN       = gd("desired_wn")
        self.ZETA             = gd("zeta")

        self.phase_durations = [
            gd("settle_duration"),
            gd("step_duration"),
            gd("recover_duration"),
            gd("step_duration"),
            0.0,   # STOP — no duration, just publishes stop and exits
        ]

        # Phase steering commands.
        # STEP_NEG steers AWAY from wall first (open space) for a clean response.
        # STEP_POS steers TOWARD wall second (more risk — run after RECOVER).
        self.phase_steering = [
            0.0,                      # SETTLE  – straight
            -self.SIDE * self.DELTA,  # STEP_NEG – away from wall
            0.0,                      # RECOVER – straight
             self.SIDE * self.DELTA,  # STEP_POS – toward wall
            0.0,                      # STOP
        ]

        # Laser filter params (consistent with wall_follower)
        self.range_min = 0.05
        self.range_max = 10.0
        self.x_min     = 0.05
        self.x_max     = 4.0

        # State machine
        self.phase_idx        = 0
        self.phase_start_time = None
        self.t0               = None
        self._row_count       = 0

        # CSV
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.csv_path   = f"/tmp/step_test_{timestamp}.csv"
        self._csv_file   = open(self.csv_path, "w", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow(
            ["time_s", "phase", "cmd_speed", "cmd_steering", "wall_dist_m", "wall_slope"]
        )
        self.get_logger().info(f"Logging to {self.csv_path}")

        # ROS interfaces
        self.drive_pub = self.create_publisher(AckermannDriveStamped, self.DRIVE_TOPIC, 1)
        self.scan_sub  = self.create_subscription(
            LaserScan, self.SCAN_TOPIC, self.scan_callback, 1
        )

        self.get_logger().info(
            f"Step test ready  side={self.SIDE}  v={self.VELOCITY} m/s  "
            f"delta={self.DELTA:.3f} rad  L={self.WHEELBASE} m  "
            f"wn={self.DESIRED_WN} rad/s  zeta={self.ZETA}"
        )
        self.get_logger().info(
            f"Expected rise time: {1.8/self.DESIRED_WN:.3f} s  "
            f"(at v={self.VELOCITY} m/s)"
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

        # compute wall distance first — needed for safety check
        wall_dist, wall_slope = self._compute_wall(msg)

        # --- SAFETY ABORT ---
        # If wall is too close OR we lost the wall during a step toward it, stop.
        if wall_dist is not None and wall_dist < WALL_ABORT_DIST_M:
            self._emergency_stop("Wall too close ({:.3f} m) — aborting".format(wall_dist))
            return
        if wall_dist is None and PHASES[self.phase_idx] == "STEP_POS":
            self._emergency_stop("Lost wall during STEP_POS — aborting")
            return

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

        # command for this phase
        speed    = self.VELOCITY
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

        # periodic flush to avoid data loss on hard crash
        self._row_count += 1
        if self._row_count % CSV_FLUSH_INTERVAL == 0:
            self._csv_file.flush()

    # ------------------------------------------------------------------
    def _emergency_stop(self, reason: str):
        self._publish_drive(0.0, 0.0)
        self._csv_file.flush()
        self._csv_file.close()
        self.get_logger().error(f"EMERGENCY STOP: {reason}")
        self.get_logger().info(f"Partial CSV saved to {self.csv_path}")
        rclpy.shutdown()

    # ------------------------------------------------------------------
    def _print_gain_guide(self):
        """
        Print gain recommendations derived from the bicycle model.

        Plant:   G(s) = K_plant / s²   where K_plant = v² / L
        Controller: C(s) = Kp + Kd*s
        Closed-loop char eq: s² + (Kd*K)*s + (Kp*K) = 0
          wn   = sqrt(Kp * K_plant)
          zeta = Kd * K_plant / (2 * wn)

        Solving for desired wn and zeta:
          Kp = wn² / K_plant
          Kd = 2 * zeta * wn / K_plant

        These gains should be normalised by v² to be speed-independent:
          Kp_norm = Kp * v²   (tune once, apply at all speeds)
          Kd_norm = Kd * v²
        """
        v      = self.VELOCITY
        L      = self.WHEELBASE
        delta  = self.DELTA
        max_s  = self.MAX_STEERING
        max_e  = self.MAX_ERR_EXPECTED
        wn     = self.DESIRED_WN
        zeta   = self.ZETA

        # True bicycle model plant gain
        K_plant = (v ** 2) / L

        # Gain formulas from pole placement
        Kp_wn  = (wn ** 2) / K_plant
        Kd_wn  = (2.0 * zeta * wn) / K_plant

        # Saturation ceiling (conservative: Kp * max_error < max_steering)
        Kp_sat = max_s / max_e

        # Final Kp — take the more conservative of bandwidth and saturation limits
        Kp = min(Kp_wn, Kp_sat)
        if Kp < Kp_wn:
            # recompute wn from the saturation-limited Kp
            wn_actual = math.sqrt(Kp * K_plant)
        else:
            wn_actual = wn
        Kd = (2.0 * zeta * wn_actual) / K_plant

        # Speed-normalised gains for deployment (scale as 1/v² automatically)
        Kp_norm = Kp * (v ** 2)
        Kd_norm = Kd * (v ** 2)

        # Saturation point
        e_sat = max_s / Kp

        # Expected wall-dist rate during step (first-order approximation)
        # At t=0+, heading error=0 so ddist/dt ≈ v * delta
        expected_dist_rate = v * delta

        sep = "=" * 65
        self.get_logger().info(sep)
        self.get_logger().info("GAIN GUIDE")
        self.get_logger().info(sep)
        self.get_logger().info(f"  Vehicle params:  v={v:.2f} m/s   L={L:.3f} m")
        self.get_logger().info(f"  K_plant = v²/L = {K_plant:.3f}  (TRUE bicycle model gain)")
        self.get_logger().info("")
        self.get_logger().info("  -- Verify from CSV --")
        self.get_logger().info(
            f"  Expected initial wall_dist rate during step: {expected_dist_rate:.4f} m/s"
        )
        self.get_logger().info(
            f"  Measure slope of wall_dist_m in STEP_NEG phase."
        )
        self.get_logger().info(
            f"  K_plant_measured = |slope| / delta = |slope| / {delta:.3f}"
        )
        self.get_logger().info(
            f"  If K_plant_measured differs significantly from {K_plant:.3f},"
        )
        self.get_logger().info(
            f"  re-run _print_gain_guide() with corrected wheelbase or v."
        )
        self.get_logger().info("")
        self.get_logger().info("  -- Pole placement (bicycle model) --")
        self.get_logger().info(f"  Desired wn:    {wn:.2f} rad/s")
        self.get_logger().info(f"  Desired zeta:  {zeta:.2f}")
        self.get_logger().info(f"  Expected rise time: {1.8/wn_actual:.3f} s")
        self.get_logger().info(f"  Kp (wn-based):  {Kp_wn:.4f}")
        self.get_logger().info(
            f"  Kp (sat ceil):  {Kp_sat:.4f}  "
            f"(max_steering={max_s} / max_err={max_e})"
        )
        if Kp_wn > Kp_sat:
            self.get_logger().warn(
                f"  *** wn-based Kp ({Kp_wn:.4f}) exceeds saturation ceiling "
                f"({Kp_sat:.4f}).  Using {Kp_sat:.4f}. "
                f"Consider reducing desired_wn or increasing max_error_expected. ***"
            )
        self.get_logger().info("")
        self.get_logger().info(f"  --> Recommended Kp:  {Kp:.4f}")
        self.get_logger().info(f"  --> Recommended Kd:  {Kd:.4f}  (zeta={zeta:.2f})")
        self.get_logger().info(f"  --> Steering saturates when |error| > {e_sat:.3f} m")
        self.get_logger().info("")
        self.get_logger().info("  -- Speed-normalised gains (for deployment) --")
        self.get_logger().info(
            f"  These are speed-independent. In your wall_follower, use:"
        )
        self.get_logger().info(
            f"    Kp_effective = Kp_norm / v²  =  {Kp_norm:.4f} / v²"
        )
        self.get_logger().info(
            f"    Kd_effective = Kd_norm / v²  =  {Kd_norm:.4f} / v²"
        )
        self.get_logger().info(
            f"  At v={v:.1f} m/s: Kp={Kp:.4f}, Kd={Kd:.4f}"
        )
        self.get_logger().info(
            f"  At v=1.0 m/s:   Kp={Kp_norm/1.0**2:.4f}, Kd={Kd_norm/1.0**2:.4f}"
        )
        self.get_logger().info(
            f"  At v=3.0 m/s:   Kp={Kp_norm/3.0**2:.4f}, Kd={Kd_norm/3.0**2:.4f}"
        )
        self.get_logger().info("")
        self.get_logger().info("  -- Tuning hints --")
        self.get_logger().info(
            "  Oscillation after step → increase Kd (or reduce desired_wn)."
        )
        self.get_logger().info(
            "  Sluggish / large overshoot → increase desired_wn (if below sat ceiling)."
        )
        self.get_logger().info(
            "  Persistent wall offset → add small Ki (integral term)."
        )
        self.get_logger().info(
            "  Servo chatter at speed → reduce desired_wn or increase zeta."
        )
        self.get_logger().info(sep)

    # ------------------------------------------------------------------
    def _compute_wall(self, msg: LaserScan):
        """OLS linear regression wall fit — consistent with wall_follower.py."""
        ranges = np.array(msg.ranges, dtype=np.float64)
        n      = len(ranges)
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

        wall_dist = abs(b) / math.sqrt(1.0 + m ** 2)
        return wall_dist, m

    # ------------------------------------------------------------------
    def _publish_drive(self, speed: float, steering: float):
        msg = AckermannDriveStamped()
        msg.drive.speed          = float(speed)
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
        node.get_logger().info("Interrupted — CSV flushed.")
    finally:
        node.destroy_node()


if __name__ == "__main__":
    main()
