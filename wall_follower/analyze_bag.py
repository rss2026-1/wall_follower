#!/usr/bin/env python3
"""
Wall follower rosbag analysis.

USAGE:
  python3 analyze_bag.py <bag_dir> [--side -1] [--desired-distance 1.0]

  --side              -1 (right wall) or +1 (left wall)   default: -1
  --desired-distance  target wall distance in meters        default: 1.0
  --scan-topic        default: /scan
  --drive-topic       default: /vesc/low_level/input/navigation
"""

import argparse
import sys
import math
import numpy as np

try:
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message
except ImportError:
    sys.exit("ERROR: source your ROS2 workspace first.")

import matplotlib.pyplot as plt

START_IDX = 27
END_IDX   = 38
MAX_ANGLE = 0.34


def compute_metrics(msg_ranges, angle_increment, side, desired_distance):
    ranges_full = np.array(msg_ranges, dtype=np.float64)

    if side == -1:
        ranges = ranges_full[START_IDX : END_IDX + 1]
        angles = np.array([angle_increment * i - (3 * np.pi / 4)
                           for i in range(START_IDX, END_IDX + 1)])
    else:
        ranges = ranges_full[-END_IDX - 1 : -START_IDX]
        angles = np.array([angle_increment * i + (3 * np.pi / 4)
                           for i in range(-END_IDX, -START_IDX + 1)])

    if len(ranges) == 0 or not np.any(np.isfinite(ranges)):
        return None

    closest_idx = np.argmin(ranges)
    x = ranges * np.cos(angles)
    y = ranges * np.sin(angles)

    try:
        oc    = np.polyfit(x, y, 1)
        coeff = np.polyfit(x, y, 1)
    except (np.linalg.LinAlgError, ValueError):
        return None

    coeff[1] = (y[closest_idx] - coeff[0] * x[closest_idx]
                - side * (coeff[0] ** 2 + (desired_distance * 1.2)) ** 0.5)
    angle_offset = float(np.arctan(coeff)[0])

    d = float(ranges[closest_idx]) - desired_distance
    distance_factor  = 2/3 * (np.pi/2 - angle_offset) * np.arctan((5 * d) ** 3)
    angle_factor     = 2/3 * 1 / (1 + abs(d)) * angle_offset

    if len(ranges_full) > 52:
        look_ahead = float(min(ranges_full[50:52]))
        look_ahead_factor = (-np.pi / (desired_distance * look_ahead)
                             if look_ahead < 2 * desired_distance else 0.0)
    else:
        look_ahead = float("nan")
        look_ahead_factor = 0.0

    raw_turn = side * (distance_factor + look_ahead_factor) + angle_factor
    steering = float(np.clip(raw_turn, -MAX_ANGLE, MAX_ANGLE))

    return {
        "distance_error":    d,
        "wall_dist":         float(ranges[closest_idx]),
        "angle_offset":      angle_offset,
        "distance_factor":   float(distance_factor),
        "angle_factor":      float(angle_factor),
        "look_ahead_factor": float(look_ahead_factor),
        "computed_steering": steering,
        "saturated":         abs(raw_turn) > MAX_ANGLE,
    }


def read_bag(bag_path, scan_topic, drive_topic):
    storage_options  = rosbag2_py.StorageOptions(uri=bag_path, storage_id="sqlite3")
    converter_options = rosbag2_py.ConverterOptions("cdr", "cdr")
    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    type_map = {m.name: m.type for m in reader.get_all_topics_and_types()}
    reader.set_filter(rosbag2_py.StorageFilter(topics=[scan_topic, drive_topic]))

    while reader.has_next():
        topic, data, ts_ns = reader.read_next()
        msg = deserialize_message(data, get_message(type_map[topic]))
        yield ts_ns, topic, msg

    del reader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("bag")
    parser.add_argument("--side",             type=int,   default=-1,  help="+1=left -1=right")
    parser.add_argument("--desired-distance", type=float, default=1.0, dest="desired_distance")
    parser.add_argument("--scan-topic",       type=str,   default="/scan", dest="scan_topic")
    parser.add_argument("--drive-topic",      type=str,
                        default="/vesc/low_level/input/navigation", dest="drive_topic")
    args = parser.parse_args()

    scan_records  = []
    drive_records = []

    print(f"Reading {args.bag} ...")
    for ts_ns, topic, msg in read_bag(args.bag, args.scan_topic, args.drive_topic):
        if topic == args.scan_topic:
            m = compute_metrics(msg.ranges, msg.angle_increment,
                                args.side, args.desired_distance)
            if m is not None:
                m["ts_ns"] = ts_ns
                scan_records.append(m)
        else:
            drive_records.append({
                "ts_ns":   ts_ns,
                "steer":   msg.drive.steering_angle,
                "speed":   msg.drive.speed,
            })

    if not scan_records:
        sys.exit("No usable scan messages found.")

    t0 = scan_records[0]["ts_ns"]
    scan_t  = np.array([(r["ts_ns"] - t0) * 1e-9 for r in scan_records])
    drive_t = np.array([(r["ts_ns"] - t0) * 1e-9 for r in drive_records])

    def col(key):
        return np.array([r[key] for r in scan_records], dtype=np.float64)

    dist_error  = col("distance_error")
    wall_dist   = col("wall_dist")
    comp_steer  = col("computed_steering")
    dist_factor = col("distance_factor")
    ang_factor  = col("angle_factor")
    la_factor   = col("look_ahead_factor")
    saturated   = np.array([r["saturated"] for r in scan_records])

    actual_steer = np.array([r["steer"] for r in drive_records])
    actual_speed = np.array([r["speed"] for r in drive_records])

    rms_error  = float(np.sqrt(np.mean(dist_error ** 2)))
    mae        = float(np.mean(np.abs(dist_error)))
    peak_error = float(np.max(np.abs(dist_error)))
    sat_pct    = float(saturated.mean()) * 100
    mean_speed = float(actual_speed.mean()) if len(actual_speed) else float("nan")

    print(f"  RMS error  : {rms_error:.4f} m")
    print(f"  MAE        : {mae:.4f} m")
    print(f"  Peak error : {peak_error:.4f} m")
    print(f"  Saturation : {sat_pct:.1f}%")
    print(f"  Mean speed : {mean_speed:.2f} m/s")

    # ---- 4-panel plot ----
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f"Wall Follower | side={'L' if args.side==1 else 'R'}  "
                 f"desired={args.desired_distance} m  RMS err={rms_error:.3f} m")

    # 1. Distance error
    ax = axes[0]
    ax.axhline(0, color="k", linewidth=0.8, linestyle="--")
    ax.plot(scan_t, dist_error, linewidth=0.9)
    ax.set_ylabel("Distance error (m)")
    ax.set_title(f"Distance error (closest range − desired)   RMS={rms_error:.3f} m  peak={peak_error:.3f} m")
    ax.grid(True, alpha=0.3)

    # 2. Wall distance vs desired
    ax = axes[1]
    ax.plot(scan_t, wall_dist, linewidth=0.9, label="wall dist")
    ax.axhline(args.desired_distance, color="r", linewidth=1.0,
               linestyle="--", label=f"desired ({args.desired_distance} m)")
    ax.set_ylabel("Distance (m)")
    ax.set_title("Wall distance")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3. Steering
    ax = axes[2]
    ax.plot(scan_t, comp_steer, linewidth=0.9, label="computed")
    if len(drive_t):
        ax.plot(drive_t, actual_steer, linewidth=0.9, alpha=0.7, label="actual cmd")
    ax.axhline( MAX_ANGLE, color="gray", linewidth=0.7, linestyle=":")
    ax.axhline(-MAX_ANGLE, color="gray", linewidth=0.7, linestyle=":")
    ax.set_ylabel("Steering (rad)")
    ax.set_title(f"Steering angle   saturation={sat_pct:.0f}%")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 4. Controller components
    ax = axes[3]
    ax.plot(scan_t, dist_factor, linewidth=0.9, label="distance_factor")
    ax.plot(scan_t, ang_factor,  linewidth=0.9, label="angle_factor")
    ax.plot(scan_t, la_factor,   linewidth=0.9, label="look_ahead_factor")
    ax.axhline(0, color="k", linewidth=0.6, linestyle="--")
    ax.set_ylabel("Contribution (rad)")
    ax.set_xlabel("Time (s)")
    ax.set_title("Steering components")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
