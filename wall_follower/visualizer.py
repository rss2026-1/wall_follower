#!/usr/bin/env python3
"""
Plot wall distance error vs time from a ROS2 bag.
Reads /scan_dist (std_msgs/msg/Float32) and plots against bag time.

Usage:
    python3 plot_wall_distance.py <path_to_bag.db3>

Example:
    python3 plot_wall_distance.py ~/racecar_ws/bags/my_run_0.db3
"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from rosbags.rosbag2 import Reader
from rosbags.typesys import Stores, get_typestore

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 plot_wall_distance.py <path_to_bag.db3>")
        sys.exit(1)

    bag_path = sys.argv[1]
    topic = "/scan_dist"

    typestore = get_typestore(Stores.ROS2_HUMBLE)

    ts, vals = [], []

    with Reader(bag_path) as reader:
        # print available topics
        print("Topics in bag:")
        for conn in reader.connections:
            print(f"  {conn.topic}  [{conn.msgtype}]")

        for connection, timestamp, rawdata in reader.messages():
            if connection.topic == topic:
                msg = typestore.deserialize_cdr(rawdata, connection.msgtype)
                ts.append(timestamp * 1e-9)   # nanoseconds → seconds
                vals.append(float(msg.data))

    if len(ts) == 0:
        print(f"No messages found on {topic}. Check topic name above.")
        sys.exit(1)

    ts   = np.array(ts)
    vals = np.array(vals)
    ts  -= ts[0]   # start from t=0

    print(f"\nExtracted {len(ts)} messages from {topic}")
    print(f"Duration: {ts[-1]:.2f} s")
    print(f"Mean distance error: {vals.mean():.4f} m")
    print(f"Std:  {vals.std():.4f} m")
    print(f"Min:  {vals.min():.4f} m")
    print(f"Max:  {vals.max():.4f} m")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.plot(ts, vals, color="blue", linewidth=1.2, alpha=0.9, label="wall distance error")
    ax.axhline(0, linewidth=0.8, linestyle="--")

    # shade error band
    ax.fill_between(ts, vals, 0,
                    where=(vals > 0), alpha=0.15, color="blue", label="too far")
    ax.fill_between(ts, vals, 0,
                    where=(vals < 0), alpha=0.15, color="orange", label="too close")

    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Distance error [m]")
    ax.set_title("Wall Distance Error vs Time — MIT RACECAR", fontsize=13)

    ax.legend()
    ax.grid(True, linewidth=0.7)

    plt.tight_layout()
    out = bag_path.replace(".db3", "_wall_distance.png")
    plt.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    print(f"\nSaved plot to: {out}")
    plt.show()

if __name__ == "__main__":
    main()
