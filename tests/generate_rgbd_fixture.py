#!/usr/bin/env python3
"""Generate a deterministic TUM-style RGB-D plane sequence without third-party packages.

This tests image loading, feature extraction, tracking and mapping with known motion.
It is a controlled fixture, not a real-world SLAM accuracy benchmark.
Calibration: fx=fy=300, cx=191.5, cy=127.5; 384x256; depth scale=5000.
"""

import argparse
from pathlib import Path
import random
import struct


def generate(output: Path) -> None:
    width, height, margin = 384, 256, 32
    source_width = width + 2 * margin
    rng = random.Random(23063)
    source = bytearray(rng.randrange(256) for _ in range(source_width * height))
    texture = bytearray(source)
    # Small blur creates stable corners while keeping the scene richly textured.
    for y in range(1, height - 1):
        for x in range(1, source_width - 1):
            total = 0
            for dy, wy in ((-1, 1), (0, 2), (1, 1)):
                for dx, wx in ((-1, 1), (0, 2), (1, 1)):
                    total += wx * wy * source[(y + dy) * source_width + x + dx]
            texture[y * source_width + x] = total // 16

    (output / "rgb").mkdir(parents=True, exist_ok=True)
    (output / "depth").mkdir(parents=True, exist_ok=True)
    depth = struct.pack(">H", 10000) * (width * height)  # PGM 16-bit samples use network order.
    associations, rgb_index, depth_index, groundtruth = [], [], [], []
    for index, shift in enumerate((0, 4, 8, 12, 16, 12, 8, 4, 0)):
        stamp = f"{index / 30.0:.9f}"
        rgb_name = f"rgb/{index:04d}.pgm"
        depth_name = f"depth/{index:04d}.pgm"
        pixels = b"".join(texture[y * source_width + margin - shift:
                                  y * source_width + margin - shift + width]
                          for y in range(height))
        (output / rgb_name).write_bytes(f"P5\n{width} {height}\n255\n".encode() + pixels)
        (output / depth_name).write_bytes(f"P5\n{width} {height}\n65535\n".encode() + depth)
        associations.append(f"{stamp} {rgb_name} {stamp} {depth_name}\n")
        rgb_index.append(f"{stamp} {rgb_name}\n")
        depth_index.append(f"{stamp} {depth_name}\n")
        groundtruth.append(f"{stamp} {-shift * 2.0 / 300.0:.9f} 0 0 0 0 0 1\n")
    (output / "associations.txt").write_text("".join(associations))
    (output / "rgb.txt").write_text("".join(rgb_index))
    (output / "depth.txt").write_text("".join(depth_index))
    (output / "groundtruth.txt").write_text("".join(groundtruth))
    (output / "calibration.txt").write_text(
        "width=384\nheight=256\nfx=300\nfy=300\ncx=191.5\ncy=127.5\ndepth_scale=5000\n")
    print(f"Wrote 9 RGB-D frames to {output}; calibration fx=fy=300, cx=191.5, cy=127.5")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="destination directory for the fixture")
    generate(parser.parse_args().output)
