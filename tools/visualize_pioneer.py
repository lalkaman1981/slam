#!/usr/bin/env python3
"""Replay this sequence and compare existing serial/SIMD results; no SLAM code changes.

Requires numpy, Pillow and imageio-ffmpeg. Run from the repository root after
slam_rgbd DATASET output/pioneer_360/{serial,serial_simd} --mode MODE.
"""
import csv
import json
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from evaluate_trajectory import evaluate, read_trajectory

dataset = Path("data/rgbd_dataset_freiburg2_pioneer_360")
output = Path("output")
runs = output / "pioneer_360"
rgb = [line.split() for line in (dataset / "rgb.txt").read_text().splitlines()
       if line.strip() and not line.startswith("#")]
times = np.array([float(row[0]) for row in rgb])
truth = np.loadtxt(dataset / "groundtruth.txt")
# The supplied ground truth repeats three timestamps; leave the dataset untouched.
truth = truth[np.unique(truth[:, 0], return_index=True)[1]]
truth = truth[(truth[:, 0] >= times[0]) & (truth[:, 0] <= times[-1])]
modes, colors = ["serial", "serial_simd"], ["#42d6d0", "#ffbf69"]
stats, paths, metrics = {}, {}, {}
key = lambda time: f"{float(time):.6f}"
for mode in modes:
    rows = list(csv.DictReader((runs / mode / "frames.csv").open()))
    stats[mode] = {key(row["timestamp"]): row for row in rows}
    trajectory = read_trajectory(runs / mode / "trajectory.txt")
    score = evaluate(truth, trajectory)
    rotation = np.array(score["rotation_estimate_to_reference"])
    translation = np.array(score["translation_estimate_to_reference_metres"])
    aligned = trajectory[:, 1:4] @ rotation.T + translation
    paths[mode] = {key(t): p for t, p in zip(trajectory[:, 0], aligned)}
    metrics[mode] = dict(paired=len(rows), tracked=len(trajectory), ate_m=score["rmse_metres"],
                        **{stage + "_s": sum(float(r[stage + "_ms"]) for r in rows) / 1000
                           for stage in ["features", "tracking", "mapping"]})
assert list(stats[modes[0]]) == list(stats[modes[1]]), "Mode comparisons require identical frames"
summary = dict(dataset=str(dataset), source_frames=len(rgb), duration_s=times[-1] - times[0],
               unpaired=len(rgb) - len(stats["serial"]), modes=metrics,
               note="Single native build, serial vs SIMD mapping. Rigid aligned ATE on tracked poses only.")
(runs / "comparison.json").write_text(json.dumps(summary, indent=2) + "\n")

W, H, fps = 1280, 800, 30
bg, panel, muted = "#101722", "#192331", "#9bacbf"
font_path = "/System/Library/Fonts/Supplemental/Arial.ttf"
font = lambda size: ImageFont.truetype(font_path, size)
small, body, title = font(16), font(20), font(30)
base = Image.new("RGB", (W, H), bg)
draw = ImageDraw.Draw(base)
draw.text((24, 18), "PIONEER 360  /  RGB-D SLAM", font=title, fill="white")
draw.text((24, 58), "Serial vs SIMD  |  same frames  |  recorded timing  |  final trajectory replay",
          font=body, fill=muted)
draw.text((24, 93), "CAMERA VIEW", font=small, fill=muted)
draw.text((700, 93), "TRAJECTORY  /  WORLD X-Y (metres)", font=small, fill=muted)
draw.rectangle((700, 116, 1256, 596), fill=panel)
lo, hi = truth[:, 1:3].min(axis=0), truth[:, 1:3].max(axis=0)
center, span = (lo + hi) / 2, max(hi - lo) + 0.8
def pixel(points):
    xy = np.asarray(points)[..., :2]
    return np.round((xy - center) * [420 / span, -420 / span] + [978, 356]).astype(int)

for axis in range(2):
    for value in np.arange(np.ceil(center[axis] - span / 2), center[axis] + span / 2, 1):
        endpoints = np.array([center - span / 2, center + span / 2])
        endpoints[:, axis] = value
        points = [tuple(p) for p in pixel(endpoints)]
        draw.line(points, fill="#293749")
        draw.text((points[0][0] + 3, points[0][1] - 18), f"{value:.0f}", font=small, fill=muted)
gt = truth[::8]
gt_pixels = [tuple(p) for p in pixel(gt[:, 1:3])]
draw.line(gt_pixels, fill="#415061", width=2)
draw.text((710, 570), "Ground truth", font=small, fill="white")
draw.text((850, 570), "Serial", font=small, fill=colors[0])
draw.text((930, 570), "SIMD", font=small, fill=colors[1])
draw.text((710, 613), "Rigid alignment; gaps are not filled with estimated poses", font=small, fill=muted)
trail = Image.new("RGBA", (W, H))
trail_draw = ImageDraw.Draw(trail)
previous, tracked = dict.fromkeys(modes), dict.fromkeys(modes, 0)
last_index, canvas = -1, None
writer = imageio_ffmpeg.write_frames(str(output / "pioneer_360.mp4"), (W, H), fps=fps,
    codec="libx264", pix_fmt_in="rgb24", pix_fmt_out="yuv420p",
    output_params=["-crf", "20", "-movflags", "+faststart"])
writer.send(None)
try:
    for frame in range(int(np.ceil((times[-1] - times[0]) * fps)) + 1):
        timestamp = min(times[0] + frame / fps, times[-1])
        index = int(np.searchsorted(times, timestamp, side="right") - 1)
        if index != last_index:
            # Consume every source frame even if multiple images fall in one video tick.
            for source in range(last_index + 1, index + 1):
                for mode, color in zip(modes, colors):
                    point = paths[mode].get(key(times[source]))
                    current = tuple(pixel(point)) if point is not None else None
                    if current is not None:
                        tracked[mode] += 1
                        trail_draw.ellipse((current[0]-2, current[1]-2, current[0]+2, current[1]+2), fill=color)
                        if previous[mode] is not None:
                            trail_draw.line([previous[mode], current], fill=color, width=3)
                    previous[mode] = current
            canvas = base.copy()
            with Image.open(dataset / rgb[index][1]) as image:
                canvas.paste(image.convert("RGB"), (24, 116))
            d = ImageDraw.Draw(canvas)
            end = int(np.searchsorted(gt[:, 0], times[index], side="right"))
            if end > 1:
                d.line(gt_pixels[:end], fill="white", width=2)
                x, y = gt_pixels[end - 1]
                d.ellipse((x-5, y-5, x+5, y+5), fill="white")
                qx, qy, qz, qw = gt[end - 1, 4:8]
                forward = np.array([2*(qx*qz+qw*qy), 2*(qy*qz-qw*qx)])
                if np.linalg.norm(forward) > 1e-6:
                    tip = (x, y) + forward / np.linalg.norm(forward) * [22, -22]
                    d.line([(x, y), tuple(tip)], fill="white", width=3)
            canvas.paste(trail, (0, 0), trail)
            d = ImageDraw.Draw(canvas)
            d.text((24, 613), f"{times[index]-times[0]:05.1f} / {times[-1]-times[0]:.1f} s   |   RGB {index+1}/{len(rgb)}",
                   font=body, fill="white")
            for n, (mode, color) in enumerate(zip(modes, colors)):
                x = 24 + n * 632
                row = stats[mode].get(key(times[index]))
                status = row["status"] if row else "UNPAIRED"
                status_color = "#ff7c86" if status == "LOST" else muted if row is None else color
                m = metrics[mode]
                d.rounded_rectangle((x, 651, x+600, 752), radius=10, fill=panel)
                d.text((x+16, 661), f"{mode.upper()}   {status}", font=body, fill=status_color)
                d.text((x+16, 691), f"Tracked so far: {tracked[mode]}   |   inliers: {row['inliers'] if row else '-'}",
                       font=small, fill="white")
                d.text((x+16, 720), f"Full run: {m['tracked']}/{m['paired']} tracked   |   mapping {m['mapping_s']*1000:.1f} ms",
                       font=small, fill=muted)
            d.text((24, 770), "LOST = tracking failed    UNPAIRED = no depth pair    White = ground truth, not SLAM",
                   font=small, fill=muted)
            last_index = index
        writer.send(np.asarray(canvas))
finally:
    writer.close()
print(output / "pioneer_360.mp4")
