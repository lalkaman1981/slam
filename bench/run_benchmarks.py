"""
Generates synthetic inputs, runs every binary from compiled/ several times and plots the times.
Each operation has one binary per variant: <op>_serial and <op>_simd run our code, <op>_opencv
runs the OpenCV counterpart (BENCHMARK.md lists how the methods differ).

usage: python3 run_benchmarks.py [--operations features pose mapping pipeline] [--sizes features=500,1000 ...]
                                 [--runs 5] [--seed 42] [--results results]
results: <results>/times.csv and <results>/{features,pose,mapping,pipeline}.png
"""
import argparse
import csv
import pathlib
import statistics
import struct
import subprocess
import zlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
COMPILED = HERE / "compiled"
INPUTS = HERE / "inputs"
RESULTS = HERE / "results"  # changed with --results

# operation -> variants, in legend order
VARIANTS = {
    "features": ["serial", "opencv"],
    "pose": ["serial", "opencv"],
    "mapping": ["serial", "opencv", "simd"],
    "pipeline": ["serial", "opencv"],
}
LABELS = {"serial": "Serial Implementation (ours)", "simd": "SIMD Implementation (ours)",
          "opencv": "OpenCV Implementation"}
COLORS = {"serial": "tab:blue", "opencv": "tab:orange", "simd": "tab:green"}

SIZES = {
    "features": list(range(500, 4000, 100)),   # side length of the square images
    "pose": list(range(500, 10001, 500)),      # correspondences
    "mapping": list(range(60, 1201, 60)),      # unknowns, 6 per keyframe
    "pipeline": list(range(500, 4000, 400)),   # side length, PIPELINE_FRAMES frames each
}
PIPELINE_FRAMES = 8

PLOTS = {
    "features": ("Serial vs OpenCV Feature Extraction and Matching Performance",
                 "Side length in pixels (square image, 2 frames)", "Extraction + Matching Time (ms)"),
    "pose": ("Serial vs OpenCV Pose Estimation Performance",
             "Number of correspondences (3D-3D pairs and 3D-2D observations)", "Pose Estimation Time (ms)"),
    "mapping": ("Serial vs SIMD vs OpenCV Mapping (Cholesky Solve) Performance",
                "System size (6 unknowns per keyframe)", "Cholesky Solve Time (ms)"),
    "pipeline": ("Serial vs OpenCV Whole Pipeline Performance",
                 f"Side length in pixels ({PIPELINE_FRAMES}-frame RGB-D sequence)", "Pipeline Time (ms)"),
}

# synthetic scene: a textured "roof" Z = PLANE_Z + max_i(a_i X + b_i Y) (metres) of three planes, so the
# points are not coplanar (a single plane leaves cv::estimateAffine3D's 12-dof fit undetermined),
# seen by a camera that moves sideways and turns a little between frames
PLANE_Z, ROOF, CELL = 2.4, [(0.25, 0.35), (0.25, -0.35), (-0.30, 0.0)], 0.09
STEP, YAW_STEP = np.array([0.04, 0.015, 0.0]), np.deg2rad(0.6)
DEPTH_SCALE = 5000.0  # raw units per metre, as person 2's DEPTH_SCALE (TUM)
POSE_K = (525.0, 525.0, 319.5, 239.5)  # 640x480 camera for the pose input


def write_png(path, image):
    """8-bit or 16-bit grayscale PNG with the standard library only"""
    height, width = image.shape
    depth = 16 if image.dtype == np.uint16 else 8
    rows = image.astype(">u2" if depth == 16 else np.uint8).view(np.uint8).reshape(height, -1)
    raw = np.hstack([np.zeros((height, 1), np.uint8), rows]).tobytes()  # filter byte 0 per row

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    header = struct.pack(">IIBBBBB", width, height, depth, 0, 0, 0, 0)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(raw, 6))
                     + chunk(b"IEND", b""))


def intrinsics(side):
    """the field of view stays the same at every size, so every image shows the same scene"""
    return 0.8 * side, 0.8 * side, side / 2.0, side / 2.0


def texture(x, y, seed):
    """random gray blocks with a smaller inner square in some of them: strong corners at any resolution"""
    ix, iy = np.floor(x / CELL).astype(np.int64), np.floor(y / CELL).astype(np.int64)
    h = ((ix * 73856093) ^ (iy * 19349663) ^ (seed * 83492791)) & 0x7FFFFFFF
    gray = 30 + h % 196
    fx, fy = x / CELL - ix, y / CELL - iy
    inner = (h // 196 % 3 == 0) & (fx > 0.3) & (fx < 0.7) & (fy > 0.3) & (fy < 0.7)
    return np.where(inner, 255 - gray, gray).astype(np.uint8)


def render_frame(side, k, seed, folder):
    """gray and depth image of frame k, written once and reused by every input that needs it"""
    gray_path, depth_path = folder / f"gray_{k}.png", folder / f"depth_{k}.png"
    if gray_path.exists() and depth_path.exists():
        return gray_path, depth_path
    fx, fy, cx, cy = intrinsics(side)
    yaw = YAW_STEP * k
    R = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])  # camera to world
    C = STEP * k
    u = (np.arange(side) - cx) / fx
    gray = np.empty((side, side), np.uint8)
    depth = np.empty((side, side), np.uint16)
    for top in range(0, side, 256):  # rows in chunks to keep memory small
        v = (np.arange(top, min(top + 256, side)) - cy) / fy
        dx, dy = np.meshgrid(u, v)
        # ray directions in the world for camera rays (dx, dy, 1), written out instead of a matmul
        wx, wy, wz = (R[i, 0] * dx + R[i, 1] * dy + R[i, 2] for i in range(3))
        # the region behind the roof is convex, so the first hit is the farthest of the three plane hits;
        # s equals the camera depth because camera rays have z = 1
        s = np.max([(PLANE_Z + a * C[0] + b * C[1] - C[2]) / (wz - a * wx - b * wy) for a, b in ROOF], axis=0)
        gray[top:top + len(v)] = texture(C[0] + s * wx, C[1] + s * wy, seed)
        depth[top:top + len(v)] = np.round(s * DEPTH_SCALE).astype(np.uint16)
    write_png(gray_path, gray)
    write_png(depth_path, depth)
    return gray_path, depth_path


def make_sequence(side, frames, seed):
    """manifest 'side fx fy cx cy depth_scale frames' plus one 'gray depth' line per frame"""
    path = INPUTS / f"sequence_{side}_{frames}_{seed}.txt"
    if path.exists():
        return path
    folder = INPUTS / f"scene_{side}_{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    lines = [" ".join(f"{x:g}" for x in (side, *intrinsics(side), DEPTH_SCALE, frames))]
    for k in range(frames):
        gray, depth = render_frame(side, k, seed, folder)
        lines.append(f"{gray.relative_to(INPUTS)} {depth.relative_to(INPUTS)}")
    path.write_text("\n".join(lines) + "\n")
    return path


def make_pose_input(n, seed):
    """n 3D-3D pairs (5 mm noise, 30% outliers) and n 3D-2D observations (1 px noise, 10% outliers)"""
    path = INPUTS / f"pose_{n}_{seed}.txt"
    if path.exists():
        return path
    rng = np.random.default_rng(seed)
    fx, fy, cx, cy = POSE_K
    rotvec = np.array([0.05, -0.08, 0.03])
    angle, axis = np.linalg.norm(rotvec), rotvec / np.linalg.norm(rotvec)
    Kx = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(angle) * Kx + (1 - np.cos(angle)) * Kx @ Kx
    t = np.array([0.12, -0.04, 0.06])

    def points():
        u, v, z = rng.uniform(20, 620, n), rng.uniform(20, 460, n), rng.uniform(1.0, 4.0, n)
        return np.column_stack([(u - cx) / fx * z, (v - cy) / fy * z, z])

    def transform(p):  # R p + t for each row, written out instead of a matmul
        return np.column_stack([p[:, 0] * R[i, 0] + p[:, 1] * R[i, 1] + p[:, 2] * R[i, 2] for i in range(3)]) + t

    a = points()
    b = transform(a) + rng.normal(0, 0.005, (n, 3))
    outliers = rng.random(n) < 0.3
    b[outliers] = np.column_stack([rng.uniform(-2, 2, n), rng.uniform(-2, 2, n), rng.uniform(1, 4, n)])[outliers]

    world = points()
    camera = transform(world)
    pixels = np.column_stack([fx * camera[:, 0] / camera[:, 2] + cx, fy * camera[:, 1] / camera[:, 2] + cy])
    pixels += rng.normal(0, 1.0, (n, 2))
    wrong = rng.random(n) < 0.1
    pixels[wrong] = np.column_stack([rng.uniform(0, 640, n), rng.uniform(0, 480, n)])[wrong]

    INPUTS.mkdir(exist_ok=True)
    with open(path, "w") as file:
        file.write(f"{fx} {fy} {cx} {cy}\n{n}\n")
        np.savetxt(file, np.hstack([a, b]), fmt="%.9g")
        file.write(f"{n}\n")
        np.savetxt(file, np.hstack([world, pixels]), fmt="%.9g")
    return path


def make_system(n, seed):
    """SPD normal matrix of a pose graph with n / 6 keyframes: chain and loop edges, prior on the first"""
    path = INPUTS / f"system_{n}_{seed}.bin"
    if path.exists():
        return path
    rng = np.random.default_rng(seed)
    nodes = n // 6
    edges = [(i, i + 1) for i in range(nodes - 1)] + [(i, i + 5) for i in range(0, nodes - 5, 3)]
    H = np.zeros((n, n))
    for i, j in edges:
        Ji, Jj = rng.normal(0, 1, (6, 6)), rng.normal(0, 1, (6, 6))
        si, sj = slice(6 * i, 6 * i + 6), slice(6 * j, 6 * j + 6)
        H[si, si] += Ji.T @ Ji
        H[sj, sj] += Jj.T @ Jj
        H[si, sj] += Ji.T @ Jj
        H[sj, si] += Jj.T @ Ji
    H[:6, :6] += 10 * np.eye(6)
    b = rng.normal(0, 1, n)
    INPUTS.mkdir(exist_ok=True)
    path.write_bytes(np.int64(n).tobytes() + H.astype("<f8").tobytes() + b.astype("<f8").tobytes())
    return path


def make_input(operation, size, seed):
    if operation == "features":
        return make_sequence(size, 2, seed)
    if operation == "pipeline":
        return make_sequence(size, PIPELINE_FRAMES, seed)
    if operation == "pose":
        return make_pose_input(size, seed)
    return make_system(size, seed)


def run_once(operation, variant, input_path):
    binary = COMPILED / f"{operation}_{variant}"
    output = subprocess.run([str(binary), "-i", str(input_path)], capture_output=True, text=True, check=True)
    return int(output.stdout.strip())


def plot(operation, medians, sizes):
    """one chart in the layout of the reference picture: time against input size, one line per implementation"""
    title, xlabel, ylabel = PLOTS[operation]
    fig, ax = plt.subplots(figsize=(10, 6))
    for variant in VARIANTS[operation]:
        ms = [medians[operation, variant, size] / 1e6 for size in sizes]
        ax.plot(sizes, ms, marker="o", markersize=4, linewidth=1.8, color=COLORS[variant], label=LABELS[variant])
    ax.set_title(title, fontsize=15)
    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(s) if i % 2 == 0 or len(sizes) <= 12 else "" for i, s in enumerate(sizes)], rotation=45)
    ax.set_ylim(bottom=0)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=12)
    fig.tight_layout()
    fig.savefig(RESULTS / f"{operation}.png", dpi=150)
    plt.close(fig)


def parse_sizes(items):
    """'features=500,1000' -> {'features': [500, 1000]}"""
    sizes = dict(SIZES)
    for item in items or []:
        operation, values = item.split("=")
        sizes[operation] = [int(v) for v in values.split(",")]
    return sizes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--operations", nargs="+", choices=list(VARIANTS), default=list(VARIANTS))
    parser.add_argument("--sizes", nargs="*", help="override per operation, e.g. features=500,1000 pose=500")
    parser.add_argument("--runs", type=int, default=5, help="measured runs, one more warm-up run is not counted")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results", default="results", help="output folder inside bench/, e.g. results/quick")
    args = parser.parse_args()

    global RESULTS
    RESULTS = HERE / args.results
    RESULTS.mkdir(parents=True, exist_ok=True)
    sizes = parse_sizes(args.sizes)

    medians = {}
    with open(RESULTS / "times.csv", "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["operation", "variant", "size", "run", "time_ns"])
        for operation in args.operations:
            for size in sizes[operation]:
                path = make_input(operation, size, args.seed)
                for variant in VARIANTS[operation]:
                    run_once(operation, variant, path)  # warm-up
                    times = [run_once(operation, variant, path) for _ in range(args.runs)]
                    writer.writerows([operation, variant, size, i, t] for i, t in enumerate(times))
                    medians[operation, variant, size] = statistics.median(times)
                    print(f"{operation:9} {variant:7} {size:>6}  median {medians[operation, variant, size] / 1e6:10.3f} ms",
                          flush=True)
            plot(operation, medians, sizes[operation])

    print(f"results in {RESULTS}")


if __name__ == "__main__":
    main()
