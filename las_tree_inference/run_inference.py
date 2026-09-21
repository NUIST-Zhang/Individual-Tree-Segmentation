"""Classify individual tree clusters from one LAS file with PointMLP Elite."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import laspy
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "classification_ModelNet40"))
from models.pointmlp import pointMLPElite  # noqa: E402
from tree_clustering import UNCLASSIFIED, cluster_from_tree_tops, confirm_tree_tops, find_possible_tree_tops  # noqa: E402


def sample_points(points: np.ndarray, count: int, generator: np.random.Generator) -> np.ndarray:
    """Return exactly ``count`` points, sampling with replacement when needed."""
    replace = len(points) < count
    indices = generator.choice(len(points), size=count, replace=replace)
    return points[indices]


def load_model(checkpoint_path: Path, class_count: int, device: torch.device) -> torch.nn.Module:
    model = pointMLPElite(num_classes=class_count).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("net", checkpoint)
    state_dict = {key[7:] if key.startswith("module.") else key: value for key, value in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def write_xyz_las(points: np.ndarray, output_path: Path) -> None:
    header = laspy.LasHeader(version="1.2", point_format=1)
    output = laspy.LasData(header)
    output.x, output.y, output.z = points[:, 0], points[:, 1], points[:, 2]
    output.write(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_las", type=Path, help="Input LAS point cloud")
    parser.add_argument("--checkpoint", type=Path, default=Path(__file__).with_name("last_checkpoint.pth"))
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).with_name("output"))
    parser.add_argument("--class-names", default="class_0,class_1,class_2", help="Comma-separated names, matching model class order")
    parser.add_argument("--height-threshold", type=float, default=5.0, help="Minimum Z for a possible tree top")
    parser.add_argument("--top-search-radius", type=float, default=0.3)
    parser.add_argument("--minimum-top-spacing", type=float, default=1.5)
    parser.add_argument("--cluster-epsilon", type=float, default=0.7)
    parser.add_argument("--min-cluster-points", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    class_names = [name.strip() for name in args.class_names.split(",") if name.strip()]
    if not args.input_las.is_file():
        raise FileNotFoundError(f"Input LAS does not exist: {args.input_las}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {args.checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = load_model(args.checkpoint, len(class_names), device)

    las = laspy.read(args.input_las)
    points = np.column_stack((las.x, las.y, las.z)).astype(np.float32)
    possible_tops = find_possible_tree_tops(points, args.height_threshold, args.top_search_radius)
    confirmed_tops = confirm_tree_tops(possible_tops, args.minimum_top_spacing)
    labels = cluster_from_tree_tops(points, confirmed_tops, args.cluster_epsilon, args.minimum_top_spacing)
    print(f"Possible tops: {len(possible_tops)}; confirmed tops: {len(confirmed_tops)}")

    grouped_points: dict[int, list[np.ndarray]] = defaultdict(list)
    rng = np.random.default_rng(args.seed)
    for cluster_id in np.unique(labels):
        if cluster_id == UNCLASSIFIED:
            continue
        cluster_points = points[labels == cluster_id]
        if len(cluster_points) < args.min_cluster_points:
            continue
        model_points = sample_points(cluster_points, 1024, rng)
        tensor = torch.from_numpy(model_points.T).unsqueeze(0).to(device)
        with torch.no_grad():
            predicted_class = int(model(tensor).argmax(dim=1).item())
        grouped_points[predicted_class].append(cluster_points)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for class_id, clusters in grouped_points.items():
        output_points = np.vstack(clusters)
        name = class_names[class_id]
        output_path = args.output_dir / f"{name}.las"
        write_xyz_las(output_points, output_path)
        print(f"{name}: {len(clusters)} clusters, {len(output_points)} points -> {output_path}")

    if not grouped_points:
        print("No cluster met --min-cluster-points; no output LAS file was written.")


if __name__ == "__main__":
    main()
