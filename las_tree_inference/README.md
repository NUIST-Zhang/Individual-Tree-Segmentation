# LAS tree-cluster inference

This folder is the minimal, self-contained inference layer for the point-cloud
experiment. It deliberately leaves the upstream PointMLP training code intact.

## Pipeline

```text
input LAS
  -> local-maximum tree-top candidates
  -> confirmed tree tops by XY spacing
  -> top-seeded downhill XY clustering
  -> each valid cluster sampled to 1,024 XYZ points
  -> PointMLP Elite prediction
  -> one LAS output per predicted class
```

## Required project components

This folder relies on two existing upstream directories at the project root:

- `classification_ModelNet40/models/pointmlp.py`: PointMLP Elite model
- `pointnet2_ops_lib/`: CUDA PointNet++ furthest-point-sampling extension

Its own required files are:

- `run_inference.py`: single command-line entry point
- `tree_clustering.py`: pure tree-top and constrained clustering functions
- `last_checkpoint.pth`: the supplied three-class model weights

## Environment

Install the upstream environment, then add LAS support:

```bash
conda env create -f environment.yml
conda activate pointmlp
pip install laspy
pip install pointnet2_ops_lib/.
```

CUDA is used automatically when PyTorch can access a GPU; otherwise PyTorch
runs on CPU. The PointNet++ sampling extension normally requires a compatible
CUDA build, so GPU inference is the expected configuration.

## Run

From the repository root:

```bash
python las_tree_inference/run_inference.py "F:\\path\\to\\input.las" --output-dir "F:\\path\\to\\output"
```

The included checkpoint has **three outputs**. Replace the default class names
with the actual training-label order before interpreting results:

```bash
python las_tree_inference/run_inference.py input.las \
  --class-names forest,farmland,buildings \
  --output-dir output
```

The output directory contains only classes with at least one accepted cluster.
Each file stores XYZ coordinates; original LAS attributes are not preserved.

## Important parameters

- `--height-threshold`: minimum Z coordinate for a candidate tree top.
- `--top-search-radius`: XY radius used to test whether a point is a local maximum.
- `--minimum-top-spacing`: XY separation used to consolidate duplicate tree tops.
- `--cluster-epsilon`: neighborhood radius during downhill cluster expansion.
- `--min-cluster-points`: discards very small clusters before model inference.

Defaults are inherited from the previous experimental scripts and must be tuned
to the LAS coordinate units, canopy density, and the preprocessing used during
model training.
