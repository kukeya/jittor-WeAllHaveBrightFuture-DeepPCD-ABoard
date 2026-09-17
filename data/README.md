# Data layout

This directory is documentation only. Do not commit the official competition
archives, extracted meshes, point-cloud caches, split JSON files, predictions,
or submission ZIPs.

Starting with the official archives, run `scripts/prepare_data.py` from the
repository root. With the commands in the top-level README, it produces:

```text
data/
  prepared/
    train/dataset_train/shapenet/  # official training meshes
    test/                          # official noisy test point clouds
  splits/
    seed_20260726_v2/              # deterministic initial split
    all_15833_seed20260726_v1/     # full official training-list manifest
  train_cache/
    surface50000_seed20260726_n15833_v1/  # generated 50,000-point surfaces
```

The training YAML files use these relative paths and pin the expected split and
cache hashes. If your archives unpack differently, use the documented command
arguments to provide the correct roots; do not edit source code to add a local
machine path.
