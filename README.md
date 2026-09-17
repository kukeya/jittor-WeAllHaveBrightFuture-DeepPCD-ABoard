# Deep Learning-Based 3D Point Cloud Denoising

## Method Design and Implementation — Track A

This repository contains the Track A implementation submitted to the Sixth CG Graphics AI Challenge. It uses three released Jittor checkpoints and a two-stage weighted ensemble for whole-cloud denoising.

The submitted Track A result was **Total 81.05**, **CD 69.86**, and **P2S 92.25**. Reproducing the result requires the official data, the released checkpoints, and the settings below.

## Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Inference](#inference)
- [Results and Reproducibility](#results-and-reproducibility)
- [License and Acknowledgement](#license-and-acknowledgement)

## Requirements

| Component | Tested version |
| --- | --- |
| OS | Ubuntu 22.04 |
| GPU | NVIDIA RTX 4090 |
| CUDA | 12.4 |
| Python | 3.10 |
| Jittor | 1.3.10.0 |
| Compiler | g++ 10 |

Released inference checkpoints:

~~~text
checkpoints/control_e32_step_00050688.pkl
checkpoints/control_e34_step_00053856.pkl
checkpoints/noisyfit_e34_step_00053856.pkl
~~~

Verify them before use:

~~~bash
(cd checkpoints && sha256sum -c SHA256SUMS)
~~~

## Installation

Create the environment from the repository root. Jittor compiles and caches CUDA operators on first use, so JITTOR_HOME must be writable. Change CUDA_HOME if CUDA is installed elsewhere.

~~~bash
conda env create -f environment.yaml
conda activate jittor

export cc_path="$(command -v g++)"
export nvcc_path=/usr/local/cuda-12.4/bin/nvcc
export CUDA_HOME=/usr/local/cuda-12.4
export JITTOR_HOME=/absolute/writable/path/jittor_cache
export conv_opt=1
mkdir -p "$JITTOR_HOME"

python scripts/check_environment.py --environment-file environment.yaml --cc-path "$cc_path" --nvcc-path "$nvcc_path" --cuda-home "$CUDA_HOME"

export WORK_ROOT=/absolute/writable/path/point_denoising_work
mkdir -p "$WORK_ROOT"
~~~

Continue only after the checker prints ENVIRONMENT_OK. Re-export the five environment variables in each new terminal before training or inference.

## Data Preparation

The official data are not redistributed here. Set the archive paths supplied by the competition organizer, then prepare the training and noisy-test trees.

~~~bash
export TRAIN_DATASET=/path/to/dataset_train.tar.gz
export TEST_DATASET=/path/to/dataset_test_noisy.zip

python scripts/prepare_data.py --train-archive "$TRAIN_DATASET" --test-archive "$TEST_DATASET" --output-dir data/prepared --split-dir data/splits/seed_20260726_v2 --seed 20260726 --val-ratio 0.05

python scripts/build_all_train_split.py --source-split data/splits/seed_20260726_v2 --output-dir data/splits/all_15833_seed20260726_v1

python scripts/build_train_surface_cache.py --mesh-root data/prepared/train/dataset_train/shapenet --train-split data/splits/all_15833_seed20260726_v1/train.json --output-dir data/train_cache/surface50000_seed20260726_n15833_v1 --select-count 15833 --num-points 50000 --seed 20260726 --workers 16
~~~

The deterministic clean-surface cache contains 15,833 arrays of shape (50000, 3). It is required for training, but not for checkpoint inference.

## Training

The two training arms use all 15,833 meshes, a patch size of 1000, batch size 10, 34 epochs, a learning rate of 5e-4, and seed 20260726. They can run on separate GPUs.

~~~bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/train.py --config configs/train/pgd_all15833_starter_laplace_condgate_huber75_d010_e34_seed20260726.yaml --run-dir "$WORK_ROOT/runs/control"

CUDA_VISIBLE_DEVICES=1 python -u scripts/train.py --config configs/train/pgd_all15833_gate_e34_noisyfit_cube24_seed20260726.yaml --run-dir "$WORK_ROOT/runs/noisyfit"
~~~

Each epoch has 1,584 steps. Resume an interrupted run by appending --resume /path/to/checkpoint.pkl to its original command.

## Inference

The commands below reproduce the released three-checkpoint inference recipe. The expected config digests allow checkpoint replay from data/prepared/test without building the multi-gigabyte training cache.

~~~bash
python scripts/denoise.py --config configs/train/pgd_all15833_starter_laplace_condgate_huber75_d010_e34_seed20260726.yaml --checkpoint checkpoints/control_e32_step_00050688.pkl --expected-config-sha256 981f6edbafc1bdc312c6ae3dfa7538965315b70cda2f85bd1bd6fbad27428c27 --input-dir data/prepared/test --output-dir "$WORK_ROOT/pred/control_e32" --patch-size 1000 --seed-k 6.0 --patch-batch-size 20 --niters 1 --normalization-mode noisy_max --fusion-mode hard_best --iteration-damping 1.0

python scripts/denoise.py --config configs/train/pgd_all15833_starter_laplace_condgate_huber75_d010_e34_seed20260726.yaml --checkpoint checkpoints/control_e34_step_00053856.pkl --expected-config-sha256 981f6edbafc1bdc312c6ae3dfa7538965315b70cda2f85bd1bd6fbad27428c27 --input-dir data/prepared/test --output-dir "$WORK_ROOT/pred/control_e34" --patch-size 1000 --seed-k 6.0 --patch-batch-size 20 --niters 1 --normalization-mode noisy_max --fusion-mode hard_best --iteration-damping 1.0

python scripts/denoise.py --config configs/train/pgd_all15833_gate_e34_noisyfit_cube24_seed20260726.yaml --checkpoint checkpoints/noisyfit_e34_step_00053856.pkl --expected-config-sha256 9e6cf10fc272a92522fd979369137ca8b7778428906c90e2b0e7fea213e7992e --input-dir data/prepared/test --output-dir "$WORK_ROOT/pred/noisyfit_e34" --patch-size 1000 --seed-k 6.0 --patch-batch-size 20 --niters 1 --normalization-mode noisy_max --fusion-mode hard_best --iteration-damping 1.0
~~~

Fuse the three prediction directories in the submitted order.

~~~bash
python scripts/ensemble_predictions.py --pred-a "$WORK_ROOT/pred/control_e32" --pred-b "$WORK_ROOT/pred/control_e34" --output-dir "$WORK_ROOT/pred/control_fused" --beta 0.25 --recipe-output "$WORK_ROOT/pred/control_recipe.json"

python scripts/ensemble_predictions.py --pred-a "$WORK_ROOT/pred/control_fused" --pred-b "$WORK_ROOT/pred/noisyfit_e34" --output-dir "$WORK_ROOT/pred/final" --beta 0.875 --recipe-output "$WORK_ROOT/pred/final_recipe.json"
~~~

The final denoised arrays and the ensemble recipe are written to $WORK_ROOT/pred/final and $WORK_ROOT/pred/final_recipe.json. The RTX 4090 setting uses patch_batch_size=20; reduce it only if GPU memory is insufficient.

## Results and Reproducibility

| Item | Submitted setting |
| --- | --- |
| Control checkpoints | e32 and e34 |
| Noisyfit checkpoint | e34 |
| Patch size | 1000 |
| Patch coverage | seed_k=6, hard_best |
| First ensemble | control-e32 + 0.25 control-e34 |
| Final ensemble | control-fused + 0.875 noisyfit-e34 |
| Training seed | 20260726 |

CD is bidirectional Chamfer distance and P2S is point-to-surface distance. A local rerun can vary numerically across CUDA and Jittor environments; it is not an independent verification of the online leaderboard score.

## License and Acknowledgement

See [LICENSE](LICENSE) and [NOTICE](NOTICE). The method is informed by [Guiding Point Cloud Denoising with Learned Structural Priors](https://github.com/git-guocc/PGD).
