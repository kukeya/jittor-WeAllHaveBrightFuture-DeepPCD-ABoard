# NCPGD-E: Noise-Conditioned Point-Cloud Denoising

NCPGD-E is the pure-Jittor point-cloud denoising solution submitted by team
"我们都有光明的未来" to Track 2 of the Sixth CG Graphics AI Challenge. It uses a
four-level Point U-Net, noise-conditioned structural priors, point-wise
displacement gates, and overlap-aware patch fusion. The implementation does
not import PyTorch, TensorFlow, PaddlePaddle, MindSpore, or JittorGeometric.

The archived A-leaderboard submission obtained `score=81.05`, `CD_score=69.86`,
and `P2S_score=92.25`. These are online leaderboard results, not a promise that
a new training run will reproduce the same score exactly: the submitted result
uses three released inference checkpoints and two-stage prediction fusion. The
required checkpoints are in `checkpoints/`; datasets, caches, logs,
predictions, and submission outputs are intentionally excluded.

## Repository layout

- `configs/`: immutable YAML configurations for the control and noisyfit runs.
- `pcdenoise/`: model, data preparation, training, inference, fusion, and
  submission code.
- `scripts/`: command-line entry points.
- `data/`: data-layout documentation only; raw data and generated caches are
  ignored.
- `checkpoints/`: the three checkpoints used by the archived inference recipe.
- `environment.yaml`: reproducible Conda environment definition.

## Environment

Tested with Ubuntu 22.04, Python 3.10, CUDA 12.4, GCC/G++ 10, and Jittor
1.3.10.0. Create the Conda environment from the repository root:

```bash
conda env create -f environment.yaml
conda activate jittor
export cc_path="$(command -v g++)"
export nvcc_path=/usr/local/cuda-12.4/bin/nvcc
export CUDA_HOME=/usr/local/cuda-12.4
export JITTOR_HOME=/path/to/a/writable/jittor_home
export conv_opt=1
python scripts/check_environment.py \
  --environment-file environment.yaml \
  --cc-path "$cc_path" \
  --nvcc-path "$nvcc_path" \
  --cuda-home "$CUDA_HOME"
```

Continue only after the checker prints `ENVIRONMENT_OK`. Jittor compiles and
caches the CUDA operators under `JITTOR_HOME` on first use. Set the five
environment variables again in every new terminal.

## Data preparation

Download the official competition archives yourself and keep them outside Git.
For example, place the training TAR and noisy test ZIP at
`/data/dataset_train.tar.gz` and `/data/dataset_test_noisy.zip`; their expected
post-preparation layout is documented in [`data/README.md`](data/README.md).

```bash
python scripts/prepare_data.py \
  --train-archive /data/dataset_train.tar.gz \
  --test-archive /data/dataset_test_noisy.zip \
  --output-dir data/prepared \
  --split-dir data/splits/seed_20260726_v2 \
  --seed 20260726 --val-ratio 0.05

python scripts/build_all_train_split.py \
  --source-split data/splits/seed_20260726_v2 \
  --output-dir data/splits/all_15833_seed20260726_v1

python scripts/build_train_surface_cache.py \
  --mesh-root data/prepared/train/dataset_train/shapenet \
  --train-split data/splits/all_15833_seed20260726_v1/train.json \
  --output-dir data/train_cache/surface50000_seed20260726_n15833_v1 \
  --select-count 15833 --num-points 50000 --seed 20260726 --workers 16
```

All locations are command-line arguments or YAML fields. The prepared data,
split, and cache must match the hashes embedded in the supplied training YAML.
On a missing file, the preparation and training programs fail with the required
path and a descriptive error rather than silently using a different dataset.

## Training

The configuration provides the model, data paths, hyperparameters, and unified
random seed (`training.seed: 20260726`). The runner writes the resolved config,
invocation, logs, manifests, and checkpoints under `--run-dir`.

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/train.py \
  --config configs/train/pgd_all15833_starter_laplace_condgate_huber75_d010_e34_seed20260726.yaml \
  --run-dir /work/runs/control
```

For the second training arm, replace the config and run directory with
`configs/train/pgd_all15833_gate_e34_noisyfit_cube24_seed20260726.yaml` and
`/work/runs/noisyfit`. The submitted recipe trained each arm for 34 epochs
(1,584 steps per epoch), with `patch_size=1000`, batch size 10, and learning
rate `5e-4`. Resume a stopped run with `--resume /path/to/checkpoint.pkl`.

## Inference and submission

Run inference on prepared test data with a trained checkpoint. This command
uses only the input directory and does not require training data when the
checkpoint's expected configuration digest is supplied.

```bash
python scripts/denoise.py \
  --config configs/train/pgd_all15833_starter_laplace_condgate_huber75_d010_e34_seed20260726.yaml \
  --checkpoint checkpoints/control_e34_step_00053856.pkl \
  --input-dir data/prepared/test --output-dir /work/pred/control_e34 \
  --patch-size 1000 --seed-k 6.0 --patch-batch-size 20 \
  --niters 1 --normalization-mode noisy_max \
  --fusion-mode hard_best --iteration-damping 1.0
```

The archival A-leaderboard recipe first fuses control epoch 32 and epoch 34
predictions with `--beta 0.25`, then fuses that result with noisyfit epoch 34
with `--beta 0.875`:

```bash
python scripts/ensemble_predictions.py \
  --pred-a /work/pred/control_e32 --pred-b /work/pred/control_e34 \
  --output-dir /work/pred/control_fused --beta 0.25 \
  --recipe-output /work/pred/control_recipe.json

python scripts/ensemble_predictions.py \
  --pred-a /work/pred/control_fused --pred-b /work/pred/noisyfit_e34 \
  --output-dir /work/pred/final --beta 0.875 \
  --recipe-output /work/pred/final_recipe.json

python scripts/validate_submission.py \
  --input-dir data/prepared/test --prediction-dir /work/pred/final \
  --input-archive /data/dataset_test_noisy.zip \
  --model-file /work/pred/final_recipe.json --run-id ncpgd_e_a81 \
  --output-zip /work/submission/result.zip \
  --manifest-path /work/submission/manifest.json \
  --expected-sample-count 200 --expected-point-count 50000
```

`patch_batch_size=20` was used on an RTX 4090 and changes throughput only; lower
it if GPU memory is insufficient.

Verify the released weights before inference:

```bash
sha256sum -c checkpoints/SHA256SUMS
```

## Metrics and result scope

The competition score is the equally weighted aggregate of CD and P2S scores.
CD measures set-level geometric coverage, while P2S measures the distance from
predicted points to the clean triangle surface. Output point count and order
must match each noisy input. This repository includes the generation and
submission-format validation path and the three inference checkpoints, but it
deliberately excludes the official dataset; therefore it cannot independently
recreate the reported online `81.05/69.86/92.25` result out of the box.

## Third-party notice

The architecture and source profile are based on
[PGD: Guiding Point Cloud Denoising with Learned Structural Priors](https://github.com/git-guocc/PGD)
at commit `b5dc8be352203cfb2a417be6d0e3e9a0cb1f196f`, which is released under the
MIT License. This repository is a Jittor implementation adapted for the
competition workflow; see [`NOTICE`](NOTICE) and [`LICENSE`](LICENSE).
