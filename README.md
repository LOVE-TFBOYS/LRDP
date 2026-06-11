# LRDP

Latent Recursive Diffusion Pyramid (LRDP) is a modular 3D medical image
registration codebase for paper experiments.

The implementation follows:

- RDP-style coarse-to-fine recursive residual flow refinement.
- DiffuseReg-style DDPM epsilon prediction for latent residual deformation.
- LDM-Morph-style latent conditioning from fixed, moving, warped moving, and
  previous-flow features.
- Self-supervised diffusion closure: deterministic residual proposals supervise
  clean latent residual flows during training, while inference samples latent
  residual flows from Gaussian noise conditioned on the registration state.

## Main Modules

- `models/registration/lrdp_model.py`: four-scale LRDP registration chain.
- `models/diffusion/diffusion.py`: DDPM flow diffusion interface.
- `models/diffusion/denoiser_swin.py`: lightweight 3D shifted-window denoiser.
- `models/diffusion/denoiser_cnn.py`: CNN denoiser fallback.
- `models/registration/flow_utils.py`: voxel-flow resizing, composition, and grid helpers.
- `losses/registration_loss.py`: unified registration objective.
- `data/`: OASIS-style NIfTI/NumPy loading, pairing, preprocessing, and augmentation.
- `configs/lrdp_default.yaml`: default paper experiment configuration.
- `configs/lrdp_ablation.yaml`: ablation presets layered on the default config.

## Configuration

Training and inference are config-driven. Command-line arguments are only for
choosing a config and overriding runtime-specific paths or a small number of
fields.

```powershell
python train.py --config configs/lrdp_default.yaml --data-root /data/TFBOYS/dataset/OASIS --save-dir checkpoints/lrdp
```

Run an ablation preset:

```powershell
python train.py --config configs/lrdp_ablation.yaml --ablation no_diffusion --data-root /data/TFBOYS/dataset/OASIS
```

Override individual fields without editing the YAML:

```powershell
python train.py --config configs/lrdp_default.yaml --set training.epochs=200 --set dataloader.batch_size=1
```

Preprocess and save the dataset before training:

```powershell
python data/preprocess_dataset.py --config configs/lrdp_default.yaml --set data.split=train
python data/preprocess_dataset.py --config configs/lrdp_default.yaml --set data.split=test
```

The processed dataset is saved compactly:

```text
processed_dir/
  images/
  labels/      optional
  masks/       optional
  manifest.csv
```

Batch inference uses the checkpoint model configuration by default:

```powershell
python infer.py --checkpoint checkpoints/lrdp/best.pt --data-root /data/TFBOYS/dataset/OASIS --output-dir outputs/lrdp
```

## Expected Dataset Layout

The default dataset reader expects an OASIS-style folder:

```text
data_root/
  imagesTr/
  imagesTs/
  labelsTr/
  masksTr/
  masksTs/
```

Train split reads `imagesTr`, `labelsTr`, and `masksTr`. Test split reads
`imagesTs` and `masksTs`; test labels are optional. Pairing, target shape,
cropping, normalization, and augmentation are controlled from the YAML config.

## Static Validation

This repository intentionally does not include random-tensor minimal demos.
Use syntax and import checks for non-toy validation:

```powershell
python -m py_compile train.py infer.py models/**/*.py data/*.py losses/*.py utils/*.py
python -c "from models import LRDPRegistrationModel; from losses import RegistrationLoss"
```
