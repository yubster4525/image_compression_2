# StyleGAN3-HVAE Neural Image Compression - Agent Guide

## Build & Test Commands
- **Run CABAC demo**: `./run_cabac_demo.sh` or `python cabac_demo.py --image [PATH] --output [DIR] --compare --resolution 256`
- **Run Gumbel-Softmax demo**: `./run_gumbel_cabac_demo.sh` or `python cabac_gumbel_demo.py --image [PATH] --output [DIR] --compare`
- **Train Gumbel-CABAC**: `./run_train_gumbel_cabac.sh` or `python train_gumbel_cabac.py --data [DATASET] --output [DIR]`
- **Single test**: `python test_stylegan3_load.py` or `python test_imagenet.py --dataset ./datasets/imagenet100/train --imagenet`
- **Test trained model**: `python test_trained_gumbel_cabac.py --model [MODEL_PATH] --input [IMAGE_PATH]`

## Code Style Guidelines
- **Imports**: Group standard library, then third-party, then local imports with blank line separations
- **Naming**: 
  - Classes: CamelCase (e.g., `HVAE_VGG_Encoder`)
  - Functions/variables: snake_case (e.g., `perceptual_weight`)
  - Constants: UPPER_CASE
- **Types**: Use type hints for function signatures when possible
- **Error handling**: Use try/except blocks with specific exceptions
- **Docstrings**: Use triple quotes with parameter descriptions
- **Structure**: Organize code into logical modules with clear separation of concerns
- **Hardware**: Support both CUDA (NVIDIA) and MPS (Apple Silicon) backends

## Parameters & Best Practices
- Set device explicitly: `--device cuda` or `--device mps`
- Respect StyleGAN3 compatibility requirements in HVAE encoder parameters
- Use lower batch sizes (4-8) for higher resolutions
- Validate using multiple metrics: PSNR, MS-SSIM, LPIPS