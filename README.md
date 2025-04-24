
# Super-Resolution with CNN and GAN

This project implements two deep learning models for single-image super-resolution (SISR): a **custom Enhanced CNN** and a **compact GAN** variant. The goal is to upscale low-resolution images (256×256) into high-resolution outputs (512×512) trained on the DIV2K dataset.

##  Tasks

- **Task A**: Enhanced CNN trained with pixel-wise MSE and evaluated using PSNR/SSIM.
- **Task B**: Compact GAN with spectral normalized discriminator and perceptual loss.

##  How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

#### Alternatively set up a clean environment via Conda.

```bash
conda env create -f environment.yml
conda activate amls2

```

### 2. Prepare your dataset

Ensure the following directory structure exists inside `./Datasets/`:

```
Datasets/
├── DIV2K_train_HR/
├── DIV2K_train_LR_bicubic/X2/
├── DIV2K_train_LR_unknown/X2/
├── DIV2K_valid_HR/
├── DIV2K_valid_LR_bicubic/X2/
├── DIV2K_valid_LR_unknown/X2/
```

> Only the first 200 images are used for training/validation for quick experimentation. (May be dependant on GPU. Models are trained on NVIDIA L4 Tensor GPU)

### 3. Run the pipeline
```bash
python main.py
```

The script will:
- Preprocess the data
- Train and test both models
- Output final results like:
```
TA:<train_psnr_A>,<test_psnr_A>;TB:<train_psnr_B>,<test_psnr_B>;
```

## 🧾 Notes

- Best model weights will be saved in the root directory (e.g., `best_cnn_bicubic.weights.h5`, `best_generator.weights.h5`).
- Adjust hyperparameters in `model.py` if needed.
- GPU recommended, especially for training Task B (GAN).
