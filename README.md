
# Super-Resolution with CNN and GAN

This project implements two deep learning models for single-image super-resolution (SISR): a **custom Enhanced CNN** and a **compact GAN** variant. The goal is to upscale low-resolution images (256×256) into high-resolution outputs (512×512) trained on the DIV2K dataset.

## 📁 Project Structure

```
.
├── A/                    # Task A: Enhanced CNN
│   ├── __init__.py
│   ├── model.py
├── B/                    # Task B: ESRGAN (GAN-based SR)
│   ├── __init__.py
│   ├── model.py
├── Datasets/             # DIV2K HR and LR images
│   ├── DIV2K_train_HR/
│   ├── DIV2K_train_LR_bicubic/
│   ├── DIV2K_valid_HR/
│   ├── DIV2K_valid_LR_bicubic/
├── main.py               # Main training/testing pipeline
├── requirements.txt      # Python dependencies
└── README.md             # This file
```

## 🧠 Tasks

- **Task A**: Enhanced CNN trained with pixel-wise MSE and evaluated using PSNR/SSIM.
- **Task B**: Compact ESRGAN with spectral normalized discriminator and perceptual loss.

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/your-username/super-resolution-project.git
cd super-resolution-project
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Prepare your dataset

Ensure the following directory structure exists inside `./Datasets/`:

```
Datasets/
├── DIV2K_train_HR/
├── DIV2K_train_LR_bicubic/X2/
├── DIV2K_valid_HR/
├── DIV2K_valid_LR_bicubic/X2/
```

> Only the first 200 images are used for training/validation for quick experimentation.

### 4. Run the pipeline
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

- Best model weights will be saved in the root directory (e.g., `best_cnn_bicubic_2.weights.h5`, `best_generator.weights.h5`).
- Adjust hyperparameters in `model.py` if needed.
- GPU recommended, especially for training Task B (GAN).
