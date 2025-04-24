import glob
import cv2
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Generic image preprocessor
def preprocess_image(image_path, target_size):
    img = cv2.imread(image_path)
    img = cv2.resize(img, target_size)
    img = img.astype(np.float32) / 255.0
    return img

def data_preprocessing(batch_size=4):
    # -------------------------------
    # Task A: Bicubic (for Enhanced CNN)
    # -------------------------------
    hr_path_A = "./Datasets/DIV2K_train_HR/*.png"
    lr_path_A = "./Datasets/DIV2K_train_LR_bicubic/X2/*.png"

    hr_images_A = [preprocess_image(p, (512, 512)) for p in sorted(glob.glob(hr_path_A))[:200]]
    lr_images_A = [preprocess_image(p, (256, 256)) for p in sorted(glob.glob(lr_path_A))[:200]]

    train_lr_A, val_lr_A, train_hr_A, val_hr_A = train_test_split(
        lr_images_A, hr_images_A, test_size=0.25, random_state=42
    )
    train_dataset_A = tf.data.Dataset.from_tensor_slices((train_lr_A, train_hr_A)).batch(batch_size)
    val_dataset_A = tf.data.Dataset.from_tensor_slices((val_lr_A, val_hr_A)).batch(batch_size)

    test_lr_A = [preprocess_image(p, (256, 256)) for p in sorted(glob.glob("./Datasets/DIV2K_valid_LR_bicubic/X2/*.png"))]
    test_hr_A = [preprocess_image(p, (512, 512)) for p in sorted(glob.glob("./Datasets/DIV2K_valid_HR/*.png"))]
    test_dataset_A = tf.data.Dataset.from_tensor_slices((test_lr_A, test_hr_A)).batch(batch_size)

    # -------------------------------
    # Task B: Unknown (for ESRGAN)
    # -------------------------------
    hr_path_B = "./Datasets/DIV2K_train_HR/*.png"
    lr_path_B = "./Datasets/DIV2K_train_LR_unknown/X2/*.png"

    hr_images_B = [preprocess_image(p, (512, 512)) for p in sorted(glob.glob(hr_path_B))[:200]]
    lr_images_B = [preprocess_image(p, (256, 256)) for p in sorted(glob.glob(lr_path_B))[:200]]

    train_lr_B, val_lr_B, train_hr_B, val_hr_B = train_test_split(
        lr_images_B, hr_images_B, test_size=0.25, random_state=42
    )
    train_dataset_B = tf.data.Dataset.from_tensor_slices((train_lr_B, train_hr_B)).batch(batch_size)
    val_dataset_B = tf.data.Dataset.from_tensor_slices((val_lr_B, val_hr_B)).batch(batch_size)

    test_lr_B = [preprocess_image(p, (256, 256)) for p in sorted(glob.glob("./Datasets/DIV2K_valid_LR_unknown/X2/*.png"))]
    test_hr_B = [preprocess_image(p, (512, 512)) for p in sorted(glob.glob("./Datasets/DIV2K_valid_HR/*.png"))]
    test_dataset_B = tf.data.Dataset.from_tensor_slices((test_lr_B, test_hr_B)).batch(batch_size)

    return (train_dataset_A, val_dataset_A, test_dataset_A), (train_dataset_B, val_dataset_B, test_dataset_B)


# ====================== Import Models ======================
from A.model import ModelA
from B.model import ModelB

# ====================== Data preprocessing =================
(data_train_A, data_val_A, data_test_A), (data_train_B, data_val_B, data_test_B) = data_preprocessing(batch_size=4)

# ====================== Task A =============================
model_A = ModelA()
acc_A_train = model_A.train(data_train_A, data_val_A)
acc_A_test = model_A.test(data_test_A)

# Clean up memory/GPU
import gc
gc.collect()
tf.keras.backend.clear_session()

# ====================== Task B =============================
model_B = ModelB()
acc_B_train = model_B.train(data_train_B, data_val_B)
acc_B_test = model_B.test(data_test_B)

# Clean up memory/GPU
gc.collect()
tf.keras.backend.clear_session()

# ====================== Final Results ======================
print('TA:{},{};TB:{},{};'.format(acc_A_train, acc_A_test, acc_B_train, acc_B_test))