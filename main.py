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
    # 1. TRAIN + VAL SETUP
    # -------------------------------
    hr_path = "./Datasets/DIV2K_train_HR/*.png"
    lr_path = "./Datasets/DIV2K_train_LR_bicubic/X2/*.png"

    lr_size = (256, 256)
    hr_size = (512, 512)

    hr_image_paths = sorted(glob.glob(hr_path))[:200]
    lr_image_paths = sorted(glob.glob(lr_path))[:200]

    hr_images = [preprocess_image(path, hr_size) for path in hr_image_paths]
    lr_images = [preprocess_image(path, lr_size) for path in lr_image_paths]

    hr_images = np.array(hr_images)
    lr_images = np.array(lr_images)

    train_lr, val_lr, train_hr, val_hr = train_test_split(
        lr_images, hr_images, test_size=0.25, random_state=42
    )

    train_dataset = tf.data.Dataset.from_tensor_slices((train_lr, train_hr)).batch(batch_size)
    val_dataset = tf.data.Dataset.from_tensor_slices((val_lr, val_hr)).batch(batch_size)

    # -------------------------------
    # 2. TEST SETUP
    # -------------------------------
    test_lr_path = sorted(glob.glob("./Datasets/DIV2K_valid_LR_bicubic/X2/*.png"))
    test_hr_path = sorted(glob.glob("./Datasets/DIV2K_valid_HR/*.png"))

    test_lr = [preprocess_image(path, lr_size) for path in test_lr_path]
    test_hr = [preprocess_image(path, hr_size) for path in test_hr_path]

    test_lr = np.array(test_lr)
    test_hr = np.array(test_hr)

    test_dataset = tf.data.Dataset.from_tensor_slices((test_lr, test_hr)).batch(batch_size)

    return train_dataset, val_dataset, test_dataset


# ====================== Import Models ======================
from A.model import ModelA
from B.model import ModelB

# ====================== Data preprocessing =================
data_train, data_val, data_test = data_preprocessing(batch_size=4)

# ====================== Task A =============================
model_A = ModelA()
acc_A_train = model_A.train(data_train, data_val)
acc_A_test = model_A.test(data_test)

# Clean up memory/GPU
import gc
gc.collect()
tf.keras.backend.clear_session()

# ====================== Task B =============================
model_B = ModelB()
acc_B_train = model_B.train(data_train, data_val, epochs=50)
acc_B_test = model_B.test(data_test)

# ====================== Final Results ======================
print('TA:{},{};TB:{},{};'.format(acc_A_train, acc_A_test, acc_B_train, acc_B_test))
