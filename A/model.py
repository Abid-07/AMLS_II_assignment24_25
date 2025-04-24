import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# -------------------------------------------------------------------
# ModelA - Bicubic Super-Resolution CNN
# Implements a residual-enhanced CNN model for x2 upscaling of bicubic-degraded images
# -------------------------------------------------------------------
class ModelA:
    def __init__(self):
        # Initialize and compile the model, define path for saving weights
        self.model = self.create_model()
        self.weights_path = "A/best_cnn_bicubic.weights.h5"

    # -------------------------------------------------------------------
    # Model Architecture
    # Input: 256x256x3 LR image → Output: 512x512x3 HR image
    # Uses residual connections and upsampling to recover lost detail
    # -------------------------------------------------------------------
    def create_model(self, input_shape=(256, 256, 3)):
        inputs = layers.Input(shape=input_shape)

        # -------- Feature Extraction (Shallow Layers) --------
        x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)
        x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
        residual = x  # Save early features for global residual connection

        # -------- Deep Feature Expansion --------
        x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
        x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
        x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
        x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
        x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)

        # -------- Global Residual Connection --------
        residual_proj = layers.Conv2D(256, (1, 1), padding='same')(residual)
        x = layers.Add()([x, residual_proj])  # Add residual to promote gradient flow

        # -------- Upsampling Block (x2 scale) --------
        x = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
        x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
        x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)

        # -------- Output Projection --------
        outputs = layers.Conv2D(3, (3, 3), padding='same', activation='sigmoid')(x)

        model = models.Model(inputs, outputs)

        # -------- Define Custom Metrics --------
        def psnr_metric(y_true, y_pred):
            return tf.image.psnr(y_true, y_pred, max_val=1.0)

        def ssim_metric(y_true, y_pred):
            return tf.image.ssim(y_true, y_pred, max_val=1.0)

        # -------- Compile with Clipping & Scheduled LR --------
        lr_schedule = ExponentialDecay(4e-4, 10000, 0.5, staircase=True)  # VDSR-inspired
        optimizer = Adam(learning_rate=lr_schedule, clipnorm=2.0)  # Clipping to stabilize training

        model.compile(
            optimizer=optimizer,
            loss='mse',  # Mean squared error to optimize pixel-level accuracy
            metrics=['mae', psnr_metric, ssim_metric]
        )
        return model

    # -------------------------------------------------------------------
    # Train the CNN model with training and validation datasets
    # Early stopping restores the best weights based on validation MAE
    # -------------------------------------------------------------------
    def train(self, train_dataset, val_dataset):
        checkpoint = ModelCheckpoint(
            filepath=self.weights_path,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        )
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )

        history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=2,
            callbacks=[checkpoint, early_stopping]
        )

        best_val_psnr = max(history.history['val_psnr_metric'])  # Extract best PSNR for reporting
        return best_val_psnr

    # -------------------------------------------------------------------
    # Evaluate model on test dataset
    # Computes average PSNR and SSIM across the entire test set
    # -------------------------------------------------------------------
    def test(self, test_dataset):
        self.model.load_weights(self.weights_path)  # Load best saved model weights

        total_psnr, total_ssim, count = 0.0, 0.0, 0
        for lr_imgs, hr_imgs in test_dataset:
            sr_imgs = self.model.predict(lr_imgs)

            for sr, hr in zip(sr_imgs, hr_imgs):
                psnr = tf.image.psnr(hr, sr, max_val=1.0).numpy()
                ssim = tf.image.ssim(hr, sr, max_val=1.0).numpy()
                total_psnr += psnr
                total_ssim += ssim
                count += 1

        avg_psnr = total_psnr / count
        avg_ssim = total_ssim / count

        # print(f"✅ Test PSNR: {avg_psnr:.2f}, SSIM: {avg_ssim:.4f}")
        return avg_psnr  # You can switch to SSIM if preferred
