import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

class ModelA:
    def __init__(self):
        self.model = self.create_model()
        self.weights_path = "best_cnn_bicubic.weights.h5"

    def create_model(self, input_shape=(256, 256, 3)):
        inputs = layers.Input(shape=input_shape)

        x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(inputs)
        x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
        residual = x

        x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
        x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
        x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
        x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)
        x = layers.Conv2D(256, (3, 3), padding='same', activation='relu')(x)

        residual_proj = layers.Conv2D(256, (1, 1), padding='same')(residual)
        x = layers.Add()([x, residual_proj])

        x = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(x)
        x = layers.Conv2D(128, (3, 3), padding='same', activation='relu')(x)
        x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
        outputs = layers.Conv2D(3, (3, 3), padding='same', activation='sigmoid')(x)

        model = models.Model(inputs, outputs)

        def psnr_metric(y_true, y_pred):
            return tf.image.psnr(y_true, y_pred, max_val=1.0)

        def ssim_metric(y_true, y_pred):
            return tf.image.ssim(y_true, y_pred, max_val=1.0)

        lr_schedule = ExponentialDecay(4e-4, 10000, 0.5, staircase=True)
        optimizer = Adam(learning_rate=lr_schedule, clipnorm=2.0)

        model.compile(optimizer=optimizer, loss='mse', metrics=['mae', psnr_metric, ssim_metric])
        return model

    def train(self, train_dataset, val_dataset):
        checkpoint = ModelCheckpoint(
            filepath=self.weights_path,
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        )
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

        history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=200,
            callbacks=[checkpoint, early_stopping]
        )

        best_val_psnr = max(history.history['val_psnr_metric'])
        return best_val_psnr

    def test(self, test_dataset):
        self.model.load_weights(self.weights_path)
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
        print(f"✅ Test PSNR: {avg_psnr:.2f}, SSIM: {avg_ssim:.4f}")
        return avg_psnr
