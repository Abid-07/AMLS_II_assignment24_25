import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model

# --------------------------------------------
# VGG19 Feature Extractor for Perceptual Loss
# Extracts feature maps from block4_conv3 layer
# --------------------------------------------
def build_vgg_extractor():
    vgg = VGG19(weights="imagenet", include_top=False, input_shape=(512, 512, 3))
    vgg.trainable = False
    return Model(inputs=vgg.input, outputs=vgg.get_layer("block4_conv3").output)


# --------------------------------------------
# Residual Dense Block (RDDB)
# Densely connected layers + residual scaling
# --------------------------------------------
def residual_dense_block(x, filters):
    inputs = x
    for _ in range(3):  # 3 conv layers with ReLU, dense connectivity
        out = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
        x = layers.Concatenate()([x, out])
    out = out * 0.1  # Residual scaling to stabilize deeper networks
    return layers.Add()([inputs, out])  # Residual skip connection


# --------------------------------------------
# Generator Network
# 6 RDDB blocks, global skip connection, x2 upsampling
# --------------------------------------------
def build_generator():
    inputs = layers.Input(shape=(256, 256, 3))
    x = layers.Conv2D(64, 3, padding='same')(inputs)
    residual = x  # Save for global skip connection

    for _ in range(6):  # Stack 6 RDDB blocks
        x = residual_dense_block(x, 64)

    x = layers.Conv2D(64, 3, padding='same')(x)
    x = layers.Add()([x, residual])  # Global skip connection

    x = layers.UpSampling2D(size=2)(x)  # Upsample from 256x256 to 512x512
    x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
    x = layers.Conv2D(3, 3, padding='same', activation='sigmoid')(x)  # Final RGB output
    return models.Model(inputs, x, name="generator")


# --------------------------------------------
# Spectral Normalization Layer
# Controls gradient norm for stable GAN training
# --------------------------------------------
class SpectralNormalization(tf.keras.layers.Layer):
    def __init__(self, layer, power_iterations=1):
        super().__init__()
        self.layer = layer
        self.power_iterations = power_iterations

    def build(self, input_shape):
        self.layer.build(input_shape)
        self.w = self.layer.kernel
        self.u = self.add_weight(shape=(1, self.w.shape[-1]), initializer='random_normal', trainable=False, name='sn_u')
        super().build(input_shape)

    def call(self, inputs):
        w_reshaped = tf.reshape(self.w, [-1, self.w.shape[-1]])
        u = self.u
        for _ in range(self.power_iterations):  # Power iteration for spectral norm
            v = tf.nn.l2_normalize(tf.matmul(u, tf.transpose(w_reshaped)))
            u = tf.nn.l2_normalize(tf.matmul(v, w_reshaped))
        self.u.assign(u)
        sigma = tf.matmul(tf.matmul(v, w_reshaped), tf.transpose(u))
        w_sn = self.w / sigma
        x = tf.nn.conv2d(inputs, w_sn, strides=self.layer.strides, padding=self.layer.padding.upper(), data_format='NHWC')
        if self.layer.use_bias:
            x = tf.nn.bias_add(x, self.layer.bias, data_format='NHWC')
        return x


# --------------------------------------------
# PatchGAN-style Discriminator
# Spectral Normalization + LeakyReLU + Dropout
# --------------------------------------------
def build_discriminator_sn():
    inputs = layers.Input(shape=(512, 512, 3))
    x = inputs
    for filters in [64, 128, 256]:
        conv = layers.Conv2D(filters, 4, strides=2, padding='same', use_bias=False)
        x = SpectralNormalization(conv)(x)
        x = layers.LeakyReLU(0.2)(x)
        x = layers.Dropout(0.3)(x)
    conv_out = layers.Conv2D(1, 4, padding='same', use_bias=False)
    x = SpectralNormalization(conv_out)(x)
    return models.Model(inputs, x, name="discriminator_sn")


# --------------------------------------------
# ESRGAN Model: Custom Keras Subclassed Model
# Combines Generator, Discriminator, and VGG for training
# --------------------------------------------
class ESRGAN(tf.keras.Model):
    def __init__(self, generator, discriminator, vgg, pixel_weight=1.0, perceptual_weight=0.006, adversarial_weight=0.001):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.vgg = vgg
        self.pixel_weight = pixel_weight
        self.perceptual_weight = perceptual_weight
        self.adversarial_weight = adversarial_weight
        self.g_loss_tracker = tf.keras.metrics.Mean(name="g_loss")
        self.d_loss_tracker = tf.keras.metrics.Mean(name="d_loss")

    def compile(self, g_optimizer, d_optimizer):
        super().compile()
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.compiled_loss = tf.keras.losses.MeanSquaredError()

    def call(self, inputs, training=False):
        return self.generator(inputs, training=training)

    def train_step(self, data):
        lr, hr = data
        with tf.GradientTape(persistent=True) as tape:
            fake_hr = self.generator(lr, training=True)
            real_logits = self.discriminator(hr, training=True)
            fake_logits = self.discriminator(fake_hr, training=True)

            # Loss calculations
            d_loss_real = tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.ones_like(real_logits), real_logits))
            d_loss_fake = tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.zeros_like(fake_logits), fake_logits))
            d_loss = d_loss_real + d_loss_fake

            adv_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(tf.ones_like(fake_logits), fake_logits))
            perceptual_loss = tf.reduce_mean(tf.abs(self.vgg(hr) - self.vgg(fake_hr)))
            pixel_loss = tf.reduce_mean(tf.abs(hr - fake_hr))  # L1 Loss

            g_loss = (self.pixel_weight * pixel_loss +
                      self.perceptual_weight * perceptual_loss +
                      self.adversarial_weight * adv_loss)

        # Optimizer updates
        g_grads = tape.gradient(g_loss, self.generator.trainable_variables)
        d_grads = tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(g_grads, self.generator.trainable_variables))
        self.d_optimizer.apply_gradients(zip(d_grads, self.discriminator.trainable_variables))

        # Metric tracking
        self.g_loss_tracker.update_state(g_loss)
        self.d_loss_tracker.update_state(d_loss)
        return {"g_loss": self.g_loss_tracker.result(), "d_loss": self.d_loss_tracker.result()}

    @property
    def metrics(self):
        return [self.g_loss_tracker, self.d_loss_tracker]



# --------------------------------------------
# ModelB: Wrapper for ESRGAN training and evaluation
# Manages generator, discriminator, optimizer setup, and metric tracking
# --------------------------------------------
class ModelB:
    def __init__(self):
        # Build core ESRGAN components
        self.gen = build_generator()
        self.disc = build_discriminator_sn()
        self.vgg = build_vgg_extractor()
        self.weights_path = "B/best_generator.weights.h5"  # Path to save/load generator weights

        # Create ESRGAN model with adjusted loss weights
        self.model = ESRGAN(
            generator=self.gen,
            discriminator=self.disc,
            vgg=self.vgg,
            pixel_weight=0.6,           # Reduced emphasis on pixel loss
            perceptual_weight=0.012,    # Maintained balance for texture perception
            adversarial_weight=0.007    # Boosted adversarial emphasis for realism
        )

        # Learning rate scheduler: Exponential decay
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=4e-4,  # High LR for faster convergence under limited time
            decay_steps=20000,
            decay_rate=0.8,
            staircase=True
        )

        # Optimizer with gradient clipping to stabilize training
        self.model.compile(
            g_optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=2.0),
            d_optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule, clipnorm=2.0)
        )

    # ----------------------------------------
    # Train the model with given dataset
    # Tracks best PSNR over validation set
    # ----------------------------------------
    def train(self, train_dataset, val_dataset, epochs=1):
        best_psnr = 0.0

        for epoch in range(epochs):
            # Training loop
            for lr, hr in train_dataset:
                self.model.train_step((lr, hr))

            # Validation loop to compute PSNR
            psnr_total, steps = 0.0, 0
            for lr_val, hr_val in val_dataset:
                sr_val = self.gen(lr_val, training=False)
                psnr_val = tf.reduce_mean(tf.image.psnr(hr_val, sr_val, max_val=1.0))
                psnr_total += psnr_val.numpy()
                steps += 1

            avg_psnr = psnr_total / steps

            # Save best model based on validation PSNR
            if avg_psnr > best_psnr:
                best_psnr = avg_psnr
                self.gen.save_weights(self.weights_path)

        return round(best_psnr, 2)  # Return best achieved validation PSNR

    # ----------------------------------------
    # Test the model on unseen data
    # Evaluates using PSNR and SSIM metrics
    # ----------------------------------------
    def test(self, test_dataset):
        self.gen.load_weights(self.weights_path)  # Load best generator weights

        # Collect all test images into arrays
        test_lr_images = []
        test_hr_images = []
        for lr, hr in test_dataset:
            test_lr_images.extend(lr.numpy())
            test_hr_images.extend(hr.numpy())

        test_lr_images = tf.convert_to_tensor(test_lr_images)
        test_hr_images = tf.convert_to_tensor(test_hr_images)

        # Generate super-resolved images
        test_sr_images = self.gen.predict(test_lr_images, batch_size=4)

        # Calculate average PSNR and SSIM across all test images
        total_psnr, total_ssim = 0.0, 0.0
        for sr, hr in zip(test_sr_images, test_hr_images):
            psnr = tf.image.psnr(hr, sr, max_val=1.0).numpy()
            ssim = tf.image.ssim(hr, sr, max_val=1.0).numpy()
            total_psnr += psnr
            total_ssim += ssim

        avg_psnr = total_psnr / len(test_lr_images)
        avg_ssim = total_ssim / len(test_lr_images)

        print(f"✅ B-Test PSNR: {avg_psnr:.2f}, SSIM: {avg_ssim:.4f}")
        return round(avg_psnr, 2)  # Return test PSNR
