"""
train.py
--------
Pipeline d'entraînement du modèle U-Net pour la segmentation
des stents biorésorbables en OCT.

Données attendues (structure Kaggle) :
    data/
    ├── train_images/
    ├── train_masks/
    ├── val_images/
    └── val_masks/

Usage :
    python train.py
"""

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from model import get_UNet

# =============================================================================
# Configuration
# =============================================================================
DATA_PATH        = '/kaggle/input/finaldb/blop'   # adapter selon l'environnement
TRAIN_FRAME_PATH = os.path.join(DATA_PATH, 'train_images')
TRAIN_MASK_PATH  = os.path.join(DATA_PATH, 'train_masks')
VAL_FRAME_PATH   = os.path.join(DATA_PATH, 'val_images')
VAL_MASK_PATH    = os.path.join(DATA_PATH, 'val_masks')

WEIGHTS_SAVE_PATH = '/kaggle/working/Pif.weights.h5'
MODEL_SAVE_PATH   = 'unet_model_with_binarized_masks.h5'

IMG_SIZE   = 256
BATCH_SIZE = 1
EPOCHS     = 10
SEED_TRAIN = 123
SEED_VAL   = 42

# =============================================================================
# Générateurs de données
# =============================================================================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    zoom_range=0.1,
    rotation_range=2,
    width_shift_range=0.1,
    height_shift_range=0.1
)
val_datagen = ImageDataGenerator(rescale=1./255)


def _make_generator(datagen, img_path, seed):
    return datagen.flow_from_directory(
        img_path,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        color_mode='grayscale',
        class_mode=None,
        seed=seed
    )


def binarize_batch_masks(mask_batch, threshold: float = 0.5) -> np.ndarray:
    """Binarise un batch de masques au seuil donné."""
    return (mask_batch > threshold).astype(np.float32)


def sync_generators_with_binarization(image_gen, mask_gen, threshold: float = 0.5):
    """Générateur qui yield (images, masques_binarisés)."""
    for img_batch, mask_batch in zip(image_gen, mask_gen):
        yield img_batch, binarize_batch_masks(mask_batch, threshold)


def sync_generators(image_gen, mask_gen):
    """Générateur simple qui yield (images, masques)."""
    for img_batch, mask_batch in zip(image_gen, mask_gen):
        yield img_batch, mask_batch


# =============================================================================
# Visualisation des données d'entraînement
# =============================================================================
def display_images_and_binarized_masks(image_gen, mask_gen, threshold=0.5, num_images=10):
    """Affiche des exemples d'images et leurs masques binarisés."""
    plt.figure(figsize=(15, 4))
    for i, (img_batch, mask_batch) in enumerate(
        sync_generators_with_binarization(image_gen, mask_gen, threshold)
    ):
        if i >= num_images:
            break
        img  = np.squeeze(img_batch[0])
        mask = np.squeeze(mask_batch[0])

        plt.subplot(2, num_images, i + 1)
        plt.imshow(img, cmap='gray')
        plt.title(f"Image {i + 1}")
        plt.axis('off')

        plt.subplot(2, num_images, num_images + i + 1)
        plt.imshow(mask, cmap='gray')
        plt.title(f"Mask {i + 1}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()


# =============================================================================
# Visualisation des courbes d'entraînement
# =============================================================================
def plot_training_curves(history):
    """Affiche les courbes de perte et de précision."""
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'],     label='Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'],     label='Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy Curve')
    plt.xlabel('Epoch')
    plt.legend()

    plt.tight_layout()
    plt.savefig('results/training_curves.png', dpi=150)
    plt.show()


# =============================================================================
# Point d'entrée principal
# =============================================================================
def main():
    # --- Chargement des données ---
    train_image_gen = _make_generator(train_datagen, TRAIN_FRAME_PATH, SEED_TRAIN)
    train_mask_gen  = _make_generator(train_datagen, TRAIN_MASK_PATH,  SEED_TRAIN)
    val_image_gen   = _make_generator(val_datagen,   VAL_FRAME_PATH,   SEED_VAL)
    val_mask_gen    = _make_generator(val_datagen,   VAL_MASK_PATH,    SEED_VAL)

    # Aperçu des données
    display_images_and_binarized_masks(train_image_gen, train_mask_gen, num_images=10)

    # --- Construction des datasets tf.data ---
    output_sig = (
        tf.TensorSpec(shape=(BATCH_SIZE, IMG_SIZE, IMG_SIZE, 1), dtype=tf.float32),
        tf.TensorSpec(shape=(BATCH_SIZE, IMG_SIZE, IMG_SIZE, 1), dtype=tf.float32),
    )

    train_dataset = tf.data.Dataset.from_generator(
        lambda: sync_generators_with_binarization(train_image_gen, train_mask_gen),
        output_signature=output_sig
    ).cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    val_dataset = tf.data.Dataset.from_generator(
        lambda: sync_generators_with_binarization(val_image_gen, val_mask_gen),
        output_signature=output_sig
    ).cache().prefetch(buffer_size=tf.data.AUTOTUNE)

    # --- Modèle ---
    model = get_UNet(img_rows=IMG_SIZE, img_cols=IMG_SIZE)
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    # --- Entraînement ---
    train_gen = sync_generators(train_image_gen, train_mask_gen)
    val_gen   = sync_generators(val_image_gen,   val_mask_gen)

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        steps_per_epoch=len(train_image_gen),
        validation_steps=len(val_image_gen)
    )

    # --- Sauvegarde ---
    model.save(MODEL_SAVE_PATH)
    print(f"Modèle sauvegardé : {MODEL_SAVE_PATH}")

    model.save_weights(WEIGHTS_SAVE_PATH)
    if os.path.exists(WEIGHTS_SAVE_PATH):
        print(f"Poids sauvegardés avec succès : {WEIGHTS_SAVE_PATH}")

    # --- Courbes ---
    os.makedirs('results', exist_ok=True)
    plot_training_curves(history)


if __name__ == '__main__':
    main()
