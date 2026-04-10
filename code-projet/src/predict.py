"""
predict.py
----------
Chargement des poids et prédiction du modèle U-Net sur les images de test.

Usage :
    python predict.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from model import get_UNet

# =============================================================================
# Configuration
# =============================================================================
DATA_PATH   = '/kaggle/input/finaldb/blop'
IMAGE_DIR   = os.path.join(DATA_PATH, 'test_images')
MASK_DIR    = os.path.join(DATA_PATH, 'test_masks')
WEIGHTS_PATH = '/kaggle/input/predicttpsynthese/Pif.weights.h5'

IMG_SIZE        = 256
BATCH_SIZE      = 1
THRESHOLD       = 0.2   # seuil de binarisation des prédictions
NUM_IMAGES_PLOT = 22    # nombre d'images à afficher


# =============================================================================
# Générateurs de données de test
# =============================================================================
def build_test_generator():
    """Construit et retourne le générateur combiné (images, masques) de test."""
    datagen = ImageDataGenerator(rescale=1./255)

    image_gen = datagen.flow_from_directory(
        directory=IMAGE_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        class_mode=None,
        seed=42,
        shuffle=False
    )
    mask_gen = datagen.flow_from_directory(
        directory=MASK_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        class_mode=None,
        seed=42,
        shuffle=False
    )
    return zip(image_gen, mask_gen)


# =============================================================================
# Chargement du modèle
# =============================================================================
def load_model_with_weights(weights_path: str):
    """Instancie le U-Net et charge les poids sauvegardés."""
    model = get_UNet(IMG_SIZE, IMG_SIZE)
    if os.path.exists(weights_path):
        model.load_weights(weights_path)
        print(f"Poids chargés depuis : {weights_path}")
    else:
        print(f"[ERREUR] Fichier de poids introuvable : {weights_path}")
    return model


# =============================================================================
# Visualisation des prédictions
# =============================================================================
def plot_predictions(model, test_generator, num_images: int = 22, threshold: float = 0.2):
    """
    Affiche côte à côte : image originale | masque réel | masque prédit.

    Args:
        model       : modèle U-Net chargé.
        test_generator : générateur (images, masques) de test.
        num_images  : nombre d'images à afficher.
        threshold   : seuil de binarisation de la prédiction.
    """
    plt.figure(figsize=(12, 4 * num_images))

    for i in range(num_images):
        images, masks = next(test_generator)

        pred        = model.predict(images[0][np.newaxis, ...])[0]
        pred_binary = (pred > threshold).astype(np.float32).squeeze()
        mask_real   = masks[0].squeeze()

        # Image originale
        plt.subplot(num_images, 3, i * 3 + 1)
        plt.imshow(images[0].squeeze(), cmap='gray')
        plt.title('Image originale')
        plt.axis('off')

        # Masque réel
        plt.subplot(num_images, 3, i * 3 + 2)
        plt.imshow(mask_real, cmap='gray')
        plt.title('Masque réel')
        plt.axis('off')

        # Masque prédit
        plt.subplot(num_images, 3, i * 3 + 3)
        plt.imshow(pred_binary, cmap='gray')
        plt.title('Prédiction')
        plt.axis('off')

    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/predictions.png', dpi=150)
    plt.show()


# =============================================================================
# Point d'entrée principal
# =============================================================================
def main():
    model          = load_model_with_weights(WEIGHTS_PATH)
    test_generator = build_test_generator()
    plot_predictions(model, test_generator, num_images=NUM_IMAGES_PLOT, threshold=THRESHOLD)


if __name__ == '__main__':
    main()
