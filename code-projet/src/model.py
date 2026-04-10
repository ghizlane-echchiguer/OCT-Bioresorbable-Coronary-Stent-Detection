"""
model.py
--------
Architecture U-Net pour la segmentation des stents biorésorbables en OCT.

Référence : Ronneberger et al., "U-Net: Convolutional Networks for Biomedical
Image Segmentation", MICCAI 2015.
"""

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
)


def get_UNet(img_rows: int = 256, img_cols: int = 256) -> Model:
    """
    Construit et retourne le modèle U-Net.

    Architecture :
        - Encodeur  : 4 blocs (Conv2D x2 + MaxPooling2D), filtres 32→64→128→256
        - Bottleneck: Conv2D x2 avec 512 filtres
        - Décodeur  : 4 blocs (Conv2DTranspose + skip connection + Conv2D x2)
        - Sortie    : Conv2D 1x1 avec activation sigmoïde (segmentation binaire)

    Args:
        img_rows (int): Hauteur des images d'entrée (défaut : 256).
        img_cols (int): Largeur des images d'entrée (défaut : 256).

    Returns:
        model (tf.keras.Model): Modèle U-Net compilable.
    """
    inputs = Input((img_rows, img_cols, 1))

    # -------------------------------------------------------------------------
    # Encodeur (chemin contractant)
    # -------------------------------------------------------------------------
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    # -------------------------------------------------------------------------
    # Bottleneck
    # -------------------------------------------------------------------------
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', padding='same')(conv5)

    # -------------------------------------------------------------------------
    # Décodeur (chemin expansif) avec skip connections
    # -------------------------------------------------------------------------
    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', padding='same')(conv9)

    # -------------------------------------------------------------------------
    # Couche de sortie : segmentation binaire pixel par pixel
    # -------------------------------------------------------------------------
    conv10 = Conv2D(1, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])
    return model
