"""
metrics.py
----------
Calcul et visualisation des métriques de segmentation :
IoU, Dice coefficient, Précision et Recall.

Usage :
    from metrics import evaluate_model
    results = evaluate_model(model, test_generator, num_images=50, threshold=0.2)
"""

import os
import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# Fonctions de calcul des métriques
# =============================================================================

def calculate_iou(mask_real: np.ndarray, mask_pred: np.ndarray) -> float:
    """
    Calcule l'Intersection over Union (IoU) entre deux masques binaires.

    Args:
        mask_real : masque de référence (vérité terrain), valeurs 0 ou 1.
        mask_pred : masque prédit binarisé, valeurs 0 ou 1.

    Returns:
        float : score IoU dans [0, 1]. Retourne 0 si l'union est nulle.
    """
    intersection = np.sum(mask_real * mask_pred)
    union        = np.sum(mask_real) + np.sum(mask_pred) - intersection
    return float(intersection / union) if union != 0 else 0.0


def calculate_dice(mask_real: np.ndarray, mask_pred: np.ndarray) -> float:
    """
    Calcule le coefficient de Dice (DSC) entre deux masques binaires.

    Args:
        mask_real : masque de référence, valeurs 0 ou 1.
        mask_pred : masque prédit binarisé, valeurs 0 ou 1.

    Returns:
        float : score Dice dans [0, 1]. Retourne 0 si les deux masques sont vides.
    """
    intersection = np.sum(mask_real * mask_pred)
    denom        = np.sum(mask_real) + np.sum(mask_pred)
    return float(2 * intersection / denom) if denom != 0 else 0.0


def calculate_precision(mask_real: np.ndarray, mask_pred: np.ndarray) -> float:
    """
    Calcule la Précision : TP / (TP + FP).

    Returns:
        float : précision dans [0, 1]. Retourne 0 si aucun pixel prédit positif.
    """
    tp = np.sum(mask_real * mask_pred)
    fp = np.sum((1 - mask_real) * mask_pred)
    return float(tp / (tp + fp)) if (tp + fp) != 0 else 0.0


def calculate_recall(mask_real: np.ndarray, mask_pred: np.ndarray) -> float:
    """
    Calcule le Recall (sensibilité) : TP / (TP + FN).

    Returns:
        float : recall dans [0, 1]. Retourne 0 si aucun pixel réel positif.
    """
    tp = np.sum(mask_real * mask_pred)
    fn = np.sum(mask_real * (1 - mask_pred))
    return float(tp / (tp + fn)) if (tp + fn) != 0 else 0.0


# =============================================================================
# Évaluation globale sur le jeu de test
# =============================================================================

def evaluate_model(model, test_generator, num_images: int = 50, threshold: float = 0.2) -> dict:
    """
    Évalue le modèle sur un ensemble d'images de test et affiche les métriques moyennes.

    Args:
        model          : modèle U-Net chargé avec ses poids.
        test_generator : générateur (images, masques) de test.
        num_images     : nombre d'images à évaluer.
        threshold      : seuil de binarisation des prédictions.

    Returns:
        dict : métriques moyennes {'iou', 'dice', 'precision', 'recall'}.
    """
    ious, dices, precisions, recalls = [], [], [], []

    for _ in range(num_images):
        images, masks = next(test_generator)

        pred        = model.predict(images[0][np.newaxis, ...])[0]
        pred_binary = (pred > threshold).astype(np.float32).squeeze()
        mask_real   = masks[0].squeeze()

        ious.append(calculate_iou(mask_real, pred_binary))
        dices.append(calculate_dice(mask_real, pred_binary))
        precisions.append(calculate_precision(mask_real, pred_binary))
        recalls.append(calculate_recall(mask_real, pred_binary))

    results = {
        'iou':       float(np.mean(ious)),
        'dice':      float(np.mean(dices)),
        'precision': float(np.mean(precisions)),
        'recall':    float(np.mean(recalls)),
    }

    print("\n===== Résultats moyens sur {} images =====".format(num_images))
    print(f"  IoU       : {results['iou']:.4f}")
    print(f"  Dice      : {results['dice']:.4f}")
    print(f"  Précision : {results['precision']:.4f}")
    print(f"  Recall    : {results['recall']:.4f}")

    return results


# =============================================================================
# Visualisation avec superposition et métriques par image
# =============================================================================

def overlay_and_evaluate(model, test_generator, num_images: int = 50,
                          start_index: int = 0, threshold: float = 0.2):
    """
    Pour chaque image de test, affiche 5 colonnes :
        1. Image originale
        2. Masque réel
        3. Masque prédit binarisé
        4. Superposition (réel=bleu, prédit=rouge)
        5. Superposition sur image originale + score IoU / Dice

    Args:
        model          : modèle U-Net chargé.
        test_generator : générateur (images, masques) de test.
        num_images     : nombre d'images à afficher.
        start_index    : nombre d'images à sauter au début du générateur.
        threshold      : seuil de binarisation.
    """
    # Avancer dans le générateur jusqu'à start_index
    for _ in range(start_index):
        next(test_generator)

    plt.figure(figsize=(20, 5 * num_images))

    for i in range(num_images):
        images, masks = next(test_generator)

        pred        = model.predict(images[0][np.newaxis, ...])[0]
        pred_binary = (pred > threshold).astype(np.float32).squeeze()
        mask_real   = masks[0].squeeze()

        iou  = calculate_iou(mask_real, pred_binary)
        dice = calculate_dice(mask_real, pred_binary)

        # Superposition masques seuls (fond noir)
        overlay = np.zeros((*pred_binary.shape, 3))
        overlay[..., 0] = pred_binary   # rouge  = prédit
        overlay[..., 2] = mask_real     # bleu   = réel
        # vert = intersection → apparaît cyan/blanc là où les deux coïncident

        # Superposition sur image originale en niveaux de gris
        overlay_img = np.stack([images[0].squeeze()] * 3, axis=-1)
        overlay_img[..., 0] = np.maximum(overlay_img[..., 0], pred_binary * 255)
        overlay_img[..., 2] = np.maximum(overlay_img[..., 2], mask_real   * 255)
        overlay_img = np.clip(overlay_img, 0, 255)

        base = i * 5
        # Col 1 : image originale
        plt.subplot(num_images, 5, base + 1)
        plt.imshow(images[0].squeeze(), cmap='gray')
        plt.title('Image originale')
        plt.axis('off')

        # Col 2 : masque réel
        plt.subplot(num_images, 5, base + 2)
        plt.imshow(mask_real, cmap='gray')
        plt.title('Masque réel')
        plt.axis('off')

        # Col 3 : masque prédit binarisé
        plt.subplot(num_images, 5, base + 3)
        plt.imshow(pred_binary, cmap='gray')
        plt.title('Masque prédit')
        plt.axis('off')

        # Col 4 : superposition masques (fond noir)
        plt.subplot(num_images, 5, base + 4)
        plt.imshow(overlay)
        plt.title('Superposition\n(réel=bleu, prédit=rouge)')
        plt.axis('off')

        # Col 5 : superposition sur image originale + métriques
        plt.subplot(num_images, 5, base + 5)
        plt.imshow(overlay_img.astype(np.uint8))
        plt.title(f'IoU: {iou:.2f} | Dice: {dice:.2f}')
        plt.axis('off')

    plt.tight_layout()
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/evaluation_overlay.png', dpi=100)
    plt.show()
