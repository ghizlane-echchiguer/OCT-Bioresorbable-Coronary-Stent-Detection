# 🔬 OCT Bioabsorbable Stent Detection — U-Net Deep Learning

> Évaluation de méthodes de segmentation de stents biorésorbables en imagerie OCT  
> **Université Clermont Auvergne** · Master 2 TSI-ITM · 2024–2025  
> Encadrante : Dr. Emilie Péry — Institut Pascal, Clermont-Ferrand  
> Réalisé par : Ghizlane Ech-chiguer & Haoidhu Assane

---

## 📌 Description

Ce projet porte sur la **détection automatique des struts de stents biorésorbables** dans des images intravasculaires obtenues par Tomographie par Cohérence Optique (OCT).

Les stents biorésorbables, en polymère, se résorbent progressivement dans l'organisme après implantation. Une segmentation précise de leurs mailles (struts) est essentielle pour :
- Vérifier le bon positionnement du stent sur la paroi artérielle
- Suivre la couverture néointimale au fil du temps
- Réduire les risques de thrombose

Le projet compare plusieurs approches (méthode Menguy 2016, Mask R-CNN 2021-2022) et propose une **nouvelle méthode basée sur U-Net**, surpassant les approches précédentes en précision et rapidité.

---

## 🗂️ Structure du projet

```
oct-stent-segmentation/
│
├── notebooks/
│   └── stent_segmentation_unet.ipynb   # Pipeline complet
│
├── report/
│   └── rapport.pdf                      # Rapport complet du projet
│
└── README.md
```

---

## 🧠 Architecture U-Net

```
Input OCT (256×256, grayscale)
        │
   ┌────▼──────────────────────────────┐
   │  ENCODEUR                         │
   │  Conv2D(32×2) → MaxPool           │
   │  Conv2D(64×2) → MaxPool           │
   │  Conv2D(128×2) → MaxPool          │
   │  Conv2D(256×2) → MaxPool          │
   └────────────────┬──────────────────┘
                    │
   ┌────────────────▼──────────────────┐
   │  BOTTLENECK — Conv2D(512×2)       │
   └────────────────┬──────────────────┘
                    │
   ┌────────────────▼──────────────────┐
   │  DÉCODEUR                         │
   │  Conv2DTranspose + Skip Conn.     │
   │  × 4 niveaux                      │
   └────────────────┬──────────────────┘
                    │
        Output Mask — Conv2D(1×1, sigmoid)
```

| Paramètre | Valeur |
|---|---|
| Framework | TensorFlow / Keras |
| Plateforme | Kaggle (GPU) |
| Taille d'entrée | 256 × 256 px (niveaux de gris) |
| Optimiseur | Adam |
| Fonction de perte | Binary Cross Entropy |
| Epochs | 10 |
| Batch size | 1 |
| Seuil de binarisation | t = 0.2 |

---

## ⚙️ Méthodologie

### 1. Données
- **6 patients** · **1408 images OCT** cartésiennes annotées manuellement par des médecins
- Deux temps d'acquisition : **J0** (jour 0) et **M6** (6 mois après implantation)
- Split : **60% train / 20% validation / 20% test**

### 2. Prétraitement
- Sélection des images en repère **cartésien** (plus adapté à l'analyse)
- Normalisation des pixels entre 0 et 1
- Binarisation des masques (seuil 0.5)

### 3. Data Augmentation (entraînement uniquement)
- Zoom, rotations, décalages horizontaux/verticaux
- `ImageDataGenerator` de Keras

---

## 📊 Résultats

### Notre modèle U-Net (50 images de test)

| Métrique | Valeur |
|---|---|
| **Dice Coefficient** | **0.78** |
| **IoU** | **0.69** |
| **Précision** | **0.72** |
| **Recall (Sensibilité)** | **0.89** |

### Comparaison avec les méthodes précédentes

| Méthode | Précision | Recall | Dice | Type |
|---|---|---|---|---|
| Menguy (2016) | ~77% | ~82% | — | Semi-automatique |
| Mask R-CNN (2021-22) | 75% | 49% | 0.60 | Automatique |
| **Notre U-Net (2024-25)** | **72%** | **89%** | **0.78** | **Automatique** |

> Notre approche améliore significativement le **Dice** (+30% vs Mask R-CNN) et le **Recall** (+82% vs Mask R-CNN), au prix d'une légère sur-segmentation.

---

## 🛠️ Stack Technique

![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=flat&logo=keras&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=flat&logo=kaggle&logoColor=white)

---

## 🚀 Utilisation

```bash
# Cloner le dépôt
git clone https://github.com/ghizlane-echchiguer/oct-stent-segmentation.git
cd oct-stent-segmentation

# Installer les dépendances
pip install tensorflow keras numpy matplotlib scikit-image scipy opencv-python
```

```python
# Lancer le notebook complet
jupyter notebook notebooks/stent_segmentation_unet.ipynb
```

> ⚠️ **Note** : Le dataset OCT (images patients) n'est pas inclus pour des raisons de confidentialité médicale. Les fichiers de poids `.h5` ne sont pas inclus en raison de leur taille.

---

## 📄 Rapport

Le rapport complet est disponible dans [`report/rapport.pdf`](report/rapport.pdf).  
Il inclut l'état de l'art, la méthodologie détaillée, les résultats quantitatifs et qualitatifs, et la comparaison avec les approches Menguy 2016 et Mask R-CNN 2021-2022.

---

## 🔗 Contexte académique

| | |
|---|---|
| **Formation** | Master 2 Traitement du Signal et des Images – Imagerie et Technologie pour la Médecine |
| **Établissement** | EUPI — Université Clermont Auvergne |
| **Laboratoire** | Institut Pascal, Clermont-Ferrand |
| **Encadrante** | Dr. Emilie Péry |
| **Année** | 2024 – 2025 |

---

## 👩‍💻 Auteures

**Ghizlane Ech-chiguer**  
[LinkedIn](https://www.linkedin.com/in/ghizlaneechchiguer/) · [GitHub](https://github.com/ghizlane-echchiguer)

**Haoidhu Assane**

---

*Projet académique — Master 2 · Université Clermont Auvergne · 2024–2025*
