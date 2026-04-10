# Structure des données

Les images ne sont pas incluses dans ce dépôt (données médicales confidentielles).

## Structure attendue

```
data/
└── blop/
    ├── train_images/
    │   └── train_images1/      # images OCT cartésiennes (format .png)
    ├── train_masks/
    │   └── train_masks1/       # masques de segmentation (format .png)
    ├── val_images/
    │   └── val_images1/
    ├── val_masks/
    │   └── val_masks1/
    ├── test_images/
    │   └── test_images1/
    └── test_masks/
        └── test_masks1/
```

## Répartition

| Ensemble    | Proportion | Rôle                              |
|-------------|-----------|-----------------------------------|
| Entraînement | 60 %      | Apprentissage du modèle           |
| Validation   | 20 %      | Suivi pendant l'entraînement      |
| Test         | 20 %      | Évaluation finale (50 images)     |

## Nomenclature des fichiers

- Images OCT : `Cartesian__<patient>__<coupe>.png`
- Masques    : `Struts__<patient>__<coupe>.png`

Données issues de 6 patients, aux temps J0 (jour 0) et M6 (mois 6).
