# Emotion Recognition Dataset

This folder contains the **Face Expression Recognition Dataset** used for training and testing emotion detection models.

## Dataset Source

**Public Dataset**: [Face Expression Recognition Dataset on Kaggle](https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset)

**Citation**: This dataset is publicly available and can be downloaded from the above Kaggle link.

## Dataset Structure

```
data/
├── train/          # Training images
│   ├── angry/      # Angry expressions
│   ├── disgust/    # Disgust expressions  
│   ├── fear/       # Fear expressions
│   ├── happy/      # Happy expressions
│   ├── neutral/    # Neutral expressions
│   ├── sad/        # Sad expressions
│   └── surprise/   # Surprise expressions
└── validation/     # Validation images
    ├── angry/      # Angry expressions
    ├── disgust/    # Disgust expressions
    ├── fear/       # Fear expressions
    ├── happy/      # Happy expressions
    ├── neutral/    # Neutral expressions
    ├── sad/        # Sad expressions
    └── surprise/   # Surprise expressions
```

## Emotion Classes

7 emotion categories:
- **angry** 😠
- **disgust** 🤢
- **fear** 😨
- **happy** 😊
- **neutral** 😐
- **sad** 😢
- **surprise** 😲

## Data Statistics

- **Total Classes**: 7 emotions
- **Image Format**: JPG
- **Training Set**: ~28,000+ images
- **Validation Set**: ~3,500+ images
- **Balanced Distribution**: ~4,000 images per emotion class

## Download

To get the complete dataset:
1. Visit: [https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset](https://www.kaggle.com/datasets/jonathanoheix/face-expression-recognition-dataset)
2. Download and extract to this folder
3. Ensure the `train/` and `validation/` subfolders are present