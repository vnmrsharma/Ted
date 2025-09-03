# Vision - Emotion Detection Systems

This folder contains three different approaches we tried for detecting emotions from facial images.

## What's in Each Folder

- **`base-yolo/`** - Simple rule-based approach using OpenCV and basic image features
- **`trained-deepface/`** - Pre-trained DeepFace models for facial analysis  
- **`trained-sentiment/`** - Custom neural network we trained from scratch

## Our Approach

We tried three different methods to see which one works best. Base-YOLO uses simple rules, Trained-Sentiment is our custom model, and Trained-DeepFace uses proven pre-trained models.

## Why We Chose Trained-DeepFace

After testing all three systems, Trained-DeepFace performed the best with 75.71% accuracy. It's the most reliable for classifying multi-class emotions through facial recognition, so we're using it for Ted's vision ability.

## Evaluation Results

![Performance Summary](evaluation/performance_summary_table.png)

*Note: All systems were tested on the same validation dataset for fair comparison.*
