# Base-YOLO - Simple Emotion Detection

This is our basic approach to detecting emotions from faces using simple rules and OpenCV.

## What's Here

- **`main.py`** - The main emotion detection script that uses OpenCV to find faces and classify emotions
- **`main-yolo.py`** - Alternative version with YOLO integration for face detection
- **`yolo11n.pt`** - YOLO model file for detecting objects (including faces)
- **`requirements.txt`** - Python packages needed to run this

## How It Works

We use OpenCV's built-in face detection to find faces in images, then apply simple rules based on image brightness, contrast, and basic features to guess the emotion. It's not super accurate but it's fast and always works.

## Why We Made This

We wanted a simple baseline system that doesn't require training or complex models. It's good for testing and comparison, even though the accuracy is lower than the other approaches.
