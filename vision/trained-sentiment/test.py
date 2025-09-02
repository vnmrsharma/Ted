#!/usr/bin/env python3
"""
Emotion Detection Model Testing
Comprehensive evaluation of the trained model on validation dataset
"""

import torch
import torch.nn as nn
from torchvision import models
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as pd
import json
import argparse
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import warnings
warnings.filterwarnings('ignore')

class EmotionTester:
    """Test the trained emotion detection model"""
    
    def __init__(self, model_path='best_emotion_model.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        self.emotion_names = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        self.emotion_colors = {
            'angry': (0, 0, 255),      # Red
            'disgust': (0, 165, 255),   # Orange
            'fear': (255, 0, 255),      # Magenta
            'happy': (0, 255, 0),       # Green
            'sad': (255, 0, 0),         # Blue
            'surprise': (0, 255, 255),  # Yellow
            'neutral': (128, 128, 128)  # Gray
        }
        
        # Load trained model
        self.model = self.load_model(model_path)
        self.model.eval()
        
        # Validation transforms
        self.transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        print(f"✅ Trained model loaded from {model_path}")
        print(f"🚀 Using device: {self.device}")
        print(f"🎯 Testing emotions: {', '.join(self.emotion_names)}")
    
    def load_model(self, model_path):
        """Load the trained emotion detection model"""
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Create the same model architecture used in training
        class AdvancedEmotionClassifier(nn.Module):
            def __init__(self, num_classes=7, pretrained=False):
                super(AdvancedEmotionClassifier, self).__init__()
                
                # Use EfficientNet-B2 as backbone
                self.backbone = models.efficientnet_b2(pretrained=pretrained)
                
                # Get the number of features from the backbone
                # EfficientNet-B2 has 1408 features in the last layer
                num_features = 1408
                
                # Create a proper classifier that works with the backbone
                self.classifier = nn.Sequential(
                    nn.AdaptiveAvgPool2d((1, 1)),
                    nn.Flatten(),
                    nn.Dropout(p=0.4),
                    nn.Linear(num_features, 1024),
                    nn.BatchNorm1d(1024),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=0.3),
                    nn.Linear(1024, 512),
                    nn.BatchNorm1d(512),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=0.2),
                    nn.Linear(512, num_classes)
                )
            
            def forward(self, x):
                # Pass through backbone (excluding the original classifier)
                x = self.backbone.features(x)
                
                # Apply our custom classifier
                x = self.classifier(x)
                
                return x
        
        # Create model instance
        model = AdvancedEmotionClassifier(num_classes=7, pretrained=False)
        
        # Load trained weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        
        return model
    
    def predict_emotion(self, image):
        """Predict emotion from image"""
        try:
            # Preprocess image
            if isinstance(image, np.ndarray):
                # Convert BGR to RGB if needed
                if len(image.shape) == 3 and image.shape[2] == 3:
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    image_rgb = image
                
                # Apply transforms
                transformed = self.transform(image=image_rgb)
                image_tensor = transformed['image'].unsqueeze(0).to(self.device)
            else:
                image_tensor = image.to(self.device)
            
            # Predict
            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_class].item()
            
            emotion = self.emotion_names[predicted_class]
            return emotion, confidence, probabilities[0].cpu().numpy()
            
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return "unknown", 0.0, np.zeros(7)
    
    def test_on_validation_dataset(self, data_dir='../data', max_samples_per_class=100):
        """Test model on validation dataset"""
        print(f"\n🧪 Testing model on validation dataset...")
        print(f"📁 Data directory: {data_dir}")
        
        validation_path = Path(data_dir) / 'validation'
        if not validation_path.exists():
            print(f"❌ Validation directory not found: {validation_path}")
            return
        
        # Results storage
        all_predictions = []
        all_emotions = []
        all_confidences = []
        all_image_paths = []
        class_results = {emotion: {'correct': 0, 'total': 0, 'confidences': []} for emotion in self.emotion_names}
        
        # Test each emotion class
        for emotion_idx, emotion_name in enumerate(self.emotion_names):
            emotion_path = validation_path / emotion_name
            if not emotion_path.exists():
                continue
            
            print(f"\n📸 Testing {emotion_name} class...")
            
            # Get image files
            image_files = list(emotion_path.glob("*.jpg"))
            if not image_files:
                print(f"   ⚠️  No images found for {emotion_name}")
                continue
            
            # Limit samples per class for faster testing
            if len(image_files) > max_samples_per_class:
                import random
                image_files = random.sample(image_files, max_samples_per_class)
            
            correct_predictions = 0
            total_predictions = len(image_files)
            
            # Test each image
            for img_file in tqdm(image_files, desc=f"Testing {emotion_name}"):
                # Load image
                image = cv2.imread(str(img_file))
                if image is None:
                    continue
                
                # Predict emotion
                predicted_emotion, confidence, all_probs = self.predict_emotion(image)
                
                # Store results
                all_predictions.append(self.emotion_names.index(predicted_emotion))
                all_emotions.append(emotion_idx)
                all_confidences.append(confidence)
                all_image_paths.append(str(img_file))
                
                # Check if correct
                is_correct = predicted_emotion == emotion_name
                if is_correct:
                    correct_predictions += 1
                
                # Update class results
                class_results[emotion_name]['total'] += 1
                if is_correct:
                    class_results[emotion_name]['correct'] += 1
                class_results[emotion_name]['confidences'].append(confidence)
            
            # Print class results
            accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
            print(f"   ✅ {emotion_name}: {correct_predictions}/{total_predictions} correct ({accuracy:.1f}%)")
        
        # Overall evaluation
        print(f"\n📊 Overall Test Results:")
        print("=" * 50)
        
        # Calculate overall accuracy
        overall_accuracy = accuracy_score(all_emotions, all_predictions) * 100
        print(f"🎯 Overall Accuracy: {overall_accuracy:.2f}%")
        
        # Per-class accuracy
        print(f"\n📈 Per-Class Accuracy:")
        for emotion_name, results in class_results.items():
            if results['total'] > 0:
                class_acc = (results['correct'] / results['total']) * 100
                avg_conf = np.mean(results['confidences']) if results['confidences'] else 0
                print(f"   {emotion_name:10}: {class_acc:5.1f}% ({results['correct']:3d}/{results['total']:3d}) | Avg Conf: {avg_conf:.3f}")
        
        # Detailed metrics
        self.generate_detailed_metrics(all_emotions, all_predictions, all_confidences, all_image_paths)
        
        return overall_accuracy, class_results
    
    def generate_detailed_metrics(self, emotions, predictions, confidences, image_paths):
        """Generate detailed evaluation metrics"""
        print(f"\n📊 Detailed Metrics:")
        print("=" * 50)
        
        # Classification report
        emotion_names = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        report = classification_report(emotions, predictions, target_names=emotion_names)
        print("Classification Report:")
        print(report)
        
        # Confusion matrix
        cm = confusion_matrix(emotions, predictions)
        self.plot_confusion_matrix(cm, emotion_names)
        
        # Confidence analysis
        self.analyze_confidence(emotions, predictions, confidences)
        
        # Save results
        self.save_test_results(emotions, predictions, confidences, image_paths, report, cm)
    
    def plot_confusion_matrix(self, cm, emotion_names):
        """Plot confusion matrix"""
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=emotion_names, yticklabels=emotion_names)
        plt.title('Emotion Detection Confusion Matrix (Validation)', fontsize=16)
        plt.ylabel('True Emotion', fontsize=14)
        plt.xlabel('Predicted Emotion', fontsize=14)
        plt.tight_layout()
        plt.savefig('test_confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("📊 Confusion matrix saved as test_confusion_matrix.png")
    
    def analyze_confidence(self, emotions, predictions, confidences):
        """Analyze prediction confidence"""
        print(f"\n🎯 Confidence Analysis:")
        
        # Overall confidence
        overall_conf = np.mean(confidences)
        print(f"   📊 Overall Average Confidence: {overall_conf:.3f}")
        
        # Confidence by correctness
        correct_mask = np.array(emotions) == np.array(predictions)
        correct_conf = np.mean(np.array(confidences)[correct_mask]) if np.any(correct_mask) else 0
        incorrect_conf = np.mean(np.array(confidences)[~correct_mask]) if np.any(~correct_mask) else 0
        
        print(f"   ✅ Correct Predictions Avg Confidence: {correct_conf:.3f}")
        print(f"   ❌ Incorrect Predictions Avg Confidence: {incorrect_conf:.3f}")
        
        # Confidence distribution
        plt.figure(figsize=(12, 5))
        
        # Confidence histogram
        plt.subplot(1, 2, 1)
        plt.hist(confidences, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title('Prediction Confidence Distribution')
        plt.xlabel('Confidence')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        # Confidence vs correctness
        plt.subplot(1, 2, 2)
        plt.scatter(confidences, correct_mask, alpha=0.6, c=['red' if not c else 'green' for c in correct_mask])
        plt.title('Confidence vs Prediction Correctness')
        plt.xlabel('Confidence')
        plt.ylabel('Correct (1) / Incorrect (0)')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('confidence_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("📊 Confidence analysis saved as confidence_analysis.png")
    
    def save_test_results(self, emotions, predictions, confidences, image_paths, report, cm):
        """Save test results to files"""
        # Save detailed results
        results = {
            'overall_accuracy': accuracy_score(emotions, predictions) * 100,
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'predictions': {
                'true_emotions': emotions,
                'predicted_emotions': predictions,
                'confidences': confidences,
                'image_paths': image_paths
            },
            'per_class_accuracy': {}
        }
        
        # Calculate per-class accuracy
        for i, emotion_name in enumerate(self.emotion_names):
            mask = np.array(emotions) == i
            if np.any(mask):
                class_acc = accuracy_score(np.array(emotions)[mask], np.array(predictions)[mask]) * 100
                results['per_class_accuracy'][emotion_name] = class_acc
        
        # Save to JSON
        with open('test_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save to CSV
        df = pd.DataFrame({
            'image_path': image_paths,
            'true_emotion': [self.emotion_names[e] for e in emotions],
            'predicted_emotion': [self.emotion_names[p] for p in predictions],
            'confidence': confidences,
            'correct': [e == p for e, p in zip(emotions, predictions)]
        })
        df.to_csv('test_results.csv', index=False)
        
        print("💾 Test results saved to test_results.json and test_results.csv")
    
    def test_single_image(self, image_path):
        """Test model on a single image"""
        if not Path(image_path).exists():
            print(f"❌ Image not found: {image_path}")
            return
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ Could not load image: {image_path}")
            return
        
        # Predict emotion
        emotion, confidence, all_probs = self.predict_emotion(image)
        
        print(f"\n🖼️  Single Image Test: {image_path}")
        print(f"🎯 Predicted Emotion: {emotion}")
        print(f"📊 Confidence: {confidence:.3f}")
        
        print(f"\n📈 All Emotion Probabilities:")
        for i, (emotion_name, prob) in enumerate(zip(self.emotion_names, all_probs)):
            print(f"   {emotion_name:10}: {prob:.3f}")
        
        return emotion, confidence, all_probs

def main():
    """Main testing function"""
    parser = argparse.ArgumentParser(description="Emotion Detection Model Testing")
    parser.add_argument("--model", default="best_emotion_model.pth", 
                       help="Path to trained model file")
    parser.add_argument("--data_dir", default="../data", 
                       help="Path to dataset directory")
    parser.add_argument("--max_samples", type=int, default=100, 
                       help="Maximum samples per class for testing")
    parser.add_argument("--single_image", default="", 
                       help="Test single image instead of full dataset")
    
    args = parser.parse_args()
    
    print("🎯 Emotion Detection Model Testing")
    print("=" * 50)
    
    try:
        # Initialize tester
        tester = EmotionTester(args.model)
        
        if args.single_image:
            # Test single image
            tester.test_single_image(args.single_image)
        else:
            # Test on validation dataset
            accuracy, class_results = tester.test_on_validation_dataset(
                args.data_dir, args.max_samples
            )
            
            print(f"\n🎉 Testing completed!")
            print(f"📊 Overall Accuracy: {accuracy:.2f}%")
        
    except Exception as e:
        print(f"❌ Testing error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
