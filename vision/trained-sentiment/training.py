#!/usr/bin/env python3
"""
Advanced Emotion Detection Model Training
Uses EfficientNet-B2 with advanced augmentation and training techniques
"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import pandas as pd
import json
import argparse
from tqdm import tqdm
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
import warnings
warnings.filterwarnings('ignore')

class AdvancedEmotionDataset(Dataset):
    """Advanced dataset with strong augmentation"""
    
    def __init__(self, data_dir, transform=None, mode='train'):
        self.data_dir = data_dir
        self.transform = transform
        self.mode = mode
        
        # Emotion mapping (7 classes)
        self.emotion_map = {
            'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3,
            'sad': 4, 'surprise': 5, 'neutral': 6
        }
        
        # Load dataset structure
        self.samples = self._load_dataset()
        print(f"📊 Loaded {len(self.samples)} {mode} samples")
        
        # Class distribution
        self._show_class_distribution()
    
    def _load_dataset(self):
        """Load dataset structure"""
        samples = []
        base_path = Path(self.data_dir) / self.mode
        
        if not base_path.exists():
            print(f"⚠️  {base_path} not found!")
            return []
        
        for emotion in self.emotion_map.keys():
            emotion_path = base_path / emotion
            if emotion_path.exists():
                for img_file in emotion_path.glob("*.jpg"):
                    samples.append({
                        'image_path': str(img_file),
                        'emotion': self.emotion_map[emotion],
                        'emotion_name': emotion
                    })
        
        return samples
    
    def _show_class_distribution(self):
        """Show class distribution"""
        emotion_counts = {}
        for sample in self.samples:
            emotion = sample['emotion_name']
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        print(f"📈 {self.mode.title()} Class Distribution:")
        for emotion, count in emotion_counts.items():
            print(f"   {emotion}: {count} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        image = cv2.imread(sample['image_path'])
        if image is None:
            # Fallback for corrupted images
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        return {
            'image': image,
            'emotion': torch.tensor(sample['emotion'], dtype=torch.long),
            'emotion_name': sample['emotion_name']
        }

class AdvancedEmotionClassifier(nn.Module):
    """Advanced emotion classification model"""
    
    def __init__(self, num_classes=7, pretrained=True):
        super(AdvancedEmotionClassifier, self).__init__()
        
        # Use EfficientNet-B2 as backbone (better than B0)
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
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize classifier weights"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Pass through backbone (excluding the original classifier)
        x = self.backbone.features(x)
        
        # Apply our custom classifier
        x = self.classifier(x)
        
        return x

class AdvancedEmotionTrainer:
    """Advanced training pipeline for emotion detection"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
        
        print(f"🚀 Training on device: {self.device}")
        
        # Advanced data transforms
        self.train_transform = A.Compose([
            A.Resize(256, 256),
            A.RandomCrop(224, 224),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.RandomRotate90(p=0.3),
            A.Rotate(limit=15, p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.OneOf([
                A.MotionBlur(blur_limit=3, p=0.3),
                A.MedianBlur(blur_limit=3, p=0.3),
                A.Blur(blur_limit=3, p=0.3),
            ], p=0.3),
            A.OneOf([
                A.CLAHE(clip_limit=2, p=0.3),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.3),
            ], p=0.5),
            A.OneOf([
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                A.ISONoise(color_shift=(0.01, 0.05), p=0.3),
                A.MultiplicativeNoise(multiplier=[0.9, 1.1], p=0.3),
            ], p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        self.val_transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
        
        # Initialize model
        self.model = AdvancedEmotionClassifier(num_classes=7, pretrained=True)
        self.model.to(self.device)
        
        # Advanced loss function with label smoothing
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # Advanced optimizer with weight decay
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay'],
            betas=(0.9, 0.999)
        )
        
        # Advanced learning rate scheduler
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config['learning_rate'],
            epochs=config['epochs'],
            steps_per_epoch=1,  # Will be updated in load_datasets
            pct_start=0.3,
            anneal_strategy='cos'
        )
        
        # Training history
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }
        
        # Best model tracking
        self.best_val_acc = 0.0
        self.best_model_path = 'best_emotion_model.pth'
    
    def load_datasets(self, data_dir):
        """Load training and validation datasets"""
        print("📥 Loading datasets...")
        
        # Load datasets
        train_dataset = AdvancedEmotionDataset(data_dir, self.train_transform, 'train')
        val_dataset = AdvancedEmotionDataset(data_dir, self.val_transform, 'validation')
        
        if len(train_dataset) == 0 or len(val_dataset) == 0:
            raise ValueError("No data found. Please check the dataset path.")
        
        # Update scheduler steps
        self.scheduler.total_steps = len(train_dataset) // self.config['batch_size'] * self.config['epochs']
        
        # Create data loaders with proper workers
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['batch_size'],
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        print(f"✅ Datasets loaded: {len(train_dataset)} train, {len(val_dataset)} val")
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch in pbar:
            images = batch['image'].to(self.device)
            emotions = batch['emotion'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, emotions)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            self.scheduler.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += emotions.size(0)
            correct += (predicted == emotions).sum().item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f"{loss.item():.4f}",
                'Acc': f"{100*correct/total:.2f}%",
                'LR': f"{self.scheduler.get_last_lr()[0]:.6f}"
            })
        
        avg_loss = total_loss / len(self.train_loader)
        avg_acc = 100 * correct / total
        
        return avg_loss, avg_acc
    
    def validate_epoch(self):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_predictions = []
        all_emotions = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc="Validation")
            for batch in pbar:
                images = batch['image'].to(self.device)
                emotions = batch['emotion'].to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                loss = self.criterion(outputs, emotions)
                
                # Statistics
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += emotions.size(0)
                correct += (predicted == emotions).sum().item()
                
                # Store predictions for metrics
                all_predictions.extend(predicted.cpu().numpy())
                all_emotions.extend(emotions.cpu().numpy())
                
                # Update progress bar
                pbar.set_postfix({
                    'Loss': f"{loss.item():.4f}",
                    'Acc': f"{100*correct/total:.2f}%"
                })
        
        avg_loss = total_loss / len(self.val_loader)
        avg_acc = 100 * correct / total
        
        return avg_loss, avg_acc, all_predictions, all_emotions
    
    def train(self, data_dir):
        """Main training loop"""
        print("🚀 Starting advanced emotion detection training...")
        
        # Load datasets
        self.load_datasets(data_dir)
        
        # Training loop
        for epoch in range(self.config['epochs']):
            print(f"\n📅 Epoch {epoch+1}/{self.config['epochs']}")
            print("=" * 50)
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc, predictions, emotions = self.validate_epoch()
            
            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Print results
            print(f"📊 Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"📊 Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            print(f"📊 Learning Rate: {self.scheduler.get_last_lr()[0]:.6f}")
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.save_model(self.best_model_path)
                print(f"💾 New best model saved! Accuracy: {val_acc:.2f}%")
            
            # Save checkpoint
            if (epoch + 1) % 10 == 0:
                checkpoint_path = f"checkpoint_epoch_{epoch+1}.pth"
                self.save_model(checkpoint_path)
                print(f"💾 Checkpoint saved: {checkpoint_path}")
        
        # Final evaluation
        print("\n🎉 Training completed!")
        self.evaluate_model(predictions, emotions)
        
        # Plot training history
        self.plot_training_history()
        
        print(f"\n🎉 Training completed! Best validation accuracy: {self.best_val_acc:.2f}%")
    
    def evaluate_model(self, predictions, emotions):
        """Evaluate model performance"""
        print("\n📈 Model Evaluation")
        print("=" * 30)
        
        # Classification report
        emotion_names = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        report = classification_report(emotions, predictions, target_names=emotion_names)
        print("Classification Report:")
        print(report)
        
        # Confusion matrix
        cm = confusion_matrix(emotions, predictions)
        self.plot_confusion_matrix(cm, emotion_names)
        
        # Save metrics
        # Convert predictions and emotions to numpy arrays if they aren't already
        if not isinstance(predictions, np.ndarray):
            predictions = np.array(predictions)
        if not isinstance(emotions, np.ndarray):
            emotions = np.array(emotions)
            
        metrics = {
            'accuracy': 100 * (predictions == emotions).sum() / len(emotions),
            'classification_report': report,
            'confusion_matrix': cm.tolist()
        }
        
        with open('training_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print("💾 Metrics saved to training_metrics.json")
    
    def plot_confusion_matrix(self, cm, emotion_names):
        """Plot confusion matrix"""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=emotion_names, yticklabels=emotion_names)
        plt.title('Emotion Detection Confusion Matrix')
        plt.ylabel('True Emotion')
        plt.xlabel('Predicted Emotion')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("📊 Confusion matrix saved as confusion_matrix.png")
    
    def plot_training_history(self):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Loss plot
        ax1.plot(self.history['train_loss'], label='Train Loss')
        ax1.plot(self.history['val_loss'], label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(self.history['train_acc'], label='Train Accuracy')
        ax2.plot(self.history['val_acc'], label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("📊 Training history saved as training_history.png")
    
    def save_model(self, filename):
        """Save model checkpoint"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'history': self.history,
            'best_val_acc': self.best_val_acc
        }
        torch.save(checkpoint, filename)
    
    def load_model(self, filename):
        """Load model checkpoint"""
        checkpoint = torch.load(filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.history = checkpoint['history']
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
        print(f"✅ Model loaded from {filename}")

def main():
    """Main training function"""
    parser = argparse.ArgumentParser(description="Advanced Emotion Detection Model Training")
    parser.add_argument("--data_dir", default="../data", help="Path to dataset directory")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    
    args = parser.parse_args()
    
    # Training configuration
    config = {
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'weight_decay': args.weight_decay
    }
    
    print("🎯 Advanced Emotion Detection Model Training")
    print("=" * 60)
    print(f"📊 Epochs: {config['epochs']}")
    print(f"📦 Batch Size: {config['batch_size']}")
    print(f"📚 Learning Rate: {config['learning_rate']}")
    print(f"⚖️  Weight Decay: {config['weight_decay']}")
    print("=" * 60)
    
    # Check if dataset exists
    if not Path(args.data_dir).exists():
        print(f"❌ Dataset not found at {args.data_dir}")
        return
    
    # Initialize trainer
    trainer = AdvancedEmotionTrainer(config)
    
    # Start training
    try:
        trainer.train(args.data_dir)
    except KeyboardInterrupt:
        print("\n⏹️  Training interrupted by user")
        trainer.save_model('interrupted_checkpoint.pth')
    except Exception as e:
        print(f"\n❌ Training error: {e}")
        trainer.save_model('error_checkpoint.pth')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
