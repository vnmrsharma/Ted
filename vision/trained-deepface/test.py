#!/usr/bin/env python3
"""
Comprehensive DeepFace Emotion Detection Testing
Generates all required test results for model evaluation
"""

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from deepface import DeepFace
from pathlib import Path
import json
import time
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tqdm import tqdm
import argparse
import warnings
warnings.filterwarnings('ignore')

class DeepFaceEmotionTester:
    """Comprehensive DeepFace emotion detection testing and evaluation"""
    
    def __init__(self, data_dir="../data"):
        self.data_dir = Path(data_dir)
        self.results_dir = Path("test_results")
        self.results_dir.mkdir(exist_ok=True)
        
        # Emotion mapping
        self.emotion_map = {
            'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3,
            'sad': 4, 'surprise': 5, 'neutral': 6
        }
        
        # Reverse mapping for results
        self.id_to_emotion = {v: k for k, v in self.emotion_map.items()}
        
        # Results storage
        self.test_results = []
        self.performance_metrics = {}
        
        # Initialize DeepFace
        self._initialize_deepface()
        
        # Test configuration
        self.test_config = {
            'max_samples_per_class': None,  # None = all samples
            'batch_size': 10,
            'analysis_timeout': 30,  # seconds
            'save_failed_images': True,
            'generate_visualizations': True
        }
    
    def _initialize_deepface(self):
        """Initialize DeepFace models"""
        print("🔧 Initializing DeepFace models...")
        try:
            # Test with a dummy image to ensure models are loaded
            dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
            _ = DeepFace.analyze(
                dummy_img, 
                actions=['emotion'],
                enforce_detection=False,
                silent=True
            )
            print("✅ DeepFace models initialized successfully")
        except Exception as e:
            print(f"❌ Error initializing DeepFace: {e}")
            raise
    
    def load_test_dataset(self, mode='validation', max_samples_per_class=None):
        """Load test dataset from the emotion data directory"""
        print(f"📂 Loading {mode} dataset...")
        
        test_samples = []
        base_path = self.data_dir / mode
        
        if not base_path.exists():
            print(f"⚠️  {base_path} not found!")
            return []
        
        for emotion in self.emotion_map.keys():
            emotion_path = base_path / emotion
            if emotion_path.exists():
                image_files = list(emotion_path.glob("*.jpg"))
                
                # Limit samples per class if specified
                if max_samples_per_class:
                    image_files = image_files[:max_samples_per_class]
                
                for img_file in image_files:
                    test_samples.append({
                        'image_path': str(img_file),
                        'true_emotion': emotion,
                        'true_emotion_id': self.emotion_map[emotion]
                    })
        
        print(f"✅ Loaded {len(test_samples)} test samples")
        
        # Show class distribution
        self._show_class_distribution(test_samples)
        
        return test_samples
    
    def _show_class_distribution(self, samples):
        """Show class distribution of test samples"""
        emotion_counts = {}
        for sample in samples:
            emotion = sample['true_emotion']
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        print("\n📊 Class Distribution:")
        for emotion, count in emotion_counts.items():
            print(f"   {emotion:10}: {count} samples")
    
    def test_single_image(self, image_path, true_emotion):
        """Test emotion detection on a single image"""
        try:
            # Load and preprocess image
            image = cv2.imread(image_path)
            if image is None:
                return {
                    'success': False,
                    'error': 'Failed to load image',
                    'true_emotion': true_emotion,
                    'image_path': image_path
                }
            
            # Convert BGR to RGB for DeepFace
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Analyze emotion using DeepFace
            start_time = time.time()
            result = DeepFace.analyze(
                rgb_image,
                actions=['emotion'],
                enforce_detection=False,
                silent=True
            )
            analysis_time = time.time() - start_time
            
            if isinstance(result, list) and len(result) > 0:
                emotion_data = result[0]
                if 'emotion' in emotion_data:
                    emotions = emotion_data['emotion']
                    
                    # Find dominant emotion
                    predicted_emotion = max(emotions.items(), key=lambda x: x[1])
                    confidence = predicted_emotion[1]
                    
                    # Check if prediction is correct
                    is_correct = predicted_emotion[0] == true_emotion
                    
                    return {
                        'success': True,
                        'true_emotion': true_emotion,
                        'predicted_emotion': predicted_emotion[0],
                        'confidence': confidence,
                        'all_emotions': emotions,
                        'is_correct': is_correct,
                        'analysis_time': analysis_time,
                        'image_path': image_path
                    }
            
            return {
                'success': False,
                'error': 'No face detected or analysis failed',
                'true_emotion': true_emotion,
                'image_path': image_path
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'true_emotion': true_emotion,
                'image_path': image_path
            }
    
    def run_batch_testing(self, test_samples, batch_size=10):
        """Run batch testing on the test dataset"""
        print(f"\n🧪 Starting batch testing on {len(test_samples)} samples...")
        
        total_samples = len(test_samples)
        successful_predictions = 0
        failed_predictions = 0
        
        # Process samples in batches
        for i in tqdm(range(0, total_samples, batch_size), desc="Testing batches"):
            batch = test_samples[i:i + batch_size]
            
            for sample in batch:
                result = self.test_single_image(
                    sample['image_path'], 
                    sample['true_emotion']
                )
                
                if result['success']:
                    successful_predictions += 1
                    self.test_results.append(result)
                else:
                    failed_predictions += 1
                    print(f"❌ Failed to analyze {sample['image_path']}: {result['error']}")
                    
                    # Save failed image for debugging if enabled
                    if self.test_config['save_failed_images']:
                        self._save_failed_image(sample['image_path'], result['error'])
        
        print(f"\n✅ Testing completed!")
        print(f"   Successful predictions: {successful_predictions}")
        print(f"   Failed predictions: {failed_predictions}")
        print(f"   Success rate: {successful_predictions/total_samples*100:.1f}%")
    
    def _save_failed_image(self, image_path, error):
        """Save failed image for debugging"""
        try:
            failed_dir = self.results_dir / "failed_images"
            failed_dir.mkdir(exist_ok=True)
            
            # Copy image to failed directory
            img_name = Path(image_path).name
            failed_path = failed_dir / f"failed_{img_name}"
            
            import shutil
            shutil.copy2(image_path, failed_path)
            
            # Save error info
            error_info = {
                'original_path': image_path,
                'error': error,
                'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            error_file = failed_dir / f"failed_{img_name}.json"
            with open(error_file, 'w') as f:
                json.dump(error_info, f, indent=2)
                
        except Exception as e:
            print(f"⚠️  Could not save failed image: {e}")
    
    def calculate_metrics(self):
        """Calculate comprehensive performance metrics"""
        if not self.test_results:
            print("⚠️  No test results to analyze")
            return
        
        print("\n📊 Calculating performance metrics...")
        
        # Extract predictions and true labels
        y_true = [result['true_emotion'] for result in self.test_results]
        y_pred = [result['predicted_emotion'] for result in self.test_results]
        
        # Calculate accuracy
        accuracy = accuracy_score(y_true, y_pred)
        
        # Generate classification report
        class_report = classification_report(
            y_true, y_pred, 
            target_names=list(self.emotion_map.keys()),
            output_dict=True
        )
        
        # Generate confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=list(self.emotion_map.keys()))
        
        # Calculate additional metrics
        total_samples = len(self.test_results)
        correct_predictions = sum(1 for r in self.test_results if r['is_correct'])
        
        # Confidence analysis
        confidences = [result['confidence'] for result in self.test_results]
        avg_confidence = np.mean(confidences)
        confidence_std = np.std(confidences)
        
        # Analysis time analysis
        analysis_times = [result['analysis_time'] for result in self.test_results]
        avg_analysis_time = np.mean(analysis_times)
        analysis_time_std = np.std(analysis_times)
        
        # Store comprehensive metrics
        self.performance_metrics = {
            'accuracy': accuracy,
            'total_samples': total_samples,
            'correct_predictions': correct_predictions,
            'incorrect_predictions': total_samples - correct_predictions,
            'classification_report': class_report,
            'confusion_matrix': cm,
            'emotion_map': self.emotion_map,
            'confidence_stats': {
                'mean': avg_confidence,
                'std': confidence_std,
                'min': min(confidences),
                'max': max(confidences)
            },
            'analysis_time_stats': {
                'mean': avg_analysis_time,
                'std': analysis_time_std,
                'min': min(analysis_times),
                'max': max(analysis_times)
            }
        }
        
        # Print summary
        print(f"✅ Overall Accuracy: {accuracy:.4f}")
        print(f"✅ Total Tested Samples: {total_samples}")
        print(f"✅ Correct Predictions: {correct_predictions}")
        print(f"✅ Average Confidence: {avg_confidence:.2f}% ± {confidence_std:.2f}%")
        print(f"✅ Average Analysis Time: {avg_analysis_time*1000:.1f} ms ± {analysis_time_std*1000:.1f} ms")
        
        # Print per-class metrics
        print("\n📈 Per-Class Performance:")
        for emotion in self.emotion_map.keys():
            if emotion in class_report:
                precision = class_report[emotion]['precision']
                recall = class_report[emotion]['recall']
                f1 = class_report[emotion]['f1-score']
                support = class_report[emotion]['support']
                print(f"   {emotion:10}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}, Support={support}")
    
    def generate_visualizations(self):
        """Generate comprehensive visualization plots"""
        if not self.performance_metrics:
            print("⚠️  No performance metrics to visualize")
            return
        
        print("\n🎨 Generating visualizations...")
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Confusion Matrix
        self._plot_confusion_matrix()
        
        # 2. Per-class Performance
        self._plot_per_class_performance()
        
        # 3. Confidence Distribution
        self._plot_confidence_distribution()
        
        # 4. Analysis Time Distribution
        self._plot_analysis_time_distribution()
        
        # 5. Emotion Accuracy Comparison
        self._plot_emotion_accuracy_comparison()
        
        # 6. Performance Summary
        self._plot_performance_summary()
        
        print("✅ Visualizations saved to test_results/ directory")
    
    def _plot_confusion_matrix(self):
        """Plot confusion matrix"""
        cm = self.performance_metrics['confusion_matrix']
        emotion_names = list(self.emotion_map.keys())
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d', 
            cmap='Blues',
            xticklabels=emotion_names,
            yticklabels=emotion_names
        )
        plt.title('DeepFace Emotion Detection - Confusion Matrix', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted Emotion', fontsize=12)
        plt.ylabel('True Emotion', fontsize=12)
        plt.tight_layout()
        plt.savefig(self.results_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_per_class_performance(self):
        """Plot per-class performance metrics"""
        class_report = self.performance_metrics['classification_report']
        emotions = list(self.emotion_map.keys())
        
        # Extract metrics
        precision = [class_report[emotion]['precision'] for emotion in emotions]
        recall = [class_report[emotion]['recall'] for emotion in emotions]
        f1 = [class_report[emotion]['f1-score'] for emotion in emotions]
        
        # Create plot
        x = np.arange(len(emotions))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
        ax.bar(x, recall, width, label='Recall', alpha=0.8)
        ax.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
        
        ax.set_xlabel('Emotions', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('DeepFace Emotion Detection - Per-Class Performance', fontsize=16, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(emotions, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.results_dir / 'per_class_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_confidence_distribution(self):
        """Plot confidence distribution for predictions"""
        confidences = [result['confidence'] for result in self.test_results]
        
        plt.figure(figsize=(10, 6))
        plt.hist(confidences, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('Confidence Score (%)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('DeepFace Emotion Detection - Confidence Distribution', fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Add statistics
        mean_conf = np.mean(confidences)
        plt.axvline(mean_conf, color='red', linestyle='--', label=f'Mean: {mean_conf:.1f}%')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'confidence_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_analysis_time_distribution(self):
        """Plot analysis time distribution"""
        analysis_times = [result['analysis_time'] * 1000 for result in self.test_results]  # Convert to ms
        
        plt.figure(figsize=(10, 6))
        plt.hist(analysis_times, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        plt.xlabel('Analysis Time (ms)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('DeepFace Emotion Detection - Analysis Time Distribution', fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3)
        
        # Add statistics
        mean_time = np.mean(analysis_times)
        plt.axvline(mean_time, color='red', linestyle='--', label=f'Mean: {mean_time:.1f} ms')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'analysis_time_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_emotion_accuracy_comparison(self):
        """Plot accuracy comparison across emotions"""
        class_report = self.performance_metrics['classification_report']
        emotions = list(self.emotion_map.keys())
        
        # Extract F1 scores
        f1_scores = [class_report[emotion]['f1-score'] for emotion in emotions]
        
        # Create plot
        plt.figure(figsize=(12, 6))
        bars = plt.bar(emotions, f1_scores, color=sns.color_palette("husl", len(emotions)))
        plt.xlabel('Emotions', fontsize=12)
        plt.ylabel('F1-Score', fontsize=12)
        plt.title('DeepFace Emotion Detection - F1-Score by Emotion', fontsize=16, fontweight='bold')
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, score in zip(bars, f1_scores):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'emotion_accuracy_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_performance_summary(self):
        """Plot overall performance summary"""
        metrics = self.performance_metrics
        
        # Create summary figure
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Overall accuracy
        ax1.pie([metrics['correct_predictions'], metrics['incorrect_predictions']], 
                labels=['Correct', 'Incorrect'], autopct='%1.1f%%', startangle=90)
        ax1.set_title('Overall Prediction Accuracy', fontweight='bold')
        
        # 2. Confidence stats
        conf_stats = metrics['confidence_stats']
        ax2.bar(['Min', 'Mean', 'Max'], 
                [conf_stats['min'], conf_stats['mean'], conf_stats['max']],
                color=['red', 'blue', 'green'])
        ax2.set_title('Confidence Statistics (%)', fontweight='bold')
        ax2.set_ylabel('Confidence (%)')
        
        # 3. Analysis time stats
        time_stats = metrics['analysis_time_stats']
        ax3.bar(['Min', 'Mean', 'Max'], 
                [time_stats['min']*1000, time_stats['mean']*1000, time_stats['max']*1000],
                color=['red', 'blue', 'green'])
        ax3.set_title('Analysis Time Statistics (ms)', fontweight='bold')
        ax3.set_ylabel('Time (ms)')
        
        # 4. Sample count
        ax4.bar(['Total', 'Successful', 'Failed'], 
                [metrics['total_samples'], metrics['correct_predictions'], 
                 metrics['incorrect_predictions']],
                color=['gray', 'green', 'red'])
        ax4.set_title('Sample Statistics', fontweight='bold')
        ax4.set_ylabel('Count')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'performance_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_results(self):
        """Save comprehensive test results and metrics"""
        if not self.test_results:
            print("⚠️  No results to save")
            return
        
        print("\n💾 Saving results...")
        
        # Save detailed results
        results_file = self.results_dir / 'test_results.json'
        with open(results_file, 'w') as f:
            json.dump(self.test_results, f, indent=2, default=str)
        
        # Save performance metrics
        metrics_file = self.results_dir / 'performance_metrics.json'
        with open(metrics_file, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            metrics_copy = self.performance_metrics.copy()
            if 'confusion_matrix' in metrics_copy:
                metrics_copy['confusion_matrix'] = metrics_copy['confusion_matrix'].tolist()
            json.dump(metrics_copy, f, indent=2, default=str)
        
        # Save summary CSV
        summary_data = []
        for result in self.test_results:
            summary_data.append({
                'image_path': result['image_path'],
                'true_emotion': result['true_emotion'],
                'predicted_emotion': result['predicted_emotion'],
                'confidence': result['confidence'],
                'is_correct': result['is_correct'],
                'analysis_time_ms': result['analysis_time'] * 1000
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_file = self.results_dir / 'test_summary.csv'
        summary_df.to_csv(summary_file, index=False)
        
        # Save test configuration
        config_file = self.results_dir / 'test_config.json'
        with open(config_file, 'w') as f:
            json.dump(self.test_config, f, indent=2)
        
        # Generate summary report
        self._generate_summary_report()
        
        print(f"✅ Results saved to {self.results_dir}/")
        print(f"   - test_results.json: Detailed results")
        print(f"   - performance_metrics.json: Performance metrics")
        print(f"   - test_summary.csv: Summary table")
        print(f"   - test_config.json: Test configuration")
        print(f"   - summary_report.txt: Human-readable summary")
        print(f"   - Multiple visualization plots")
    
    def _generate_summary_report(self):
        """Generate human-readable summary report"""
        report_file = self.results_dir / 'summary_report.txt'
        
        with open(report_file, 'w') as f:
            f.write("DEEPFACE EMOTION DETECTION - TEST SUMMARY REPORT\n")
            f.write("=" * 60 + "\n\n")
            
            f.write(f"Test Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total Samples: {self.performance_metrics['total_samples']}\n")
            f.write(f"Overall Accuracy: {self.performance_metrics['accuracy']:.4f}\n")
            f.write(f"Correct Predictions: {self.performance_metrics['correct_predictions']}\n")
            f.write(f"Incorrect Predictions: {self.performance_metrics['incorrect_predictions']}\n\n")
            
            f.write("PER-CLASS PERFORMANCE:\n")
            f.write("-" * 30 + "\n")
            class_report = self.performance_metrics['classification_report']
            for emotion in self.emotion_map.keys():
                if emotion in class_report:
                    precision = class_report[emotion]['precision']
                    recall = class_report[emotion]['recall']
                    f1 = class_report[emotion]['f1-score']
                    support = class_report[emotion]['support']
                    f.write(f"{emotion:10}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}, Support={support}\n")
            
            f.write(f"\nCONFIDENCE STATISTICS:\n")
            f.write("-" * 30 + "\n")
            conf_stats = self.performance_metrics['confidence_stats']
            f.write(f"Mean Confidence: {conf_stats['mean']:.2f}% ± {conf_stats['std']:.2f}%\n")
            f.write(f"Confidence Range: {conf_stats['min']:.1f}% - {conf_stats['max']:.1f}%\n")
            
            f.write(f"\nANALYSIS TIME STATISTICS:\n")
            f.write("-" * 30 + "\n")
            time_stats = self.performance_metrics['analysis_time_stats']
            f.write(f"Mean Analysis Time: {time_stats['mean']*1000:.1f} ms ± {time_stats['std']*1000:.1f} ms\n")
            f.write(f"Time Range: {time_stats['min']*1000:.1f} ms - {time_stats['max']*1000:.1f} ms\n")
    
    def run_complete_test(self, mode='validation', max_samples_per_class=None, batch_size=10):
        """Run complete testing pipeline"""
        print("🚀 Starting DeepFace Emotion Detection Testing Pipeline")
        print("=" * 60)
        
        try:
            # Load test dataset
            test_samples = self.load_test_dataset(mode, max_samples_per_class)
            if not test_samples:
                print("❌ No test samples loaded")
                return
            
            # Update test configuration
            self.test_config['max_samples_per_class'] = max_samples_per_class
            self.test_config['batch_size'] = batch_size
            
            # Run batch testing
            self.run_batch_testing(test_samples, batch_size)
            
            # Calculate metrics
            self.calculate_metrics()
            
            # Generate visualizations
            if self.test_config['generate_visualizations']:
                self.generate_visualizations()
            
            # Save results
            self.save_results()
            
            print("\n🎉 Testing pipeline completed successfully!")
            
        except Exception as e:
            print(f"❌ Error in testing pipeline: {e}")
            raise

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='DeepFace Emotion Detection Testing')
    parser.add_argument('--data-dir', type=str, default='../data', 
                       help='Path to emotion dataset directory (default: ../data)')
    parser.add_argument('--mode', type=str, default='validation', choices=['train', 'validation'],
                       help='Dataset mode to test (default: validation)')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum samples per class (default: all)')
    parser.add_argument('--batch-size', type=int, default=10,
                       help='Batch size for testing (default: 10)')
    
    args = parser.parse_args()
    
    try:
        # Create tester and run tests
        tester = DeepFaceEmotionTester(data_dir=args.data_dir)
        tester.run_complete_test(
            mode=args.mode,
            max_samples_per_class=args.max_samples,
            batch_size=args.batch_size
        )
        
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
