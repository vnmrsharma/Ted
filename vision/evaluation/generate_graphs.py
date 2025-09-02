#!/usr/bin/env python3
"""
Generate comparison graphs from evaluation results
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

def load_evaluation_results():
    """Load evaluation results from JSON files"""
    evaluation_dir = Path(__file__).parent
    
    # Load system comparison
    comparison_path = evaluation_dir / "system_comparison.json"
    if comparison_path.exists():
        with open(comparison_path, 'r') as f:
            comparison_data = json.load(f)
        return comparison_data
    else:
        print("❌ System comparison file not found. Run evaluate_separate.py first.")
        return None

def create_accuracy_comparison_chart(data):
    """Create accuracy comparison bar chart"""
    plt.figure(figsize=(10, 6))
    
    systems = [item['System'] for item in data]
    accuracies = [float(item['Overall Accuracy'].rstrip('%')) for item in data]
    
    bars = plt.bar(systems, accuracies, color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{acc:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.title('Emotion Detection System Accuracy Comparison', fontsize=16, fontweight='bold')
    plt.xlabel('System', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.ylim(0, max(accuracies) * 1.2)
    plt.grid(axis='y', alpha=0.3)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    return plt.gcf()

def create_processing_time_chart(data):
    """Create processing time comparison chart"""
    plt.figure(figsize=(10, 6))
    
    systems = [item['System'] for item in data]
    # Convert time strings to float values
    times = []
    for item in data:
        time_str = item['Avg Processing Time']
        if time_str.endswith('s'):
            time_val = float(time_str.rstrip('s'))
        else:
            time_val = 0.0
        times.append(time_val)
    
    bars = plt.bar(systems, times, color=['#FFE66D', '#95E1D3', '#F38181'])
    
    # Add value labels on bars
    for bar, time_val in zip(bars, times):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0001, 
                f'{time_val:.3f}s', ha='center', va='bottom', fontweight='bold')
    
    plt.title('Average Processing Time Comparison', fontsize=16, fontweight='bold')
    plt.xlabel('System', fontsize=12)
    plt.ylabel('Processing Time (seconds)', fontsize=12)
    plt.ylim(0, max(times) * 1.2 if max(times) > 0 else 0.002)
    plt.grid(axis='y', alpha=0.3)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    return plt.gcf()

def create_performance_radar_chart(data):
    """Create radar chart comparing multiple performance metrics"""
    # Prepare data for radar chart
    systems = [item['System'] for item in data]
    
    # Normalize metrics for radar chart (0-1 scale)
    accuracies = [float(item['Overall Accuracy'].rstrip('%')) / 100 for item in data]
    
    # Convert time to inverse scale (faster = better)
    times = []
    for item in data:
        time_str = item['Avg Processing Time']
        if time_str.endswith('s'):
            time_val = float(time_str.rstrip('s'))
        else:
            time_val = 0.0
        times.append(time_val)
    
    # Normalize and invert time (faster = higher score)
    max_time = max(times) if max(times) > 0 else 0.001
    time_scores = [1 - (t / max_time) for t in times]
    
    # Number of variables
    categories = ['Accuracy', 'Speed']
    N = len(categories)
    
    # Create angles for each category
    angles = [n / float(N) * 2 * 3.14159 for n in range(N)]
    angles += angles[:1]  # Complete the circle
    
    # Create subplot
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    
    # Plot each system
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    for i, system in enumerate(systems):
        values = [accuracies[i], time_scores[i]]
        values += values[:1]  # Complete the circle
        
        ax.plot(angles, values, 'o-', linewidth=2, label=system, color=colors[i])
        ax.fill(angles, values, alpha=0.25, color=colors[i])
    
    # Set labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1)
    ax.set_title('Performance Comparison Radar Chart', size=16, y=1.08)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    return plt.gcf()

def create_summary_table(data):
    """Create a summary table visualization"""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare table data
    table_data = []
    for item in data:
        table_data.append([
            item['System'],
            item['Overall Accuracy'],
            item['Total Predictions'],
            item['Correct Predictions'],
            item['Avg Processing Time']
        ])
    
    # Create table
    table = ax.table(cellText=table_data,
                    colLabels=['System', 'Accuracy', 'Total', 'Correct', 'Avg Time'],
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.25, 0.15, 0.15, 0.15, 0.15])
    
    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Color header row
    for i in range(len(table_data[0])):
        table[(0, i)].set_facecolor('#4ECDC4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Color alternating rows
    for i in range(1, len(table_data) + 1):
        for j in range(len(table_data[0])):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#F7F7F7')
    
    plt.title('Emotion Detection Systems Performance Summary', fontsize=16, fontweight='bold', pad=20)
    plt.tight_layout()
    return plt.gcf()

def main():
    """Main function to generate all graphs"""
    print("📊 Generating comparison graphs...")
    
    # Load data
    data = load_evaluation_results()
    if not data:
        return
    
    # Set style
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # Create charts
    charts = []
    
    # 1. Accuracy comparison
    print("  📈 Creating accuracy comparison chart...")
    accuracy_chart = create_accuracy_comparison_chart(data)
    charts.append(('accuracy_comparison.png', accuracy_chart))
    
    # 2. Processing time comparison
    print("  ⏱️  Creating processing time chart...")
    time_chart = create_processing_time_chart(data)
    charts.append(('processing_time_comparison.png', time_chart))
    
    # 3. Performance radar chart
    print("  🎯 Creating performance radar chart...")
    radar_chart = create_performance_radar_chart(data)
    charts.append(('performance_radar.png', radar_chart))
    
    # 4. Summary table
    print("  📋 Creating summary table...")
    table_chart = create_summary_table(data)
    charts.append(('performance_summary_table.png', table_chart))
    
    # Save all charts
    output_dir = Path(__file__).parent
    for filename, chart in charts:
        output_path = output_dir / filename
        chart.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  💾 Saved: {filename}")
        plt.close(chart)
    
    print(f"\n✅ All graphs generated successfully!")
    print(f"📁 Check the evaluation folder for: {', '.join([f[0] for f in charts])}")

if __name__ == "__main__":
    main()
