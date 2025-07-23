#!/usr/bin/env python3
"""
Folder Structure Guide for Image Classification Project
This script explains the numbered output folder structure for better organization.
"""

import os

def print_folder_structure():
    """Print the organized folder structure with explanations"""
    
    print("=" * 80)
    print("ORGANIZED OUTPUT FOLDER STRUCTURE")
    print("=" * 80)
    
    folder_structure = {
        "00_feature_extraction": {
            "description": "Feature extraction analysis and visualizations",
            "contents": [
                "feature_analysis_plots.png",
                "feature_viz_true_*.png",
                "feature_viz_false_*.png",
                "feature_correlation_heatmap.png",
                "feature_importance_plot.png"
            ]
        },
        "01_train_model": {
            "description": "Step 1: Model training and parameter tuning results",
            "contents": [
                "training_accuracy_vs_var_smoothing.png",
                "tuning_results.csv",
                "confusion_matrix_enhanced.png"
            ]
        },
        "02_test_validation": {
            "description": "Step 2: Test validation and detailed results",
            "contents": [
                "test_results.csv",
                "test_results.png",
                "confusion_matrix_enhanced.png"
            ]
        },
        "03_summary": {
            "description": "Step 3: Basic classification summary",
            "contents": [
                "summary.pdf"
            ]
        },
        "04_classification_summary": {
            "description": "Step 4: Detailed classification summary",
            "contents": [
                "summary_detailed.pdf"
            ]
        },
        "05_model_comparison": {
            "description": "Step 5: Comprehensive model comparison and evaluation",
            "contents": [
                "model_comparison_performance.png",
                "top_models_radar_chart.png",
                "confusion_matrix_comparison.png",
                "model_comparison_results.csv",
                "comprehensive_evaluation.pdf",
                "detailed_comparison_table.png",
                "summary_statistics_table.png"
            ]
        }
    }
    
    print("\n📁 OUTPUT FOLDER ORGANIZATION:")
    print("-" * 50)
    
    for folder, info in folder_structure.items():
        print(f"\n🔹 {folder}/")
        print(f"   Purpose: {info['description']}")
        print("   Contents:")
        for content in info['contents']:
            print(f"     • {content}")
    
    print("\n" + "=" * 80)
    print("FOLDER NAMING CONVENTION:")
    print("=" * 80)
    print("• 00_* = Pre-processing/Feature extraction")
    print("• 01_* = Step 1: Training and tuning")
    print("• 02_* = Step 2: Validation and testing")
    print("• 03_* = Step 3: Basic summaries")
    print("• 04_* = Step 4: Detailed summaries")
    print("• 05_* = Step 5: Model comparison")
    print("\nThis numbering system makes it easy to:")
    print("• Identify which step each folder belongs to")
    print("• Understand the processing order")
    print("• Find specific outputs quickly")
    print("• Maintain organized research documentation")

def create_folder_structure():
    """Create the numbered folder structure"""
    
    folders = [
        "output/00_feature_extraction",
        "output/01_train_model", 
        "output/02_test_validation",
        "output/03_summary",
        "output/04_classification_summary",
        "output/05_model_comparison"
    ]
    
    print("\n📂 Creating folder structure...")
    for folder in folders:
        os.makedirs(folder, exist_ok=True)
        print(f"   ✅ Created: {folder}")
    
    print("\n🎉 All folders created successfully!")
    print("   You can now run your main.py script and outputs will be organized by step.")

if __name__ == "__main__":
    print_folder_structure()
    create_folder_structure() 