import os
import cv2
import numpy as np
import joblib
import time
from datetime import datetime
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from feature_extraction import FeatureExtractor, extract_features_from_dataset, create_feature_analysis_plots
import csv
import pandas as pd
import matplotlib.backends.backend_pdf
from fpdf import FPDF

# Constants
IMAGE_SIZE = (64, 64)
TRUE_DIR = 'data/true'
FALSE_DIR = 'data/false'
MODEL_PATH = 'models/model_enhanced.pkl'
LOG_FILE = 'logs/training_enhanced_log.txt'

# --- Parameter Tuning ---
VAR_SMOOTHING_VALUES = [1e-9, 1e-8, 1e-7, 1e-6]  # Nilai var_smoothing untuk tuning GaussianNB

# Output directory constants for better organization
OUTPUT_DIRS = {
    'feature_extraction': 'output/00_feature_extraction',
    'train_model': 'output/01_train_model',
    'test_validation': 'output/02_test_validation',
    'summary': 'output/03_summary',
    'classification_summary': 'output/04_classification_summary',
    'model_comparison': 'output/05_model_comparison'
}

def log_info(message):
    """Log information with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] {message}"
    print(log_message)
    with open(LOG_FILE, 'a') as f:
        f.write(log_message + '\n')

def load_dataset_with_features():
    """Load dataset using advanced feature extraction"""
    log_info("Starting enhanced dataset loading with feature extraction...")
    
    # Extract features from dataset
    X, y, feature_names = extract_features_from_dataset(TRUE_DIR, FALSE_DIR, IMAGE_SIZE)
    
    # Dataset statistics
    total_samples = len(X)
    true_samples = np.sum(y == 1)
    false_samples = np.sum(y == 0)
    
    log_info(f"Enhanced Dataset Statistics:")
    log_info(f"  Total samples: {total_samples}")
    log_info(f"  True samples (class 1): {true_samples}")
    log_info(f"  False samples (class 0): {false_samples}")
    log_info(f"  Class balance: {true_samples/total_samples*100:.1f}% true, {false_samples/total_samples*100:.1f}% false")
    log_info(f"  Feature dimensions: {X.shape[1]} (extracted features)")
    log_info(f"  Feature breakdown:")
    
    # Count feature types
    color_features = len([n for n in feature_names if 'color' in n])
    texture_features = len([n for n in feature_names if 'glcm' in n or 'lbp' in n])
    shape_features = len([n for n in feature_names if 'shape' in n or 'edge' in n])
    
    log_info(f"    Color features: {color_features}")
    log_info(f"    Texture features: {texture_features}")
    log_info(f"    Shape features: {shape_features}")
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    log_info(f"Train/Test Split:")
    log_info(f"  Training samples: {len(X_train)} ({len(X_train)/total_samples*100:.1f}%)")
    log_info(f"  Test samples: {len(X_test)} ({len(X_test)/total_samples*100:.1f}%)")
    log_info(f"  Training class distribution: {np.sum(y_train == 1)} true, {np.sum(y_train == 0)} false")
    log_info(f"  Test class distribution: {np.sum(y_test == 1)} true, {np.sum(y_test == 0)} false")
    
    return X_train, X_test, y_train, y_test, feature_names

def train_model(X_train, y_train, var_smoothing=1e-9):
    log_info(f"Starting enhanced model training (var_smoothing={var_smoothing})...")
    start_time = time.time()
    
    model = GaussianNB(var_smoothing=var_smoothing)
    model.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    log_info(f"Enhanced model training completed in {training_time:.2f} seconds")
    
    # Model information
    log_info(f"Enhanced Model Information:")
    log_info(f"  Model type: Gaussian Naive Bayes")
    log_info(f"  Number of classes: {len(model.classes_)}")
    log_info(f"  Class labels: {model.classes_}")
    log_info(f"  Prior probabilities: {model.class_prior_}")
    log_info(f"  Feature count: {X_train.shape[1]}")
    
    return model

def load_model():
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        log_info(f"Loaded enhanced model from {MODEL_PATH}")
        return model
    else:
        log_info("Enhanced model file not found. Please train the model first.")
        return None

def test_model(model, X_test, y_test):
    log_info("Starting enhanced model evaluation...")
    start_time = time.time()
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    evaluation_time = time.time() - start_time
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    log_info(f"Enhanced Model Evaluation Results:")
    log_info(f"  Evaluation time: {evaluation_time:.2f} seconds")
    log_info(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    log_info(f"  Confusion Matrix:")
    log_info(f"    True Negatives: {conf_matrix[0,0]}")
    log_info(f"    False Positives: {conf_matrix[0,1]}")
    log_info(f"    False Negatives: {conf_matrix[1,0]}")
    log_info(f"    True Positives: {conf_matrix[1,1]}")
    
    # Calculate additional metrics
    precision = conf_matrix[1,1] / (conf_matrix[1,1] + conf_matrix[0,1]) if (conf_matrix[1,1] + conf_matrix[0,1]) > 0 else 0
    recall = conf_matrix[1,1] / (conf_matrix[1,1] + conf_matrix[1,0]) if (conf_matrix[1,1] + conf_matrix[1,0]) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    log_info(f"  Precision: {precision:.4f}")
    log_info(f"  Recall: {recall:.4f}")
    log_info(f"  F1-Score: {f1_score:.4f}")
    
    print("Enhanced Accuracy:", accuracy)
    print("Enhanced Classification Report:\n", classification_report(y_test, y_pred))
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'confusion_matrix': conf_matrix,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }

def plot_confusion_matrix(conf_matrix, save_path='output/01_train_model/confusion_matrix_enhanced.png'):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['False', 'True'], 
                yticklabels=['False', 'True'])
    plt.title('Enhanced Model Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    log_info(f"Enhanced confusion matrix plot saved to {save_path}")

def save_test_results_table(y_true, y_pred, y_proba, save_path='output/02_test_validation/test_results.csv'):
    """
    Save test results (ground truth, prediction, probabilities) as a CSV file.
    """
    df = pd.DataFrame({
        'Ground Truth': y_true,
        'Prediction': y_pred,
        'Probability False': y_proba[:, 0],
        'Probability True': y_proba[:, 1]
    })
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    log_info(f"Test results table saved to {save_path}")

def save_test_results_image(y_true, y_pred, y_proba, save_path='output/02_test_validation/test_results.png', max_rows=20):
    """
    Save test results (ground truth, prediction, probabilities) as an image (PNG).
    """
    df = pd.DataFrame({
        'Ground Truth': y_true,
        'Prediction': y_pred,
        'Probability False': y_proba[:, 0],
        'Probability True': y_proba[:, 1]
    })
    df = df.head(max_rows)
    fig, ax = plt.subplots(figsize=(10, 0.5 * max_rows + 1))
    ax.axis('off')
    table = ax.table(cellText=df.values, colLabels=df.columns, loc='center')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    log_info(f"Test results table image saved to {save_path}")

def save_classification_summary_pdf(results, output_dir, extra_info=None):
    """
    Save a PDF summary of classification results to output_dir/summary.pdf
    """
    os.makedirs(output_dir, exist_ok=True)
    pdf_path = os.path.join(output_dir, 'summary.pdf')
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Classification Results Summary", ln=True, align='C')
    pdf.ln(10)
    if extra_info:
        for line in extra_info:
            pdf.cell(200, 8, txt=line, ln=True)
        pdf.ln(5)
    pdf.cell(200, 8, txt=f"Accuracy: {results['accuracy']:.4f}", ln=True)
    pdf.cell(200, 8, txt=f"Precision: {results['precision']:.4f}", ln=True)
    pdf.cell(200, 8, txt=f"Recall: {results['recall']:.4f}", ln=True)
    pdf.cell(200, 8, txt=f"F1-Score: {results['f1_score']:.4f}", ln=True)
    pdf.ln(5)
    pdf.cell(200, 8, txt="Confusion Matrix:", ln=True)
    cm = results['confusion_matrix']
    pdf.cell(200, 8, txt=f"  TN: {cm[0,0]}  FP: {cm[0,1]}", ln=True)
    pdf.cell(200, 8, txt=f"  FN: {cm[1,0]}  TP: {cm[1,1]}", ln=True)
    pdf.ln(5)
    pdf.cell(200, 8, txt="See CSV/PNG outputs for detailed results.", ln=True)
    pdf.output(pdf_path)
    log_info(f"Classification summary PDF saved to {pdf_path}")
    return pdf_path

def save_detailed_classification_summary(results, output_dir, extra_info=None):
    """
    Save a detailed summary of classification results as a PDF in output_dir/summary_detailed.pdf
    """
    import os
    from fpdf import FPDF
    os.makedirs(output_dir, exist_ok=True)
    pdf_path = os.path.join(output_dir, 'summary_detailed.pdf')
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Detailed Classification Results Summary", ln=True, align='C')
    pdf.ln(10)
    if extra_info:
        for line in extra_info:
            pdf.cell(200, 8, txt=line, ln=True)
        pdf.ln(5)
    pdf.cell(200, 8, txt=f"Accuracy: {results['accuracy']:.4f}", ln=True)
    pdf.cell(200, 8, txt=f"Precision: {results['precision']:.4f}", ln=True)
    pdf.cell(200, 8, txt=f"Recall: {results['recall']:.4f}", ln=True)
    pdf.cell(200, 8, txt=f"F1-Score: {results['f1_score']:.4f}", ln=True)
    pdf.ln(5)
    pdf.cell(200, 8, txt="Confusion Matrix:", ln=True)
    cm = results['confusion_matrix']
    pdf.cell(200, 8, txt=f"  TN: {cm[0,0]}  FP: {cm[0,1]}", ln=True)
    pdf.cell(200, 8, txt=f"  FN: {cm[1,0]}  TP: {cm[1,1]}", ln=True)
    pdf.ln(5)
    pdf.cell(200, 8, txt="Sample Predictions:", ln=True)
    y_pred = results['predictions']
    y_proba = results['probabilities']
    max_rows = min(10, len(y_pred))
    for i in range(max_rows):
        pdf.cell(200, 8, txt=f"Sample {i+1}: Pred={y_pred[i]}, Prob_False={y_proba[i,0]:.2f}, Prob_True={y_proba[i,1]:.2f}", ln=True)
    pdf.ln(5)
    pdf.cell(200, 8, txt="See CSV/PNG outputs for full details.", ln=True)
    pdf.output(pdf_path)
    log_info(f"Detailed classification summary PDF saved to {pdf_path}")
    return pdf_path

"""Step 1: Training Model Awal & Tuning Parameter Sederhana"""
def train_model_step():
    log_info("=" * 60)
    log_info("Starting Enhanced Image Classification with Feature Extraction")
    log_info("=" * 60)
    
    # Load dataset with feature extraction
    X_train, X_test, y_train, y_test, feature_names = load_dataset_with_features()

    # Create feature analysis plots
    create_feature_analysis_plots(X_train, feature_names)

    # --- Tuning var_smoothing ---
    best_accuracy = 0
    best_model = None
    best_vs = None
    best_results = None
    train_output_dir = OUTPUT_DIRS['train_model']  # Step 1 outputs
    tuning_results = []  # Untuk menyimpan hasil tuning
    for vs in VAR_SMOOTHING_VALUES:
        log_info(f"\n--- Training with var_smoothing={vs} ---")
        model = train_model(X_train, y_train, var_smoothing=vs)
        results = test_model(model, X_test, y_test)
        acc = results['accuracy']
        log_info(f"Accuracy for var_smoothing={vs}: {acc:.4f}")
        tuning_results.append((vs, acc))
        if acc > best_accuracy:
            best_accuracy = acc
            best_model = model
            best_vs = vs
            best_results = results
    log_info(f"\nBest var_smoothing: {best_vs} with accuracy: {best_accuracy:.4f}")
    
    # Save the best model
    joblib.dump(best_model, MODEL_PATH)
    log_info(f"Best model saved to {MODEL_PATH}")

    # Only save training/tuning outputs in train_output_dir
    vs_values = [x[0] for x in tuning_results]
    acc_values = [x[1] for x in tuning_results]
    plt.figure(figsize=(8,5))
    plt.plot(vs_values, acc_values, marker='o')
    plt.xscale('log')
    plt.xlabel('var_smoothing (log scale)')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs var_smoothing (GaussianNB)')
    plt.grid(True)
    plt.tight_layout()
    os.makedirs(train_output_dir, exist_ok=True)
    plt.savefig(os.path.join(train_output_dir, 'training_accuracy_vs_var_smoothing.png'), dpi=300)
    plt.close()
    log_info(f'Accuracy vs var_smoothing (GaussianNB) graph saved in {train_output_dir}/training_accuracy_vs_var_smoothing.png')

    # --- Simpan hasil tuning ke CSV ---
    csv_path = os.path.join(train_output_dir, 'tuning_results.csv')
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['var_smoothing', 'accuracy'])
        for vs, acc in tuning_results:
            writer.writerow([vs, acc])
    log_info(f'Tuning results saved to {csv_path}')

    log_info("=" * 60)
    log_info("Enhanced training and evaluation completed")
    log_info("=" * 60)
    return best_model, X_train, X_test, y_train, y_test, best_results

def validate_and_save_results(model, X_test, y_test, output_dir):
    """
    Step 2: Validate model on test set and save all outputs to output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    log_info("=" * 60)
    log_info("Starting Model Validation and Performance Logging (Step 2)")
    log_info("=" * 60)
    results = test_model(model, X_test, y_test)
    plot_confusion_matrix(results['confusion_matrix'], save_path=os.path.join(output_dir, 'confusion_matrix_enhanced.png'))
    save_test_results_table(
        y_true=y_test,
        y_pred=results['predictions'],
        y_proba=results['probabilities'],
        save_path=os.path.join(output_dir, 'test_results.csv')
    )
    save_test_results_image(
        y_true=y_test,
        y_pred=results['predictions'],
        y_proba=results['probabilities'],
        save_path=os.path.join(output_dir, 'test_results.png'),
        max_rows=20
    )
    log_info("Validation and result saving completed.")
    log_info("=" * 60)
    return results

def step5_model_comparison_evaluation(X_train, X_test, y_train, y_test):
    """
    Step 5: Comprehensive Model Comparison and Performance Evaluation
    Compares different Naive Bayes variants and other classifiers
    """
    log_info("=" * 60)
    log_info("Starting Step 5: Model Comparison and Performance Evaluation")
    log_info("=" * 60)
    
    # Create output directory for Step 5
    step5_output_dir = OUTPUT_DIRS['model_comparison']
    os.makedirs(step5_output_dir, exist_ok=True)
    
    # Define models to compare
    models = {
        'GaussianNB (var_smoothing=1e-9)': GaussianNB(var_smoothing=1e-9),
        'GaussianNB (var_smoothing=1e-8)': GaussianNB(var_smoothing=1e-8),
        'GaussianNB (var_smoothing=1e-7)': GaussianNB(var_smoothing=1e-7),
        'GaussianNB (var_smoothing=1e-6)': GaussianNB(var_smoothing=1e-6),
        'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
        'RandomForest': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    # Store results for comparison
    comparison_results = []
    
    log_info("Training and evaluating multiple models...")
    
    for model_name, model in models.items():
        log_info(f"\n--- Evaluating {model_name} ---")
        
        # Train model
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        # Evaluate model
        eval_start_time = time.time()
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        eval_time = time.time() - eval_start_time
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        # Calculate additional metrics
        tn, fp, fn, tp = conf_matrix.ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Store results
        result = {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'training_time': training_time,
            'evaluation_time': eval_time,
            'confusion_matrix': conf_matrix,
            'predictions': y_pred,
            'probabilities': y_pred_proba
        }
        comparison_results.append(result)
        
        log_info(f"  Accuracy: {accuracy:.4f}")
        log_info(f"  Precision: {precision:.4f}")
        log_info(f"  Recall: {recall:.4f}")
        log_info(f"  F1-Score: {f1_score:.4f}")
        log_info(f"  Training time: {training_time:.2f}s")
        log_info(f"  Evaluation time: {eval_time:.2f}s")
    
    # Create comprehensive comparison plots
    create_model_comparison_plots(comparison_results, step5_output_dir)
    
    # Save comparison results to CSV
    save_comparison_results_csv(comparison_results, step5_output_dir)
    
    # Create comprehensive evaluation PDF
    create_comprehensive_evaluation_pdf(comparison_results, step5_output_dir)
    
    # Create detailed comparison table
    create_detailed_comparison_table(comparison_results, step5_output_dir)
    
    # Find best model
    best_model_result = max(comparison_results, key=lambda x: x['accuracy'])
    log_info(f"\nBest performing model: {best_model_result['model_name']}")
    log_info(f"Best accuracy: {best_model_result['accuracy']:.4f}")
    
    log_info("=" * 60)
    log_info("Step 5: Model comparison and evaluation completed")
    log_info("=" * 60)
    
    return comparison_results, best_model_result

def create_model_comparison_plots(comparison_results, output_dir):
    """Create comprehensive comparison plots for Step 5"""
    
    # Extract data for plotting
    model_names = [result['model_name'] for result in comparison_results]
    accuracies = [result['accuracy'] for result in comparison_results]
    precisions = [result['precision'] for result in comparison_results]
    recalls = [result['recall'] for result in comparison_results]
    f1_scores = [result['f1_score'] for result in comparison_results]
    training_times = [result['training_time'] for result in comparison_results]
    eval_times = [result['evaluation_time'] for result in comparison_results]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Step 5: Comprehensive Model Performance Comparison', fontsize=16, fontweight='bold')
    
    # 1. Accuracy comparison
    axes[0, 0].bar(model_names, accuracies, color='skyblue', alpha=0.7)
    axes[0, 0].set_title('Accuracy Comparison')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Precision comparison
    axes[0, 1].bar(model_names, precisions, color='lightgreen', alpha=0.7)
    axes[0, 1].set_title('Precision Comparison')
    axes[0, 1].set_ylabel('Precision')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Recall comparison
    axes[0, 2].bar(model_names, recalls, color='lightcoral', alpha=0.7)
    axes[0, 2].set_title('Recall Comparison')
    axes[0, 2].set_ylabel('Recall')
    axes[0, 2].tick_params(axis='x', rotation=45)
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. F1-Score comparison
    axes[1, 0].bar(model_names, f1_scores, color='gold', alpha=0.7)
    axes[1, 0].set_title('F1-Score Comparison')
    axes[1, 0].set_ylabel('F1-Score')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Training time comparison
    axes[1, 1].bar(model_names, training_times, color='plum', alpha=0.7)
    axes[1, 1].set_title('Training Time Comparison')
    axes[1, 1].set_ylabel('Training Time (seconds)')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Evaluation time comparison
    axes[1, 2].bar(model_names, eval_times, color='lightblue', alpha=0.7)
    axes[1, 2].set_title('Evaluation Time Comparison')
    axes[1, 2].set_ylabel('Evaluation Time (seconds)')
    axes[1, 2].tick_params(axis='x', rotation=45)
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison_performance.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create radar chart for top metrics
    create_radar_chart(comparison_results, output_dir)
    
    # Create heatmap for confusion matrices
    create_confusion_matrix_heatmap(comparison_results, output_dir)
    
    log_info(f"Model comparison plots saved to {output_dir}")

def create_radar_chart(comparison_results, output_dir):
    """Create radar chart for top performing models"""
    
    # Select top 4 models by accuracy
    top_models = sorted(comparison_results, key=lambda x: x['accuracy'], reverse=True)[:4]
    
    # Metrics to plot
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    # Number of variables
    N = len(metrics)
    
    # Compute angle for each axis
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Complete the circle
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Plot for each model
    colors = ['red', 'blue', 'green', 'orange']
    for i, (result, color) in enumerate(zip(top_models, colors)):
        values = [result[metric] for metric in metrics]
        values += values[:1]  # Complete the circle
        
        ax.plot(angles, values, 'o-', linewidth=2, label=result['model_name'], color=color)
        ax.fill(angles, values, alpha=0.25, color=color)
    
    # Set the labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 1)
    ax.set_title('Top 4 Models Performance Radar Chart', size=16, y=1.1)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'top_models_radar_chart.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_confusion_matrix_heatmap(comparison_results, output_dir):
    """Create heatmap comparing confusion matrices"""
    
    # Select top 4 models
    top_models = sorted(comparison_results, key=lambda x: x['accuracy'], reverse=True)[:4]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Confusion Matrix Comparison - Top 4 Models', fontsize=16, fontweight='bold')
    
    for i, (result, ax) in enumerate(zip(top_models, axes.flat)):
        conf_matrix = result['confusion_matrix']
        
        # Create heatmap
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['False', 'True'], 
                   yticklabels=['False', 'True'], ax=ax)
        ax.set_title(f"{result['model_name']}\nAccuracy: {result['accuracy']:.4f}")
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()

def save_comparison_results_csv(comparison_results, output_dir):
    """Save comparison results to CSV file"""
    
    csv_path = os.path.join(output_dir, 'model_comparison_results.csv')
    
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['model_name', 'accuracy', 'precision', 'recall', 'f1_score', 
                     'training_time', 'evaluation_time']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in comparison_results:
            writer.writerow({
                'model_name': result['model_name'],
                'accuracy': result['accuracy'],
                'precision': result['precision'],
                'recall': result['recall'],
                'f1_score': result['f1_score'],
                'training_time': result['training_time'],
                'evaluation_time': result['evaluation_time']
            })
    
    log_info(f"Model comparison results saved to {csv_path}")

def create_comprehensive_evaluation_pdf(comparison_results, output_dir):
    """Create comprehensive evaluation PDF with detailed analysis"""
    
    pdf_path = os.path.join(output_dir, 'comprehensive_evaluation.pdf')
    
    # Create PDF
    pdf = FPDF()
    pdf.add_page()
    
    # Title
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(0, 10, 'Step 5: Comprehensive Model Evaluation Report', ln=True, align='C')
    pdf.ln(10)
    
    # Date and time
    pdf.set_font('Arial', '', 10)
    pdf.cell(0, 10, f'Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', ln=True)
    pdf.ln(5)
    
    # Executive Summary
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Executive Summary', ln=True)
    pdf.set_font('Arial', '', 10)
    
    # Find best model
    best_model = max(comparison_results, key=lambda x: x['accuracy'])
    worst_model = min(comparison_results, key=lambda x: x['accuracy'])
    
    pdf.cell(0, 10, f'Best performing model: {best_model["model_name"]}', ln=True)
    pdf.cell(0, 10, f'Best accuracy: {best_model["accuracy"]:.4f} ({best_model["accuracy"]*100:.2f}%)', ln=True)
    pdf.cell(0, 10, f'Worst performing model: {worst_model["model_name"]}', ln=True)
    pdf.cell(0, 10, f'Worst accuracy: {worst_model["accuracy"]:.4f} ({worst_model["accuracy"]*100:.2f}%)', ln=True)
    pdf.ln(5)
    
    # Performance Analysis
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Performance Analysis', ln=True)
    pdf.set_font('Arial', '', 10)
    
    # Calculate statistics
    accuracies = [r['accuracy'] for r in comparison_results]
    avg_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    
    pdf.cell(0, 10, f'Average accuracy across all models: {avg_accuracy:.4f} ({avg_accuracy*100:.2f}%)', ln=True)
    pdf.cell(0, 10, f'Standard deviation of accuracy: {std_accuracy:.4f}', ln=True)
    pdf.cell(0, 10, f'Accuracy range: {min(accuracies):.4f} - {max(accuracies):.4f}', ln=True)
    pdf.ln(5)
    
    # Model-specific Analysis
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Model-Specific Analysis', ln=True)
    pdf.set_font('Arial', '', 10)
    
    for result in comparison_results:
        pdf.cell(0, 10, f'{result["model_name"]}:', ln=True)
        pdf.cell(20, 10, '')
        pdf.cell(0, 10, f'  Accuracy: {result["accuracy"]:.4f} ({result["accuracy"]*100:.2f}%)', ln=True)
        pdf.cell(20, 10, '')
        pdf.cell(0, 10, f'  Precision: {result["precision"]:.4f} ({result["precision"]*100:.2f}%)', ln=True)
        pdf.cell(20, 10, '')
        pdf.cell(0, 10, f'  Recall: {result["recall"]:.4f} ({result["recall"]*100:.2f}%)', ln=True)
        pdf.cell(20, 10, '')
        pdf.cell(0, 10, f'  F1-Score: {result["f1_score"]:.4f} ({result["f1_score"]*100:.2f}%)', ln=True)
        pdf.cell(20, 10, '')
        pdf.cell(0, 10, f'  Training time: {result["training_time"]:.4f}s', ln=True)
        pdf.cell(20, 10, '')
        pdf.cell(0, 10, f'  Evaluation time: {result["evaluation_time"]:.4f}s', ln=True)
        pdf.ln(5)
    
    # Recommendations
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Recommendations', ln=True)
    pdf.set_font('Arial', '', 10)
    
    # Find fastest and most accurate models
    fastest_model = min(comparison_results, key=lambda x: x['training_time'])
    most_precise_model = max(comparison_results, key=lambda x: x['precision'])
    most_recall_model = max(comparison_results, key=lambda x: x['recall'])
    
    pdf.cell(0, 10, f'For speed: {fastest_model["model_name"]} (training time: {fastest_model["training_time"]:.4f}s)', ln=True)
    pdf.cell(0, 10, f'For precision: {most_precise_model["model_name"]} (precision: {most_precise_model["precision"]:.4f})', ln=True)
    pdf.cell(0, 10, f'For recall: {most_recall_model["model_name"]} (recall: {most_recall_model["recall"]:.4f})', ln=True)
    pdf.ln(5)
    
    # Technical Details
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(0, 10, 'Technical Details', ln=True)
    pdf.set_font('Arial', '', 10)
    
    pdf.cell(0, 10, f'Total models evaluated: {len(comparison_results)}', ln=True)
    pdf.cell(0, 10, f'Dataset size: {len(comparison_results[0]["predictions"])} test samples', ln=True)
    pdf.cell(0, 10, f'Feature dimensions: 142 (color, texture, shape features)', ln=True)
    pdf.cell(0, 10, f'Evaluation method: Train/Test split (80/20)', ln=True)
    pdf.ln(5)
    
    # Save PDF
    pdf.output(pdf_path)
    log_info(f"Comprehensive evaluation PDF saved to {pdf_path}")

def create_detailed_comparison_table(comparison_results, output_dir):
    """Create detailed comparison table with all metrics"""
    
    # Create detailed table image
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis('tight')
    ax.axis('off')
    
    # Prepare data for table
    table_data = []
    headers = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'Training Time (s)', 'Eval Time (s)']
    
    for result in comparison_results:
        row = [
            result['model_name'],
            f"{result['accuracy']:.4f}",
            f"{result['precision']:.4f}",
            f"{result['recall']:.4f}",
            f"{result['f1_score']:.4f}",
            f"{result['training_time']:.4f}",
            f"{result['evaluation_time']:.4f}"
        ]
        table_data.append(row)
    
    # Create table
    table = ax.table(cellText=table_data, colLabels=headers, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    # Style the table
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Highlight best values
    best_accuracy_idx = max(range(len(comparison_results)), key=lambda i: comparison_results[i]['accuracy'])
    best_precision_idx = max(range(len(comparison_results)), key=lambda i: comparison_results[i]['precision'])
    best_recall_idx = max(range(len(comparison_results)), key=lambda i: comparison_results[i]['recall'])
    best_f1_idx = max(range(len(comparison_results)), key=lambda i: comparison_results[i]['f1_score'])
    
    # Highlight best accuracy
    table[(best_accuracy_idx + 1, 1)].set_facecolor('#FFD700')
    table[(best_precision_idx + 1, 2)].set_facecolor('#FFD700')
    table[(best_recall_idx + 1, 3)].set_facecolor('#FFD700')
    table[(best_f1_idx + 1, 4)].set_facecolor('#FFD700')
    
    plt.title('Step 5: Detailed Model Comparison Table', fontsize=16, fontweight='bold', pad=20)
    
    # Save table
    table_path = os.path.join(output_dir, 'detailed_comparison_table.png')
    plt.savefig(table_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create summary statistics table
    create_summary_statistics_table(comparison_results, output_dir)
    
    log_info(f"Detailed comparison table saved to {table_path}")

def create_summary_statistics_table(comparison_results, output_dir):
    """Create summary statistics table"""
    
    # Calculate summary statistics
    accuracies = [r['accuracy'] for r in comparison_results]
    precisions = [r['precision'] for r in comparison_results]
    recalls = [r['recall'] for r in comparison_results]
    f1_scores = [r['f1_score'] for r in comparison_results]
    training_times = [r['training_time'] for r in comparison_results]
    eval_times = [r['evaluation_time'] for r in comparison_results]
    
    # Create summary table
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis('tight')
    ax.axis('off')
    
    summary_data = [
        ['Metric', 'Mean', 'Std', 'Min', 'Max', 'Best Model'],
        ['Accuracy', f"{np.mean(accuracies):.4f}", f"{np.std(accuracies):.4f}", 
         f"{min(accuracies):.4f}", f"{max(accuracies):.4f}", 
         comparison_results[np.argmax(accuracies)]['model_name']],
        ['Precision', f"{np.mean(precisions):.4f}", f"{np.std(precisions):.4f}", 
         f"{min(precisions):.4f}", f"{max(precisions):.4f}", 
         comparison_results[np.argmax(precisions)]['model_name']],
        ['Recall', f"{np.mean(recalls):.4f}", f"{np.std(recalls):.4f}", 
         f"{min(recalls):.4f}", f"{max(recalls):.4f}", 
         comparison_results[np.argmax(recalls)]['model_name']],
        ['F1-Score', f"{np.mean(f1_scores):.4f}", f"{np.std(f1_scores):.4f}", 
         f"{min(f1_scores):.4f}", f"{max(f1_scores):.4f}", 
         comparison_results[np.argmax(f1_scores)]['model_name']],
        ['Training Time (s)', f"{np.mean(training_times):.4f}", f"{np.std(training_times):.4f}", 
         f"{min(training_times):.4f}", f"{max(training_times):.4f}", 
         comparison_results[np.argmin(training_times)]['model_name']],
        ['Eval Time (s)', f"{np.mean(eval_times):.4f}", f"{np.std(eval_times):.4f}", 
         f"{min(eval_times):.4f}", f"{max(eval_times):.4f}", 
         comparison_results[np.argmin(eval_times)]['model_name']]
    ]
    
    # Create table
    table = ax.table(cellText=summary_data, cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    # Style the table
    for i in range(len(summary_data[0])):
        table[(0, i)].set_facecolor('#2196F3')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.title('Step 5: Summary Statistics Across All Models', fontsize=16, fontweight='bold', pad=20)
    
    # Save summary table
    summary_path = os.path.join(output_dir, 'summary_statistics_table.png')
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    log_info(f"Summary statistics table saved to {summary_path}")

def main():
    # Step 1: Training and tuning
    best_model, X_train, X_test, y_train, y_test, best_results = train_model_step()

    # Step 2: Validation and test result saving
    results = validate_and_save_results(
        model=best_model,
        X_test=X_test,
        y_test=y_test,
        output_dir=OUTPUT_DIRS['test_validation']
    )

    # Step 3: Save summary PDF in new output/summary/ directory
    summary_dir = OUTPUT_DIRS['summary']
    extra_info = [
        f"Model: GaussianNB (var_smoothing={best_model.var_smoothing})",
        f"Test samples: {len(X_test)}",
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    ]
    save_classification_summary_pdf(results, summary_dir, extra_info=extra_info)

    # Step 4: Save detailed summary PDF in new output/classification_summary/ directory
    detailed_summary_dir = OUTPUT_DIRS['classification_summary']
    save_detailed_classification_summary(results, detailed_summary_dir, extra_info=extra_info)

    # Reload model and test again (optional)
    reloaded_model = load_model()
    if reloaded_model:
        print("Testing reloaded enhanced model:")
        test_model(reloaded_model, X_test, y_test)
    
    # Step 5: Model comparison and performance evaluation
    comparison_results, best_model_result = step5_model_comparison_evaluation(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test
    )
    
    log_info("All steps completed successfully!")
    log_info("=" * 60)

if __name__ == "__main__":
    main() 