import os
import cv2
import numpy as np
import joblib
import time
from datetime import datetime
from sklearn.naive_bayes import GaussianNB
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

def plot_confusion_matrix(conf_matrix, save_path='output/train_model/confusion_matrix_enhanced.png'):
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

def save_test_results_table(y_true, y_pred, y_proba, save_path='output/test_validation_model/test_results.csv'):
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

def save_test_results_image(y_true, y_pred, y_proba, save_path='output/test_validation_model/test_results.png', max_rows=20):
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
    train_output_dir = 'output/train_model'  # Step 1 outputs
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

def main():
    # Step 1: Training and tuning
    best_model, X_train, X_test, y_train, y_test, best_results = train_model_step()

    # Step 2: Validation and test result saving
    results = validate_and_save_results(
        model=best_model,
        X_test=X_test,
        y_test=y_test,
        output_dir='output/test_validation_model'
    )

    # Step 3: Save summary PDF in new output/summary/ directory
    summary_dir = 'output/summary'
    extra_info = [
        f"Model: GaussianNB (var_smoothing={best_model.var_smoothing})",
        f"Test samples: {len(X_test)}",
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    ]
    save_classification_summary_pdf(results, summary_dir, extra_info=extra_info)

    # Step 4: Save detailed summary PDF in new output/classification_summary/ directory
    detailed_summary_dir = 'output/classification_summary'
    save_detailed_classification_summary(results, detailed_summary_dir, extra_info=extra_info)

    # Reload model and test again (optional)
    reloaded_model = load_model()
    if reloaded_model:
        print("Testing reloaded enhanced model:")
        test_model(reloaded_model, X_test, y_test)
    # Next steps can be added here

if __name__ == "__main__":
    main() 