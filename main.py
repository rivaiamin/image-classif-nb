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
from feature_extraction import FeatureExtractor, extract_features_from_dataset, log_feature_info

# Constants
IMAGE_SIZE = (64, 64)
TRUE_DIR = 'data/true'
FALSE_DIR = 'data/false'
MODEL_PATH = 'models/model_enhanced.pkl'
LOG_FILE = 'logs/training_enhanced_log.txt'

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

def train_model(X_train, y_train):
    log_info("Starting enhanced model training...")
    start_time = time.time()
    
    model = GaussianNB()
    model.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    log_info(f"Enhanced model training completed in {training_time:.2f} seconds")
    
    # Save the trained model
    joblib.dump(model, MODEL_PATH)
    log_info(f"Enhanced model saved to {MODEL_PATH}")
    
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

def plot_confusion_matrix(conf_matrix, save_path='confusion_matrix_enhanced.png'):
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

def create_feature_analysis_plots(X, feature_names, save_path='feature_analysis.png'):
    """Create feature analysis plots"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Feature Analysis - Color, Texture, and Shape Features', fontsize=16)
    
    # Feature type breakdown
    color_features = [n for n in feature_names if 'color' in n]
    texture_features = [n for n in feature_names if 'glcm' in n or 'lbp' in n]
    shape_features = [n for n in feature_names if 'shape' in n or 'edge' in n]
    
    feature_counts = [len(color_features), len(texture_features), len(shape_features)]
    feature_labels = ['Color', 'Texture', 'Shape']
    
    axes[0, 0].pie(feature_counts, labels=feature_labels, autopct='%1.1f%%', startangle=90)
    axes[0, 0].set_title('Feature Type Distribution')
    
    # Feature importance (variance)
    feature_variance = np.var(X, axis=0)
    top_features_idx = np.argsort(feature_variance)[-10:]  # Top 10 features
    top_feature_names = [feature_names[i] for i in top_features_idx]
    
    axes[0, 1].barh(range(len(top_feature_names)), feature_variance[top_features_idx])
    axes[0, 1].set_yticks(range(len(top_feature_names)))
    axes[0, 1].set_yticklabels(top_feature_names)
    axes[0, 1].set_title('Top 10 Features by Variance')
    axes[0, 1].set_xlabel('Variance')
    
    # Feature correlation heatmap (sample)
    sample_features = X[:, :20]  # First 20 features for visualization
    correlation_matrix = np.corrcoef(sample_features.T)
    
    im = axes[1, 0].imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
    axes[1, 0].set_title('Feature Correlation Matrix (First 20 Features)')
    plt.colorbar(im, ax=axes[1, 0])
    
    # Feature distribution
    axes[1, 1].hist(X.flatten(), bins=50, alpha=0.7, edgecolor='black')
    axes[1, 1].set_title('Overall Feature Distribution')
    axes[1, 1].set_xlabel('Feature Values')
    axes[1, 1].set_ylabel('Frequency')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    log_info(f"Feature analysis plot saved to {save_path}")

def main():
    # Initialize log file
    log_info("=" * 60)
    log_info("Starting Enhanced Image Classification with Feature Extraction")
    log_info("=" * 60)
    
    # Load dataset with feature extraction
    X_train, X_test, y_train, y_test, feature_names = load_dataset_with_features()

    # Create feature analysis plots
    create_feature_analysis_plots(X_train, feature_names)

    # Train and save model
    model = train_model(X_train, y_train)

    # Evaluate model
    results = test_model(model, X_test, y_test)
    
    # Plot confusion matrix
    plot_confusion_matrix(results['confusion_matrix'])

    # Example of re-loading the model and testing again
    log_info("Testing reloaded enhanced model...")
    reloaded_model = load_model()
    if reloaded_model:
        print("Testing reloaded enhanced model:")
        test_model(reloaded_model, X_test, y_test)
    
    log_info("=" * 60)
    log_info("Enhanced training and evaluation completed")
    log_info("=" * 60)

if __name__ == "__main__":
    main() 