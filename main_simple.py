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

# Constants
IMAGE_SIZE = (64, 64)
TRUE_DIR = 'data/true'
FALSE_DIR = 'data/false'
MODEL_PATH = 'model.pkl'
LOG_FILE = 'training_log.txt'

def log_info(message):
    """Log information with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] {message}"
    print(log_message)
    with open(LOG_FILE, 'a') as f:
        f.write(log_message + '\n')

def load_images_from_folder(folder, label):
    data = []
    labels = []
    valid_extensions = ('.png', '.jpg', '.jpeg')
    total_files = 0
    loaded_files = 0
    
    if not os.path.exists(folder):
        log_info(f"Warning: Directory {folder} does not exist")
        return data, labels
    
    for filename in os.listdir(folder):
        total_files += 1
        if filename.lower().endswith(valid_extensions):
            path = os.path.join(folder, filename)
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, IMAGE_SIZE)
                features = img.flatten()
                data.append(features)
                labels.append(label)
                loaded_files += 1
            else:
                log_info(f"Warning: Could not load image {path}")
    
    log_info(f"Folder {folder}: {loaded_files}/{total_files} images loaded successfully")
    return data, labels

def load_dataset():
    log_info("Starting dataset loading process...")
    
    # Load true and false images
    true_data, true_labels = load_images_from_folder(TRUE_DIR, 1)
    false_data, false_labels = load_images_from_folder(FALSE_DIR, 0)
    
    # Combine datasets
    X = np.array(true_data + false_data)
    y = np.array(true_labels + false_labels)
    
    # Dataset statistics
    total_samples = len(X)
    true_samples = np.sum(y == 1)
    false_samples = np.sum(y == 0)
    
    log_info(f"Dataset Statistics:")
    log_info(f"  Total samples: {total_samples}")
    log_info(f"  True samples (class 1): {true_samples}")
    log_info(f"  False samples (class 0): {false_samples}")
    log_info(f"  Class balance: {true_samples/total_samples*100:.1f}% true, {false_samples/total_samples*100:.1f}% false")
    log_info(f"  Feature dimensions: {X.shape[1]} (flattened {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]} images)")
    
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    log_info(f"Train/Test Split:")
    log_info(f"  Training samples: {len(X_train)} ({len(X_train)/total_samples*100:.1f}%)")
    log_info(f"  Test samples: {len(X_test)} ({len(X_test)/total_samples*100:.1f}%)")
    log_info(f"  Training class distribution: {np.sum(y_train == 1)} true, {np.sum(y_train == 0)} false")
    log_info(f"  Test class distribution: {np.sum(y_test == 1)} true, {np.sum(y_test == 0)} false")
    
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    log_info("Starting model training...")
    start_time = time.time()
    
    model = GaussianNB()
    model.fit(X_train, y_train)
    
    training_time = time.time() - start_time
    log_info(f"Model training completed in {training_time:.2f} seconds")
    
    # Save the trained model
    joblib.dump(model, MODEL_PATH)
    log_info(f"Model saved to {MODEL_PATH}")
    
    # Model information
    log_info(f"Model Information:")
    log_info(f"  Model type: Gaussian Naive Bayes")
    log_info(f"  Number of classes: {len(model.classes_)}")
    log_info(f"  Class labels: {model.classes_}")
    log_info(f"  Prior probabilities: {model.class_prior_}")
    
    return model

def load_model():
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        log_info(f"Loaded model from {MODEL_PATH}")
        return model
    else:
        log_info("Model file not found. Please train the model first.")
        return None

def test_model(model, X_test, y_test):
    log_info("Starting model evaluation...")
    start_time = time.time()
    
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    evaluation_time = time.time() - start_time
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    log_info(f"Model Evaluation Results:")
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
    
    print("Accuracy:", accuracy)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'confusion_matrix': conf_matrix,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }

def plot_confusion_matrix(conf_matrix, save_path='confusion_matrix.png'):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['False', 'True'], 
                yticklabels=['False', 'True'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    log_info(f"Confusion matrix plot saved to {save_path}")

def main():
    # Initialize log file
    log_info("=" * 50)
    log_info("Starting Image Classification with Naive Bayes")
    log_info("=" * 50)
    
    # Load dataset
    X_train, X_test, y_train, y_test = load_dataset()

    # Train and save model
    model = train_model(X_train, y_train)

    # Evaluate model
    results = test_model(model, X_test, y_test)
    
    # Plot confusion matrix
    plot_confusion_matrix(results['confusion_matrix'])

    # Example of re-loading the model and testing again
    log_info("Testing reloaded model...")
    reloaded_model = load_model()
    if reloaded_model:
        print("Testing reloaded model:")
        test_model(reloaded_model, X_test, y_test)
    
    log_info("=" * 50)
    log_info("Training and evaluation completed")
    log_info("=" * 50)

if __name__ == "__main__":
    main()
