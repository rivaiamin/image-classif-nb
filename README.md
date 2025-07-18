# Image Classification with Naive Bayes

This project implements a binary image classifier using Gaussian Naive Bayes algorithm with comprehensive logging and statistical analysis for research purposes.

## Project Overview

The system classifies images into two categories (True/False) using machine learning techniques. It includes detailed logging, statistical analysis, and performance evaluation to support research and experimentation.

## Process Flow

### 1. **Initialization and Setup**
```
┌─────────────────────────────────────┐
│           Main Execution            │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│      Initialize Logging System      │
│  - Create training_log.txt file     │
│  - Set up timestamp logging         │
└─────────────────┬───────────────────┘
```

### 2. **Dataset Loading Process**
```
                  │
                  ▼
┌─────────────────────────────────────┐
│        Load Dataset Function        │
│  - Load images from data/true/      │
│  - Load images from data/false/     │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│      Image Processing Pipeline      │
│  For each image:                    │
│  ├─ Read image (grayscale)          │
│  ├─ Resize to 64x64 pixels          │
│  ├─ Flatten to 1D array (4096 dim)  │
│  └─ Add to dataset with label       │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│      Dataset Statistics Logging     │
│  - Total samples count              │
│  - Class distribution               │
│  - Feature dimensions               │
│  - Loading success rates            │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│        Train/Test Split             │
│  - 80% training, 20% testing        │
│  - Stratified sampling              │
│  - Random state for reproducibility │
└─────────────────┬───────────────────┘
```

### 3. **Model Training Process**
```
                  │
                  ▼
┌─────────────────────────────────────┐
│        Model Training Function      │
│  - Initialize GaussianNB model      │
│  - Record training start time       │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│         Fit Model to Data           │
│  - Train on X_train, y_train        │
│  - Learn class priors               │
│  - Estimate feature distributions   │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│      Training Statistics Logging    │
│  - Training time measurement        │
│  - Model information logging        │
│  - Class labels and priors          │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│         Save Trained Model          │
│  - Serialize model to model.pkl     │
│  - Enable model reuse               │
└─────────────────┬───────────────────┘
```

### 4. **Model Evaluation Process**
```
                  │
                  ▼
┌─────────────────────────────────────┐
│        Model Evaluation Function    │
│  - Load test data                   │
│  - Record evaluation start time     │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│         Make Predictions            │
│  - Predict classes (y_pred)         │
│  - Get prediction probabilities     │
│  - Calculate confidence scores      │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│      Calculate Performance Metrics  │
│  ├─ Accuracy Score                  │
│  ├─ Confusion Matrix                │
│  ├─ Precision                       │
│  ├─ Recall                          │
│  └─ F1-Score                        │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│      Evaluation Results Logging     │
│  - All metrics with timestamps      │
│  - Detailed confusion matrix        │
│  - Performance breakdown            │
└─────────────────┬───────────────────┘
```

### 5. **Visualization and Output**
```
                  │
                  ▼
┌─────────────────────────────────────┐
│      Generate Confusion Matrix      │
│  - Create heatmap visualization     │
│  - Save as confusion_matrix.png     │
│  - Professional plotting with seaborn│
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│         Model Reload Test           │
│  - Load saved model from file       │
│  - Verify model persistence         │
│  - Re-run evaluation for validation │
└─────────────────┬───────────────────┘
```

## Data Flow Diagram

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Raw       │    │  Processed  │    │   Training  │
│  Images     │───▶│   Features  │───▶│    Data     │
│ (64x64)     │    │  (4096 dim) │    │   (80%)     │
└─────────────┘    └─────────────┘    └─────────────┘
                           │                   │
                           ▼                   ▼
                   ┌─────────────┐    ┌─────────────┐
                   │   Test      │    │   Naive     │
                   │   Data      │    │   Bayes     │
                   │   (20%)     │    │   Model     │
                   └─────────────┘    └─────────────┘
                                              │
                                              ▼
                   ┌─────────────┐    ┌─────────────┐
                   │  Model      │    │ Performance │
                   │  Predictions│◀───│  Evaluation │
                   └─────────────┘    └─────────────┘
                                              │
                                              ▼
                   ┌─────────────┐    ┌─────────────┐
                   │  Logs &     │    │  Confusion  │
                   │  Statistics │    │   Matrix    │
                   │  (txt file) │    │   (PNG)     │
                   └─────────────┘    └─────────────┘
```

## Key Components

### **Data Processing**
- **Image Loading**: Reads images from `data/true/` and `data/false/` directories
- **Preprocessing**: Converts to grayscale, resizes to 64x64 pixels
- **Feature Extraction**: Flattens images to 4096-dimensional feature vectors
- **Data Splitting**: 80% training, 20% testing with stratification

### **Model Training**
- **Algorithm**: Gaussian Naive Bayes
- **Learning**: Estimates class priors and feature distributions
- **Persistence**: Saves trained model to `model.pkl`

### **Evaluation Metrics**
- **Accuracy**: Overall correct predictions percentage
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed breakdown of predictions

### **Logging System**
- **Timestamped Logs**: All events logged with timestamps
- **File Output**: Complete log saved to `training_log.txt`
- **Console Display**: Real-time logging to console
- **Research Documentation**: Comprehensive experiment tracking

## Output Files

1. **`training_log.txt`**: Complete research log with all statistics and timestamps
2. **`model.pkl`**: Serialized trained model for reuse
3. **`confusion_matrix.png`**: Visual representation of model performance
4. **`requirements.txt`**: Python dependencies

## Usage

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare Data**:
   - Place true images in `data/true/`
   - Place false images in `data/false/`
   - Supported formats: PNG, JPG, JPEG

3. **Run Training**:
   ```bash
   python main.py
   ```

4. **Review Results**:
   - Check `training_log.txt` for detailed statistics
   - View `confusion_matrix.png` for visual performance
   - Use `model.pkl` for future predictions

## Research Benefits

- **Reproducibility**: Fixed random state and comprehensive logging
- **Transparency**: Complete visibility into data processing and model performance
- **Documentation**: Automatic generation of research logs
- **Analysis**: Rich statistical information for research papers
- **Visualization**: Professional plots for presentations and reports

## Technical Details

- **Image Size**: 64x64 pixels (4096 features after flattening)
- **Color Space**: Grayscale (single channel)
- **Algorithm**: Gaussian Naive Bayes
- **Cross-validation**: Train/test split (80/20)
- **Random State**: 42 (for reproducibility) 