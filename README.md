# Image Classification with Naive Bayes: Hawar Daun Bakteri

This project implements a binary image classifier using Gaussian Naive Bayes algorithm with **advanced feature extraction**, comprehensive logging, and statistical analysis for research purposes. The project is a part of my research on Hawar Daun Bakteri. The goal is to classify images of Hawar Daun Bakteri into two categories: True (Hawar Daun Bakteri) and False (Bukan Hawar Daun Bakteri).

## Project Overview

The system classifies images into two categories (True/False) using machine learning techniques with **enhanced feature extraction** including color, texture, and shape analysis. It includes detailed logging, statistical analysis, and performance evaluation to support research and experimentation.

## Process **Flow**

### 1. **Initialization and Setup**
```
┌─────────────────────────────────────┐
│           Main Execution            │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│      Initialize Enhanced Logging    │
│  - Create logs/training_enhanced_log.txt │
│  - Set up timestamp logging         │
│  - Feature extraction logging       │
└─────────────────┬───────────────────┘
```

### 2. **Enhanced Dataset Loading Process**
```
                  │
                  ▼
┌─────────────────────────────────────┐
│    Advanced Feature Extraction      │
│  - Load images from data/true/      │
│  - Load images from data/false/     │
│  - Extract color, texture, shape    │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│      Multi-Feature Processing      │
│  For each image:                    │
│  ├─ Color Features                  │
│  │  ├─ BGR, HSV, LAB color spaces  │
│  │  ├─ Color histograms & moments   │
│  │  └─ Dominant color analysis     │
│  ├─ Texture Features                │
│  │  ├─ GLCM (Gray Level Co-occurrence) │
│  │  ├─ LBP (Local Binary Pattern)  │
│  │  └─ Texture statistics          │
│  └─ Shape Features                  │
│    ├─ Edge detection & contours     │
│    ├─ Shape metrics (area, perimeter) │
│    └─ Geometric properties         │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│      Enhanced Dataset Statistics    │
│  - Total samples count              │
│  - Class distribution               │
│  - Feature type breakdown           │
│  - Color/Texture/Shape feature counts │
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

### 3. **Enhanced Model Training Process**
```
                  │
                  ▼
┌─────────────────────────────────────┐
│    Enhanced Model Training Function │
│  - Initialize GaussianNB model      │
│  - Record training start time       │
│  - Handle multi-dimensional features │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│         Fit Model to Data           │
│  - Train on X_train, y_train        │
│  - Learn class priors               │
│  - Estimate feature distributions   │
│  - Handle complex feature space     │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│    Enhanced Training Statistics     │
│  - Training time measurement        │
│  - Model information logging        │
│  - Class labels and priors          │
│  - Feature count analysis           │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│         Save Enhanced Model         │
│  - Serialize model to models/model_enhanced.pkl │
│  - Enable model reuse               │
└─────────────────┬───────────────────┘
```

### 4. **Enhanced Model Evaluation Process**
```
                  │
                  ▼
┌─────────────────────────────────────┐
│    Enhanced Model Evaluation        │
│  - Load test data                   │
│  - Record evaluation start time     │
│  - Handle multi-feature predictions │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│         Make Predictions            │
│  - Predict classes (y_pred)         │
│  - Get prediction probabilities     │
│  - Calculate confidence scores      │
│  - Multi-feature classification     │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│    Calculate Performance Metrics    │
│  ├─ Accuracy Score                  │
│  ├─ Confusion Matrix                │
│  ├─ Precision                       │
│  ├─ Recall                          │
│  └─ F1-Score                        │
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│    Enhanced Results Logging         │
│  - All metrics with timestamps      │
│  - Detailed confusion matrix        │
│  - Performance breakdown            │
│  - Feature analysis results         │
└─────────────────┬───────────────────┘
```

### 5. **Advanced Visualization and Analysis**
```
                  │
                  ▼
┌─────────────────────────────────────┐
│    Enhanced Confusion Matrix        │
│  - Create heatmap visualization     │
│  - Save as confusion_matrix_enhanced.png │
│  - Professional plotting with seaborn│
└─────────────────┬───────────────────┘
                  │
                  ▼
┌─────────────────────────────────────┐
│      Feature Analysis Plots         │
│  - Feature type distribution        │
│  - Top features by variance         │
│  - Feature correlation matrix       │
│  - Overall feature distribution     │
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

## Enhanced Data Flow Diagram

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│   Raw       │    │  Enhanced   │    │   Training  │
│  Images     │───▶│  Features   │───▶│    Data     │
│ (64x64)     │    │ (Multi-dim) │    │   (80%)     │
└─────────────┘    └─────────────┘    └─────────────┘
                           │                   │
                           ▼                   ▼
                   ┌─────────────┐    ┌─────────────┐
                   │   Test      │    │  Enhanced   │
                   │   Data      │    │   Naive     │
                   │   (20%)     │    │   Bayes     │
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
                   │  Enhanced   │    │  Advanced   │
                   │  Logs &     │    │  Visualizations │
                   │  Statistics │    │   (PNG)     │
                   │  (txt file) │    │             │
                   └─────────────┘    └─────────────┘
```

## Key Components

### **Enhanced Data Processing**
- **Image Loading**: Reads images from `data/true/` and `data/false/` directories
- **Advanced Feature Extraction**: 
  - **Color Features**: BGR, HSV, LAB color spaces, histograms, moments, dominant colors
  - **Texture Features**: GLCM (Gray Level Co-occurrence Matrix), LBP (Local Binary Pattern)
  - **Shape Features**: Edge detection, contours, geometric properties
- **Multi-dimensional Features**: Complex feature vectors combining color, texture, and shape
- **Data Splitting**: 80% training, 20% testing with stratification

### **Enhanced Model Training**
- **Algorithm**: Gaussian Naive Bayes with multi-feature support
- **Learning**: Estimates class priors and complex feature distributions
- **Persistence**: Saves trained model to `models/model_enhanced.pkl`

### **Advanced Evaluation Metrics**
- **Accuracy**: Overall correct predictions percentage
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed breakdown of predictions
- **Feature Analysis**: Variance analysis and correlation studies

### **Enhanced Logging System**
- **Timestamped Logs**: All events logged with timestamps
- **File Output**: Complete log saved to `logs/training_enhanced_log.txt`
- **Console Display**: Real-time logging to console
- **Feature Logging**: Detailed feature extraction statistics
- **Research Documentation**: Comprehensive experiment tracking

## Output Files

1. **`logs/training_enhanced_log.txt`**: Complete research log with all statistics and timestamps
2. **`models/model_enhanced.pkl`**: Serialized trained model for reuse
3. **`confusion_matrix_enhanced.png`**: Visual representation of model performance
4. **`feature_analysis.png`**: Comprehensive feature analysis plots
5. **`requirements.txt`**: Python dependencies

## Usage

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare Data**:
   - Place true images in `data/true/`
   - Place false images in `data/false/`
   - Supported formats: PNG, JPG, JPEG

3. **Run Enhanced Training**:
   ```bash
   python main.py
   ```

4. **Review Results**:
   - Check `logs/training_enhanced_log.txt` for detailed statistics
   - View `confusion_matrix_enhanced.png` for visual performance
   - Analyze `feature_analysis.png` for feature insights
   - Use `models/model_enhanced.pkl` for future predictions

## Research Status

**⚠️ Research in Progress** - This project is currently under active development and research. The enhanced feature extraction system has been implemented and is being evaluated. Key areas of ongoing research include:

- **Feature Selection**: Optimizing the combination of color, texture, and shape features
- **Model Performance**: Evaluating the impact of multi-dimensional features on classification accuracy
- **Dataset Expansion**: Collecting additional samples for improved training
- **Algorithm Comparison**: Testing against other classification algorithms
- **Parameter Tuning**: Optimizing model parameters for better performance

## Research Benefits

- **Advanced Feature Extraction**: Multi-dimensional analysis combining color, texture, and shape
- **Reproducibility**: Fixed random state and comprehensive logging
- **Transparency**: Complete visibility into data processing and model performance
- **Documentation**: Automatic generation of research logs
- **Analysis**: Rich statistical information and feature analysis for research papers
- **Visualization**: Professional plots for presentations and reports

## Technical Details

- **Image Size**: 64x64 pixels (resized for processing)
- **Feature Types**: Color (histograms, moments, dominant colors), Texture (GLCM, LBP), Shape (edges, contours, geometric properties)
- **Algorithm**: Gaussian Naive Bayes with multi-feature support
- **Cross-validation**: Train/test split (80/20)
- **Random State**: 42 (for reproducibility)
- **Enhanced Logging**: Comprehensive feature extraction and model performance tracking 