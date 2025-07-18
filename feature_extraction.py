import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.feature import graycomatrix, graycoprops
from skimage import measure
import os
from datetime import datetime
from sklearn.cluster import KMeans

class FeatureExtractor:
    def __init__(self, image_size=(64, 64)):
        self.image_size = image_size
        self.feature_names = []
        
    def extract_color_features(self, image):
        """Extract color-based features"""
        features = []
        feature_names = []
        
        # Convert to different color spaces
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Color histograms
        for i, color_space in enumerate(['BGR', 'HSV', 'LAB']):
            if color_space == 'BGR':
                img_space = image
            elif color_space == 'HSV':
                img_space = hsv
            else:
                img_space = lab
                
            for j in range(3):  # For each channel
                hist = cv2.calcHist([img_space], [j], None, [256], [0, 256])
                hist = hist.flatten() / hist.sum()  # Normalize
                
                # Statistical moments
                mean_val = np.mean(hist)
                std_val = np.std(hist)
                skewness = np.mean(((hist - mean_val) / std_val) ** 3) if std_val > 0 else 0
                kurtosis = np.mean(((hist - mean_val) / std_val) ** 4) if std_val > 0 else 0
                
                features.extend([mean_val, std_val, skewness, kurtosis])
                feature_names.extend([
                    f'{color_space}_ch{j}_mean', f'{color_space}_ch{j}_std',
                    f'{color_space}_ch{j}_skew', f'{color_space}_ch{j}_kurt'
                ])
        
        # Color moments (mean, std for each channel)
        for i in range(3):
            channel = image[:, :, i]
            features.extend([np.mean(channel), np.std(channel)])
            feature_names.extend([f'color_moment_mean_ch{i}', f'color_moment_std_ch{i}'])
        
        # Dominant colors (simplified)
        pixels = image.reshape(-1, 3)
        kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
        kmeans.fit(pixels)
        dominant_colors = kmeans.cluster_centers_
        
        for i, color in enumerate(dominant_colors):
            features.extend(color)
            feature_names.extend([f'dominant_color_{i}_B', f'dominant_color_{i}_G', f'dominant_color_{i}_R'])
        
        return features, feature_names
    
    def extract_texture_features(self, image):
        """Extract texture-based features using GLCM"""
        features = []
        feature_names = []
        
        # Convert to grayscale for texture analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # GLCM properties
        distances = [1, 2, 3]
        angles = [0, 45, 90, 135]
        
        for distance in distances:
            for angle in angles:
                # Convert angle to radians
                angle_rad = np.radians(angle)
                
                # Calculate GLCM
                glcm = graycomatrix(gray, [distance], [angle_rad], levels=256, symmetric=True, normed=True)
                
                # Calculate GLCM properties
                contrast = graycoprops(glcm, 'contrast')[0, 0]
                dissimilarity = graycoprops(glcm, 'dissimilarity')[0, 0]
                homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
                energy = graycoprops(glcm, 'energy')[0, 0]
                correlation = graycoprops(glcm, 'correlation')[0, 0]
                
                features.extend([contrast, dissimilarity, homogeneity, energy, correlation])
                feature_names.extend([
                    f'glcm_contrast_d{distance}_a{angle}',
                    f'glcm_dissimilarity_d{distance}_a{angle}',
                    f'glcm_homogeneity_d{distance}_a{angle}',
                    f'glcm_energy_d{distance}_a{angle}',
                    f'glcm_correlation_d{distance}_a{angle}'
                ])
        
        # Local Binary Pattern (simplified)
        lbp_features = self._calculate_lbp(gray)
        features.extend(lbp_features)
        feature_names.extend([f'lbp_hist_{i}' for i in range(len(lbp_features))])
        
        return features, feature_names
    
    def _calculate_lbp(self, gray_image):
        """Calculate Local Binary Pattern histogram"""
        # Simplified LBP implementation
        height, width = gray_image.shape
        lbp_image = np.zeros((height-2, width-2), dtype=np.uint8)
        
        for i in range(1, height-1):
            for j in range(1, width-1):
                center = gray_image[i, j]
                code = 0
                # 8-neighbor LBP
                neighbors = [
                    gray_image[i-1, j-1], gray_image[i-1, j], gray_image[i-1, j+1],
                    gray_image[i, j+1], gray_image[i+1, j+1], gray_image[i+1, j],
                    gray_image[i+1, j-1], gray_image[i, j-1]
                ]
                
                for k, neighbor in enumerate(neighbors):
                    if neighbor >= center:
                        code |= (1 << k)
                
                lbp_image[i-1, j-1] = code
        
        # Calculate histogram
        hist, _ = np.histogram(lbp_image.flatten(), bins=256, range=(0, 256))
        hist = hist / hist.sum()  # Normalize
        
        # Use first 16 bins for simplicity
        return hist[:16]
    
    def extract_shape_features(self, image):
        """Extract shape-based features"""
        features = []
        feature_names = []
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Shape features
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)
            compactness = (perimeter ** 2) / (4 * np.pi * area) if area > 0 else 0
            
            # Bounding rectangle
            x, y, w, h = cv2.boundingRect(largest_contour)
            aspect_ratio = w / h if h > 0 else 0
            extent = area / (w * h) if w * h > 0 else 0
            
            # Convex hull
            hull = cv2.convexHull(largest_contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            
            # Moments
            moments = cv2.moments(largest_contour)
            if moments['m00'] != 0:
                cx = moments['m10'] / moments['m00']
                cy = moments['m01'] / moments['m00']
            else:
                cx, cy = 0, 0
            
            features.extend([area, perimeter, compactness, aspect_ratio, extent, solidity, cx, cy])
            feature_names.extend([
                'shape_area', 'shape_perimeter', 'shape_compactness',
                'shape_aspect_ratio', 'shape_extent', 'shape_solidity',
                'shape_centroid_x', 'shape_centroid_y'
            ])
        else:
            # No contours found
            features.extend([0, 0, 0, 0, 0, 0, 0, 0])
            feature_names.extend([
                'shape_area', 'shape_perimeter', 'shape_compactness',
                'shape_aspect_ratio', 'shape_extent', 'shape_solidity',
                'shape_centroid_x', 'shape_centroid_y'
            ])
        
        # Edge density
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        features.append(edge_density)
        feature_names.append('edge_density')
        
        return features, feature_names
    
    def extract_all_features(self, image):
        """Extract all features: color, texture, and shape"""
        # Resize image to standard size
        image = cv2.resize(image, self.image_size)
        
        # Extract different feature types
        color_features, color_names = self.extract_color_features(image)
        texture_features, texture_names = self.extract_texture_features(image)
        shape_features, shape_names = self.extract_shape_features(image)
        
        # Combine all features
        all_features = color_features + texture_features + shape_features
        all_names = color_names + texture_names + shape_names
        
        return all_features, all_names
    
    def visualize_features(self, image, save_path='feature_visualization.png'):
        """Create comprehensive feature visualization"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Feature Extraction Visualization', fontsize=16)
        
        # Original image
        axes[0, 0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Color histogram
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        for i, (color, channel) in enumerate([('Hue', 0), ('Saturation', 1), ('Value', 2)]):
            hist = cv2.calcHist([hsv], [channel], None, [256], [0, 256])
            axes[0, 1].plot(hist, label=color)
        axes[0, 1].set_title('HSV Color Histogram')
        axes[0, 1].legend()
        
        # Grayscale for texture
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        axes[0, 2].imshow(gray, cmap='gray')
        axes[0, 2].set_title('Grayscale Image')
        axes[0, 2].axis('off')
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        axes[1, 0].imshow(edges, cmap='gray')
        axes[1, 0].set_title('Edge Detection')
        axes[1, 0].axis('off')
        
        # GLCM visualization (simplified)
        glcm = graycomatrix(gray, [1], [0], levels=256, symmetric=True, normed=True)
        axes[1, 1].imshow(glcm[:, :, 0, 0], cmap='hot')
        axes[1, 1].set_title('GLCM Matrix')
        
        # Feature distribution (placeholder)
        axes[1, 2].text(0.5, 0.5, 'Feature\nDistribution\n(Generated during\nfull extraction)', 
                        ha='center', va='center', transform=axes[1, 2].transAxes)
        axes[1, 2].set_title('Feature Distribution')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path

def log_feature_info(message):
    """Log feature extraction information"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_message = f"[{timestamp}] FEATURE: {message}"
    print(log_message)
    with open('feature_extraction_log.txt', 'a') as f:
        f.write(log_message + '\n')

def extract_features_from_dataset(true_dir, false_dir, image_size=(64, 64)):
    """Extract features from entire dataset"""
    extractor = FeatureExtractor(image_size)
    
    all_features = []
    all_labels = []
    feature_names = None
    
    log_feature_info("Starting feature extraction from dataset...")
    
    # Process true images
    true_count = 0
    for filename in os.listdir(true_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            path = os.path.join(true_dir, filename)
            image = cv2.imread(path)
            if image is not None:
                features, names = extractor.extract_all_features(image)
                all_features.append(features)
                all_labels.append(1)
                true_count += 1
                
                if feature_names is None:
                    feature_names = names
                
                # Create visualization for first few images
                if true_count <= 3:
                    viz_path = f'feature_viz_true_{true_count}.png'
                    extractor.visualize_features(image, viz_path)
                    log_feature_info(f"Created visualization: {viz_path}")
    
    # Process false images
    false_count = 0
    for filename in os.listdir(false_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            path = os.path.join(false_dir, filename)
            image = cv2.imread(path)
            if image is not None:
                features, names = extractor.extract_all_features(image)
                all_features.append(features)
                all_labels.append(0)
                false_count += 1
                
                # Create visualization for first few images
                if false_count <= 3:
                    viz_path = f'feature_viz_false_{false_count}.png'
                    extractor.visualize_features(image, viz_path)
                    log_feature_info(f"Created visualization: {viz_path}")
    
    log_feature_info(f"Feature extraction completed:")
    log_feature_info(f"  True images processed: {true_count}")
    log_feature_info(f"  False images processed: {false_count}")
    log_feature_info(f"  Total features per image: {len(feature_names)}")
    log_feature_info(f"  Feature types: Color ({len([n for n in feature_names if 'color' in n])})")
    log_feature_info(f"  Feature types: Texture ({len([n for n in feature_names if 'glcm' in n or 'lbp' in n])})")
    log_feature_info(f"  Feature types: Shape ({len([n for n in feature_names if 'shape' in n or 'edge' in n])})")
    
    return np.array(all_features), np.array(all_labels), feature_names

if __name__ == "__main__":
    # Test feature extraction
    test_image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    extractor = FeatureExtractor()
    features, names = extractor.extract_all_features(test_image)
    print(f"Extracted {len(features)} features")
    print(f"Feature names: {names[:10]}...")  # Show first 10 