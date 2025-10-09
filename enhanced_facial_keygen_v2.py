"""
Enhanced Facial Key Generation Model v2.0
Implementation of Professor's Advanced Pipeline:
Face → Features (100–300) → Slots → Process using LFSR → Output → Reinsert → Repeat → Final output → Numeric keys

This version includes:
1. Enhanced feature extraction with multiple facial analysis techniques
2. Better discriminative power between different face images
3. Improved LFSR processing with dynamic feedback
4. Statistical validation of generated keys
"""

import cv2
import numpy as np
import hashlib
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import mediapipe as mp
from typing import List, Tuple, Dict
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class EnhancedFacialKeygenModel:
    def __init__(self, target_features: int = 150, slot_count: int = 8, lfsr_rounds: int = 4):
        """
        Initialize the Enhanced Facial Keygen Model
        
        Args:
            target_features: Target number of facial features (100-300)
            slot_count: Number of processing slots for LFSR
            lfsr_rounds: Number of LFSR processing rounds
        """
        self.target_features = min(max(target_features, 100), 300)
        self.slot_count = slot_count
        self.lfsr_rounds = lfsr_rounds
        
        # Initialize MediaPipe components
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_face_detection = mp.solutions.face_detection
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.7
        )
        
        # Processing components
        self.scaler = StandardScaler()
        self.feature_scaler = MinMaxScaler()
        self.pca = None
        self.is_trained = False
        
        # LFSR configurations
        self.lfsr_configs = self._generate_lfsr_configs()
        
        print(f"Enhanced Facial Keygen Model v2.0 Initialized:")
        print(f"- Target Features: {self.target_features}")
        print(f"- Slot Count: {self.slot_count}")
        print(f"- LFSR Rounds: {self.lfsr_rounds}")

    def _generate_lfsr_configs(self) -> List[Dict]:
        """Generate LFSR configurations for each slot"""
        configs = []
        
        # Different primitive polynomials and configurations
        base_configs = [
            {'polynomial': [8, 6, 5, 4], 'initial_state': 0xFF},
            {'polynomial': [8, 7, 6, 1], 'initial_state': 0xAA},
            {'polynomial': [8, 6, 5, 1], 'initial_state': 0x55},
            {'polynomial': [8, 7, 2, 1], 'initial_state': 0xCC},
            {'polynomial': [8, 6, 4, 3], 'initial_state': 0x33},
            {'polynomial': [8, 5, 4, 3], 'initial_state': 0xF0},
            {'polynomial': [8, 7, 5, 3], 'initial_state': 0x0F},
            {'polynomial': [8, 6, 3, 2], 'initial_state': 0x99},
        ]
        
        for i in range(self.slot_count):
            config = base_configs[i % len(base_configs)].copy()
            # Add slot-specific modifications
            config['feedback_mask'] = (0xFF << (i % 4)) & 0xFFFFFFFF
            config['slot_id'] = i
            configs.append(config)
        
        return configs

    def extract_comprehensive_features(self, image_path: str) -> np.ndarray:
        """
        Extract comprehensive facial features using multiple techniques
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        features = []
        
        # 1. MediaPipe Face Mesh Features
        mesh_features = self._extract_mesh_features(rgb_image)
        features.extend(mesh_features)
        
        # 2. Geometric Features
        geometric_features = self._extract_geometric_features(rgb_image, h, w)
        features.extend(geometric_features)
        
        # 3. Texture Features
        texture_features = self._extract_texture_features(image)
        features.extend(texture_features)
        
        # 4. Statistical Features
        statistical_features = self._extract_statistical_features(image)
        features.extend(statistical_features)
        
        # 5. Edge and Gradient Features
        edge_features = self._extract_edge_features(image)
        features.extend(edge_features)
        
        feature_vector = np.array(features, dtype=np.float32)
        print(f"Extracted {len(feature_vector)} comprehensive features")
        
        return feature_vector

    def _extract_mesh_features(self, rgb_image) -> List[float]:
        """Extract features from MediaPipe face mesh"""
        features = []
        
        results = self.face_mesh.process(rgb_image)
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            
            # Extract landmark coordinates with more detail
            for landmark in face_landmarks.landmark:
                features.extend([landmark.x, landmark.y, landmark.z])
            
            # Add landmark-based geometric calculations
            landmarks_array = np.array([[lm.x, lm.y, lm.z] for lm in face_landmarks.landmark])
            
            # Centroid
            centroid = np.mean(landmarks_array, axis=0)
            features.extend(centroid.tolist())
            
            # Standard deviations
            std_devs = np.std(landmarks_array, axis=0)
            features.extend(std_devs.tolist())
            
            # Distances from centroid
            distances = np.linalg.norm(landmarks_array - centroid, axis=1)
            features.extend([np.mean(distances), np.std(distances), np.max(distances), np.min(distances)])
        
        return features

    def _extract_geometric_features(self, rgb_image, h, w) -> List[float]:
        """Extract geometric features from face"""
        features = []
        
        # Face detection for bounding box
        detection_results = self.face_detection.process(rgb_image)
        
        if detection_results.detections:
            detection = detection_results.detections[0]
            bbox = detection.location_data.relative_bounding_box
            
            # Bounding box features
            features.extend([bbox.xmin, bbox.ymin, bbox.width, bbox.height])
            
            # Aspect ratios and proportions
            face_aspect_ratio = bbox.width / bbox.height if bbox.height > 0 else 1.0
            features.append(face_aspect_ratio)
            
            # Position in image
            center_x = bbox.xmin + bbox.width / 2
            center_y = bbox.ymin + bbox.height / 2
            features.extend([center_x, center_y])
            
            # Area and perimeter
            area = bbox.width * bbox.height
            perimeter = 2 * (bbox.width + bbox.height)
            features.extend([area, perimeter])
        else:
            features.extend([0.0] * 10)  # Padding if no detection
        
        return features

    def _extract_texture_features(self, image) -> List[float]:
        """Extract texture-based features"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        features = []
        
        # Resize for consistent processing
        gray_resized = cv2.resize(gray, (64, 64))
        
        # Local Binary Pattern (simplified version)
        lbp_features = []
        for i in range(1, gray_resized.shape[0]-1):
            for j in range(1, gray_resized.shape[1]-1):
                center = gray_resized[i, j]
                pattern = 0
                for di, dj in [(-1,-1), (-1,0), (-1,1), (0,1), (1,1), (1,0), (1,-1), (0,-1)]:
                    if gray_resized[i+di, j+dj] >= center:
                        pattern += 1
                lbp_features.append(pattern)
        
        # Statistical measures of LBP
        lbp_array = np.array(lbp_features)
        features.extend([
            np.mean(lbp_array),
            np.std(lbp_array),
            np.median(lbp_array),
            np.max(lbp_array),
            np.min(lbp_array)
        ])
        
        # Histogram features
        hist = cv2.calcHist([gray_resized], [0], None, [16], [0, 256])
        hist_normalized = hist.flatten() / np.sum(hist)
        features.extend(hist_normalized.tolist())
        
        return features

    def _extract_statistical_features(self, image) -> List[float]:
        """Extract statistical features from image"""
        features = []
        
        # Convert to different color spaces and extract statistics
        color_spaces = [
            ('BGR', image),
            ('HSV', cv2.cvtColor(image, cv2.COLOR_BGR2HSV)),
            ('LAB', cv2.cvtColor(image, cv2.COLOR_BGR2LAB)),
            ('GRAY', cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        ]
        
        for name, img in color_spaces:
            if len(img.shape) == 3:
                for channel in range(img.shape[2]):
                    ch = img[:, :, channel]
                    features.extend([
                        np.mean(ch),
                        np.std(ch),
                        np.median(ch),
                        np.percentile(ch, 25),
                        np.percentile(ch, 75)
                    ])
            else:
                features.extend([
                    np.mean(img),
                    np.std(img),
                    np.median(img),
                    np.percentile(img, 25),
                    np.percentile(img, 75)
                ])
        
        return features

    def _extract_edge_features(self, image) -> List[float]:
        """Extract edge and gradient features"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        features = []
        
        # Sobel gradients
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Gradient magnitude and direction
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        direction = np.arctan2(grad_y, grad_x)
        
        features.extend([
            np.mean(magnitude),
            np.std(magnitude),
            np.mean(direction),
            np.std(direction)
        ])
        
        # Canny edges
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        features.append(edge_density)
        
        # Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        features.extend([np.mean(laplacian), np.std(laplacian)])
        
        return features

    def train_feature_processor(self, image_paths: List[str]):
        """Train the feature processing pipeline"""
        print(f"Training enhanced feature processor on {len(image_paths)} images...")
        
        all_features = []
        for img_path in image_paths:
            try:
                features = self.extract_comprehensive_features(img_path)
                all_features.append(features)
            except Exception as e:
                print(f"Warning: Could not process {img_path}: {e}")
                continue
        
        if not all_features:
            raise ValueError("No valid features extracted from training images")
        
        # Handle different feature vector lengths
        max_len = max(len(f) for f in all_features)
        padded_features = []
        
        for features in all_features:
            if len(features) < max_len:
                padded = np.zeros(max_len)
                padded[:len(features)] = features
                padded_features.append(padded)
            else:
                padded_features.append(features[:max_len])
        
        feature_matrix = np.vstack(padded_features)
        print(f"Training on feature matrix shape: {feature_matrix.shape}")
        
        # Fit scalers
        self.scaler.fit(feature_matrix)
        scaled_features = self.scaler.transform(feature_matrix)
        
        # PCA with appropriate number of components
        max_components = min(scaled_features.shape[0]-1, scaled_features.shape[1], self.target_features)
        
        if max_components < self.target_features:
            print(f"Adjusting target features from {self.target_features} to {max_components}")
            self.target_features = max_components
        
        self.pca = PCA(n_components=self.target_features)
        self.pca.fit(scaled_features)
        
        # Fit feature scaler for final normalization
        pca_features = self.pca.transform(scaled_features)
        self.feature_scaler.fit(pca_features)
        
        self.is_trained = True
        print(f"Enhanced feature processor trained:")
        print(f"- Final feature count: {self.target_features}")
        print(f"- Explained variance ratio: {np.sum(self.pca.explained_variance_ratio_):.4f}")

    def process_image_to_slots(self, image_path: str) -> List[np.ndarray]:
        """Process image through the complete feature extraction and slot creation pipeline"""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train_feature_processor() first.")
        
        # Extract and process features
        raw_features = self.extract_comprehensive_features(image_path)
        
        # Handle feature length consistency
        expected_length = self.scaler.n_features_in_
        if len(raw_features) < expected_length:
            padded_features = np.zeros(expected_length)
            padded_features[:len(raw_features)] = raw_features
            raw_features = padded_features
        else:
            raw_features = raw_features[:expected_length]
        
        # Apply transformations
        scaled_features = self.scaler.transform(raw_features.reshape(1, -1))
        pca_features = self.pca.transform(scaled_features).flatten()
        final_features = self.feature_scaler.transform(pca_features.reshape(1, -1)).flatten()
        
        # Create slots
        slot_size = len(final_features) // self.slot_count
        remainder = len(final_features) % self.slot_count
        
        slots = []
        idx = 0
        
        for i in range(self.slot_count):
            # Distribute remainder evenly among first few slots
            current_slot_size = slot_size + (1 if i < remainder else 0)
            slot_data = final_features[idx:idx + current_slot_size]
            slots.append(slot_data)
            idx += current_slot_size
        
        print(f"Created {len(slots)} slots from {len(final_features)} features")
        return slots

    def advanced_lfsr_process(self, slot_data: np.ndarray, config: Dict, round_num: int) -> np.ndarray:
        """Advanced LFSR processing with dynamic feedback"""
        if len(slot_data) == 0:
            return slot_data
        
        # Convert to integer state
        normalized = (slot_data - np.min(slot_data)) / (np.max(slot_data) - np.min(slot_data) + 1e-10)
        state = (normalized * 0xFFFFFFFF).astype(np.uint32)
        
        polynomial = config['polynomial']
        feedback_mask = config['feedback_mask']
        
        # Dynamic LFSR with round-dependent modifications
        for iteration in range(32 + round_num * 16):  # Increasing complexity each round
            # Calculate feedback using polynomial
            feedback = 0
            for tap in polynomial:
                if tap < len(state):
                    feedback ^= state[tap]
            
            # Apply feedback mask and round-specific transformations
            feedback ^= (feedback_mask >> (iteration % 32)) & 0xFF
            feedback ^= (round_num << (iteration % 4)) & 0xFF
            
            # Shift register and insert feedback
            state = np.roll(state, 1)
            state[0] = feedback
        
        # Convert back to normalized float
        processed = state.astype(np.float64) / 0xFFFFFFFF
        
        return processed.astype(np.float32)

    def generate_advanced_keys(self, final_slots: List[np.ndarray]) -> Dict[str, any]:
        """Generate advanced cryptographic keys from processed slots"""
        # Combine all slot data
        combined_data = np.concatenate(final_slots)
        
        keys = {}
        
        # Primary cryptographic hashes
        data_bytes = combined_data.tobytes()
        keys['sha256'] = hashlib.sha256(data_bytes).hexdigest()
        keys['sha512'] = hashlib.sha512(data_bytes).hexdigest()
        keys['md5'] = hashlib.md5(data_bytes).hexdigest()
        
        # Numeric key representations
        keys['primary_numeric'] = int(keys['sha256'][:16], 16)  # 64-bit from SHA256
        keys['secondary_numeric'] = int(keys['md5'][:8], 16)    # 32-bit from MD5
        
        # Slot-specific keys
        slot_keys = []
        for i, slot in enumerate(final_slots):
            slot_bytes = slot.tobytes()
            slot_hash = hashlib.sha256(slot_bytes + str(i).encode()).hexdigest()
            slot_keys.append(int(slot_hash[:8], 16))  # 32-bit per slot
        
        keys['slot_keys'] = slot_keys
        keys['slot_combined'] = sum(slot_keys) % (2**64)  # 64-bit combined
        
        # Statistical keys for validation
        keys['statistical'] = {
            'mean': float(np.mean(combined_data)),
            'std': float(np.std(combined_data)),
            'entropy': float(-np.sum(combined_data * np.log2(combined_data + 1e-10))),
            'variance': float(np.var(combined_data)),
            'kurtosis': float(self._calculate_kurtosis(combined_data)),
            'skewness': float(self._calculate_skewness(combined_data))
        }
        
        # Binary representations
        keys['binary_key'] = ''.join(format(int(x * 255), '08b') for x in combined_data[:32])
        keys['hex_key'] = ''.join(format(int(x * 255), '02x') for x in combined_data[:32])
        
        return keys

    def _calculate_kurtosis(self, data):
        """Calculate kurtosis of data"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        normalized = (data - mean) / std
        return np.mean(normalized**4) - 3

    def _calculate_skewness(self, data):
        """Calculate skewness of data"""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0
        normalized = (data - mean) / std
        return np.mean(normalized**3)

    def full_enhanced_pipeline(self, image_path: str) -> Dict[str, any]:
        """Execute the complete enhanced pipeline"""
        print(f"\n=== Enhanced Pipeline Processing: {image_path} ===")
        
        # Step 1: Create initial slots
        print("Step 1: Extracting features and creating slots...")
        slots = self.process_image_to_slots(image_path)
        
        # Step 2: Multi-round LFSR processing with reinsertion
        print(f"Step 2: Multi-round LFSR processing ({self.lfsr_rounds} rounds)...")
        
        current_slots = slots
        for round_num in range(self.lfsr_rounds):
            print(f"  Round {round_num + 1}/{self.lfsr_rounds}")
            
            processed_slots = []
            for i, slot in enumerate(current_slots):
                config = self.lfsr_configs[i % len(self.lfsr_configs)]
                processed_slot = self.advanced_lfsr_process(slot, config, round_num)
                processed_slots.append(processed_slot)
            
            # Reinsertion: Use processed slots as input for next round
            current_slots = processed_slots
            
            if round_num < self.lfsr_rounds - 1:
                print(f"    Reinserting processed data for next round...")
        
        # Step 3: Generate final keys
        print("Step 3: Generating advanced cryptographic keys...")
        final_keys = self.generate_advanced_keys(current_slots)
        
        # Add processing metadata
        final_keys['metadata'] = {
            'image_path': image_path,
            'target_features': self.target_features,
            'slot_count': self.slot_count,
            'lfsr_rounds': self.lfsr_rounds,
            'timestamp': datetime.now().isoformat(),
            'model_version': 'Enhanced v2.0'
        }
        
        print("=== Enhanced Pipeline Complete ===\n")
        return final_keys

def demo_enhanced_model():
    """Demonstration of the enhanced facial keygen model"""
    
    # Initialize enhanced model
    model = EnhancedFacialKeygenModel(
        target_features=150,  # Reasonable target for limited training data
        slot_count=8,         # Fewer slots for better distribution
        lfsr_rounds=4         # More rounds for better diffusion
    )
    
    # Get available images
    captures_dir = "captures"
    if not os.path.exists(captures_dir):
        print(f"Error: {captures_dir} directory not found!")
        return
    
    image_files = [f for f in os.listdir(captures_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not image_files:
        print(f"Error: No image files found in {captures_dir}!")
        return
    
    image_paths = [os.path.join(captures_dir, f) for f in image_files]
    print(f"Processing images: {image_files}")
    
    # Train the enhanced model
    try:
        model.train_feature_processor(image_paths)
    except Exception as e:
        print(f"Error during training: {e}")
        return
    
    # Process each image with enhanced pipeline
    results = {}
    print(f"\n{'='*60}")
    print("ENHANCED FACIAL KEY GENERATION RESULTS")
    print(f"{'='*60}")
    
    for img_path in image_paths:
        try:
            keys = model.full_enhanced_pipeline(img_path)
            results[os.path.basename(img_path)] = keys
            
            print(f"\n--- KEYS FOR: {os.path.basename(img_path).upper()} ---")
            print(f"Primary Numeric Key:    {keys['primary_numeric']}")
            print(f"Secondary Numeric Key:  {keys['secondary_numeric']}")
            print(f"Slot Combined Key:      {keys['slot_combined']}")
            print(f"SHA256 (first 32):      {keys['sha256'][:32]}...")
            print(f"Statistical Entropy:    {keys['statistical']['entropy']:.6f}")
            print(f"Statistical Mean:       {keys['statistical']['mean']:.6f}")
            print(f"Statistical Std:        {keys['statistical']['std']:.6f}")
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
    
    # Save detailed results
    output_file = "enhanced_facial_keygen_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n{'='*60}")
    print(f"Detailed results saved to: {output_file}")
    print(f"{'='*60}")
    
    # Key uniqueness analysis
    if len(results) > 1:
        print(f"\nKEY UNIQUENESS ANALYSIS:")
        primary_keys = [results[img]['primary_numeric'] for img in results.keys()]
        unique_keys = len(set(primary_keys))
        print(f"Total images processed: {len(results)}")
        print(f"Unique primary keys: {unique_keys}")
        print(f"Uniqueness ratio: {unique_keys/len(results)*100:.1f}%")
    
    return results

if __name__ == "__main__":
    demo_enhanced_model()
