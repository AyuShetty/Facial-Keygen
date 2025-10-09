"""
Advanced Facial Key Generation Model
Based on Professor's Instructions:
Face → Features (100–300) → Slots → Process using LFSR → Output → Reinsert → Repeat → Final output → Numeric keys

This implementation uses a multi-stage LFSR processing approach with feature slot management
for enhanced cryptographic key generation from facial biometrics.
"""

import cv2
import numpy as np
import hashlib
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import mediapipe as mp
from typing import List, Tuple, Dict
import json
import os
from datetime import datetime

class FacialKeygenModel:
    def __init__(self, feature_count: int = 200, slot_count: int = 16, lfsr_rounds: int = 5):
        """
        Initialize the Advanced Facial Keygen Model
        
        Args:
            feature_count: Number of facial features to extract (100-300)
            slot_count: Number of processing slots for LFSR
            lfsr_rounds: Number of LFSR processing rounds
        """
        self.feature_count = min(max(feature_count, 100), 300)  # Ensure 100-300 range
        self.slot_count = slot_count
        self.lfsr_rounds = lfsr_rounds
        
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # LFSR Configuration - Different tap configurations for each slot
        self.lfsr_taps = self._generate_lfsr_taps()
        
        # Feature processing components
        self.scaler = StandardScaler()
        self.pca = None
        self.is_trained = False
        
        print(f"Initialized Facial Keygen Model:")
        print(f"- Feature Count: {self.feature_count}")
        print(f"- Slot Count: {self.slot_count}")
        print(f"- LFSR Rounds: {self.lfsr_rounds}")

    def _generate_lfsr_taps(self) -> List[List[int]]:
        """Generate different LFSR tap configurations for each slot"""
        # Primitive polynomials for different LFSR lengths
        base_taps = [
            [7, 6],      # 8-bit
            [8, 6, 1, 0], # 9-bit  
            [9, 7],      # 10-bit
            [10, 8, 3, 2], # 11-bit
            [11, 10, 8, 5], # 12-bit
            [12, 11, 10, 4], # 13-bit
            [13, 12, 11, 8], # 14-bit
            [14, 13, 12, 2], # 15-bit
        ]
        
        taps = []
        for i in range(self.slot_count):
            # Cycle through different tap configurations
            base_tap = base_taps[i % len(base_taps)]
            # Scale taps based on feature count per slot
            slot_size = self.feature_count // self.slot_count
            scaled_taps = [(tap * slot_size) // max(base_tap) for tap in base_tap]
            taps.append(scaled_taps)
        
        return taps

    def extract_facial_features(self, image_path: str) -> np.ndarray:
        """
        Extract comprehensive facial features from image
        
        Args:
            image_path: Path to the facial image
            
        Returns:
            Extracted feature vector
        """
        # Load and preprocess image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert BGR to RGB for MediaPipe
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe Face Mesh
        results = self.face_mesh.process(rgb_image)
        
        if not results.multi_face_landmarks:
            raise ValueError("No face detected in image")
        
        # Extract 3D landmark coordinates
        face_landmarks = results.multi_face_landmarks[0]
        
        # Convert to numpy array
        landmarks = []
        for landmark in face_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])
        
        feature_vector = np.array(landmarks, dtype=np.float32)
        
        # Add additional geometric features
        geometric_features = self._compute_geometric_features(face_landmarks, image.shape)
        feature_vector = np.concatenate([feature_vector, geometric_features])
        
        print(f"Extracted {len(feature_vector)} raw features from face")
        return feature_vector

    def _compute_geometric_features(self, landmarks, image_shape) -> np.ndarray:
        """Compute additional geometric features from face landmarks"""
        h, w = image_shape[:2]
        
        # Convert landmarks to pixel coordinates
        points = []
        for landmark in landmarks.landmark:
            points.append([landmark.x * w, landmark.y * h])
        points = np.array(points)
        
        # Compute geometric features
        features = []
        
        # Eye aspect ratios
        left_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        right_eye_indices = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        
        if len(points) > max(max(left_eye_indices), max(right_eye_indices)):
            left_eye_points = points[left_eye_indices]
            right_eye_points = points[right_eye_indices]
            
            # Eye aspect ratios
            left_ear = self._eye_aspect_ratio(left_eye_points[:6])  # Use first 6 points
            right_ear = self._eye_aspect_ratio(right_eye_points[:6])
            features.extend([left_ear, right_ear])
        
        # Face symmetry measures
        center_x = w // 2
        left_side = points[points[:, 0] < center_x]
        right_side = points[points[:, 0] >= center_x]
        
        if len(left_side) > 0 and len(right_side) > 0:
            symmetry = np.abs(np.mean(left_side[:, 0]) - (w - np.mean(right_side[:, 0])))
            features.append(symmetry / w)  # Normalize
        
        # Bounding box ratios
        if len(points) > 0:
            bbox_width = np.max(points[:, 0]) - np.min(points[:, 0])
            bbox_height = np.max(points[:, 1]) - np.min(points[:, 1])
            aspect_ratio = bbox_width / bbox_height if bbox_height > 0 else 1.0
            features.append(aspect_ratio)
        
        return np.array(features, dtype=np.float32)

    def _eye_aspect_ratio(self, eye_points) -> float:
        """Calculate eye aspect ratio from eye landmark points"""
        if len(eye_points) < 6:
            return 0.0
        
        # Vertical eye landmarks
        A = np.linalg.norm(eye_points[1] - eye_points[5])
        B = np.linalg.norm(eye_points[2] - eye_points[4])
        
        # Horizontal eye landmark
        C = np.linalg.norm(eye_points[0] - eye_points[3])
        
        # Eye aspect ratio
        ear = (A + B) / (2.0 * C) if C > 0 else 0.0
        return ear

    def train_feature_processor(self, image_paths: List[str]):
        """
        Train the feature processing pipeline (PCA + Scaler)
        
        Args:
            image_paths: List of training image paths
        """
        print(f"Training feature processor on {len(image_paths)} images...")
        
        # Extract features from all training images
        all_features = []
        for img_path in image_paths:
            try:
                features = self.extract_facial_features(img_path)
                all_features.append(features)
            except Exception as e:
                print(f"Warning: Could not process {img_path}: {e}")
                continue
        
        if not all_features:
            raise ValueError("No valid features extracted from training images")
        
        # Stack all features
        feature_matrix = np.vstack(all_features)
        print(f"Training on feature matrix shape: {feature_matrix.shape}")
        
        # Fit scaler
        self.scaler.fit(feature_matrix)
        
        # Apply scaling
        scaled_features = self.scaler.transform(feature_matrix)
        
        # Fit PCA to reduce to target feature count
        # Adjust n_components based on available data
        max_components = min(scaled_features.shape[0], scaled_features.shape[1])
        actual_components = min(self.feature_count, max_components - 1)
        
        if actual_components < self.feature_count:
            print(f"Warning: Reducing feature count from {self.feature_count} to {actual_components} due to limited training data")
            self.feature_count = actual_components
        
        self.pca = PCA(n_components=self.feature_count)
        self.pca.fit(scaled_features)
        
        self.is_trained = True
        print(f"Feature processor trained. Reduced to {self.feature_count} features")
        print(f"Explained variance ratio: {np.sum(self.pca.explained_variance_ratio_):.4f}")

    def process_features_to_slots(self, features: np.ndarray) -> List[np.ndarray]:
        """
        Process features and distribute them into slots
        
        Args:
            features: Processed feature vector
            
        Returns:
            List of feature slots
        """
        if not self.is_trained:
            raise ValueError("Feature processor not trained. Call train_feature_processor() first.")
        
        # Apply scaling and PCA
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        features_reduced = self.pca.transform(features_scaled).flatten()
        
        # Ensure we have exactly the target number of features
        if len(features_reduced) != self.feature_count:
            # Pad or truncate as needed
            if len(features_reduced) < self.feature_count:
                padding = np.zeros(self.feature_count - len(features_reduced))
                features_reduced = np.concatenate([features_reduced, padding])
            else:
                features_reduced = features_reduced[:self.feature_count]
        
        # Distribute features into slots
        # Adjust slot count if we have fewer features than slots
        actual_slot_count = min(self.slot_count, self.feature_count)
        slot_size = self.feature_count // actual_slot_count
        
        if slot_size == 0:  # If we have more slots than features
            # Each slot gets at most one feature, duplicate features as needed
            slots = []
            for i in range(actual_slot_count):
                feature_idx = i % self.feature_count
                slots.append(np.array([features_reduced[feature_idx]]))
        else:
            slots = []
            for i in range(actual_slot_count):
                start_idx = i * slot_size
                if i == actual_slot_count - 1:  # Last slot gets remaining features
                    end_idx = self.feature_count
                else:
                    end_idx = (i + 1) * slot_size
                
                slot_features = features_reduced[start_idx:end_idx]
                # Ensure each slot has at least one feature
                if len(slot_features) == 0:
                    slot_features = np.array([features_reduced[-1]])  # Use last feature
                slots.append(slot_features)
        
        print(f"Distributed {self.feature_count} features into {len(slots)} slots (requested {self.slot_count})")
        return slots

    def lfsr_process_slot(self, slot_data: np.ndarray, taps: List[int], rounds: int = 1) -> np.ndarray:
        """
        Process a single slot using LFSR
        
        Args:
            slot_data: Input slot data
            taps: LFSR tap positions
            rounds: Number of processing rounds
            
        Returns:
            Processed slot data
        """
        # Convert to integer representation for LFSR
        # Scale and convert to unsigned integers
        if len(slot_data) == 0:
            return np.array([])
        
        slot_min, slot_max = np.min(slot_data), np.max(slot_data)
        if slot_max > slot_min:
            normalized = (slot_data - slot_min) / (slot_max - slot_min)
        else:
            normalized = np.ones_like(slot_data) * 0.5
        
        # Convert to 32-bit integers
        state = (normalized * 0xFFFFFFFF).astype(np.uint32)
        
        # Ensure we have valid taps for the slot size
        valid_taps = [tap for tap in taps if tap < len(state)]
        if not valid_taps:
            valid_taps = [0]  # Default to first position if no valid taps
        
        # LFSR processing
        for round_num in range(rounds):
            if len(state) <= 1:
                # For single element, just apply XOR with itself
                state = np.array([state[0] ^ 0xAAAAAAAA], dtype=np.uint32)
            else:
                new_state = state.copy()
                
                for i in range(len(state)):
                    feedback = 0
                    for tap in valid_taps:
                        feedback ^= state[tap]
                    
                    # Shift and insert feedback
                    new_state = np.roll(new_state, 1)
                    new_state[0] = feedback
                
                state = new_state
        
        # Convert back to float representation
        processed_data = state.astype(np.float32) / 0xFFFFFFFF
        
        return processed_data

    def generate_numeric_keys(self, slots: List[np.ndarray]) -> Dict[str, any]:
        """
        Generate final numeric keys from processed slots
        
        Args:
            slots: List of processed slot data
            
        Returns:
            Dictionary containing various numeric key formats
        """
        # Concatenate all slot data
        combined_data = np.concatenate(slots)
        
        # Generate multiple key formats
        keys = {}
        
        # 1. Direct hash-based key
        data_bytes = combined_data.tobytes()
        sha256_hash = hashlib.sha256(data_bytes).hexdigest()
        keys['sha256_hex'] = sha256_hash
        keys['sha256_int'] = int(sha256_hash, 16)
        
        # 2. Statistical-based keys
        keys['mean_key'] = float(np.mean(combined_data))
        keys['std_key'] = float(np.std(combined_data))
        keys['entropy_key'] = float(-np.sum(combined_data * np.log2(combined_data + 1e-10)))
        
        # 3. Quantized keys
        quantized = np.round(combined_data * 255).astype(np.uint8)
        keys['quantized_sum'] = int(np.sum(quantized))
        keys['quantized_xor'] = int(np.bitwise_xor.reduce(quantized))
        
        # 4. Slot-based keys
        slot_keys = []
        for i, slot in enumerate(slots):
            slot_hash = hashlib.md5(slot.tobytes()).hexdigest()
            slot_keys.append(int(slot_hash[:8], 16))  # First 8 hex chars as int
        
        keys['slot_keys'] = slot_keys
        keys['slot_combined'] = sum(slot_keys) % (2**32)  # 32-bit combined key
        
        return keys

    def full_pipeline(self, image_path: str) -> Dict[str, any]:
        """
        Execute the complete pipeline: Face → Features → Slots → LFSR → Keys
        
        Args:
            image_path: Path to input facial image
            
        Returns:
            Generated numeric keys
        """
        print(f"\n=== Starting Full Pipeline for: {image_path} ===")
        
        # Step 1: Extract facial features
        print("Step 1: Extracting facial features...")
        raw_features = self.extract_facial_features(image_path)
        
        # Step 2: Process features and create slots
        print("Step 2: Processing features into slots...")
        slots = self.process_features_to_slots(raw_features)
        
        # Step 3: Process each slot with LFSR
        print("Step 3: Processing slots with LFSR...")
        processed_slots = []
        
        for round_num in range(self.lfsr_rounds):
            print(f"  LFSR Round {round_num + 1}/{self.lfsr_rounds}")
            current_slots = slots if round_num == 0 else processed_slots
            new_slots = []
            
            for i, slot in enumerate(current_slots):
                taps = self.lfsr_taps[i % len(self.lfsr_taps)]
                processed_slot = self.lfsr_process_slot(slot, taps, rounds=1)
                new_slots.append(processed_slot)
            
            processed_slots = new_slots
            
            # Reinsert processed data back into the pipeline for next round
            if round_num < self.lfsr_rounds - 1:
                print("  Reinserting processed data for next round...")
        
        # Step 4: Generate final numeric keys
        print("Step 4: Generating final numeric keys...")
        final_keys = self.generate_numeric_keys(processed_slots)
        
        # Add metadata
        final_keys['metadata'] = {
            'image_path': image_path,
            'feature_count': self.feature_count,
            'slot_count': self.slot_count,
            'lfsr_rounds': self.lfsr_rounds,
            'timestamp': datetime.now().isoformat()
        }
        
        print("=== Pipeline Complete ===\n")
        return final_keys

def demo_pipeline():
    """Demonstration of the new facial keygen model"""
    
    # Initialize model
    model = FacialKeygenModel(feature_count=200, slot_count=16, lfsr_rounds=3)
    
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
    print(f"Found images: {image_files}")
    
    # Train the feature processor
    try:
        model.train_feature_processor(image_paths)
    except Exception as e:
        print(f"Error during training: {e}")
        return
    
    # Process each image
    results = {}
    for img_path in image_paths:
        try:
            keys = model.full_pipeline(img_path)
            results[os.path.basename(img_path)] = keys
            
            print(f"\n--- Keys for {os.path.basename(img_path)} ---")
            print(f"SHA256 (first 16 chars): {keys['sha256_hex'][:16]}...")
            print(f"Statistical keys - Mean: {keys['mean_key']:.6f}, Std: {keys['std_key']:.6f}")
            print(f"Quantized sum: {keys['quantized_sum']}")
            print(f"Slot combined key: {keys['slot_combined']}")
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue
    
    # Save results
    output_file = "facial_keygen_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to: {output_file}")
    
    return results

if __name__ == "__main__":
    demo_pipeline()
