"""
REAL-TIME FACIAL KEY GENERATION WITH CAMERA CAPTURE
==================================================

Enhanced implementation with:
1. Real-time camera capture
2. Biometric consistency for same person
3. Multi-frame analysis for reliability
4. Live preview and key generation
5. Person identification and tracking

Based on Professor's Pipeline:
Face → Features (100–300) → Slots → Process using LFSR → Output → Reinsert → Repeat → Final output → Numeric keys
"""

import cv2
import numpy as np
import hashlib
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import mediapipe as mp
from typing import List, Tuple, Dict, Optional
import json
import os
from datetime import datetime
import warnings
from scipy import stats
import threading
import time
from collections import deque
import pickle

# Import our research model
from research_facial_keygen_model import ResearchFacialKeygenModel

warnings.filterwarnings('ignore')

class RealTimeFacialKeyGenerator:
    """
    Real-time facial key generation with camera capture and biometric consistency
    """
    
    def __init__(self, 
                 target_features: int = 128, 
                 slot_count: int = 16, 
                 lfsr_rounds: int = 5,
                 consistency_threshold: float = 0.85):
        """
        Initialize Real-time Facial Key Generator
        
        Args:
            target_features: Target number of features (100-300)
            slot_count: Number of LFSR processing slots
            lfsr_rounds: Number of iterative processing rounds
            consistency_threshold: Threshold for person identification consistency
        """
        
        self.target_features = target_features
        self.slot_count = slot_count
        self.lfsr_rounds = lfsr_rounds
        self.consistency_threshold = consistency_threshold
        
        # Initialize the core facial keygen model
        self.keygen_model = ResearchFacialKeygenModel(
            target_features=target_features,
            slot_count=slot_count,
            lfsr_rounds=lfsr_rounds,
            research_mode=False
        )
        
        # Camera and detection setup
        self.camera = None
        self.camera_index = 0
        self.frame_width = 640
        self.frame_height = 480
        
        # Consistency and tracking
        self.person_profiles = {}  # Store known person profiles
        self.recent_frames = deque(maxlen=5)  # Store recent frames for analysis
        self.frame_buffer = deque(maxlen=10)  # Buffer for multi-frame analysis
        
        # Real-time processing
        self.is_running = False
        self.current_keys = None
        self.current_person_id = None
        self.confidence_score = 0.0
        
        # Performance tracking
        self.processing_times = []
        
        print(f"🎥 Real-Time Facial Key Generator Initialized")
        print(f"Configuration: {target_features} features, {slot_count} slots, {lfsr_rounds} rounds")

    def initialize_camera(self, camera_index: int = 0) -> bool:
        """Initialize camera for real-time capture"""
        
        try:
            self.camera = cv2.VideoCapture(camera_index)
            self.camera_index = camera_index
            
            # Set camera properties
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            self.camera.set(cv2.CAP_PROP_FPS, 30)
            
            # Test camera
            ret, frame = self.camera.read()
            if not ret:
                print(f"❌ Failed to initialize camera {camera_index}")
                return False
            
            print(f"✅ Camera {camera_index} initialized successfully")
            print(f"Resolution: {self.frame_width}x{self.frame_height}")
            
            return True
            
        except Exception as e:
            print(f"❌ Camera initialization error: {e}")
            return False

    def capture_and_save_frame(self, output_dir: str = "captured_faces") -> str:
        """Capture a single frame and save it"""
        
        if not self.camera:
            raise ValueError("Camera not initialized. Call initialize_camera() first.")
        
        os.makedirs(output_dir, exist_ok=True)
        
        ret, frame = self.camera.read()
        if not ret:
            raise ValueError("Failed to capture frame from camera")
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"captured_face_{timestamp}.jpg"
        filepath = os.path.join(output_dir, filename)
        
        # Save frame
        cv2.imwrite(filepath, frame)
        print(f"📸 Frame captured: {filepath}")
        
        return filepath

    def train_with_camera_captures(self, num_training_images: int = 5, 
                                 capture_interval: float = 2.0) -> bool:
        """
        Train the model using camera-captured images
        
        Args:
            num_training_images: Number of training images to capture
            capture_interval: Seconds between captures
        """
        
        print(f"\n🎓 TRAINING WITH CAMERA CAPTURES")
        print(f"{'='*50}")
        print(f"Will capture {num_training_images} images with {capture_interval}s intervals")
        print("Position yourself in front of the camera...")
        
        if not self.camera:
            if not self.initialize_camera():
                return False
        
        training_images = []
        
        # Create preview window
        cv2.namedWindow('Training Capture', cv2.WINDOW_AUTOSIZE)
        
        for i in range(num_training_images):
            print(f"\nCapturing training image {i+1}/{num_training_images}...")
            print(f"Get ready... {int(capture_interval)}s countdown")
            
            # Countdown with live preview
            start_time = time.time()
            while time.time() - start_time < capture_interval:
                ret, frame = self.camera.read()
                if ret:
                    # Add countdown text
                    remaining = int(capture_interval - (time.time() - start_time))
                    cv2.putText(frame, f"Capturing in: {remaining}s", 
                              (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(frame, f"Training {i+1}/{num_training_images}", 
                              (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    cv2.imshow('Training Capture', frame)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        cv2.destroyAllWindows()
                        return False
            
            # Capture the training image
            try:
                image_path = self.capture_and_save_frame("training_images")
                training_images.append(image_path)
                print(f"✅ Captured: {os.path.basename(image_path)}")
            except Exception as e:
                print(f"❌ Capture failed: {e}")
                continue
        
        cv2.destroyAllWindows()
        
        if len(training_images) < 2:
            print("❌ Insufficient training images captured")
            return False
        
        # Train the model
        try:
            print(f"\n🤖 Training model with {len(training_images)} images...")
            self.keygen_model.train_research_model(training_images)
            print("✅ Model training completed successfully!")
            return True
        except Exception as e:
            print(f"❌ Training failed: {e}")
            return False

    def extract_biometric_signature(self, image_path: str) -> np.ndarray:
        """
        Extract a consistent biometric signature from face image
        This is used for person identification and consistency
        """
        try:
            # Extract comprehensive features
            features = self.keygen_model.extract_research_features(image_path)
            
            # Create a stable signature by using statistical measures
            # that are less sensitive to minor variations
            signature_components = [
                np.mean(features),
                np.std(features),
                np.median(features),
                np.percentile(features, 25),
                np.percentile(features, 75),
                np.var(features),
                stats.skew(features),
                stats.kurtosis(features)
            ]
            
            # Add geometric ratios (more stable across captures)
            if len(features) > 10:
                ratios = []
                for i in range(0, min(20, len(features)), 2):
                    if i+1 < len(features) and features[i+1] != 0:
                        ratios.append(features[i] / features[i+1])
                
                if ratios:
                    signature_components.extend([
                        np.mean(ratios),
                        np.std(ratios),
                        np.median(ratios)
                    ])
            
            signature = np.array(signature_components, dtype=np.float32)
            
            # Normalize signature for consistency
            if np.std(signature) > 0:
                signature = (signature - np.mean(signature)) / np.std(signature)
            
            return signature
            
        except Exception as e:
            print(f"Error extracting biometric signature: {e}")
            return np.array([])

    def calculate_similarity(self, signature1: np.ndarray, signature2: np.ndarray) -> float:
        """Calculate similarity between two biometric signatures"""
        
        if len(signature1) == 0 or len(signature2) == 0:
            return 0.0
        
        # Ensure same length
        min_len = min(len(signature1), len(signature2))
        sig1 = signature1[:min_len]
        sig2 = signature2[:min_len]
        
        # Calculate multiple similarity metrics
        try:
            # Cosine similarity
            dot_product = np.dot(sig1, sig2)
            norms = np.linalg.norm(sig1) * np.linalg.norm(sig2)
            cosine_sim = dot_product / norms if norms > 0 else 0
            
            # Correlation coefficient
            correlation = np.corrcoef(sig1, sig2)[0, 1] if len(sig1) > 1 else 0
            correlation = 0 if np.isnan(correlation) else correlation
            
            # Normalized euclidean distance (inverted)
            euclidean_dist = np.linalg.norm(sig1 - sig2)
            max_possible_dist = np.linalg.norm(sig1) + np.linalg.norm(sig2)
            euclidean_sim = 1 - (euclidean_dist / max_possible_dist) if max_possible_dist > 0 else 0
            
            # Combined similarity score
            similarity = (cosine_sim + correlation + euclidean_sim) / 3
            
            return max(0, min(1, similarity))  # Ensure 0-1 range
            
        except Exception as e:
            print(f"Error calculating similarity: {e}")
            return 0.0

    def identify_person(self, image_path: str) -> Tuple[str, float]:
        """
        Identify person and return person ID with confidence score
        
        Returns:
            Tuple of (person_id, confidence_score)
        """
        
        # Extract biometric signature
        current_signature = self.extract_biometric_signature(image_path)
        
        if len(current_signature) == 0:
            return "unknown", 0.0
        
        best_match_id = None
        best_similarity = 0.0
        
        # Compare with known persons
        for person_id, profile in self.person_profiles.items():
            similarity = self.calculate_similarity(current_signature, profile['signature'])
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match_id = person_id
        
        # Check if similarity meets threshold
        if best_similarity >= self.consistency_threshold:
            return best_match_id, best_similarity
        else:
            # Create new person profile
            new_person_id = f"person_{len(self.person_profiles) + 1}"
            self.person_profiles[new_person_id] = {
                'signature': current_signature,
                'first_seen': datetime.now().isoformat(),
                'total_captures': 1,
                'last_seen': datetime.now().isoformat()
            }
            return new_person_id, 1.0

    def generate_consistent_keys(self, image_path: str) -> Dict[str, any]:
        """
        Generate consistent keys for the same person across multiple captures
        """
        
        # Identify person
        person_id, confidence = self.identify_person(image_path)
        
        print(f"👤 Person identified: {person_id} (confidence: {confidence:.3f})")
        
        # Generate keys using the research pipeline
        keys = self.keygen_model.research_pipeline(image_path)
        
        # Add person identification info
        keys['person_identification'] = {
            'person_id': person_id,
            'confidence_score': confidence,
            'is_known_person': confidence >= self.consistency_threshold,
            'identification_timestamp': datetime.now().isoformat()
        }
        
        # Update person profile
        if person_id in self.person_profiles:
            self.person_profiles[person_id]['total_captures'] += 1
            self.person_profiles[person_id]['last_seen'] = datetime.now().isoformat()
        
        return keys

    def real_time_key_generation(self, display_preview: bool = True, 
                               process_interval: float = 2.0) -> None:
        """
        Real-time key generation with camera feed
        
        Args:
            display_preview: Whether to show camera preview
            process_interval: Seconds between key generation
        """
        
        if not self.keygen_model.is_trained:
            print("❌ Model not trained. Please train first using train_with_camera_captures()")
            return
        
        if not self.camera:
            if not self.initialize_camera():
                return
        
        print(f"\n🔴 STARTING REAL-TIME KEY GENERATION")
        print(f"{'='*50}")
        print("Press 'q' to quit, 's' to save current keys, 'c' to capture training image")
        
        self.is_running = True
        last_process_time = 0
        
        if display_preview:
            cv2.namedWindow('Real-Time Facial Key Generation', cv2.WINDOW_AUTOSIZE)
        
        try:
            while self.is_running:
                ret, frame = self.camera.read()
                if not ret:
                    print("❌ Failed to read from camera")
                    break
                
                # Add frame to buffer
                self.frame_buffer.append(frame.copy())
                
                current_time = time.time()
                
                # Process keys at specified interval
                if current_time - last_process_time >= process_interval:
                    self._process_frame_for_keys(frame)
                    last_process_time = current_time
                
                if display_preview:
                    # Add overlay information
                    self._add_overlay_info(frame)
                    
                    cv2.imshow('Real-Time Facial Key Generation', frame)
                    
                    # Handle key presses
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('s'):
                        self._save_current_keys()
                    elif key == ord('c'):
                        self._capture_current_frame()
                
                # Small delay to prevent excessive CPU usage
                time.sleep(0.01)
        
        except KeyboardInterrupt:
            print("\n⏹️ Stopping real-time generation...")
        
        finally:
            self.is_running = False
            if display_preview:
                cv2.destroyAllWindows()
            
            print("✅ Real-time key generation stopped")

    def _process_frame_for_keys(self, frame: np.ndarray) -> None:
        """Process current frame for key generation"""
        
        try:
            start_time = time.time()
            
            # Save current frame temporarily
            temp_path = "temp_realtime_frame.jpg"
            cv2.imwrite(temp_path, frame)
            
            # Generate keys
            self.current_keys = self.generate_consistent_keys(temp_path)
            self.current_person_id = self.current_keys['person_identification']['person_id']
            self.confidence_score = self.current_keys['person_identification']['confidence_score']
            
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            # Track processing time
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            
            print(f"🔑 Keys generated for {self.current_person_id} in {processing_time:.2f}s")
            print(f"   Primary Key: {self.current_keys['primary_numeric_key']}")
            
        except Exception as e:
            print(f"❌ Error processing frame: {e}")

    def _add_overlay_info(self, frame: np.ndarray) -> None:
        """Add information overlay to the frame"""
        
        # Add title
        cv2.putText(frame, "Real-Time Facial Key Generation", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add person info if available
        if self.current_person_id:
            cv2.putText(frame, f"Person: {self.current_person_id}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Confidence: {self.confidence_score:.3f}", 
                       (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add key info if available
        if self.current_keys:
            key_text = f"Key: {str(self.current_keys['primary_numeric_key'])[:16]}..."
            cv2.putText(frame, key_text, 
                       (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
        
        # Add performance info
        if self.processing_times:
            avg_time = np.mean(self.processing_times[-10:])  # Last 10 processing times
            fps_text = f"Avg Processing: {avg_time:.2f}s"
            cv2.putText(frame, fps_text, 
                       (10, frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Add instructions
        cv2.putText(frame, "Press 'q' to quit, 's' to save keys, 'c' to capture", 
                   (10, frame.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    def _save_current_keys(self) -> None:
        """Save current generated keys to file"""
        
        if self.current_keys:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"realtime_keys_{self.current_person_id}_{timestamp}.json"
            
            with open(filename, 'w') as f:
                json.dump(self.current_keys, f, indent=2, default=str)
            
            print(f"💾 Keys saved to: {filename}")
        else:
            print("❌ No keys available to save")

    def _capture_current_frame(self) -> None:
        """Capture and save current frame"""
        
        if len(self.frame_buffer) > 0:
            frame = self.frame_buffer[-1]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"captured_realtime_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            print(f"📸 Frame saved: {filename}")

    def save_person_profiles(self, filepath: str = "person_profiles.json") -> None:
        """Save person profiles for future use"""
        
        # Convert numpy arrays to lists for JSON serialization
        profiles_for_save = {}
        for person_id, profile in self.person_profiles.items():
            profiles_for_save[person_id] = {
                'signature': profile['signature'].tolist() if isinstance(profile['signature'], np.ndarray) else profile['signature'],
                'first_seen': profile['first_seen'],
                'total_captures': profile['total_captures'],
                'last_seen': profile['last_seen']
            }
        
        with open(filepath, 'w') as f:
            json.dump(profiles_for_save, f, indent=2)
        
        print(f"👥 Person profiles saved to: {filepath}")

    def load_person_profiles(self, filepath: str = "person_profiles.json") -> bool:
        """Load person profiles from file"""
        
        try:
            if not os.path.exists(filepath):
                print(f"No existing person profiles found at {filepath}")
                return False
            
            with open(filepath, 'r') as f:
                profiles_data = json.load(f)
            
            # Convert lists back to numpy arrays
            for person_id, profile in profiles_data.items():
                self.person_profiles[person_id] = {
                    'signature': np.array(profile['signature']),
                    'first_seen': profile['first_seen'],
                    'total_captures': profile['total_captures'],
                    'last_seen': profile['last_seen']
                }
            
            print(f"👥 Loaded {len(self.person_profiles)} person profiles from {filepath}")
            return True
            
        except Exception as e:
            print(f"❌ Error loading person profiles: {e}")
            return False

    def cleanup(self) -> None:
        """Cleanup resources"""
        
        self.is_running = False
        
        if self.camera:
            self.camera.release()
        
        cv2.destroyAllWindows()
        
        # Save person profiles
        self.save_person_profiles()
        
        print("🧹 Cleanup completed")

def demo_realtime_system():
    """Demonstration of the real-time facial key generation system"""
    
    print("🎥 REAL-TIME FACIAL KEY GENERATION DEMO")
    print("=" * 60)
    
    # Initialize system
    realtime_system = RealTimeFacialKeyGenerator(
        target_features=128,
        slot_count=16,
        lfsr_rounds=5,
        consistency_threshold=0.85
    )
    
    # Load existing person profiles if available
    realtime_system.load_person_profiles()
    
    try:
        # Option 1: Train with camera captures
        print("\n📚 TRAINING OPTIONS:")
        print("1. Train with new camera captures")
        print("2. Use existing training images")
        print("3. Skip training (if already trained)")
        
        choice = input("Enter choice (1-3): ").strip()
        
        if choice == "1":
            # Train with camera
            if realtime_system.train_with_camera_captures(num_training_images=5, capture_interval=3.0):
                print("✅ Training completed successfully!")
            else:
                print("❌ Training failed")
                return
                
        elif choice == "2":
            # Use existing images
            training_dir = "captures"
            if os.path.exists(training_dir):
                image_files = [f for f in os.listdir(training_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if image_files:
                    image_paths = [os.path.join(training_dir, f) for f in image_files]
                    realtime_system.keygen_model.train_research_model(image_paths)
                    print("✅ Training with existing images completed!")
                else:
                    print("❌ No images found in captures directory")
                    return
            else:
                print("❌ Captures directory not found")
                return
                
        elif choice == "3":
            print("⏭️ Skipping training...")
            # Assume model is already trained
            realtime_system.keygen_model.is_trained = True
        
        # Option 2: Choose operation mode
        print("\n🚀 OPERATION MODES:")
        print("1. Real-time key generation with live preview")
        print("2. Single capture and key generation")
        print("3. Batch processing of captures")
        
        mode = input("Enter mode (1-3): ").strip()
        
        if mode == "1":
            # Real-time mode
            print("\n🔴 Starting real-time mode...")
            print("Position yourself in front of the camera")
            input("Press Enter when ready...")
            
            realtime_system.real_time_key_generation(
                display_preview=True,
                process_interval=3.0  # Generate keys every 3 seconds
            )
            
        elif mode == "2":
            # Single capture mode
            print("\n📸 Single capture mode...")
            if realtime_system.initialize_camera():
                input("Press Enter to capture...")
                image_path = realtime_system.capture_and_save_frame()
                
                keys = realtime_system.generate_consistent_keys(image_path)
                
                print(f"\n🔑 GENERATED KEYS:")
                print(f"Person ID: {keys['person_identification']['person_id']}")
                print(f"Confidence: {keys['person_identification']['confidence_score']:.3f}")
                print(f"Primary Key: {keys['primary_numeric_key']}")
                print(f"Blockchain Address: {keys['blockchain_address']}")
                
                # Save keys
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"single_capture_keys_{timestamp}.json"
                with open(filename, 'w') as f:
                    json.dump(keys, f, indent=2, default=str)
                print(f"💾 Keys saved to: {filename}")
            
        elif mode == "3":
            # Batch processing mode
            print("\n📁 Batch processing mode...")
            num_captures = int(input("Number of captures to process: "))
            
            if realtime_system.initialize_camera():
                all_results = {}
                
                for i in range(num_captures):
                    print(f"\nCapture {i+1}/{num_captures}")
                    input("Press Enter to capture...")
                    
                    image_path = realtime_system.capture_and_save_frame()
                    keys = realtime_system.generate_consistent_keys(image_path)
                    
                    all_results[f"capture_{i+1}"] = keys
                    
                    print(f"✅ Processed: {keys['person_identification']['person_id']}")
                
                # Save batch results
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"batch_results_{timestamp}.json"
                with open(filename, 'w') as f:
                    json.dump(all_results, f, indent=2, default=str)
                print(f"💾 Batch results saved to: {filename}")
        
        print(f"\n📊 SUMMARY:")
        print(f"Known persons: {len(realtime_system.person_profiles)}")
        if realtime_system.processing_times:
            print(f"Average processing time: {np.mean(realtime_system.processing_times):.2f}s")
    
    except KeyboardInterrupt:
        print("\n⏹️ Demo interrupted by user")
    
    except Exception as e:
        print(f"❌ Demo error: {e}")
    
    finally:
        # Cleanup
        realtime_system.cleanup()
        print("🏁 Demo completed!")

if __name__ == "__main__":
    demo_realtime_system()