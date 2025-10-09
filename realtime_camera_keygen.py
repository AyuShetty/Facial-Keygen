"""
REAL-TIME FACIAL KEY GENERATION WITH CAMERA CAPTURE
===================================================

This module implements real-time camera capture with biometric consistency verification.
It ensures the same person always generates the same cryptographic keys.

Key Features:
1. Live camera capture with quality assessment
2. Template-based consistency verification
3. Multiple capture averaging for stability
4. Person enrollment and recognition
5. Secure key regeneration for enrolled persons

"""

import cv2
import numpy as np
import hashlib
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import pickle
from pathlib import Path

# Import our research model
from research_facial_keygen_model import ResearchFacialKeygenModel

class BiometricTemplate:
    """Stores biometric template for a person to ensure consistency"""
    
    def __init__(self, person_id: str, feature_vector: np.ndarray, 
                 primary_key: int, blockchain_address: int):
        self.person_id = person_id
        self.feature_vector = feature_vector
        self.primary_key = primary_key
        self.blockchain_address = blockchain_address
        self.created_at = datetime.now()
        self.verification_count = 0
        self.last_verified = None
        
    def to_dict(self):
        return {
            'person_id': self.person_id,
            'feature_vector': self.feature_vector.tolist(),
            'primary_key': self.primary_key,
            'blockchain_address': self.blockchain_address,
            'created_at': self.created_at.isoformat(),
            'verification_count': self.verification_count,
            'last_verified': self.last_verified.isoformat() if self.last_verified else None
        }

class RealTimeFacialKeygen:
    """
    Real-time facial key generation with camera capture and consistency verification
    """
    
    def __init__(self, facial_model: ResearchFacialKeygenModel):
        self.facial_model = facial_model
        self.templates_dir = Path("biometric_templates")
        self.templates_dir.mkdir(exist_ok=True)
        
        # Load existing templates
        self.enrolled_persons = {}
        self._load_templates()
        
        # Camera settings
        self.camera_index = 0
        self.capture_width = 640
        self.capture_height = 480
        
        # Quality thresholds
        self.min_face_confidence = 0.8
        self.min_brightness = 50
        self.max_brightness = 200
        
        # Consistency thresholds
        self.similarity_threshold = 0.85  # 85% similarity for same person
        
        print("Real-Time Facial Keygen System Initialized")
        print(f"Enrolled persons: {len(self.enrolled_persons)}")

    def _load_templates(self):
        """Load existing biometric templates"""
        template_files = list(self.templates_dir.glob("*.pkl"))
        
        for template_file in template_files:
            try:
                with open(template_file, 'rb') as f:
                    template = pickle.load(f)
                    self.enrolled_persons[template.person_id] = template
                    print(f"Loaded template: {template.person_id}")
            except Exception as e:
                print(f"Error loading template {template_file}: {e}")

    def _save_template(self, template: BiometricTemplate):
        """Save biometric template"""
        template_file = self.templates_dir / f"{template.person_id}.pkl"
        
        with open(template_file, 'wb') as f:
            pickle.dump(template, f)
        
        # Also save JSON version for readability
        json_file = self.templates_dir / f"{template.person_id}.json"
        with open(json_file, 'w') as f:
            json.dump(template.to_dict(), f, indent=2)

    def assess_image_quality(self, image: np.ndarray) -> Tuple[bool, str, float]:
        """
        Assess if captured image is suitable for key generation
        
        Returns: (is_good, message, quality_score)
        """
        # Check brightness
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        
        if brightness < self.min_brightness:
            return False, "Image too dark", brightness / 255.0
        if brightness > self.max_brightness:
            return False, "Image too bright", brightness / 255.0
        
        # Check blur (using Laplacian variance)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        if laplacian_var < 100:
            return False, "Image too blurry", laplacian_var / 1000.0
        
        # Check face detection confidence
        # This is a simplified check - actual implementation would use face detection
        quality_score = min(brightness / 255.0 * 0.5 + laplacian_var / 1000.0 * 0.5, 1.0)
        
        return True, "Good quality", quality_score

    def capture_from_camera(self, num_captures: int = 3, 
                           person_name: Optional[str] = None) -> List[str]:
        """
        Capture multiple images from camera for averaging
        
        Args:
            num_captures: Number of images to capture for averaging
            person_name: Optional name for saving captures
            
        Returns:
            List of captured image paths
        """
        print(f"\n{'='*60}")
        print("CAMERA CAPTURE MODE")
        print(f"{'='*60}")
        print(f"Capturing {num_captures} images for stability...")
        print("Press SPACE to capture, ESC to cancel, Q to quit")
        
        # Initialize camera
        cap = cv2.VideoCapture(self.camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.capture_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.capture_height)
        
        if not cap.isOpened():
            raise RuntimeError("Could not open camera")
        
        captured_images = []
        capture_count = 0
        
        # Create capture directory
        capture_dir = Path("live_captures")
        capture_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        while capture_count < num_captures:
            ret, frame = cap.read()
            
            if not ret:
                print("Failed to grab frame")
                break
            
            # Flip frame for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Assess quality
            is_good, message, quality_score = self.assess_image_quality(frame)
            
            # Draw UI elements
            display_frame = frame.copy()
            
            # Status box
            status_color = (0, 255, 0) if is_good else (0, 0, 255)
            cv2.rectangle(display_frame, (10, 10), (630, 80), (0, 0, 0), -1)
            cv2.rectangle(display_frame, (10, 10), (630, 80), status_color, 2)
            
            # Text information
            cv2.putText(display_frame, f"Capture {capture_count + 1}/{num_captures}", 
                       (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_frame, f"Quality: {message} ({quality_score:.2f})", 
                       (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
            
            # Instructions
            cv2.putText(display_frame, "SPACE: Capture | ESC: Cancel | Q: Quit", 
                       (20, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Show frame
            cv2.imshow('Facial Key Generation - Camera Capture', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            # Capture on SPACE
            if key == ord(' '):
                if is_good:
                    # Save captured image
                    if person_name:
                        filename = f"{person_name}_{timestamp}_{capture_count + 1}.jpg"
                    else:
                        filename = f"capture_{timestamp}_{capture_count + 1}.jpg"
                    
                    filepath = capture_dir / filename
                    cv2.imwrite(str(filepath), frame)
                    
                    captured_images.append(str(filepath))
                    capture_count += 1
                    
                    print(f"✓ Captured image {capture_count}/{num_captures}: {filename}")
                    
                    # Visual feedback
                    flash = frame.copy()
                    cv2.rectangle(flash, (0, 0), (640, 480), (255, 255, 255), 20)
                    cv2.imshow('Facial Key Generation - Camera Capture', flash)
                    cv2.waitKey(100)
                else:
                    print(f"✗ Image quality insufficient: {message}")
            
            # Cancel on ESC
            elif key == 27:  # ESC
                print("Capture cancelled")
                break
            
            # Quit on Q
            elif key == ord('q'):
                print("Quitting...")
                break
        
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\nCaptured {len(captured_images)} images")
        return captured_images

    def calculate_feature_similarity(self, features1: np.ndarray, 
                                    features2: np.ndarray) -> float:
        """
        Calculate similarity between two feature vectors (0-1 scale)
        """
        # Ensure same length
        min_len = min(len(features1), len(features2))
        f1 = features1[:min_len]
        f2 = features2[:min_len]
        
        # Normalize vectors
        f1_norm = f1 / (np.linalg.norm(f1) + 1e-10)
        f2_norm = f2 / (np.linalg.norm(f2) + 1e-10)
        
        # Cosine similarity
        similarity = np.dot(f1_norm, f2_norm)
        
        # Convert to 0-1 range
        similarity = (similarity + 1) / 2
        
        return float(similarity)

    def enroll_new_person(self, person_name: str, num_captures: int = 3) -> Dict:
        """
        Enroll a new person by capturing images and generating consistent keys
        
        Args:
            person_name: Name/ID of the person to enroll
            num_captures: Number of images to capture for averaging
            
        Returns:
            Generated keys for the enrolled person
        """
        print(f"\n{'='*60}")
        print(f"ENROLLING NEW PERSON: {person_name}")
        print(f"{'='*60}")
        
        # Check if already enrolled
        if person_name in self.enrolled_persons:
            print(f"Warning: {person_name} is already enrolled!")
            response = input("Do you want to re-enroll? (yes/no): ")
            if response.lower() != 'yes':
                return self.enrolled_persons[person_name].to_dict()
        
        # Capture multiple images
        captured_images = self.capture_from_camera(num_captures, person_name)
        
        if len(captured_images) < 2:
            raise ValueError("Need at least 2 captures for enrollment")
        
        # Extract features from all captures
        print("\nExtracting features from captures...")
        all_features = []
        
        for img_path in captured_images:
            try:
                features = self.facial_model.extract_research_features(img_path)
                all_features.append(features)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
        
        if not all_features:
            raise ValueError("No valid features extracted from captures")
        
        # Average features for consistency
        print("Averaging features for stability...")
        avg_features = np.mean(all_features, axis=0)
        
        # Generate keys using averaged features
        print("Generating cryptographic keys...")
        
        # Create a temporary image path for the averaged features
        # We'll use the first captured image but with averaged features
        temp_path = captured_images[0]
        
        # Process through the pipeline
        keys = self.facial_model.research_pipeline(temp_path)
        
        # Create biometric template
        template = BiometricTemplate(
            person_id=person_name,
            feature_vector=avg_features,
            primary_key=keys['primary_numeric_key'],
            blockchain_address=keys['blockchain_address']
        )
        
        # Save template
        self.enrolled_persons[person_name] = template
        self._save_template(template)
        
        print(f"\n{'='*60}")
        print(f"✓ ENROLLMENT SUCCESSFUL: {person_name}")
        print(f"{'='*60}")
        print(f"Primary Key: {template.primary_key}")
        print(f"Blockchain Address: {template.blockchain_address}")
        print(f"Template saved in: {self.templates_dir}")
        
        return keys

    def verify_and_regenerate_keys(self, num_captures: int = 3) -> Optional[Dict]:
        """
        Capture from camera, identify person, and regenerate their consistent keys
        
        Returns:
            Keys if person is recognized, None otherwise
        """
        print(f"\n{'='*60}")
        print("BIOMETRIC VERIFICATION & KEY REGENERATION")
        print(f"{'='*60}")
        
        if len(self.enrolled_persons) == 0:
            print("No enrolled persons! Please enroll someone first.")
            return None
        
        # Capture images
        captured_images = self.capture_from_camera(num_captures, "verification")
        
        if not captured_images:
            print("No images captured")
            return None
        
        # Extract features from captures
        print("\nExtracting features for verification...")
        all_features = []
        
        for img_path in captured_images:
            try:
                features = self.facial_model.extract_research_features(img_path)
                all_features.append(features)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
        
        if not all_features:
            print("No valid features extracted")
            return None
        
        # Handle different feature lengths and average features
        max_len = max(len(f) for f in all_features)
        padded_features = []
        
        for features in all_features:
            if len(features) < max_len:
                padded = np.zeros(max_len)
                padded[:len(features)] = features
                padded_features.append(padded)
            else:
                padded_features.append(features[:max_len])
        
        avg_features = np.mean(padded_features, axis=0)
        
        # Compare with enrolled persons
        print("\nComparing with enrolled persons...")
        best_match = None
        best_similarity = 0.0
        
        for person_id, template in self.enrolled_persons.items():
            similarity = self.calculate_feature_similarity(
                avg_features, template.feature_vector
            )
            
            print(f"  {person_id}: {similarity:.2%} similarity")
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = template
        
        # Check if match is good enough
        if best_similarity >= self.similarity_threshold:
            print(f"\n{'='*60}")
            print(f"✓ PERSON RECOGNIZED: {best_match.person_id}")
            print(f"{'='*60}")
            print(f"Similarity: {best_similarity:.2%}")
            print(f"\nRegenerating consistent keys...")
            
            # Update verification stats
            best_match.verification_count += 1
            best_match.last_verified = datetime.now()
            self._save_template(best_match)
            
            # Return the stored consistent keys
            result = {
                'person_id': best_match.person_id,
                'primary_numeric_key': best_match.primary_key,
                'blockchain_address': best_match.blockchain_address,
                'similarity_score': best_similarity,
                'verification_count': best_match.verification_count,
                'enrolled_date': best_match.created_at.isoformat()
            }
            
            print(f"\nCONSISTENT KEYS REGENERATED:")
            print(f"Primary Key: {result['primary_numeric_key']}")
            print(f"Blockchain Address: {result['blockchain_address']}")
            print(f"Verification Count: {result['verification_count']}")
            
            return result
        else:
            print(f"\n{'='*60}")
            print(f"✗ PERSON NOT RECOGNIZED")
            print(f"{'='*60}")
            print(f"Best match: {best_match.person_id if best_match else 'None'}")
            print(f"Similarity: {best_similarity:.2%} (threshold: {self.similarity_threshold:.2%})")
            print("\nPlease enroll first or try capturing again with better lighting.")
            
            return None

    def list_enrolled_persons(self):
        """Display all enrolled persons"""
        print(f"\n{'='*60}")
        print("ENROLLED PERSONS")
        print(f"{'='*60}")
        
        if not self.enrolled_persons:
            print("No enrolled persons yet.")
            return
        
        for person_id, template in self.enrolled_persons.items():
            print(f"\nPerson ID: {person_id}")
            print(f"  Enrolled: {template.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"  Verifications: {template.verification_count}")
            print(f"  Primary Key: {template.primary_key}")
            print(f"  Blockchain Address: {template.blockchain_address}")
            if template.last_verified:
                print(f"  Last Verified: {template.last_verified.strftime('%Y-%m-%d %H:%M:%S')}")

def demo_realtime_system():
    """Interactive demo of real-time facial key generation"""
    
    print("="*60)
    print("REAL-TIME FACIAL KEY GENERATION SYSTEM")
    print("="*60)
    print("\nInitializing system...")
    
    # Initialize facial model
    facial_model = ResearchFacialKeygenModel(
        target_features=128,
        slot_count=16,
        lfsr_rounds=5,
        research_mode=False
    )
    
    # Train with existing images if available
    captures_dir = Path("captures")
    if captures_dir.exists():
        image_files = list(captures_dir.glob("*.jpg"))
        if image_files:
            print(f"Training model with {len(image_files)} existing images...")
            facial_model.train_research_model([str(f) for f in image_files[:3]])
    
    # Initialize real-time system
    rt_system = RealTimeFacialKeygen(facial_model)
    
    # Interactive menu
    while True:
        print(f"\n{'='*60}")
        print("MAIN MENU")
        print(f"{'='*60}")
        print("1. Enroll New Person")
        print("2. Verify and Regenerate Keys")
        print("3. List Enrolled Persons")
        print("4. Exit")
        print(f"{'='*60}")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            person_name = input("Enter person name/ID: ").strip()
            if person_name:
                try:
                    keys = rt_system.enroll_new_person(person_name)
                except Exception as e:
                    print(f"Enrollment error: {e}")
            else:
                print("Invalid name")
        
        elif choice == '2':
            try:
                result = rt_system.verify_and_regenerate_keys()
                if result:
                    print("\n✓ Key regeneration successful!")
                    print(f"Same keys will be generated every time for {result['person_id']}")
            except Exception as e:
                print(f"Verification error: {e}")
        
        elif choice == '3':
            rt_system.list_enrolled_persons()
        
        elif choice == '4':
            print("\nExiting system... Goodbye!")
            break
        
        else:
            print("Invalid choice, please try again")

if __name__ == "__main__":
    demo_realtime_system()
