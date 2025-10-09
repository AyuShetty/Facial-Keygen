import cv2
import numpy as np
import hashlib
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D

# Initialize models and components
def initialize_models():
    # Face detection model
    face_cascade = cv2.CascadeClassifier()
    if not face_cascade.load(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'):
        raise RuntimeError("Error loading face detection model")
    
    # Feature extraction model
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    x = GlobalAveragePooling2D()(base_model.output)
    feature_model = Model(inputs=base_model.input, outputs=x)
    
    # Fixed projection matrix (reproducible)
    rng = np.random.default_rng(seed=42)
    projection_matrix = rng.normal(size=(1280, 256))
    
    return face_cascade, feature_model, projection_matrix

# Main processing pipeline
def generate_face_key():
    # Initialize models
    face_cascade, feature_model, projection_matrix = initialize_models()
    
    # Capture face image
    cap = cv2.VideoCapture(0)
    
    # Allow camera warm-up and capture frame
    for _ in range(30):
        ret, frame = cap.read()
    
    cap.release()
    
    if not ret:
        raise RuntimeError("Failed to capture image")
    
    # Detect and preprocess face
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(faces) == 0:
        raise RuntimeError("No face detected")
    
    # Get the largest face
    x, y, w, h = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
    face_img = frame[y:y+h, x:x+w]
    
    # Preprocess image for MobileNetV2
    processed_img = cv2.resize(face_img, (224, 224))
    processed_img = cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB)
    processed_img = processed_img.astype(np.float32) / 127.5 - 1.0
    processed_img = np.expand_dims(processed_img, axis=0)
    
    # Extract and project features
    features = feature_model.predict(processed_img, verbose=0)
    projected = np.dot(features, projection_matrix).flatten()
    
    # Convert to fixed-point integers (fixed syntax error)
    state = [int(val * (1 << 20)) & 0xFFFFFFFF for val in projected]
    
    # LFSR Configuration (taps selected for good diffusion)
    TAPS = [0, 254, 165, 97, 61]  # Optimal for 256-element LFSR
    
    # Run LFSR rounds
    for _ in range(1024):
        feedback = 0
        for tap in TAPS:
            feedback ^= state[tap]
        state = state[1:] + [feedback]
    
    # Generate final key
    state_bytes = b''.join(val.to_bytes(4, 'big') for val in state)
    return hashlib.sha256(state_bytes).hexdigest()

if __name__ == "__main__":
    try:
        face_key = generate_face_key()
        print("Generated Face Key:", face_key)
    except Exception as e:
        print("Error:", str(e))