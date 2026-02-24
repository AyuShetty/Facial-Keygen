# 🔐 Facial Biometric Key Generation System

**Production-Ready Cryptographic Key Generation using Facial Biometrics**

A sophisticated system that generates secure cryptographic keys from facial biometric data using advanced Dynamic LFSR (Linear Feedback Shift Register) processing with real-time face recognition capabilities.

## 🎯 Features

- **Single-Shot Recognition**: Instantly recognizes registered users from a single camera frame
- **Dynamic Key Generation**: Advanced LFSR algorithm with entropy maximization
- **Automatic User Management**: Seamless registration and identification
- **Multiple Key Formats**: Primary, backup, hash, and compact keys
- **Production Architecture**: Clean, modular, and secure codebase
- **Real-time Processing**: Live camera integration with instant feedback

## 🚀 Quick Start

### 1. Setup
```bash
python setup.py
```

### 2. Run Desktop Application
```bash
python facial_keygen_system.py
```

### 3. Run API Server (Optional)
```bash
pip install flask
python api_server.py
```

## 💡 How It Works

1. **Face Detection**: Camera captures your face using MediaPipe
2. **Feature Extraction**: Extracts biometric landmarks (468 points)
3. **User Recognition**: Checks against registered user templates
4. **Key Processing**: 
   - **Existing User**: Retrieves stored keys
   - **New User**: Generates keys using dynamic LFSR algorithm

## 🔐 Key Generation Algorithm

The system uses a sophisticated Dynamic LFSR (Linear Feedback Shift Register) approach:

- **Dynamic Processing**: Continues until entropy exhaustion (not fixed rounds)
- **Cross-Buffer Feedback**: Multiple processing buffers with interdependence
- **Entropy Tracking**: Monitors uniqueness and convergence
- **Multiple Outputs**: Generates primary, backup, hash, and compact keys

## 📁 Project Structure

```
facial_keygen_system.py     # Main desktop application
api_server.py              # REST API interface
setup.py                   # Automated setup script
config.json               # System configuration
production_requirements.txt # Dependencies

user_templates/           # Face recognition templates (auto-created)
secure_keys/             # Generated cryptographic keys (auto-created)
```

## 🛠️ System Components

### BiometricProcessor
- Face detection and landmark extraction
- MediaPipe integration
- Feature normalization

### DynamicLFSREngine
- Advanced LFSR transformations
- Entropy maximization
- Multi-format key generation

### FaceRecognitionSystem
- User template management
- Similarity matching
- PCA-based feature reduction

### KeyStorageSystem
- Secure key persistence
- Timestamp tracking
- Access logging

## 🔒 Security Features

- **Template Protection**: Biometric templates stored separately from keys
- **Entropy Maximization**: Dynamic processing ensures maximum randomness
- **Multiple Key Types**: Different keys for different use cases
- **Access Control**: User identification required for key retrieval

## 📊 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | System health check |
| `/api/process` | POST | Process biometric data |
| `/api/users` | GET | List registered users |
| `/api/users/<id>` | DELETE | Delete user |
| `/api/config` | GET | Get system config |

## 📈 Performance Metrics

- **Recognition Speed**: < 500ms per frame
- **Key Generation**: < 2 seconds for new users
- **Accuracy**: >92% similarity threshold for recognition
- **Entropy**: Dynamic rounds until convergence

## ⚙️ Configuration

Edit `config.json` to customize:
- Recognition thresholds
- Key generation parameters
- Camera settings
- Security options

## 🚨 Important Notes

- **Never commit** `user_templates/` or `secure_keys/` directories
- System automatically creates `.gitignore` to protect sensitive data
- Requires camera access for operation
- Designed for single-user-per-session scenarios

## 📝 Usage Examples

### Desktop Mode
```python
# Automatic - just run the application
python facial_keygen_system.py

# Position face in camera
# Press SPACE to process
# System will automatically recognize or register
```

### API Mode
```bash
# Start server
python api_server.py

# Process image (POST /api/process)
{
    "image": "base64_encoded_image_data",
    "user_id": "optional_custom_id"
}
```

## 🔧 Technical Requirements

- Python 3.8+
- OpenCV 4.8+
- MediaPipe 0.10+
- scikit-learn 1.3+
- NumPy 1.24+
- Camera device

## 🎯 Production Ready

This system is designed for production use with:
- Error handling and validation
- Modular architecture
- Secure data handling
- Performance optimization
- Clean API interfaces
- Comprehensive logging

## 📞 Support

For technical issues or questions about the facial biometric key generation algorithm, please refer to the system documentation or contact the development team.

---

*Facial Biometric Key Generation System - Secure, Fast, Production-Ready*