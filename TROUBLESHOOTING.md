# 🔧 CAMERA SYSTEM TROUBLESHOOTING

## ❓ **COMMON ISSUES & SOLUTIONS**

### **Issue 1: "Could not open camera"**

**Problem**: Camera not accessible

**Solutions**:
1. **Check if camera is being used by another app**:
   - Close Zoom, Teams, Skype, etc.
   - Close browser tabs using camera
   
2. **Check camera permissions**:
   - Windows Settings → Privacy → Camera
   - Allow desktop apps to access camera
   
3. **Try different camera index**:
   Edit `realtime_camera_keygen.py`:
   ```python
   self.camera_index = 0  # Try 0, 1, or 2
   ```

### **Issue 2: Camera opens but freezes**

**Solutions**:
1. Press **ESC** to cancel
2. Restart the program
3. Check if camera drivers are updated

### **Issue 3: "Image too dark" / "Image too bright"**

**Solutions**:
- Adjust room lighting
- Move closer/farther from light source
- Use natural daylight if possible
- Avoid direct overhead lights

### **Issue 4: "Image too blurry"**

**Solutions**:
- Stay still while capturing
- Clean camera lens
- Ensure camera is in focus
- Move to well-lit area

### **Issue 5: Person not recognized after enrollment**

**Solutions**:
1. **Check similarity threshold** (default 85%):
   ```python
   self.similarity_threshold = 0.85  # Lower to 0.75 for easier matching
   ```

2. **Try enrolling again** with better lighting
3. **Capture in similar conditions** as enrollment

---

## 🎮 **QUICK TEST SCRIPT**

If you want to test just the camera without the full system:

```python
import cv2

# Test camera access
cap = cv2.VideoCapture(0)  # Try 0, 1, or 2

if not cap.isOpened():
    print("Camera not accessible!")
else:
    print("Camera working! Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if ret:
            cv2.imshow('Camera Test', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
```

Save as `test_camera.py` and run it to verify camera works.

---

## 📞 **CONTACT FOR HELP**

If issues persist:
1. Check camera is physically connected
2. Update camera drivers
3. Restart computer
4. Try external webcam if available
