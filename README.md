## Real-Time Driver Drowsiness Detection

### Overview
This module implements a real-time driver drowsiness detection system using a trained YOLO-based model. It processes live webcam input to detect signs of fatigue such as eye closure and yawning, and computes a drowsiness score based on temporal patterns.

An alert is triggered when sustained drowsiness is detected, making the system suitable for safety-critical monitoring applications.

---

### Features
- Real-time object detection using a trained YOLO model  
- Detection of key drowsiness indicators:
  - Eye closure
  - Yawning  
- Sliding window temporal analysis  
- Computation of:
  - PERCLOS (Percentage of Eye Closure)
  - Yawn rate (events per second)  
- Weighted drowsiness scoring mechanism  
- Stability check to reduce false positives  
- Audio alert system for driver warning  

---

### Methodology

#### Detection
Each frame from the webcam is processed using the YOLO model to identify:
- `eyeclosed`
- `yawn`

#### Temporal Analysis
A fixed-size sliding window is maintained over recent frames:
- Tracks eye closure occurrences  
- Tracks yawning events  

#### Metrics Computed
- **PERCLOS**  
  Ratio of frames where eyes are detected as closed  

- **Yawn Rate**  
  Frequency of yawning events per second  

---

### Drowsiness Score

A weighted score is computed:
Drowsiness Score = 0.7 × PERCLOS + 0.3 × Yawn Rate


#### Alert Logic
- If the score exceeds a threshold  
- And remains stable for a fixed duration  
→ A warning alert is triggered  

---

### Requirements

Install dependencies:
pip install ultralytics opencv-python


---

### Usage

1. Place your trained model (`best.pt`) in the project directory  

2. Run the script:
python scripts/realtime.py


3. Press `q` to exit the application  

---

### Parameters

| Parameter              | Description                        | Default |
|----------------------|----------------------------------|--------|
| WINDOW_SIZE          | Number of frames in sliding window | 120    |
| FPS_ASSUMED          | Estimated frames per second        | 30     |
| W_PERCLOS            | Weight for eye closure             | 0.7    |
| W_YAWN               | Weight for yawning                 | 0.3    |
| DROWSY_SCORE_THRESHOLD | Detection threshold              | 0.5    |
| STABILITY_SECONDS    | Required duration above threshold  | 2      |

---

### Output

- Live video feed with detection bounding boxes  
- Real-time status:
  - ALERT  
  - MONITORING  
  - DROWSY  
- Display of computed metrics:
  - PERCLOS  
  - Yawn rate  
  - Drowsiness score  

---

### Notes
- Designed for controlled lighting conditions for optimal performance  
- Accuracy depends on model quality and dataset  
- Windows-specific alert uses `winsound`  

---

### Demo




