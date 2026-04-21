Real-Time Driver Drowsiness Detection

Overview

This module implements a real-time driver drowsiness detection system using a trained YOLO-based model. It processes live webcam input to detect signs of fatigue such as eye closure and yawning, and computes a drowsiness score based on temporal patterns.
An alert is triggered when sustained drowsiness is detected, making the system suitable for safety-critical monitoring applications.

Features

Real-time object detection using a trained YOLO model

    Detection of key drowsiness indicators:
      
        Eye closure
        Yawning
    
    Sliding window temporal analysis
    
    Computation of:
    
        PERCLOS (Percentage of Eye Closure)
        
        Yawn rate (events per second)
    
    Weighted drowsiness scoring mechanism
    
    Stability check to reduce false positives
    
    Audio alert system for driver warning


Methodology

    Detection:
    
    Each frame from the webcam is processed using the YOLO model to identify:
    
        eyeclosed
        
        yawn
    
    Temporal Analysis:
    
        A fixed-size sliding window is maintained over recent frames:
        
        Tracks eye closure occurrences
        
        Tracks yawning events
    
    Metrics Computed:
    
        PERCLOS
        Ratio of frames where eyes are detected as closed
        
        Yawn Rate
        Frequency of yawning events per second

Drowsiness Score

    A weighted score is computed:
    
      Drowsiness Score = 0.7 × PERCLOS + 0.3 × Yawn Rate
    
    Alert Logic
    
      If the score exceeds a threshold
      And remains stable for a fixed duration
      → A warning alert is triggered
