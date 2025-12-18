import cv2
import mediapipe as mp
import numpy as np
import json
import os
from datetime import datetime
from flask import Flask, render_template_string, request, jsonify
import threading
import base64

app = Flask(__name__)

class AITailorSystem:
    def __init__(self):
        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Camera and display settings
        self.cap = None
        self.calibration_distance = 50
        self.pixel_to_cm_ratio = 0
        self.calibrated = False
        self.measurements_data = {}
        self.current_user = "Guest"
        self.current_measurements = None
        
    def calibrate_system(self, frame):
        """Calibrate using face width - simplified version"""
        h, w, _ = frame.shape
        # Assume average face is roughly 1/8th of frame width
        face_width_pixels = w / 8
        average_face_width_cm = 15.5
        self.pixel_to_cm_ratio = average_face_width_cm / face_width_pixels
        self.calibrated = True
        return True
    
    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def extract_measurements(self, landmarks, frame_shape):
        """Extract body measurements from pose landmarks"""
        h, w, _ = frame_shape
        
        landmarks_dict = {
            'nose': 0,
            'left_shoulder': 11,
            'right_shoulder': 12,
            'left_elbow': 13,
            'right_elbow': 14,
            'left_wrist': 15,
            'right_wrist': 16,
            'left_hip': 23,
            'right_hip': 24,
            'left_knee': 25,
            'right_knee': 26,
            'left_ankle': 27,
            'right_ankle': 28,
            'left_ear': 7,
            'right_ear': 8
        }
        
        measurements = {}
        
        if self.pixel_to_cm_ratio == 0:
            return None
        
        try:
            # Shoulder width
            left_shoulder = [landmarks[landmarks_dict['left_shoulder']].x * w, 
                           landmarks[landmarks_dict['left_shoulder']].y * h]
            right_shoulder = [landmarks[landmarks_dict['right_shoulder']].x * w, 
                            landmarks[landmarks_dict['right_shoulder']].y * h]
            shoulder_width = self.calculate_distance(left_shoulder, right_shoulder)
            measurements['Shoulder Width'] = round(shoulder_width * self.pixel_to_cm_ratio, 1)
            
            # Arm length
            left_shoulder_pos = [landmarks[landmarks_dict['left_shoulder']].x * w,
                               landmarks[landmarks_dict['left_shoulder']].y * h]
            left_wrist_pos = [landmarks[landmarks_dict['left_wrist']].x * w,
                            landmarks[landmarks_dict['left_wrist']].y * h]
            arm_length = self.calculate_distance(left_shoulder_pos, left_wrist_pos)
            measurements['Left Arm Length'] = round(arm_length * self.pixel_to_cm_ratio, 1)
            
            # Torso length
            left_hip = [landmarks[landmarks_dict['left_hip']].x * w,
                       landmarks[landmarks_dict['left_hip']].y * h]
            torso_length = self.calculate_distance(left_shoulder_pos, left_hip)
            measurements['Torso Length'] = round(torso_length * self.pixel_to_cm_ratio, 1)
            
            # Inseam
            left_ankle = [landmarks[landmarks_dict['left_ankle']].x * w,
                         landmarks[landmarks_dict['left_ankle']].y * h]
            inseam = self.calculate_distance(left_hip, left_ankle)
            measurements['Inseam'] = round(inseam * self.pixel_to_cm_ratio, 1)
            
            # Hip width
            right_hip = [landmarks[landmarks_dict['right_hip']].x * w,
                        landmarks[landmarks_dict['right_hip']].y * h]
            hip_width = self.calculate_distance(left_hip, right_hip)
            measurements['Hip Width'] = round(hip_width * self.pixel_to_cm_ratio, 1)
            
            # Leg length
            leg_length = self.calculate_distance(left_hip, left_ankle)
            measurements['Leg Length'] = round(leg_length * self.pixel_to_cm_ratio, 1)
            
            return measurements
        
        except Exception as e:
            print(f"Error calculating measurements: {e}")
            return None
    
    def process_frame(self, frame):
        """Process frame and return measurements"""
        if not self.calibrated:
            self.calibrate_system(frame)
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        
        if results.pose_landmarks:
            measurements = self.extract_measurements(results.pose_landmarks, frame.shape)
            self.current_measurements = measurements
            return True, measurements
        
        return False, None
    
    def save_measurements(self, user_name, measurements):
        """Save measurements to JSON"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if user_name not in self.measurements_data:
            self.measurements_data[user_name] = []
        
        measurement_record = {
            'timestamp': timestamp,
            'measurements': measurements
        }
        
        self.measurements_data[user_name].append(measurement_record)
        
        filename = f"measurements_{user_name}.json"
        with open(filename, 'w') as f:
            json.dump(self.measurements_data[user_name], f, indent=4)
        
        return True

tailor_system = AITailorSystem()

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>AI Tailor - Body Measurement</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        
        body {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        
        .container {
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            max-width: 800px;
            width: 100%;
            padding: 30px;
        }
        
        h1 {
            text-align: center;
            color: #333;
            margin-bottom: 10px;
            font-size: 2em;
        }
        
        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 30px;
            font-size: 0.9em;
        }
        
        .video-container {
            background: #000;
            border-radius: 15px;
            overflow: hidden;
            margin-bottom: 25px;
            aspect-ratio: 4/3;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        
        video {
            width: 100%;
            height: 100%;
            object-fit: cover;
        }
        
        .controls {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 15px;
            margin-bottom: 25px;
        }
        
        button {
            padding: 15px;
            border: none;
            border-radius: 10px;
            font-size: 1em;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        
        .btn-primary {
            background: #667eea;
            color: white;
            grid-column: 1 / -1;
        }
        
        .btn-primary:hover {
            background: #5568d3;
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(102, 126, 234, 0.4);
        }
        
        .btn-secondary {
            background: #48bb78;
            color: white;
        }
        
        .btn-secondary:hover {
            background: #38a169;
            transform: translateY(-2px);
        }
        
        .btn-danger {
            background: #f56565;
            color: white;
        }
        
        .btn-danger:hover {
            background: #e53e3e;
            transform: translateY(-2px);
        }
        
        .status-box {
            background: #f7fafc;
            border-left: 5px solid #667eea;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
        }
        
        .status-item {
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            font-size: 0.95em;
        }
        
        .status-label {
            color: #666;
            font-weight: 500;
        }
        
        .status-value {
            color: #333;
            font-weight: 600;
        }
        
        .measurements-box {
            background: #f0f4ff;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
        }
        
        .measurements-box h3 {
            color: #667eea;
            margin-bottom: 15px;
            font-size: 1.2em;
        }
        
        .measurement-item {
            display: flex;
            justify-content: space-between;
            padding: 10px 0;
            border-bottom: 1px solid #e2e8f0;
            font-size: 0.95em;
        }
        
        .measurement-item:last-child {
            border-bottom: none;
        }
        
        .measurement-name {
            color: #555;
        }
        
        .measurement-value {
            color: #667eea;
            font-weight: 600;
            font-size: 1.1em;
        }
        
        .input-group {
            margin-bottom: 15px;
        }
        
        input {
            width: 100%;
            padding: 12px;
            border: 2px solid #e2e8f0;
            border-radius: 8px;
            font-size: 1em;
            transition: border-color 0.3s;
        }
        
        input:focus {
            outline: none;
            border-color: #667eea;
        }
        
        .alert {
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            font-weight: 500;
        }
        
        .alert-success {
            background: #c6f6d5;
            color: #22543d;
            border-left: 4px solid #48bb78;
        }
        
        .alert-error {
            background: #fed7d7;
            color: #742a2a;
            border-left: 4px solid #f56565;
        }
        
        .alert-info {
            background: #bee3f8;
            color: #2c5282;
            border-left: 4px solid #4299e1;
        }
        
        @media (max-width: 600px) {
            .container {
                padding: 20px;
            }
            
            h1 {
                font-size: 1.5em;
            }
            
            .controls {
                grid-template-columns: 1fr;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>👕 AI Tailor</h1>
        <p class="subtitle">Automatic Body Measurement System</p>
        
        <div id="alert"></div>
        
        <div class="video-container">
            <video id="video" playsinline autoplay></video>
        </div>
        
        <div class="status-box">
            <div class="status-item">
                <span class="status-label">Calibration Status:</span>
                <span class="status-value" id="calibration-status">Not Started</span>
            </div>
            <div class="status-item">
                <span class="status-label">Body Detection:</span>
                <span class="status-value" id="detection-status">Waiting...</span>
            </div>
        </div>
        
        <div class="controls">
            <button class="btn-primary" onclick="calibrate()">📏 Calibrate System</button>
            <button class="btn-secondary" onclick="measureNow()">📊 Take Measurement</button>
            <button class="btn-danger" onclick="stopCamera()">⏹️ Stop Camera</button>
        </div>
        
        <div class="input-group">
            <input type="text" id="username" placeholder="Enter your name (optional)" value="Guest">
        </div>
        
        <button class="btn-primary" onclick="saveMeasurements()" style="width: 100%;">💾 Save My Measurements</button>
        
        <div class="measurements-box" id="measurements-box" style="display: none;">
            <h3>📐 Current Measurements (cm)</h3>
            <div id="measurements-list"></div>
        </div>
    </div>
    
    <script>
        const video = document.getElementById('video');
        let calibrated = false;
        let currentMeasurements = null;
        
        // Start camera
        navigator.mediaDevices.getUserMedia({ 
            video: { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } } 
        })
        .then(stream => {
            video.srcObject = stream;
            showAlert('Camera started! Click "Calibrate System" to begin.', 'info');
        })
        .catch(err => {
            showAlert('Error accessing camera: ' + err.message, 'error');
        });
        
        function calibrate() {
            fetch('/calibrate')
                .then(res => res.json())
                .then(data => {
                    calibrated = true;
                    document.getElementById('calibration-status').textContent = '✅ Calibrated';
                    showAlert('System calibrated successfully!', 'success');
                });
        }
        
        function measureNow() {
            if (!calibrated) {
                showAlert('Please calibrate the system first!', 'error');
                return;
            }
            
            fetch('/measure')
                .then(res => res.json())
                .then(data => {
                    if (data.success) {
                        currentMeasurements = data.measurements;
                        displayMeasurements(data.measurements);
                        document.getElementById('detection-status').textContent = '✅ Body Detected';
                        showAlert('Measurements captured!', 'success');
                    } else {
                        document.getElementById('detection-status').textContent = '❌ No Body Detected';
                        showAlert('Could not detect body. Make sure your full body is visible.', 'error');
                    }
                });
        }
        
        function displayMeasurements(measurements) {
            const box = document.getElementById('measurements-box');
            const list = document.getElementById('measurements-list');
            list.innerHTML = '';
            
            for (let [key, value] of Object.entries(measurements)) {
                const item = document.createElement('div');
                item.className = 'measurement-item';
                item.innerHTML = `
                    <span class="measurement-name">${key}</span>
                    <span class="measurement-value">${value} cm</span>
                `;
                list.appendChild(item);
            }
            
            box.style.display = 'block';
        }
        
        function saveMeasurements() {
            if (!currentMeasurements) {
                showAlert('No measurements to save. Click "Take Measurement" first!', 'error');
                return;
            }
            
            const username = document.getElementById('username').value || 'Guest';
            
            fetch('/save', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ user: username, measurements: currentMeasurements })
            })
            .then(res => res.json())
            .then(data => {
                showAlert(`Measurements saved for ${username}!`, 'success');
            });
        }
        
        function stopCamera() {
            const stream = video.srcObject;
            const tracks = stream.getTracks();
            tracks.forEach(track => track.stop());
            showAlert('Camera stopped.', 'info');
        }
        
        function showAlert(message, type) {
            const alertBox = document.getElementById('alert');
            alertBox.className = `alert alert-${type}`;
            alertBox.textContent = message;
            
            setTimeout(() => {
                alertBox.textContent = '';
                alertBox.className = '';
            }, 4000);
        }
    </script>
</body>
</html>
'''

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/calibrate')
def calibrate():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if ret:
        tailor_system.calibrate_system(frame)
        cap.release()
        return jsonify({'success': True, 'message': 'Calibrated'})
    cap.release()
    return jsonify({'success': False})

@app.route('/measure')
def measure():
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    if ret:
        success, measurements = tailor_system.process_frame(frame)
        cap.release()
        if success and measurements:
            return jsonify({'success': True, 'measurements': measurements})
    cap.release()
    return jsonify({'success': False})

@app.route('/save', methods=['POST'])
def save():
    data = request.json
    tailor_system.save_measurements(data['user'], data['measurements'])
    return jsonify({'success': True})

if __name__ == '__main__':
    print("🚀 AI Tailor System Starting...")
    print("📱 Open your browser and go to: http://localhost:5000")
    print("Press Ctrl+C to stop")
    app.run(debug=False, host='0.0.0.0', port=5000)