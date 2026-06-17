import cv2
import mediapipe as mp
import pickle
import numpy as np
from flask import Flask, render_template, Response
from flask_socketio import SocketIO

app = Flask(__name__)
# Async mode threading agar video tidak lag saat kirim data socket
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# --- LOAD MODEL ---
try:
    model = pickle.load(open('models/bisindo_model.pkl', 'rb'))
    print("✅ Model loaded successfully!")
except:
    print("❌ Error: Model tidak ditemukan! Pastikan sudah jalankan train.py")
    model = None

# --- KONFIGURASI MEDIAPIPE ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=0,       # Mode ringan (Cepat)
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)
mp_draw = mp.solutions.drawing_utils

# --- KONFIGURASI STYLE LANDMARK (CYAN / BIRU MUDA) ---
# Agar terlihat modern dan rapi seperti di gambar referensi
landmark_style = mp_draw.DrawingSpec(color=(248, 189, 56), thickness=2, circle_radius=2) # Titik Cyan
connection_style = mp_draw.DrawingSpec(color=(255, 255, 255), thickness=2) # Garis Cyan

def gen_frames():
    cap = cv2.VideoCapture(0)
    # SET RESOLUSI KE 640x480 (Rasio 4:3 Standard Webcam)
    # Ini PENTING agar gambar tidak di-zoom crop oleh CSS nanti
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    frame_count = 0
    
    while True:
        success, frame = cap.read()
        if not success: break
        
        # Mirror frame (biar seperti cermin)
        frame = cv2.flip(frame, 1)
        frame_count += 1
        
        # LOGIKA SKIP FRAME (Optimasi CPU)
        # Deteksi tangan hanya jalan setiap 2 frame sekali
        if frame_count % 2 == 0:
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)
            
            if results.multi_hand_landmarks:
                data_row = []
                available_hands = results.multi_hand_landmarks[:2]
                
                for hand_landmarks in available_hands:
                    # Gambar Landmark yang RAPI (Cyan Style)
                    mp_draw.draw_landmarks(
                        frame, 
                        hand_landmarks, 
                        mp_hands.HAND_CONNECTIONS,
                        landmark_drawing_spec=landmark_style,
                        connection_drawing_spec=connection_style
                    )

                    # Normalisasi Data
                    base_x = hand_landmarks.landmark[0].x
                    base_y = hand_landmarks.landmark[0].y
                    for lm in hand_landmarks.landmark:
                        data_row.extend([lm.x - base_x, lm.y - base_y])

                # Padding Data (Isi 0 jika cuma 1 tangan)
                while len(data_row) < 84:
                    data_row.append(0.0)
                
                # Prediksi
                if model:
                    try:
                        prediction = model.predict([data_row])[0]
                        proba_list = model.predict_proba([data_row])[0]
                        proba = max(proba_list) * 100
                        
                        socketio.emit('update_letter', {
                            'letter': str(prediction),
                            'confidence': int(proba)
                        })
                    except Exception as e:
                        pass
        
        # Encode ke JPEG (Kualitas 70% agar ringan di browser)
        ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Host 0.0.0.0 agar bisa diakses local network jika perlu
    socketio.run(app, debug=False, host='0.0.0.0', port=5000)