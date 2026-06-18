import cv2
import mediapipe as mp
import csv
import os

target_label = input("Masukkan Huruf (A-Z): ").upper()
if not os.path.exists('data'): os.makedirs('data')

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,             
    model_complexity=0,          
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_draw = mp.solutions.drawing_utils

file_path = f'data/{target_label}.csv'
with open(file_path, 'a', newline='') as f:
    writer = csv.writer(f)
    print(f"--- MEREKAM '{target_label}' (Tekan 'Q' Stop) ---")
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1)
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        if results.multi_hand_landmarks:

            data_row = [] 
            
            available_hands = results.multi_hand_landmarks[:2]
            
            for hand_landmarks in available_hands:
                base_x = hand_landmarks.landmark[0].x
                base_y = hand_landmarks.landmark[0].y
                
                for lm in hand_landmarks.landmark:
                    data_row.extend([lm.x - base_x, lm.y - base_y])
            
            while len(data_row) < 84:
                data_row.append(0.0)
            
            writer.writerow([target_label] + data_row)
            
            for h in available_hands:
                mp_draw.draw_landmarks(frame, h, mp_hands.HAND_CONNECTIONS)
        
        cv2.imshow('Perekam Data (Stabil)', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
