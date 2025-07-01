import cv2
from deepface import DeepFace

cap = cv2.VideoCapture(0)

reference_img_path = "phudinan.jpg"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # ย่อขนาดภาพเป็น 640x480 เลย
    small_frame = cv2.resize(frame, (640, 480))

    try:
        results = DeepFace.analyze(small_frame, actions=['emotion'], enforce_detection=False)

        for face in results:
            x, y, w, h = face['region']['x'], face['region']['y'], face['region']['w'], face['region']['h']
            emotion = face['dominant_emotion'].lower()

            face_img = small_frame[y:y+h, x:x+w]

            verification = DeepFace.verify(face_img, reference_img_path, enforce_detection=False)

            if verification['verified']:
                name = "Phudinan Phukakun"
            else:
                name = "Unknown"

            if emotion == 'happy':
                color = (0, 255, 255)  # เหลือง
            elif emotion == 'sad':
                color = (255, 0, 0)    # น้ำเงิน
            elif emotion == 'neutral':
                color = (0, 255, 0)    # เขียว
            else:
                continue

            cv2.rectangle(small_frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(small_frame, emotion.capitalize(), (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            cv2.putText(small_frame, name, (x, y + h + 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    except Exception as e:
        print("ไม่พบใบหน้าหรือเกิดข้อผิดพลาด:", e)

    cv2.imshow('Emotion Detection & Recognition', small_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
