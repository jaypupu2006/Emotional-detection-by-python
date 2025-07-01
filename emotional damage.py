import cv2
from deepface import DeepFace

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

        for face in result:
            x, y, w, h = face['region']['x'], face['region']['y'], face['region']['w'], face['region']['h']
            emotion = face['dominant_emotion'].lower()

            # Set color based on emotion
            if emotion == 'happy':
                color = (0, 255, 255)  # Yellow
            elif emotion == 'sad':
                color = (255, 0, 0)    # Blue
            elif emotion == 'neutral':
                color = (0, 255 , 0)    # Red
            else:
                continue  # Skip other emotions

            # Draw rectangle and emotion label
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, emotion.capitalize(), (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    except Exception as e:
        print("No face detected or error:", e)

    cv2.imshow('Emotion Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()