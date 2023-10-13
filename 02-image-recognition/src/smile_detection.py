import cv2
import torch
from options import Options

def smile_detection(model):
    opt = Options().parse()

    model_path = opt.modelpath
    model.load_model(model_path)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    cap = cv2.VideoCapture(0)

    while True:
        grab, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5)

        # Draw rectangles around faces
        for (x, y, w, h) in faces:
            face_roi = gray[y:y+h, x:x+w]

            face_roi = cv2.resize(face_roi, (64, 64))
            face_roi = face_roi / 255.0
            roi = torch.tensor(face_roi, dtype=torch.float32).unsqueeze(0)

            (not_smile, smile) = model.predict(roi)
            label = "smiling" if smile > not_smile else "not smiling"
            cv2.putText(frame, label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow('Smile Detector', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


