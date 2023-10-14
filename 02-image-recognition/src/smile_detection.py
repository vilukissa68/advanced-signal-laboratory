import cv2
import torch

def smile_detection(model, opt):

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    cap = cv2.VideoCapture(0)

    while True:
        grab, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5)

        # Draw rectangles around faces
        for (x, y, w, h) in faces:
            if model.opt.nc == 1:
                face_roi = gray[y:y+h, x:x+w]
            else:
                face_roi = color[y:y+h, x:x+w]

            face_roi = cv2.resize(face_roi, (64, 64))
            face_roi = face_roi / 255.0

            # X, Y, C -> 0, C, X, Y
            roi = torch.tensor(face_roi, dtype=model.dtype).to(opt.device)
            roi = roi.permute(2, 0, 1).unsqueeze(0)

            # 0 = not smiling, 1 = smiling
            prediction = model.predict(roi)
            label = "smiling" if prediction == 1 else "not smiling"

            #label = "smiling" if smile > not_smile else "not smiling"
            cv2.putText(frame, label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow('Smile Detector', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


