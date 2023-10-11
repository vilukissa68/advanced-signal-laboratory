import cv2
import torch


def smile_detection():

    # TODO: koulutetun datan lisääminen
    # model_path = '../models/20231009-214704_experiment_name_SimpleCNN_3x64_3x32_2x16_1x8.model'
    # model = torch.load(model_path)

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    cap = cv2.VideoCapture(0)

    while True:
        grab, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5)

        # (not_smile, smile) = model.predict()
        label = "Not Smiling" # if not_smile, "smiling" if smile

        # Draw rectangles around faces
        for (x, y, w, h) in faces:
            cv2.putText(frame, label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow('Smile Detector', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


smile_detection()
