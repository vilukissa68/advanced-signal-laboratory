import cv2
import torch

def smile_detection(model, opt):

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    cap = cv2.VideoCapture(0)

    while True:
        grab, frame = cap.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        color = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5)

        # Draw rectangles around faces
        for (x, y, w, h) in faces:
            scaling_factor = opt.scaling_factor

            expanded_w = int(w * scaling_factor)
            expanded_h = int(h * scaling_factor)

            # Calculate x, y to keep the center the same
            expanded_x = max(0, x - (expanded_w - w) // 2)
            expanded_y = max(0, y - (expanded_h - h) // 2)

            if model.opt.nc == 1:
                face_roi = gray[expanded_y:expanded_y+expanded_h,
                                expanded_x:expanded_x+expanded_w]

            else:
                face_roi = color[expanded_y:expanded_y+expanded_h,
                                 expanded_x:expanded_x+expanded_w]

            face_roi = cv2.resize(face_roi, (64, 64))
            face_roi = face_roi / 255.0

            # X, Y, C -> 0, C, X, Y
            roi = torch.tensor(face_roi, dtype=model.dtype).to(opt.device)

            if model.opt.nc == 1:
                roi = roi.unsqueeze(0).unsqueeze(0)
            else:
                roi = roi.permute(2, 0, 1).unsqueeze(0)

            # 0 = not smiling, 1 = smiling
            prediction = model.predict(roi)

            # Sigmoid to binary
            prediction = int(prediction > 0.5)

            if prediction == 1:
                label = "smiling"
            elif prediction == 0:
                label = "not smiling"
            else:
                label = "predicting"

            cv2.putText(frame, label, (expanded_x, expanded_y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.rectangle(frame, (expanded_x, expanded_y),
                          (expanded_x + expanded_w, expanded_y + expanded_h),
                          (0, 255, 0), 2)

        cv2.imshow('Smile Detector', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


