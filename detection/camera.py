import cv2

from yolo_detector import FaceDetector

from mask_classifier import MaskClassifier



face_detector = FaceDetector(
    r"D:\Face-Mask-Detection\ai_models\yolo\model.pt"
)


mask_classifier = MaskClassifier(
    r"D:\Face-Mask-Detection\ai_models\mobilenet\mask_classifier_mobilenetv2.h5"
)



camera = cv2.VideoCapture(0)


while True:


    ret, frame = camera.read()


    if not ret:
        break



    faces = face_detector.detect(
        frame
    )


    for face in faces:


        x1,y1,x2,y2 = face["box"]


        face_img = frame[
            y1:y2,
            x1:x2
        ]


        label, confidence = (
            mask_classifier.predict(
                face_img
            )
        )


        text = (
            f"{label}: {confidence:.2f}"
        )


        cv2.rectangle(
            frame,
            (x1,y1),
            (x2,y2),
            (0,255,0),
            2
        )


        cv2.putText(
            frame,
            text,
            (x1,y1-10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0,255,0),
            2
        )


    cv2.imshow(
        "Face Mask Detection",
        frame
    )


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



camera.release()

cv2.destroyAllWindows()