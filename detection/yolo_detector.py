from ultralytics import YOLO


class FaceDetector:


    def __init__(self, model_path):

        self.model = YOLO(
            model_path
        )


    def detect(self, frame):

        results = self.model(
            frame,
            verbose=False
        )


        faces=[]


        for result in results:

            boxes = result.boxes


            for box in boxes:

                x1,y1,x2,y2 = (
                    box.xyxy[0]
                    .cpu()
                    .numpy()
                )


                confidence = (
                    box.conf[0]
                    .cpu()
                    .numpy()
                )


                faces.append(
                    {
                        "box":
                        [
                            int(x1),
                            int(y1),
                            int(x2),
                            int(y2)
                        ],

                        "confidence":
                        float(confidence)
                    }
                )


        return faces