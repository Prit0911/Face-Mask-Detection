import cv2
import numpy as np

from tensorflow.keras.models import load_model


class MaskClassifier:

    def __init__(self, model_path):

        self.model = load_model(
            model_path
        )


        self.classes = [
            "mask_weared_incorrect",
            "with_mask",
            "without_mask"
        ]


    def predict(self, face):

        face = cv2.resize(
            face,
            (224,224)
        )


        face = cv2.cvtColor(
            face,
            cv2.COLOR_BGR2RGB
        )


        face = face / 255.0


        face = np.expand_dims(
            face,
            axis=0
        )


        prediction = self.model.predict(
            face,
            verbose=0
        )


        class_id = np.argmax(
            prediction
        )


        confidence = np.max(
            prediction
        )


        return (
            self.classes[class_id],
            confidence
        )