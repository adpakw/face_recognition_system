import os
from typing import Any, Dict, List, Tuple

import cv2
from tqdm import tqdm

from app.clients.json_processor import JsonProcessor
from app.clients.video_processors.opencv_reader import OpenCVVideoReader
from app.models.ssd import PeopleDetectorModel

class PeopleDetector:
    def __init__(self):
        self.detection_model = PeopleDetectorModel()


    def draw_detections(self, image, detections):
        """Отрисовка bounding boxes и скоров на изображении"""
        for detection in detections:
            cv2.rectangle(image, (detection['x1'], detection['y1']), (detection['x2'], detection['y2']), (0, 255, 0), 2)
            label = f"{detection['score']:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)

            cv2.rectangle(image, (detection['x1'], detection['y1'] - 20), (detection['x1'] + w, detection['y1']), (0, 255, 0), -1)

            cv2.putText(
                image, label, (detection['x1'], detection['y1'] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1
            )

        return image

    def detect_on_image(self, image_path):
        original_image = cv2.imread(image_path)
        image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        people_boxes = self.detection_model.detect_people(image)

        result_image = self.draw_detections(original_image.copy(), people_boxes)

        cv2.imshow("Detection Results", result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return people_boxes

    def test_people_detector_img(self):
        print(self.detect_on_image("data/dog.jpeg"))
        print(self.detect_on_image("data/input.jpg"))

    def process(
        self, frame, show_video=False
    ):
        """Обработка одного кадра"""
        result = {"frame": None, "people_boxes": None}

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result["people_boxes"] = self.detection_model.detect_people(frame_rgb)

        if show_video:
            result_frame = self.draw_detections(frame.copy(), result["people_boxes"])
            result["frame"] = result_frame

        return result
