import cv2
import numpy as np
from app.models.yunet import YuNet


class FaceDetector:
    def __init__(
        self,
        model_path: str = "app/models/weights/face_detection_yunet_2023mar.onnx",
        score_threshold=0.3,
        nms_threshold=0.3,
        top_k=5000,
        device="cpu",
        confidence_threshold = 0.7
    ):
        """
        Инициализация детектора лиц с использованием YuNet

        Args:
            model_path (str): Путь к файлу модели YuNet (.onnx)
                            Если None, будет использована встроенная в OpenCV версия
            input_size (tuple): Размер входного изображения (ширина, высота)
            conf_threshold (float): Порог уверенности для детекции
        """
        self.detection_model = YuNet(
            model_path=model_path,
            score_threshold=score_threshold,
            nms_threshold=nms_threshold,
            top_k=top_k,
            device=device
        )
        self.confidence_threshold = confidence_threshold

    def process(self, frame: np.ndarray, people_bboxes: list = None, show_video=False):
        """Обработка одного кадра"""
        result = {"frame": None, "result_dicts": None}

        face_bboxes = []
        result["result_dicts"] = []
        for person_bbox in people_bboxes:
            face_bbox = self.detection_model.detect_faces(
                frame, person_bbox=person_bbox, confidence_threshold=self.confidence_threshold
            )
            face_bboxes.append(face_bbox[0])
            result["result_dicts"].append({"person_bbox": person_bbox,
                                           "face_bbox": face_bbox})

        if show_video:
            result["frame"] = self.visualize(frame.copy(), face_bboxes)

        return result

    def visualize(
        self,
        image: np.ndarray,
        faces: list,
        color: tuple = (0, 255, 0),
        thickness: int = 2,
    ) -> np.ndarray:
        """
        Визуализация обнаруженных лиц на изображении

        Args:
            image (np.ndarray): Исходное изображение
            faces (list): Список обнаруженных лиц
            color (tuple): Цвет bounding box (B, G, R)
            thickness (int): Толщина линий

        Returns:
            np.ndarray: Изображение с отрисованными bounding boxes и landmarks
        """
        vis_image = image.copy()

        for face in faces:
            # Рисуем bounding box
            cv2.rectangle(
                vis_image,
                (face["x1"], face["y1"]),
                (face["x2"], face["y2"]),
                color,
                thickness,
            )

            # Добавляем confidence score
            label = f"{face['score']:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(
                vis_image,
                (face["x1"], face["y1"] - 20),
                (face["x1"] + w, face["y1"]),
                color,
                -1,
            )
            cv2.putText(
                vis_image,
                label,
                (face["x1"], face["y1"] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
            )

        return vis_image
