from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np

from app.models.yunet import YuNet
from app.utils.config_reader import ConfigReader


class FaceDetector:
    def __init__(self, config: Optional[ConfigReader] = None):
        """
        Инициализация детектора лиц
        """
        if config is None:
            config = ConfigReader()

        detector_config = config.get_pipeline_step_config("face_detector")

        model_name = config.get_config().pipeline["face_detector"].model
        self.detection_model = self._init_model(model_name, detector_config)

        self.confidence_threshold = detector_config["cfg"]["confidence_threshold"]
        self.face_padding = detector_config["cfg"]["face_padding"]

        self.config = config

    def _init_model(self, model_name: str, config: Dict[str, Any]):
        """Инициализирует модель детекции лиц на основе имени и конфига"""
        if model_name == "YuNet":
            return YuNet(
                model_path=config["cfg"]["model_path"],
                score_threshold=config["cfg"]["score_threshold"],
                nms_threshold=config["cfg"]["nms_threshold"],
                top_k=config["cfg"]["top_k"],
            )
        else:
            raise ValueError(f"Unsupported face detection model: {model_name}")

    def process(
        self,
        frame: np.ndarray,
        people_bboxes: Optional[List[Dict]] = None,
        show_video: bool = False,
    ) -> Dict[str, Union[np.ndarray, List[Dict]]]:
        """
        Обработка одного кадра с возможностью работы как с bbox людей, так и без них

        Args:
            frame: Входное изображение (BGR)
            people_bboxes: Опциональный список bbox людей. Если None, ищем лица на всем изображении
            show_video: Флаг отображения результата

        Returns:
            Словарь с результатами:
            {
                "frame": изображение с отрисованными bbox (если show_video=True),
                "result_dicts": список словарей с информацией о найденных лицах
            }
        """
        result = {"frame": frame, "result_dicts": []}

        if people_bboxes is None:
            faces = self._detect_faces_on_frame(frame)
            result["result_dicts"] = [{"face_bbox": face} for face in faces]
        else:
            result["result_dicts"] = self._detect_faces_in_people_bboxes(
                frame, people_bboxes
            )

        if show_video:
            faces = [
                r["face_bbox"]
                for r in result["result_dicts"]
                if r["face_bbox"] is not None
            ]
            result["frame"] = self.visualize(frame.copy(), faces)

        return result

    def _detect_faces_on_frame(self, frame: np.ndarray) -> List[Dict]:
        """Детекция лиц на всем изображении"""
        h, w = frame.shape[:2]
        faces = self.detection_model.detect_faces(
            frame,
            input_size=(w, h),
            confidence_threshold=self.confidence_threshold,
        )

        processed_faces = []
        for face in faces:
            processed_face = self._process_face_bbox(face, (0, 0), frame.shape)
            processed_faces.append(processed_face)

        return processed_faces

    def _detect_faces_in_people_bboxes(
        self, frame: np.ndarray, people_bboxes: List[Dict]
    ) -> List[Dict]:
        """Детекция лиц внутри bbox людей"""
        result = []

        for person_bbox in people_bboxes:
            x1, y1, x2, y2 = (
                person_bbox["x1"],
                person_bbox["y1"],
                person_bbox["x2"],
                person_bbox["y2"],
            )
            person_img = frame[y1:y2, x1:x2]

            if person_img.size == 0:
                result.append({"person_bbox": person_bbox, "face_bbox": None})
                continue

            h, w = person_img.shape[:2]
            faces = self.detection_model.detect_faces(
                person_img,
                input_size=(w, h),
                confidence_threshold=self.confidence_threshold,
            )

            if faces:
                face_bbox = self._process_face_bbox(faces[0], (x1, y1), frame.shape)
                result.append({"person_bbox": person_bbox, "face_bbox": face_bbox})
            else:
                result.append({"person_bbox": person_bbox, "face_bbox": None})

        return result

    def _process_face_bbox(
        self, face_bbox: Dict[str, Any], offset: tuple, frame_shape: tuple
    ) -> Dict:
        """Обработка и нормализация bounding box лица"""
        x_offset, y_offset = offset

        face_bbox = {
            "x1": face_bbox["x1"] + x_offset,
            "y1": face_bbox["y1"] + y_offset,
            "x2": face_bbox["x2"] + x_offset,
            "y2": face_bbox["y2"] + y_offset,
            "score": face_bbox["score"],
        }

        width = face_bbox["x2"] - face_bbox["x1"]
        height = face_bbox["y2"] - face_bbox["y1"]

        x_padding = int(width * self.face_padding)
        y_padding = int(height * self.face_padding)

        face_bbox["x1"] -= x_padding
        face_bbox["y1"] -= y_padding
        face_bbox["x2"] += x_padding
        face_bbox["y2"] += y_padding

        face_bbox["x1"] = max(0, min(face_bbox["x1"], frame_shape[1] - 1))
        face_bbox["y1"] = max(0, min(face_bbox["y1"], frame_shape[0] - 1))
        face_bbox["x2"] = max(0, min(face_bbox["x2"], frame_shape[1] - 1))
        face_bbox["y2"] = max(0, min(face_bbox["y2"], frame_shape[0] - 1))

        return face_bbox

    def visualize(
        self,
        image: np.ndarray,
        faces: list,
        color: tuple = (0, 255, 0),
        thickness: int = 2,
    ) -> np.ndarray:
        """
        Визуализация обнаруженных лиц на изображении
        """
        vis_image = image.copy()

        for face in faces:
            cv2.rectangle(
                vis_image,
                (face["x1"], face["y1"]),
                (face["x2"], face["y2"]),
                color,
                thickness,
            )

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
