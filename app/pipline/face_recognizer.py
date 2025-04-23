from typing import Literal

import cv2
import numpy as np

from app.clients.image_dataset import ImageDataset
from app.models.arcface import ArcFace
from app.utils.config_reader import ConfigReader
from app.models.vgg_face import VGG_16
from app.models.sface import SFace

class FaceRecognizer:
    def __init__(
        self,
        device: str = "cuda",
        model: Literal["ArcFace", "VGG-Face", "SFace"] = None,
        confidence_threshold=0.5,
    ):
        if model is None:
            self.config_reader = ConfigReader()
            face_recognizer_config = self.config_reader.get_face_recognizer_config()
            self._choose_model(
                face_recognizer_config.model, face_recognizer_config.device
            )
        else:
            self._choose_model(model, device)
        self.image_dataset = ImageDataset()

        self.images_names, self.images_embs = self.image_dataset.read_features()

    def _choose_model(self, model, device):
        if model == "ArcFace":
            self.face_embedding_extractor = ArcFace(device)
        elif model == "VGG-Face":
            self.face_embedding_extractor = VGG_16(device)
        elif model == "SFace":
            self.face_embedding_extractor = SFace()
        else:
            ValueError

    def process(self, frame, result_dicts, show_video=False):
        result = {"frame": None, "result_dicts": None}
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output_frame = frame.copy()

        result["result_dicts"] = []
        for person in result_dicts:
            if person["face_bbox"] is None:
                person["person_id"] = None
                result["result_dicts"].append(person)
                continue
            # Вырезаем область лица
            x1, y1, x2, y2 = (
                person["face_bbox"]["x1"],
                person["face_bbox"]["y1"],
                person["face_bbox"]["x2"],
                person["face_bbox"]["y2"],
            )
            face_img = frame_rgb[y1:y2, x1:x2]

            if isinstance(self.face_embedding_extractor, SFace):
                face_emb = self.face_embedding_extractor.get_embeddings(frame, (x1, y1, x2, y2))
                print(face_emb)
            else:
                face_emb = self.face_embedding_extractor.get_embeddings(face_img)

            score, image_name = self._compare_embeddings(face_emb)

            person["person_id"] = {"name": image_name, "score": score}
            result["result_dicts"].append(person)

            if show_video:
                output_frame = self._draw_person_info(output_frame, person)

        if show_video:
            result["frame"] = output_frame

        return result

    def _draw_person_info(self, frame, person):
        """Отображает информацию о персонаже на кадре"""
        # Получаем координаты bbox персонажа
        p_x1, p_y1 = person["person_bbox"]["x1"], person["person_bbox"]["y1"]
        p_x2, p_y2 = person["person_bbox"]["x2"], person["person_bbox"]["y2"]

        # Получаем информацию для отображения
        person_id = person["person_id"]
        if person_id is None:
            return frame

        # Формируем текст для отображения
        text = f"ID: {person_id['name']} ({person_id['score']:.2f})"
        (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)

        # Вычисляем позицию текста (над bbox)
        text_y = max(p_y1 - 10, 20)  # Не выходим за верхнюю границу кадра

        # Рисуем текст
        cv2.putText(
            frame,
            text,
            (p_x2 - w, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

        return frame

    # def _compare_embeddings(self, embedding):
    #     sims = np.dot(embedding, self.images_embs.T)
    #     pare_index = np.argmax(sims)
    #     score = sims[pare_index]
    #     image_name = self.images_names[pare_index]
    #     return score, image_name

    def _compare_embeddings(self, embedding):
        emb = embedding / np.linalg.norm(embedding)
        known_embs = self.images_embs / np.linalg.norm(self.images_embs, axis=1, keepdims=True)
        
        # Косинусное сходство (чем ближе к 1, тем более похожи)
        similarities = np.dot(known_embs, emb)
        pare_index = np.argmax(similarities)
        score = similarities[pare_index]
        image_name = self.images_names[pare_index]
        return score, image_name