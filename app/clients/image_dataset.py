import os
import cv2
import numpy as np
import shutil
from typing import Optional, Tuple, Literal
from app.models.yunet import YuNet
from app.models.arcface import ArcFace
from app.models.vgg_face import VGG_16
from app.utils.config_reader import ConfigReader
from app.models.sface import SFace

class ImageDataset:
    def __init__(
        self,
        face_detector: YuNet = YuNet(),
        face_encoder: Literal["ArcFace", "VGG-Face"] = "ArcFace",
        backup_dir: str = "datasets/backup",
        add_persons_dir: str = "datasets/new_persons",
        faces_save_dir: str = "datasets/data",
        features_path: str = "datasets/face_features/feature.npz",
    ):
        """
        Initialize the face dataset manager.

        Args:
            detector: Face detector model (e.g., RetinaFace)
            feature_extractor: Face feature extraction model
            backup_dir: Directory to save original images after processing
            add_persons_dir: Directory with new persons' images
            faces_save_dir: Directory to save extracted faces
            features_path: Path to .npz file with face features
        """
        self.face_detector = face_detector

        self.config_reader = ConfigReader()
        face_recognizer_config = self.config_reader.get_face_recognizer_config()

        self._choose_model(
            model=face_recognizer_config.model, device=face_recognizer_config.device
        )

        self.backup_dir = self._ensure_dir_exists(backup_dir)
        self.add_persons_dir = self._ensure_dir_exists(add_persons_dir)
        self.faces_save_dir = self._ensure_dir_exists(faces_save_dir)
        self._ensure_dir_exists(os.path.dirname(features_path))
        self.features_path = features_path

    @staticmethod
    def _ensure_dir_exists(dir_path: str) -> str:
        """Ensure directory exists, create if not. Returns normalized path."""
        os.makedirs(dir_path, exist_ok=True)
        return os.path.normpath(dir_path)

    def _choose_model(self, model, device):
        if model == "ArcFace":
            self.face_encoder = ArcFace(device)
        elif model == "VGG-Face":
            self.face_encoder = VGG_16(device)
        elif model == "SFace":
            self.face_embedding_extractor = SFace()
        else:
            ValueError

    def add_persons(self) -> None:
        """
        Add new persons to the face recognition database.
        """
        # Extract faces and features from new images
        new_names, new_embs = self._process_new_persons(
            self.add_persons_dir, self.faces_save_dir
        )
        print(new_names, new_embs)

        if len(new_names) == 0:
            print("No new persons found!")
            return

        # Merge with existing features if available
        if os.path.exists(self.features_path):
            old_names, old_embs = self._load_features(self.features_path)
            all_names = np.hstack((old_names, new_names))
            all_embs = np.vstack((old_embs, new_embs))
        else:
            all_names, all_embs = new_names, new_embs
            print("Created new features file")

        # Save combined features
        self._save_features(self.features_path, all_names, all_embs)

        # Backup original images
        self._backup_original_images(self.add_persons_dir, self.backup_dir)

        print("Successfully added new persons!")

    def _process_new_persons(
        self, input_dir: str, output_dir: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process images of new persons: detect faces, extract and save features.
        """
        names = []
        embeddings = []

        for person_name in os.listdir(input_dir):
            person_dir = os.path.join(input_dir, person_name)
            face_dir = os.path.join(output_dir, person_name)
            os.makedirs(face_dir, exist_ok=True)

            for img_name in self._list_image_files(person_dir):
                img_path = os.path.join(person_dir, img_name)
                img = cv2.imread(img_path)

                h, w = img.shape[:2]

                # Detect faces
                bboxes = self.face_detector.detect_faces(img, input_size=(w, h))

                for i, bbox in enumerate(bboxes):
                    # Extract face
                    face = img[bbox["y1"] : bbox["y2"], bbox["x1"] : bbox["x2"]]

                    # Save face
                    face_id = len(os.listdir(face_dir))
                    cv2.imwrite(os.path.join(face_dir, f"{face_id}.jpg"), face)

                    # Extract features
                    if isinstance(self.face_encoder, SFace):
                        embedding = self.face_encoder.get_embeddings(img, (bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]))
                    else:
                        embedding = self.face_encoder.get_embeddings(face)
                    names.append(person_name)
                    embeddings.append(embedding)

        return np.array(names), np.array(embeddings)

    @staticmethod
    def _list_image_files(dir_path: str) -> list:
        """List all image files in directory."""
        return [
            f
            for f in os.listdir(dir_path)
            if f.lower().endswith(("png", "jpg", "jpeg"))
        ]

    @staticmethod
    def _load_features(path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load existing features from .npz file."""
        data = np.load(path)
        return data["images_name"], data["images_emb"]

    @staticmethod
    def _save_features(path: str, names: np.ndarray, embs: np.ndarray) -> None:
        """Save features to .npz file."""
        np.savez_compressed(path, images_name=names, images_emb=embs)

    @staticmethod
    def _backup_original_images(src_dir: str, dst_dir: str) -> None:
        """Move processed images to backup directory."""
        os.makedirs(dst_dir, exist_ok=True)
        for item in os.listdir(src_dir):
            src = os.path.join(src_dir, item)
            dst = os.path.join(dst_dir, item)
            shutil.move(src, dst)

    def read_features(self):
        try:
            data = np.load(self.features_path, allow_pickle=True)
            images_name = data["images_name"]
            images_emb = data["images_emb"]

            return images_name, images_emb
        except:
            return None
