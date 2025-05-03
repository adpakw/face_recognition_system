import os
import cv2
import numpy as np
import shutil
import faiss
from typing import Optional, Tuple, List
from pathlib import Path
from app.utils.config_reader import ConfigReader
from app.pipeline.face_detector import FaceDetector
from app.pipeline.face_recognizer import FaceRecognizer
import torch

class ImageDataset:
    def __init__(self, config: Optional[ConfigReader] = None):
        """
        Класс для работы с датасетом лиц
        """
        if config is None:
            self.config = ConfigReader()
        else:
            self.config = config
        self._init_paths()
        
        self.face_detector = FaceDetector(config)
        self.face_recognizer = FaceRecognizer(config)
        
        self.dimensions = self._get_embedding_dimension()
        self.index = self._init_faiss_index()
        self.names = np.array([])
        
        self._load_existing_data()

    def _init_paths(self):
        """Инициализация путей из конфига"""
        general_config = self.config.get_general_config()
        
        self.backup_dir = self._ensure_dir_exists(
            general_config.backup_dir
        )
        self.add_persons_dir = self._ensure_dir_exists(
            general_config.add_persons_dir
        )
        self.faces_save_dir = self._ensure_dir_exists(
            general_config.faces_save_dir
        )
        
        features_dir = general_config.features_dir
        os.makedirs(features_dir, exist_ok=True)
        
        self.features_path = os.path.join(features_dir, "feature.faiss")
        self.names_path = os.path.join(features_dir, "names.npy")

    def _get_embedding_dimension(self) -> int:
        """Определяет размерность эмбеддингов на основе модели"""
        recognizer_config = self.config.get_pipeline_step_config("face_recognizer")
        model_name = recognizer_config["name"]
        
        if model_name == "VGG-Face":
            return 4096
        elif model_name == "Facenet":
            return 128
        elif model_name == "Facenet512":
            return 512
        elif model_name == "ArcFace":
            return 512
        elif model_name == "SFace":
            return 128

    def _init_faiss_index(self):
        """Инициализирует Faiss индекс"""
        index = faiss.IndexFlatIP(self.dimensions)
        gpu_id = self.config.get_general_config().gpu_id
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        res = faiss.StandardGpuResources()
        res.setTempMemory(3 * 1024 * 1024 * 1024)
        return faiss.index_cpu_to_gpu(res, gpu_id, index)

    def add_persons(self) -> None:
        """
        Добавляет новых людей в базу данных с проверкой размерности
        """
        new_names, new_embs = self._process_new_persons()
        
        if len(new_names) == 0:
            print("No new persons found!")
            return

        new_embs = np.array(new_embs).astype('float32')
        
        if new_embs.size > 0 and new_embs.shape[1] != self.dimensions:
            print(f"Error: Expected embedding dimension {self.dimensions}, got {new_embs.shape[1]}")
            return

        faiss.normalize_L2(new_embs)
        new_names = np.array(new_names)

        if self.index.ntotal == 0:
            print(f"Adding new embeddings with shape: {new_embs.shape}")
            self.index.add(new_embs)
            self.names = new_names
        else:
            if new_embs.shape[1] != self.index.d:
                print(f"Dimension mismatch: index expects {self.index.d}, got {new_embs.shape[1]}")
                return
                
            print(f"Merging {new_embs.shape[0]} new embeddings with existing index")
            self.index.add(new_embs)
            self.names = np.concatenate((self.names, new_names))

        self._save_data()
        self._backup_original_images()
        print(f"Successfully added {len(new_names)} new faces")

    def _process_new_persons(self) -> Tuple[List[str], List[np.ndarray]]:
        """Обрабатывает новые изображения с правильной обработкой путей"""
        names = []
        embeddings = []

        for person_name in os.listdir(self.add_persons_dir):
            person_dir = os.path.join(self.add_persons_dir, person_name)
            
            if not os.path.isdir(person_dir):
                continue
                
            face_dir = os.path.join(self.faces_save_dir, person_name)
            os.makedirs(face_dir, exist_ok=True)

            for img_name in self._list_image_files(person_dir):
                img_path = os.path.join(person_dir, img_name)
                
                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        print(f"Warning: Cannot read image {img_path}")
                        continue

                    h, w = img.shape[:2]
                    detection_result = self.face_detector.process(img, None)
                    
                    for face_info in detection_result["result_dicts"]:
                        if face_info["face_bbox"] is None:
                            continue

                        bbox = face_info["face_bbox"]
                        face = img[bbox["y1"]:bbox["y2"], bbox["x1"]:bbox["x2"]]
                        
                        if face.shape[0] < 10 or face.shape[1] < 10:
                            continue

                        face_id = len(os.listdir(face_dir))
                        face_path = os.path.join(face_dir, f"{face_id}.jpg")
                        cv2.imwrite(face_path, face)

                        embedding = self.face_recognizer.extract_embedding(
                            cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                        )
                        
                        if embedding.shape[0] != self.dimensions:
                            print(f"Warning: Wrong embedding dimension {embedding.shape[0]} (expected {self.dimensions})")
                            continue

                        names.append(person_name)
                        embeddings.append(embedding)

                except Exception as e:
                    print(f"Error processing {img_path}: {str(e)}")
                    continue

        return names, embeddings

    def search(self, query_embedding: np.ndarray, k: int = 1, threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Поиск в базе эмбеддингов"""
        if self.index.ntotal == 0:
            return np.array([]), np.array([]), np.array([])
            
        query_embedding = query_embedding.astype('float32').reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        
        scores, indices = self.index.search(query_embedding, k)
        scores = scores[0]
        indices = indices[0]
        
        mask = scores >= threshold
        return scores[mask], indices[mask], self.names[indices[mask]]

    def read_features(self) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Возвращает все сохраненные эмбеддинги и имена"""
        if os.path.exists(self.features_path) and os.path.exists(self.names_path):
            all_embeddings = self.index.reconstruct_n(0, self.index.ntotal)
            return self.names, all_embeddings
        return None

    @staticmethod
    def _list_image_files(dir_path: str) -> List[str]:
        """Возвращает список полных путей к изображениям"""
        if not os.path.isdir(dir_path):
            return []
            
        return [
            f for f in os.listdir(dir_path)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]

    @staticmethod
    def _ensure_dir_exists(dir_path: str) -> str:
        os.makedirs(dir_path, exist_ok=True)
        return os.path.normpath(dir_path)

    def _backup_original_images(self):
        """Перемещает обработанные изображения в backup"""
        os.makedirs(self.backup_dir, exist_ok=True)
        for item in os.listdir(self.add_persons_dir):
            src = os.path.join(self.add_persons_dir, item)
            dst = os.path.join(self.backup_dir, item)
            shutil.move(src, dst)

    def _save_data(self):
        """Сохраняет данные на диск"""
        cpu_index = faiss.index_gpu_to_cpu(self.index)
        faiss.write_index(cpu_index, self.features_path)
        np.save(self.names_path, self.names)

    def _load_existing_data(self):
        """Загружает существующие данные"""
        if os.path.exists(self.features_path) and os.path.exists(self.names_path):
            try:
                cpu_index = faiss.read_index(self.features_path)
                gpu_id = self.config.get_general_config().gpu_id
                self.index = faiss.index_cpu_to_gpu(
                    faiss.StandardGpuResources(), 
                    gpu_id, 
                    cpu_index
                )
                self.names = np.load(self.names_path)
            except Exception as e:
                print(f"Error loading existing data: {e}")
                self.index = self._init_faiss_index()
                self.names = np.array([])