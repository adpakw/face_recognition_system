import cv2
import numpy as np


class YuNet:
    def __init__(
        self,
        model_path: str = "app/models/weights/face_detection_yunet_2023mar.onnx",
        score_threshold=0.3,
        nms_threshold=0.3,
        top_k=5000,
        device: str = "cpu",
    ):
        """
        Инициализация детектора лиц с YuNet

        Args:
            model_path (str): Путь к ONNX модели YuNet
            input_size (tuple): Размер входного изображения для модели (ширина, высота)
            device (str): Устройство для вычислений ('cuda' или 'cpu')
        """
        self.device = device.lower()

        # Проверяем доступность CUDA
        cuda_available = cv2.cuda.getCudaEnabledDeviceCount() > 0
        if self.device == "cuda" and not cuda_available:
            print("Warning: CUDA requested but not available, falling back to CPU")
            self.device = "cpu"

        # Выбираем бэкенд в зависимости от устройства
        if self.device == "cuda":
            backend = cv2.dnn.DNN_BACKEND_CUDA
            target = cv2.dnn.DNN_TARGET_CUDA
            device_info = "CUDA"
        else:
            backend = cv2.dnn.DNN_BACKEND_OPENCV
            target = cv2.dnn.DNN_TARGET_CPU
            device_info = "CPU"

        print(f"YuNet face detector initialized on {device_info} device")

        # Загрузка модели YuNet
        self.model = cv2.FaceDetectorYN.create(
            model=model_path,
            config="",
            input_size=(320, 320),
            score_threshold=score_threshold,
            nms_threshold=nms_threshold,
            top_k=top_k,
            backend_id=backend,
            target_id=target,
        )

    def detect_faces(self, image, input_size = (320, 320), confidence_threshold=0.7):
        """
        Детектирование лиц в области человека

        Args:
            original_image (np.ndarray): Оригинальное изображение (BGR)
            person_bbox (dict): Bounding box человека в формате {'x1', 'y1', 'x2', 'y2'}
            confidence_threshold (float): Порог уверенности для детекции

        Returns:
            list: Список обнаруженных лиц в формате [{'x1', 'y1', 'x2', 'y2', 'score'}, ...]
                  Координаты приведены к оригинальному изображению
        # """
        self.model.setInputSize(input_size)  # Обновляем размер входного изображения

        # Детекция лиц
        _, faces = self.model.detect(image)
        if faces is None:
            return []

        # Преобразование координат к оригинальному изображению
        detected_faces = []
        for face in faces:
            if face[-1] < confidence_threshold:  # face[-1] - confidence score
                continue

            # Координаты лица относительно вырезанного изображения
            fx1, fy1, fw, fh = map(int, face[:4])
            fx2, fy2 = fx1 + fw, fy1 + fh

            detected_faces.append(
                {"x1": fx1, "y1": fy1, "x2": fx2, "y2": fy2, "score": face[-1]}
            )

        return detected_faces
