import cv2
import numpy as np
import torch
import torchvision
from torchvision.models.detection import SSD300_VGG16_Weights


class PeopleDetectorModel:
    def __init__(self, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Инициализация детектора людей с SSD300

        Args:
            device (str): Устройство для вычислений ('cuda' или 'cpu')
        """
        self.device = torch.device(device)
        print(f"Using device: {self.device}")

        self.model = torchvision.models.detection.ssd300_vgg16(
            weights=SSD300_VGG16_Weights.DEFAULT
        )
        self.model.eval().to(self.device)

        self.resize_size = (300, 300)

        self.transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToPILImage(),
                torchvision.transforms.Resize(self.resize_size),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    def detect_people(self, image, confidence_threshold=0.5):
        """
        Детектирование людей на изображении

        Args:
            image (): Путь к изображению
            confidence_threshold (float): Порог уверенности для детекции

        Returns:
            tuple: (изображение с bounding boxes, список bounding boxes)
        """
        image = image.copy()
        orig_height, orig_width = image.shape[:2]
        # self.show_tensor_cv(self.transform(image))

        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            outputs = self.model(image_tensor)

        boxes = outputs[0]["boxes"].cpu().numpy()
        scores = outputs[0]["scores"].cpu().numpy()
        labels = outputs[0]["labels"].cpu().numpy()

        people_boxes = []
        for box, score, label in zip(boxes, scores, labels):
            if label == 1 and score > confidence_threshold:  # label 1 - person
                x1, y1, x2, y2 = box
                scale_x = orig_width / self.resize_size[0]
                scale_y = orig_height / self.resize_size[1]

                x1 = int(x1 * scale_x)
                y1 = int(y1 * scale_y)
                x2 = int(x2 * scale_x)
                y2 = int(y2 * scale_y)

                x1 = max(0, min(x1, orig_width - 1))
                y1 = max(0, min(y1, orig_height - 1))
                x2 = max(0, min(x2, orig_width - 1))
                y2 = max(0, min(y2, orig_height - 1))

                people_boxes.append(((x1, y1, x2, y2), score))

        return people_boxes

    def show_tensor_cv(self, tensor):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        tensor = tensor * std + mean

        tensor = torch.clamp(tensor, 0, 1)
        np_img = tensor.numpy()

        np_img = np.transpose(np_img, (1, 2, 0))

        np_img = (np_img * 255).astype(np.uint8)

        bgr_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)

        cv2.imshow("Tensor Image", bgr_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
