from app.models.resnet import IResNet, IBasicBlock
import torch
import torchvision


class ArcFace:
    def __init__(
        self,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        arch: str = "r50",
    ):
        self.device = torch.device(device)
        print(f"Using device: {self.device}")

        if arch == "r50":
            self.model = IResNet(IBasicBlock, [3, 4, 14, 3])
            path = "app/models/weights/arcface_r50.pth"

        weight = torch.load(path, map_location=self.device)

        self.model.load_state_dict(weight)
        self.model.to(self.device).eval()

        self.resize_size = (112, 112)

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

    def get_embeddings(self, image):
        image = image.copy()

        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            emb_img_face = self.model(image_tensor).cpu().numpy().flatten()

        return emb_img_face
