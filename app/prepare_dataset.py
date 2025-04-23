from app.clients.image_dataset import ImageDataset
from app.models.yunet import YuNet
from app.models.arcface import ArcFace


def main():

    image_dataset = ImageDataset()

    image_dataset.add_persons()

    print(image_dataset.read_features())


if __name__ == "__main__":
    main()
