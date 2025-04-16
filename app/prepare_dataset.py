from app.clients.image_dataset import ImageDataset
from app.models.yunet import YuNet
from app.models.arcface import ArcFace


def main():

    image_dataset = ImageDataset(
        face_detector=YuNet(),
        face_feature_extractor=ArcFace(),
        backup_dir="datasets/backup",
        add_persons_dir="datasets/new_persons",
        faces_save_dir="datasets/data",
        features_path="datasets/face_features/feature.npz",
    )

    image_dataset.add_persons()

    print(image_dataset.read_features())


if __name__ == "__main__":
    main()
