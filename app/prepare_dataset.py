from app.clients.image_dataset import ImageDataset
from app.utils.config_reader import ConfigReader
import tensorflow as tf

def main():
    config = ConfigReader("app/configs/pipeline_conf.yaml")
    image_dataset = ImageDataset(config)
    image_dataset.add_persons()

    print(image_dataset.read_features())


if __name__ == "__main__":
    main()
