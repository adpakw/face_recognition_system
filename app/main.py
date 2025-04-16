from app.pipline.people_detector import PeopleDetector
from app.pipline.pipline import AutomaticIdentificationPipeline
from app.clients.video_processors.opencv_writer import OpenCVVideoWriter

def main():
    people_detector = PeopleDetector()

    # people_detector.test_people_detector_img()

    # people_detector.test_people_detector_video2()

    # pipline = AutomaticIdentificationPipeline({"people_detector": people_detector})
    # pipline = AutomaticIdentificationPipeline(config_path="app/configs/pipline_conf.yaml"
    # )

    # pipline.test_people_detector_video()

    # video_writer = OpenCVVideoWriter()
    # video_writer.get_result_video("data/input_task.mp4", "results/input_task", preview=True)


if __name__ == "__main__":
    main()
    def read_features(self):
        try:
            data = np.load(self.features_path, allow_pickle=True)
            images_name = data["images_name"]
            images_emb = data["images_emb"]

            return images_name, images_emb
        except:
            return None