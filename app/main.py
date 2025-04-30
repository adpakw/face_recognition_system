from app.pipeline.people_detector import PeopleDetector
from app.pipeline.pipeline import AutomaticIdentificationPipeline
from app.clients.video_processors.opencv_writer import OpenCVVideoWriter
from app.utils.config_reader import ConfigReader
import os
import time
import tensorflow as tf
def main():
    # people_detector = PeopleDetector()

    # people_detector.test_people_detector_img()

    # people_detector.test_people_detector_video2()

    # pipline = AutomaticIdentificationPipeline({"people_detector": people_detector})
    tf.config.set_visible_devices([], 'GPU')
    baseline_cfg = ConfigReader("app/configs/pipeline_conf.yaml")
    pipline = AutomaticIdentificationPipeline(baseline_cfg)

    start = time.time()
    for vid in os.listdir("cuts"):
        if vid[0] != ".":
            pipline.process_video(
                f"cuts/{vid}", 30, True, True, True, True
            )
    print("time:",time.time()-start)

    # pipline.test_people_detector_video2()

    # video_writer = OpenCVVideoWriter()
    # video_writer.get_result_video("data/input_task.mp4", "results/input_task", preview=True)


if __name__ == "__main__":
    main()
