from app.pipeline.people_detector import PeopleDetector
from app.pipeline.pipeline import AutomaticIdentificationPipeline
from app.clients.video_processors.opencv_writer import OpenCVVideoWriter
from app.utils.config_reader import ConfigReader
import os
import time
def main():
    baseline_cfg = ConfigReader("app/configs/pipeline_conf.yaml")
    pipline = AutomaticIdentificationPipeline(baseline_cfg)

    start = time.time()
    for vid in os.listdir("cuts"):
        if vid[0] != ".":
            pipline.process_video(
                f"cuts/{vid}", 30, True, True, True, True
            )
    print("time:",time.time()-start)


if __name__ == "__main__":
    main()
