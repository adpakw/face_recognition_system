import os
import time

from app.metrics.metrics_calculator import IdentificationMetricsCalculator
from app.pipeline.pipeline import AutomaticIdentificationPipeline
from app.utils.config_reader import ConfigReader


def main():
    baseline_cfg = ConfigReader("app/configs/pipeline_conf.yaml")
    pipline = AutomaticIdentificationPipeline(baseline_cfg)

    start = time.time()
    for vid in os.listdir("videos"):
        if vid[0] != ".":
            pipline.process_video(f"videos/{vid}", 30, True, True, True, True)
    print("time:", time.time() - start)

    calculator = IdentificationMetricsCalculator(
        results_dir="results", videos_dir="videos", annotations_dir="annotations"
    )

    calculator.calculate_all_metrics()

    calculator.print_metrics()

    calculator.save_metrics_to_csv("results/identification_metrics.csv")


if __name__ == "__main__":
    main()
