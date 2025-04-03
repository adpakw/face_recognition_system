import os
from typing import Any, Dict, List, Tuple

import cv2
from tqdm import tqdm

from app.clients.json_processor import JsonProcessor
from app.clients.video_processors.opencv import OpenCVVideoProcessor
from app.models.ssd import PeopleDetectorModel


class PeopleDetector:
    def __init__(self, show_progress: bool = True):
        self.detection_model = PeopleDetectorModel()
        self.json_processor = JsonProcessor(output_root="people_detection_results")
        self.default_show_progress = show_progress

    def draw_boxes_with_scores(self, image, boxes_with_scores):
        """Отрисовка bounding boxes и скоров на изображении"""
        for box, score in boxes_with_scores:
            x1, y1, x2, y2 = map(int, box)

            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{score:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)

            cv2.rectangle(image, (x1, y1 - 20), (x1 + w, y1), (0, 255, 0), -1)

            cv2.putText(
                image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1
            )

        return image

    def detect_on_image(self, image_path):
        original_image = cv2.imread(image_path)
        image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

        people_boxes = self.detection_model.detect_people(image)

        result_image = self.draw_boxes_with_scores(original_image.copy(), people_boxes)

        cv2.imshow("Detection Results", result_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        return people_boxes

    def test_people_detector_img(self):
        print(self.detect_on_image("data/dog.jpeg"))
        print(self.detect_on_image("data/input.jpg"))

    def process_frame(
        self, frame, frame_num, timestamp, show_video=False, save_in_json=False
    ):
        """Обработка одного кадра"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        people_boxes = self.detection_model.detect_people(frame_rgb)

        result = {"frame": None, "file_path": None}

        if save_in_json:
            result["file_path"] = self.json_processor.save_detections(
                video_path=self.current_video_path,
                frame_num=frame_num,
                timestamp=timestamp,
                boxes_with_scores=people_boxes,
            )

        if show_video:
            result_frame = self.draw_boxes_with_scores(frame.copy(), people_boxes)
            result["frame"] = result_frame

        return result

    def detect_on_video(
        self,
        video_path: str,
        target_fps: float = 10,
        show_video: bool = False,
        save_in_json: bool = False,
        progress_bar: bool = True,
    ) -> Tuple[List[str], Dict[str, Any]]:
        """
        Обработка видео с детекцией людей

        :param video_path: путь к видеофайлу
        :param target_fps: целевая частота кадров
        :param show_video: показывать ли видео в реальном времени
        :param save_in_json: сохранять ли результаты в JSON
        :param progress_bar: показывать ли прогресс-бар
        :return: (список путей к JSON-файлам, информация о видео)
        """
        self.current_video_path = video_path
        json_files = []

        with OpenCVVideoProcessor(video_path, target_fps=target_fps) as vp:
            video_info = vp.get_video_info()

            total_frames = int(
                video_info["total_frames"]
                / (video_info["original_fps"] / video_info["target_fps"])
            )
            duration = video_info["duration"]

            print(f"\nProcessing video: {os.path.basename(video_path)}")
            print(
                f"Duration: {duration:.2f}s, Frames: {total_frames}, Target FPS: {target_fps}"
            )

            pbar = tqdm(
                total=total_frames,
                desc="Processing frames",
                unit="frame",
                disable=not progress_bar,
            )

            try:
                for frame, timestamp, frame_num in vp.frames():
                    result = self.process_frame(
                        frame,
                        frame_num,
                        timestamp,
                        show_video=show_video,
                        save_in_json=save_in_json,
                    )

                    if result["file_path"] and result["file_path"] not in json_files:
                        json_files.append(result["file_path"])

                    if show_video and result["frame"] is not None:
                        cv2.imshow("Detection Results", result["frame"])
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break

                    pbar.update(1)

            finally:
                pbar.close()
                if save_in_json:
                    self.json_processor.finalize()
                if show_video:
                    cv2.destroyAllWindows()

        print("\nProcessing complete!")
        print(f"Video info: {video_info}")
        print(f"Generated {len(json_files)} JSON files:")
        for file_path in json_files:
            print(f"- {file_path}")

        return json_files, video_info

    def test_people_detector_video(self):
        self.detect_on_video(
            "data/input_task.mp4", 10, show_video=False, save_in_json=True
        )
