from app.models.ssd import PeopleDetectorModel
from app.clients.video_processors.opencv import OpenCVVideoProcessor
import cv2


class PeopleDetector:
    def __init__(self):
        self.detection_model = PeopleDetectorModel()

    def draw_boxes_with_scores(self, image, boxes_with_scores):
        """Отрисовка bounding boxes и скоров на изображении"""
        for box, score in boxes_with_scores:
            x1, y1, x2, y2 = map(int, box)
            
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            label = f"{score:.2f}"
            (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            
            cv2.rectangle(image, (x1, y1 - 20), (x1 + w, y1), (0, 255, 0), -1)
            
            cv2.putText(image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        
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


    def detect_on_video(self, video_path, target_fps):
        with OpenCVVideoProcessor(video_path, target_fps=target_fps) as vp:
            print("Video info:", vp.get_video_info())
            
            for frame, timestamp, frame_num in vp.frames():
                # print(f"Frame #{frame_num} at {timestamp:.2f}s")
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                people_boxes = self.detection_model.detect_people(frame_rgb)
                result_frame = self.draw_boxes_with_scores(frame.copy(), people_boxes)
                cv2.imshow("5 FPS", result_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

    def test_people_detector_video(self):
        self.detect_on_video("data/input_task.mp4", 10)

    def save_in_json(self, video_path, data):
        pass