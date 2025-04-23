from collections import defaultdict
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple
from app.metrics.iou import BBox,IOUCalculator
from app.clients.frame_finder import FrameFinder
@dataclass
class Detection:
    bbox: BBox
    confidence: float
    is_true_positive: bool = False

class MAPCalculator(IOUCalculator):
    def __init__(self, cvat_xml_path: str, frame_finder: FrameFinder, iou_threshold: float = 0.5):
        """
        Инициализация калькулятора mAP
        
        :param cvat_xml_path: путь к XML файлу разметки CVAT
        :param frame_finder: экземпляр FrameFinder для доступа к предсказаниям
        :param iou_threshold: порог IoU для определения true positive
        """
        super().__init__(cvat_xml_path, frame_finder)
        self.iou_threshold = iou_threshold
        
    def evaluate_for_map(self) -> Dict[str, Dict[str, List[Detection]]]:
        """
        Оценка детекций для расчета mAP
        
        :return: словарь с детекциями по классам
        """
        all_detections = defaultdict(list)
        
        for frame_num in self.gt_boxes.keys():
            gt_boxes = self.gt_boxes.get(frame_num, [])
            pred_data = self.frame_finder.find_frames(frame_num)
            
            if not pred_data:
                continue
                
            gt_by_class = defaultdict(list)
            for box in gt_boxes:
                gt_by_class[box.label].append(box)
            
            pred_boxes = []
            for person in pred_data.get('people', []):
                if 'person' in person:
                    pred_boxes.append(Detection(
                        bbox=BBox(
                            x1=person['person']['x1'],
                            y1=person['person']['y1'],
                            x2=person['person']['x2'],
                            y2=person['person']['y2'],
                            label='person_bbox',
                            frame=frame_num
                        ),
                        confidence=person['person']['confidence']
                    ))
                if 'face' in person and person['face'] is not None:
                    pred_boxes.append(Detection(
                        bbox=BBox(
                            x1=person['face']['x1'],
                            y1=person['face']['y1'],
                            x2=person['face']['x2'],
                            y2=person['face']['y2'],
                            label='face_bbox',
                            frame=frame_num
                        ),
                        confidence=person['face']['confidence']
                    ))
            
            for class_name in gt_by_class.keys():
                class_gt = gt_by_class[class_name]
                class_preds = [d for d in pred_boxes if d.bbox.label == class_name]
                
                class_preds.sort(key=lambda x: x.confidence, reverse=True)
                
                used_gt = [False] * len(class_gt)
                
                for detection in class_preds:
                    best_iou = 0.0
                    best_gt_idx = -1
                    
                    for i, gt_box in enumerate(class_gt):
                        if used_gt[i]:
                            continue
                            
                        iou = self.calculate_iou(detection.bbox, gt_box)
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = i
                    
                    if best_iou >= self.iou_threshold:
                        detection.is_true_positive = True
                        used_gt[best_gt_idx] = True
                    
                    all_detections[class_name].append(detection)
        
        return all_detections
    
    def calculate_ap(self, detections: List[Detection], total_gt: int) -> float:
        """
        Расчет Average Precision для одного класса
        
        :param detections: список детекций класса
        :param total_gt: общее количество GT объектов этого класса
        :return: значение AP
        """
        if not detections or total_gt == 0:
            return 0.0
        
        # Сортируем детекции по confidence
        detections.sort(key=lambda x: x.confidence, reverse=True)
        
        tp = np.zeros(len(detections))
        fp = np.zeros(len(detections))
        
        for i, detection in enumerate(detections):
            if detection.is_true_positive:
                tp[i] = 1
            else:
                fp[i] = 1
        
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        recall = tp_cumsum / total_gt
        precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)
        
        precision = np.maximum.accumulate(precision[::-1])[::-1]
        
        change_indices = np.where(recall[:-1] != recall[1:])[0] + 1
        change_indices = np.insert(change_indices, 0, 0)
        change_indices = np.append(change_indices, len(recall) - 1)
        
        ap = 0.0
        for i in range(len(change_indices) - 1):
            start = change_indices[i]
            end = change_indices[i + 1]
            ap += (recall[end] - recall[start]) * precision[end]
        
        return ap
    
    def calculate_map(self) -> Dict[str, float]:
        """
        Расчет mAP для всех классов
        
        :return: словарь с AP для каждого класса и mAP
        """
        class_detections = self.evaluate_for_map()
        class_stats = {}
        
        total_gt = defaultdict(int)
        for frame_num, boxes in self.gt_boxes.items():
            for box in boxes:
                total_gt[box.label] += 1
        
        aps = []
        for class_name, detections in class_detections.items():
            ap = self.calculate_ap(detections, total_gt.get(class_name, 0))
            class_stats[f'AP_{class_name}'] = ap
            aps.append(ap)
        
        class_stats['mAP'] = np.mean(aps) if aps else 0.0
        
        return class_stats