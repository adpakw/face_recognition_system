import xml.etree.ElementTree as ET
from typing import List, Dict, Tuple, Optional
import numpy as np
from dataclasses import dataclass
from app.clients.frame_finder import FrameFinder

@dataclass
class BBox:
    x1: float
    y1: float
    x2: float
    y2: float
    label: str
    frame: int

class IOUCalculator:
    def __init__(self, cvat_xml_path: str, frame_finder: FrameFinder):
        """
        Инициализация калькулятора IoU
        
        :param cvat_xml_path: путь к XML файлу разметки CVAT
        :param frame_finder: экземпляр FrameFinder для доступа к предсказаниям
        """
        self.frame_finder = frame_finder
        self.gt_boxes = self._parse_cvat_xml(cvat_xml_path)
        
    def _parse_cvat_xml(self, xml_path: str) -> Dict[int, List[BBox]]:
        """Парсинг XML файла CVAT"""
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        gt_boxes = {}
        
        for track in root.findall('.//track'):
            label = track.get('label')
            for box in track.findall('box'):
                frame = int(box.get('frame'))
                xtl = float(box.get('xtl'))
                ytl = float(box.get('ytl'))
                xbr = float(box.get('xbr'))
                ybr = float(box.get('ybr'))
                
                bbox = BBox(
                    x1=xtl, y1=ytl,
                    x2=xbr, y2=ybr,
                    label=label,
                    frame=frame
                )
                
                if frame not in gt_boxes:
                    gt_boxes[frame] = []
                gt_boxes[frame].append(bbox)
                
        return gt_boxes
    
    @staticmethod
    def calculate_iou(box1: BBox, box2: BBox) -> float:
        """Расчет IoU для двух bounding box"""
        x_left = max(box1.x1, box2.x1)
        y_top = max(box1.y1, box2.y1)
        x_right = min(box1.x2, box2.x2)
        y_bottom = min(box1.y2, box2.y2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        box1_area = (box1.x2 - box1.x1) * (box1.y2 - box1.y1)
        box2_area = (box2.x2 - box2.x1) * (box2.y2 - box2.y1)
        
        union_area = box1_area + box2_area - intersection_area
        
        iou = intersection_area / union_area
        return iou
    
    def evaluate_frame(self, frame_num: int) -> Dict[str, float]:
        """
        Оценка метрик для одного кадра
        
        :param frame_num: номер кадра
        :return: словарь с метриками {'person_iou', 'face_iou', 'person_count_diff', 'face_count_diff'}
        """
        results = {
            'person_iou': 0.0,
            'face_iou': 0.0,
            'person_count_diff': 0,
            'face_count_diff': 0
        }
        
        gt_boxes = self.gt_boxes.get(frame_num, [])
        pred_data = self.frame_finder.find_frames(frame_num)
        
        if not pred_data:
            return results
            
        pred_boxes = []
        for person in pred_data.get('people', []):
            if 'person' in person:
                pred_boxes.append(BBox(
                    x1=person['person']['x1'],
                    y1=person['person']['y1'],
                    x2=person['person']['x2'],
                    y2=person['person']['y2'],
                    label='person_bbox',
                    frame=frame_num
                ))
            if 'face' in person and person['face'] is not None:
                pred_boxes.append(BBox(
                    x1=person['face']['x1'],
                    y1=person['face']['y1'],
                    x2=person['face']['x2'],
                    y2=person['face']['y2'],
                    label='face_bbox',
                    frame=frame_num
                ))
        
        gt_persons = [b for b in gt_boxes if b.label == 'person_bbox']
        gt_faces = [b for b in gt_boxes if b.label == 'face_bbox']
        pred_persons = [b for b in pred_boxes if b.label == 'person_bbox']
        pred_faces = [b for b in pred_boxes if b.label == 'face_bbox']
        
        results['person_count_diff'] = len(pred_persons) - len(gt_persons)
        results['face_count_diff'] = len(pred_faces) - len(gt_faces)
        
        if gt_persons and pred_persons:
            ious = []
            for gt in gt_persons:
                best_iou = 0.0
                for pred in pred_persons:
                    iou = self.calculate_iou(gt, pred)
                    if iou > best_iou:
                        best_iou = iou
                ious.append(best_iou)
            results['person_iou'] = np.mean(ious) if ious else 0.0
        
        if gt_faces and pred_faces:
            ious = []
            for gt in gt_faces:
                best_iou = 0.0
                for pred in pred_faces:
                    iou = self.calculate_iou(gt, pred)
                    if iou > best_iou:
                        best_iou = iou
                ious.append(best_iou)
            results['face_iou'] = np.mean(ious) if ious else 0.0
        
        return results
    
    def evaluate_all_frames(self) -> Dict[str, List[float]]:
        """
        Оценка метрик для всех кадров с разметкой
        
        :return: словарь с метриками по всем кадрам
        """
        all_results = {
            'person_iou': [],
            'face_iou': [],
            'person_count_diff': [],
            'face_count_diff': []
        }
        
        for frame_num in self.gt_boxes.keys():
            frame_results = self.evaluate_frame(frame_num)
            for key in all_results:
                all_results[key].append(frame_results[key])
        
        return all_results
    
    def get_summary_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Сводная статистика по всем метрикам
        
        :return: словарь с mean, std, min, max для каждой метрики
        """
        all_results = self.evaluate_all_frames()
        stats = {}
        
        for metric, values in all_results.items():
            if not values:  # если нет значений для метрики
                stats[metric] = {
                    'mean': 0.0,
                    'std': 0.0,
                    'min': 0.0,
                    'max': 0.0
                }
                continue
                
            stats[metric] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values))
            }
        
        return stats