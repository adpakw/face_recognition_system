import json
import os
import csv
import pandas as pd
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import glob
from app.clients.frame_finder import FrameFinder
import cv2


class IdentificationMetricsCalculator:
    def __init__(self, results_dir: str, videos_dir: str, annotations_dir: str):
        """
        Инициализация калькулятора метрик идентификации
        
        :param results_dir: путь к папке с результатами идентификации (baseline_results)
        :param videos_dir: путь к папке с видео (cuts)
        :param annotations_dir: путь к папке с разметкой (annotations)
        """
        self.results_dir = results_dir
        self.videos_dir = videos_dir
        self.annotations_dir = annotations_dir
        self.metrics = {
            'total': defaultdict(int),
            'per_video': {}
        }
        
    def _get_video_fps(self, video_path: str) -> float:
        """
        Получение FPS видео (заглушка - в реальности нужно использовать OpenCV)
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Не удалось открыть видео: {self.video_path}")

        return cap.get(cv2.CAP_PROP_FPS)
    
    def _time_to_frames(self, time_str: str, fps: float) -> int:
        """Конвертация времени в формат HH:MM:SS в номер кадра"""
        h, m, s = map(float, time_str.split(':'))
        total_seconds = h * 3600 + m * 60 + s
        return int(total_seconds * fps) + 1
    
    def _load_annotations(self, video_name: str) -> List[Tuple[int, int, List[str]]]:
        """Загрузка разметки из CSV файла с использованием pandas"""
        csv_path = os.path.join(self.annotations_dir, f"{video_name}.csv")
        if not os.path.exists(csv_path):
            return []
        
        fps = self._get_video_fps(os.path.join(self.videos_dir, f"{video_name}.mp4"))
        annotations = []
        
        try:
            df = pd.read_csv(csv_path)
            
            if 'from' not in df.columns or 'to' not in df.columns or 'persons' not in df.columns:
                raise ValueError("CSV файл не содержит необходимых колонок: 'from', 'to', 'persons'")
            
            df['start_frame'] = df['from'].apply(lambda x: self._time_to_frames(x, fps))
            df['end_frame'] = df['to'].apply(lambda x: self._time_to_frames(x, fps))
            
            df['persons_list'] = df['persons'].apply(
                lambda x: [p.strip() for p in x.split(',')] if pd.notna(x) and isinstance(x, str) else []
            )
            
            annotations = list(zip(df['start_frame'], df['end_frame'], df['persons_list']))
            
        except Exception as e:
            print(f"Ошибка при чтении файла {csv_path}: {str(e)}")
            
        return annotations
    
    def _get_ground_truth(self, frame_number: int, annotations: List[Tuple[int, int, List[str]]]) -> List[str]:
        """Получение ground truth для конкретного кадра"""
        for start, end, persons in annotations:
            if start <= frame_number <= end:
                return persons
        return []
    
    def _process_video(self, video_name: str):
        """Обработка одного видео и расчет метрик"""
        video_metrics = defaultdict(int)
        annotations = self._load_annotations(video_name)
        if not annotations:
            print(f"Не найдены аннотации для видео {video_name}")
            return
        
        video_results_dir = os.path.join(self.results_dir, video_name)
        if not os.path.exists(video_results_dir):
            print(f"Не найдены результаты для видео {video_name}")
            return
            
        json_files = glob.glob(os.path.join(video_results_dir, "*.json"))
        if not json_files:
            print(f"Не найдены JSON файлы для видео {video_name}")
            return
            
        frame_finder = FrameFinder(json_files)
        
        for start_frame, end_frame, _ in annotations:
            for frame_number in range(start_frame, end_frame + 1):
                frame_data = frame_finder.find_frames(frame_number)
                if frame_data is None or 'people' not in frame_data:
                    continue
                    
                ground_truth = self._get_ground_truth(frame_number, annotations)
                detected_people = frame_data['people']
                
                predicted_names = set()
                for person in detected_people:
                    if 'person_id' in person and person['person_id'] is not None:
                        person_id = person['person_id']
                        if 'name' in person_id and person_id['name'].lower() != 'unknown':
                            predicted_names.add(person_id['name'])
                
                gt_set = set(ground_truth)
                
                for name in gt_set:
                    if name in predicted_names:
                        video_metrics['tp'] += 1
                    else:
                        video_metrics['fn'] += 1
                        
                for name in predicted_names:
                    if name not in gt_set:
                        video_metrics['fp'] += 1
        
        tp = video_metrics['tp']
        fp = video_metrics['fp']
        fn = video_metrics['fn']
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
        
        video_metrics.update({
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'accuracy': accuracy
        })
        
        self.metrics['per_video'][video_name] = video_metrics
        
        for key in ['tp', 'fp', 'fn']:
            self.metrics['total'][key] += video_metrics[key]
    
    def calculate_all_metrics(self):
        """Расчет метрик для всех видео"""
        annotation_files = glob.glob(os.path.join(self.annotations_dir, "*.csv"))
        
        for ann_file in annotation_files:
            video_name = os.path.splitext(os.path.basename(ann_file))[0]
            self._process_video(video_name)
        
        tp = self.metrics['total']['tp']
        fp = self.metrics['total']['fp']
        fn = self.metrics['total']['fn']
        
        total_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        total_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        total_f1 = 2 * (total_precision * total_recall) / (total_precision + total_recall) if (total_precision + total_recall) > 0 else 0
        total_accuracy = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
        
        self.metrics['total'].update({
            'precision': total_precision,
            'recall': total_recall,
            'f1': total_f1,
            'accuracy': total_accuracy
        })
    
    def get_metrics(self) -> Dict:
        """Возвращает рассчитанные метрики"""
        return self.metrics
    
    def print_metrics(self):
        """Вывод метрик в читаемом формате"""
        print("Общие метрики:")
        total = self.metrics['total']
        print(f"Accuracy: {total['accuracy']:.4f}")
        print(f"Precision: {total['precision']:.4f}")
        print(f"Recall: {total['recall']:.4f}")
        print(f"F1-score: {total['f1']:.4f}")
        print(f"TP: {total['tp']}, FP: {total['fp']}, FN: {total['fn']}")
        print("\nМетрики по видео:")
        
        for video_name, metrics in self.metrics['per_video'].items():
            print(f"\nВидео: {video_name}")
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall: {metrics['recall']:.4f}")
            print(f"F1-score: {metrics['f1']:.4f}")
            print(f"TP: {metrics['tp']}, FP: {metrics['fp']}, FN: {metrics['fn']}")
    
    def save_metrics_to_csv(self, output_path: str):
        """Сохранение метрик в CSV файл"""
        try:
            data = []
            for video_name, metrics in self.metrics['per_video'].items():
                row = {
                    'video': video_name,
                    'accuracy': metrics['accuracy'],
                    'precision': metrics['precision'],
                    'recall': metrics['recall'],
                    'f1': metrics['f1'],
                    'tp': metrics['tp'],
                    'fp': metrics['fp'],
                    'fn': metrics['fn']
                }
                data.append(row)
            
            total = self.metrics['total']
            data.append({
                'video': 'TOTAL',
                'accuracy': total['accuracy'],
                'precision': total['precision'],
                'recall': total['recall'],
                'f1': total['f1'],
                'tp': total['tp'],
                'fp': total['fp'],
                'fn': total['fn']
            })
            
            df = pd.DataFrame(data)
            df.to_csv(output_path, index=False)
            print(f"Метрики успешно сохранены в {output_path}")
            
        except Exception as e:
            print(f"Ошибка при сохранении метрик в CSV: {str(e)}")