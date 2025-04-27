import json
import numpy as np
from typing import Dict, List, Tuple
from collections import defaultdict
from app.clients.frame_finder import FrameFinder
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class VideoIdentificationEvaluator:
    def __init__(self, known_identities: List[str], frame_finder: FrameFinder):
        """
        :param known_identities: список всех возможных ID людей в видео
        :param frame_finder: экземпляр FrameFinder для доступа к предсказаниям
        """
        self.known_ids = known_identities
        self.frame_finder = frame_finder
        self.id_to_idx = {id_: i for i, id_ in enumerate(known_identities)}

        self.confusion_matrix = np.zeros((len(known_identities), len(known_identities)))
        self.per_person_stats = {
            id_: {"tp": 0, "fp": 0, "fn": 0, "tn": 0} for id_ in known_identities
        }
        self.frame_results = []
        self.total_samples = 0
        self.correct_samples = 0

    def process_video(self, ground_truth: Dict[int, List[str]]) -> Dict:
        """
        Обработка всего видео

        :param ground_truth: словарь {номер_кадра: [список_ID_людей]}
        :return: итоговые метрики
        """
        for frame_num, true_ids in ground_truth.items():
            frame_data = self.frame_finder.find_frames(frame_num)
            if not frame_data:
                continue

            pred_ids = []
            for person in frame_data.get("people", []):
                if "person_id" in person and person["person_id"]:
                    pred_id = person["person_id"]["name"]
                    confidence = person["person_id"].get("confidence", 1.0)
                    pred_ids.append((pred_id, confidence))

            self._process_frame(frame_num, true_ids, pred_ids)

        return self._calculate_final_metrics()

    def _process_frame(
        self, frame_num: int, true_ids: List[str], pred_ids: List[Tuple[str, float]]
    ):
        """Обработка одного кадра"""
        frame_result = {
            "frame": frame_num,
            "true_positives": 0,
            "false_positives": 0,
            "false_negatives": 0,
            "true_negatives": 0,
        }

        for id_ in self.known_ids:
            in_gt = id_ in true_ids
            in_pred = id_ in [pid[0] for pid in pred_ids]

            if in_gt and in_pred:
                self.per_person_stats[id_]["tp"] += 1
                frame_result["true_positives"] += 1
                self.correct_samples += 1
            elif in_gt and not in_pred:
                self.per_person_stats[id_]["fn"] += 1
                frame_result["false_negatives"] += 1
            elif not in_gt and in_pred:
                self.per_person_stats[id_]["fp"] += 1
                frame_result["false_positives"] += 1
            else:
                self.per_person_stats[id_]["tn"] += 1
                frame_result["true_negatives"] += 1
                self.correct_samples += 1

            self.total_samples += 1

        self.frame_results.append(frame_result)

    def _calculate_final_metrics(self) -> Dict:
        """Расчет итоговых метрик"""
        total_tp = sum(stats["tp"] for stats in self.per_person_stats.values())
        total_fp = sum(stats["fp"] for stats in self.per_person_stats.values())
        total_fn = sum(stats["fn"] for stats in self.per_person_stats.values())
        total_tn = sum(stats["tn"] for stats in self.per_person_stats.values())

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )
        accuracy = (
            self.correct_samples / self.total_samples if self.total_samples > 0 else 0
        )

        person_metrics = {}
        for id_, stats in self.per_person_stats.items():
            tp, fp, fn, tn = stats["tp"], stats["fp"], stats["fn"], stats["tn"]
            p = tp / (tp + fp) if (tp + fp) > 0 else 0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0
            f = 2 * (p * r) / (p + r) if (p + r) > 0 else 0
            acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0

            person_metrics[id_] = {
                "accuracy": acc,
                "precision": p,
                "recall": r,
                "f1": f,
                "support": tp + fn,
            }

        return {
            "overall": {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "true_positives": total_tp,
                "false_positives": total_fp,
                "false_negatives": total_fn,
                "true_negatives": total_tn,
                "total_samples": self.total_samples,
                "correct_samples": self.correct_samples,
            },
            "per_person": person_metrics,
            "confusion_matrix": self.confusion_matrix.tolist(),
            "processed_frames": len(self.frame_results),
        }

    def save_results(self, output_path: str):
        """Сохранение результатов в JSON"""
        results = {
            "metrics": self._calculate_final_metrics(),
            "frame_results": self.frame_results,
        }
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

    def print_summary(self, persons):
        """Вывод сводки по метрикам"""
        metrics = self._calculate_final_metrics()

        print("\n=== Результаты оценки ===")
        # print(f"Accuracy: {metrics['overall']['accuracy']:.4f} ({metrics['overall']['correct_samples']}/{metrics['overall']['total_samples']})")
        print(f"Precision: {metrics['overall']['precision']:.4f}")
        print(f"Recall: {metrics['overall']['recall']:.4f}")
        print(f"F1-score: {metrics['overall']['f1']:.4f}")

        print("\nПо персонам:")
        for person in persons:
            m = metrics["per_person"][person]
            print(f"{person}:")
            print(f"  Accuracy: {m['accuracy']:.4f}")
            print(f"  Precision: {m['precision']:.4f}")
            print(f"  Recall: {m['recall']:.4f}")
            print(f"  F1: {m['f1']:.4f}")
            print(f"  Support: {m['support']}")