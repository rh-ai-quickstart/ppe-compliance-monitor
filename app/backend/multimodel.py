import cv2
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from ultralytics import YOLO
from collections import defaultdict
import os


class MultiModalAIDemo:
    """Core video analysis pipeline for detection, summaries, and chat context."""

    def __init__(self, video_path):
        """Initialize the demo with a local video path."""
        self.video_path = video_path
        self.cap = None
        self.anomaly_detector = None
        self.summarizer = None
        self.description_buffer = []
        self.frame_count = 0
        self.class_names = [
            "Hardhat",
            "Mask",
            "NO-Hardhat",
            "NO-Mask",
            "NO-Safety Vest",
            "Person",
            "Safety Cone",
            "Safety Vest",
            "machinery",
            "vehicle",
        ]
        self.ppe_stats = defaultdict(lambda: {"compliant": 0, "non_compliant": 0})
        self.latest_detection = defaultdict(int)
        self.latest_summary = ""

    def setup_components(self):
        """Load models and initialize runtime components."""
        self.cap = cv2.VideoCapture(self.video_path)
        default_model_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "models", "ppe.pt")
        )
        model_path = os.getenv("MODEL_PATH", default_model_path)
        self.anomaly_detector = YOLO(model_path)
        print("Model classes:", self.anomaly_detector.names)
        self.class_names = list(self.anomaly_detector.names.values())
        print("Using class names:", self.class_names)

        model_name = "google/flan-t5-base"
        self.summarizer_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.summarizer_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.summarizer_model.to(self.device)

    def format_detection_description(self, detections):
        """Build a short, human-readable description from detection counts."""
        description = "Detected: "
        for item in [
            "Person",
            "Hardhat",
            "Safety Vest",
            "Mask",
            "NO-Hardhat",
            "NO-Safety Vest",
            "NO-Mask",
        ]:
            if detections[item] > 0:
                description += f"{item}: {detections[item]}, "

        return description.rstrip(", ")

    def append_description(self, description):
        """Append a description to the rolling buffer with bounds."""
        self.description_buffer.append(description)
        if len(self.description_buffer) > 50:
            self.description_buffer.pop(0)

    def generate_image_description(self, frame):
        """Run detection on a frame and return a short description string."""
        results = self.anomaly_detector(frame, stream=True)
        detections = defaultdict(int)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                currentClass = self.class_names[cls]
                detections[currentClass] += 1

        description = self.format_detection_description(detections)
        self.append_description(description)
        return description

    def generate_summary(self, descriptions):
        """Summarize PPE compliance over a list of detection descriptions."""
        total_stats = defaultdict(int)
        frame_count = len(descriptions)

        for desc in descriptions:
            for item in [
                "Person",
                "Hardhat",
                "Safety Vest",
                "Mask",
                "NO-Hardhat",
                "NO-Safety Vest",
                "NO-Mask",
            ]:
                count = desc.count(item)
                total_stats[item] += count

        summary = "Safety Trends Summary:\n\n"
        summary += f"Total observations: {frame_count} frames\n\n"

        if total_stats["Person"] > 0:
            hardhat_compliance = (
                total_stats["Hardhat"]
                / (total_stats["Hardhat"] + total_stats["NO-Hardhat"])
                if (total_stats["Hardhat"] + total_stats["NO-Hardhat"]) > 0
                else 0
            )
            vest_compliance = (
                total_stats["Safety Vest"]
                / (total_stats["Safety Vest"] + total_stats["NO-Safety Vest"])
                if (total_stats["Safety Vest"] + total_stats["NO-Safety Vest"]) > 0
                else 0
            )
            mask_compliance = (
                total_stats["Mask"] / (total_stats["Mask"] + total_stats["NO-Mask"])
                if (total_stats["Mask"] + total_stats["NO-Mask"]) > 0
                else 0
            )

            summary += "Compliance rates:\n"
            summary += f"\n• Hardhat compliance: {hardhat_compliance:.2%} ({total_stats['Hardhat']} out of {total_stats['Hardhat'] + total_stats['NO-Hardhat']} detections)"
            summary += f"\n• Safety Vest compliance: {vest_compliance:.2%} ({total_stats['Safety Vest']} out of {total_stats['Safety Vest'] + total_stats['NO-Safety Vest']} detections)"
            summary += f"\n• Mask compliance: {mask_compliance:.2%} ({total_stats['Mask']} out of {total_stats['Mask'] + total_stats['NO-Mask']} detections)"

            overall_compliance = (
                hardhat_compliance + vest_compliance + mask_compliance
            ) / 3
            summary += f"\n\nOverall PPE compliance: {overall_compliance:.2%}\n"

            summary += "\nRecommendations:\n"
            if overall_compliance < 0.8:
                summary += f"\n• Critical: Immediate action required. {total_stats['NO-Hardhat'] + total_stats['NO-Safety Vest'] + total_stats['NO-Mask']} PPE violations detected."
                summary += "\n• Conduct an emergency safety briefing."
                summary += "\n• Increase on-site safety inspections."
            elif overall_compliance < 0.95:
                summary += f"\n• Warning: Improvement needed. {total_stats['NO-Hardhat'] + total_stats['NO-Safety Vest'] + total_stats['NO-Mask']} PPE violations detected."
                summary += "\n• Reinforce PPE policies through team meetings."
                summary += "\n• Consider additional PPE training sessions."
            else:
                summary += (
                    "\n• Good compliance observed. Maintain current safety protocols."
                )
                summary += "\n• Continue regular safety reminders and training."
        else:
            summary += "\n• No people detected in the observed period."
            summary += "\n• Check camera positioning and functionality."

        return summary

    def get_latest_detection(self):
        """Return the most recent detection counts."""
        return self.latest_detection

    def get_latest_summary(self):
        """Return the most recent summary."""
        return self.latest_summary

    def capture_and_update(self, resize_to=None):
        """Capture a frame, optionally resize, update detection state, and return frame data."""
        success, frame = self.cap.read()
        if not success:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return None, []

        if resize_to:
            frame = cv2.resize(frame, resize_to)

        detections = []
        counts = defaultdict(int)
        results = self.anomaly_detector(frame, stream=True)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = self.class_names[cls]
                counts[class_name] += 1
                detections.append(
                    {
                        "bbox": (x1, y1, x2, y2),
                        "confidence": conf,
                        "class_id": cls,
                        "class_name": class_name,
                    }
                )

        self.latest_detection = counts
        description = self.format_detection_description(counts)
        self.append_description(description)

        if self.frame_count % 50 == 0:
            self.latest_summary = self.generate_summary(self.description_buffer)

        self.frame_count += 1
        return frame, detections

    def generate_frames(self):
        """Backward-compatible wrapper to update state for one frame."""
        self.capture_and_update()
