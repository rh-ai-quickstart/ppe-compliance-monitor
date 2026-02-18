import cv2
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from ultralytics import YOLO
from collections import defaultdict
from datetime import datetime
import os
import tempfile
from minio_client import download_file, is_minio_enabled
from deep_sort_realtime.deepsort_tracker import DeepSort
from database import (
    init_database, 
    insert_person, 
    update_person_last_seen, 
    insert_observation
)


def get_model_path():
    """
    Get model path, downloading from MinIO if enabled.
    
    When MINIO_ENABLED=true: Downloads model from MinIO to a temp directory.
        - Used for local development with podman-compose
    When MINIO_ENABLED=false: Uses MODEL_PATH environment variable.
        - Used for Kubernetes/OpenShift where files are pre-downloaded to PVC
    """
    if is_minio_enabled():
        bucket = os.getenv("MINIO_MODEL_BUCKET", "models")
        object_name = os.getenv("MINIO_MODEL_KEY", "ppe.pt")
        local_path = os.path.join(tempfile.gettempdir(), "minio_cache", "models", object_name)
        return download_file(bucket, object_name, local_path)
    else:
        # Fallback to local files for development without MinIO
        default_model_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "models", "ppe.pt")
        )
        return os.getenv("MODEL_PATH", default_model_path)


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
        
        # Object tracking and per-person PPE tracking
        self.tracker = None
        self.person_history = {}  # {track_id: {"first_seen": datetime, "last_seen": datetime}}
        self.person_observations = []  # List of per-person PPE observations with timestamps
        self.latest_tracked_persons = []  # Most recent frame's tracked persons with PPE status
        
        # State-change tracking: only record when PPE status changes
        # {track_id: (hardhat, vest, mask)} - last known PPE state for each person
        self.person_last_state = {}

    def setup_components(self):
        """Load models and initialize runtime components."""
        self.cap = cv2.VideoCapture(self.video_path)
        model_path = get_model_path()
        self.anomaly_detector = YOLO(model_path)
        print("Model classes:", self.anomaly_detector.names)
        self.class_names = list(self.anomaly_detector.names.values())
        print("Using class names:", self.class_names)

        model_name = "google/flan-t5-base"
        self.summarizer_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.summarizer_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.summarizer_model.to(self.device)
        
        # Initialize DeepSORT tracker for person tracking
        # max_age: frames to keep track alive without detection
        # n_init: frames before track is confirmed
        self.tracker = DeepSort(max_age=30, n_init=3)
        print("Object tracker initialized (DeepSORT)")
        
        # Initialize SQLite database for persistent storage
        init_database()
        print("SQLite database initialized")

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

    def get_latest_tracked_persons(self):
        """Return the most recent tracked persons with their PPE status."""
        return self.latest_tracked_persons

    def get_person_history(self):
        """Return the history of all tracked persons."""
        return self.person_history

    def get_person_observations(self):
        """Return all person observations (for database storage in Step 2)."""
        return self.person_observations

    def get_unique_person_count(self):
        """Return the count of unique persons tracked so far."""
        return len(self.person_history)

    @staticmethod
    def _boxes_overlap(box1, box2):
        """
        Check if two bounding boxes overlap.
        
        Args:
            box1: (x1, y1, x2, y2) tuple
            box2: (x1, y1, x2, y2) tuple
            
        Returns:
            True if boxes overlap, False otherwise
        """
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Check if boxes don't overlap
        if x2_1 < x1_2 or x2_2 < x1_1:  # One is to the left of the other
            return False
        if y2_1 < y1_2 or y2_2 < y1_1:  # One is above the other
            return False
        
        return True

    @staticmethod
    def _calculate_iou(box1, box2):
        """
        Calculate Intersection over Union (IoU) between two bounding boxes.
        
        Args:
            box1: (x1, y1, x2, y2) tuple
            box2: (x1, y1, x2, y2) tuple
            
        Returns:
            IoU value between 0 and 1
        """
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        if union == 0:
            return 0.0
        
        return intersection / union

    def _associate_ppe_to_person(self, person_bbox, all_detections):
        """
        Determine PPE status for a specific person based on bounding box overlap.
        
        Args:
            person_bbox: (x1, y1, x2, y2) bounding box of the tracked person
            all_detections: list of all YOLO detections in the frame
            
        Returns:
            dict with PPE status: {"hardhat": True/False/None, "vest": True/False/None, "mask": True/False/None}
        """
        status = {
            "hardhat": None,  # None means unknown/not visible
            "vest": None,
            "mask": None
        }
        
        # PPE class mapping
        ppe_mapping = {
            "Hardhat": ("hardhat", True),
            "NO-Hardhat": ("hardhat", False),
            "Safety Vest": ("vest", True),
            "NO-Safety Vest": ("vest", False),
            "Mask": ("mask", True),
            "NO-Mask": ("mask", False),
        }
        
        for det in all_detections:
            class_name = det["class_name"]
            if class_name in ppe_mapping:
                ppe_bbox = det["bbox"]
                
                # Check if PPE bbox overlaps with person bbox
                if self._boxes_overlap(person_bbox, ppe_bbox):
                    ppe_type, ppe_value = ppe_mapping[class_name]
                    # Only update if not already set, or if this detection has higher confidence
                    if status[ppe_type] is None:
                        status[ppe_type] = ppe_value
        
        return status

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

        # --- Object Tracking and PPE Association ---
        # Prepare person detections for the tracker
        # DeepSORT expects: list of ([x, y, w, h], confidence, class_name)
        person_detections_for_tracker = []
        for det in detections:
            if det["class_name"] == "Person":
                x1, y1, x2, y2 = det["bbox"]
                # Convert to [x, y, width, height] format
                w = x2 - x1
                h = y2 - y1
                person_detections_for_tracker.append(
                    ([x1, y1, w, h], det["confidence"], "person")
                )
        
        # Update tracker with person detections
        tracked_persons = []
        now = datetime.now()
        
        if self.tracker is not None and person_detections_for_tracker:
            tracks = self.tracker.update_tracks(person_detections_for_tracker, frame=frame)
            
            for track in tracks:
                if not track.is_confirmed():
                    continue
                
                track_id = track.track_id
                # Get bounding box in (x1, y1, x2, y2) format
                ltrb = track.to_ltrb()
                person_bbox = (int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3]))
                
                # Update person history (first/last seen)
                if track_id not in self.person_history:
                    self.person_history[track_id] = {
                        "first_seen": now,
                        "last_seen": now
                    }
                    # Persist new person to SQLite
                    insert_person(track_id, now, now)
                else:
                    self.person_history[track_id]["last_seen"] = now
                    # Update last_seen in SQLite
                    update_person_last_seen(track_id, now)
                
                # Associate PPE with this person
                ppe_status = self._associate_ppe_to_person(person_bbox, detections)
                
                tracked_person = {
                    "track_id": track_id,
                    "bbox": person_bbox,
                    "hardhat": ppe_status["hardhat"],
                    "vest": ppe_status["vest"],
                    "mask": ppe_status["mask"],
                    "timestamp": now
                }
                tracked_persons.append(tracked_person)
                
                # --- State-Change Recording ---
                # Only record observation if PPE state has changed (or first appearance)
                current_state = (ppe_status["hardhat"], ppe_status["vest"], ppe_status["mask"])
                last_state = self.person_last_state.get(track_id)
                
                # Record if: new person OR state changed
                if last_state is None or last_state != current_state:
                    # Record observation for historical tracking
                    self.person_observations.append({
                        "track_id": track_id,
                        "timestamp": now,
                        "hardhat": ppe_status["hardhat"],
                        "vest": ppe_status["vest"],
                        "mask": ppe_status["mask"],
                        "bbox": person_bbox
                    })
                    # Persist observation to SQLite
                    insert_observation(
                        track_id=track_id,
                        timestamp=now,
                        hardhat=ppe_status["hardhat"],
                        vest=ppe_status["vest"],
                        mask=ppe_status["mask"]
                    )
                    # Update last known state
                    self.person_last_state[track_id] = current_state
                    
                    if last_state is None:
                        print(f"[STATE] New person {track_id}: hardhat={ppe_status['hardhat']}, vest={ppe_status['vest']}, mask={ppe_status['mask']}")
                    else:
                        print(f"[STATE] Person {track_id} state changed: {last_state} -> {current_state}")
                
                # Limit observation history in memory (keep last 1000 observations)
                if len(self.person_observations) > 1000:
                    self.person_observations = self.person_observations[-1000:]
        
        self.latest_tracked_persons = tracked_persons
        # --- End Object Tracking ---

        if self.frame_count % 50 == 0:
            self.latest_summary = self.generate_summary(self.description_buffer)

        self.frame_count += 1
        return frame, detections

    def generate_frames(self):
        """Backward-compatible wrapper to update state for one frame."""
        self.capture_and_update()
