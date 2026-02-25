import cv2
from collections import defaultdict
from datetime import datetime

from database import (
    init_database,
    insert_person,
    update_person_last_seen,
    insert_observation,
)
from logger import get_logger
from runtime import Runtime

log = get_logger(__name__)


class MultiModalAIDemo:
    """Core video analysis pipeline for detection, summaries, and chat context."""

    def __init__(self, video_path):
        """Initialize the demo with a local video path."""
        self.video_path = video_path
        self.cap = None
        self.runtime = None
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
        self.person_history = {}  # {track_id: {"first_seen": datetime, "last_seen": datetime}}
        self.person_observations = []  # List of per-person PPE observations with timestamps
        self.latest_tracked_persons = []  # Most recent frame's tracked persons with PPE status

        # State-change tracking: only record when PPE status changes
        # {track_id: (hardhat, vest, mask)} - last known PPE state for each person
        self.person_last_state = {}

        # Throttle update_person_last_seen: only write to DB every N frames (not every frame)
        self._last_seen_update_interval = (
            30  # Update last_seen in DB every ~1 sec at 30fps
        )
        self._frames_since_last_seen_update = 0

    def setup_components(self):
        """Load models and initialize runtime components."""
        self.cap = cv2.VideoCapture(self.video_path)
        self.runtime = Runtime()
        log.info(f"Model classes: {self.runtime.CLASSES}")
        self.class_names = list(self.runtime.CLASSES.values())
        log.info(f"Using class names: {self.class_names}")

        # Initialize DeepSORT tracker for person tracking (works with OVMS detections).
        # We use DeepSORT instead of Ultralytics because: (1) OVMS already does detection;
        # using Ultralytics would duplicate inference. (2) Ultralytics tracking is bundled
        # with its detection in model.track(); it cannot accept external detections.
        from deep_sort_realtime.deepsort_tracker import DeepSort

        self.tracker = DeepSort(max_age=30, n_init=3)
        print("Object tracking enabled (DeepSORT)")

        # Initialize PostgreSQL database for persistent storage
        init_database()
        print("PostgreSQL database initialized")

    def format_detection_description(
        self, detections_class_count: dict[str, int]
    ) -> str:
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
            if detections_class_count[item] > 0:
                description += f"{item}: {detections_class_count[item]}, "

        return description.rstrip(", ")

    def append_description(self, description):
        """Append a description to the rolling buffer with bounds."""
        self.description_buffer.append(description)
        if len(self.description_buffer) > 50:
            self.description_buffer.pop(0)

    def generate_image_description(self, frame):
        """Run detection on a frame and return a short description string."""
        detections = self.runtime.run(frame)
        detections_class_count = defaultdict(int)

        for d in detections:
            if d.class_name in ["Safety Cone", "Safety Vest", "machinery", "vehicle"]:
                continue
            detections_class_count[d.class_name] += 1

        description = self.format_detection_description(detections_class_count)
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

    @staticmethod
    def _boxes_overlap(box1, box2):
        """Check if two bounding boxes overlap."""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        if x2_1 < x1_2 or x2_2 < x1_1:
            return False
        if y2_1 < y1_2 or y2_2 < y1_1:
            return False
        return True

    def _associate_ppe_to_person(self, person_bbox, all_detections):
        """
        Determine PPE status for a specific person based on bounding box overlap.

        Returns dict with PPE status: {"hardhat": True/False/None, "vest": True/False/None, "mask": True/False/None}
        """
        status = {"hardhat": None, "vest": None, "mask": None}
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
                if self._boxes_overlap(person_bbox, ppe_bbox):
                    ppe_type, ppe_value = ppe_mapping[class_name]
                    if status[ppe_type] is None:
                        status[ppe_type] = ppe_value
        return status

    def capture_and_update(self, resize_to=None):
        """Capture a frame, optionally resize, update detection state, and return frame data."""
        success, frame = self.cap.read()
        if not success:
            log.debug("End of video reached, looping back to start")
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return None, []

        if resize_to:
            frame = cv2.resize(frame, resize_to)

        # Run OVMS detection (via Runtime)
        runtime_detections = self.runtime.run(frame)

        # Build detections list and collect Person detections for DeepSORT
        detections = []
        counts = defaultdict(int)
        person_detections_for_tracker = []  # ([left, top, w, h], confidence, "Person")

        for d in runtime_detections:
            if d.class_name in ["Safety Cone", "Safety Vest", "machinery", "vehicle"]:
                continue
            counts[d.class_name] += 1
            x, y, w, h = d.bbox
            x1 = round(x * d.scale)
            y1 = round(y * d.scale)
            x2 = round((x + w) * d.scale)
            y2 = round((y + h) * d.scale)
            detections.append(
                {
                    "bbox": (x1, y1, x2, y2),
                    "confidence": d.confidence,
                    "class_id": d.class_id,
                    "class_name": d.class_name,
                }
            )
            # Only track Person detections
            if d.class_name == "Person" and d.confidence > 0.5:
                width = x2 - x1
                height = y2 - y1
                if width > 0 and height > 0:
                    person_detections_for_tracker.append(
                        ([x1, y1, width, height], d.confidence, "Person")
                    )

        # Run DeepSORT to get track IDs for persons
        tracked_person_boxes = {}  # {track_id: (x1, y1, x2, y2)}
        if person_detections_for_tracker:
            tracks = self.tracker.update_tracks(
                person_detections_for_tracker, frame=frame
            )
            for track in tracks:
                if not track.is_confirmed():
                    continue
                track_id = int(track.track_id)
                ltrb = track.to_ltrb()  # [left, top, right, bottom]
                if ltrb is not None:
                    x1, y1, x2, y2 = map(int, ltrb)
                    tracked_person_boxes[track_id] = (x1, y1, x2, y2)

        self.latest_detection = counts
        description = self.format_detection_description(counts)
        self.append_description(description)

        # Add track_id to Person detections in output (for display)
        for det in detections:
            if det["class_name"] == "Person":
                # Find matching track by bbox overlap
                for tid, pbox in tracked_person_boxes.items():
                    if self._boxes_overlap(det["bbox"], pbox):
                        det["track_id"] = tid
                        break

        # --- Object Tracking and PPE Association ---
        tracked_persons = []
        now = datetime.now()
        self._frames_since_last_seen_update += 1
        do_last_seen_db_update = (
            self._frames_since_last_seen_update >= self._last_seen_update_interval
        )

        for track_id, person_bbox in tracked_person_boxes.items():
            # Update person history (first/last seen) - in-memory every frame
            if track_id not in self.person_history:
                self.person_history[track_id] = {
                    "first_seen": now,
                    "last_seen": now,
                }
                insert_person(track_id, now, now)
            else:
                self.person_history[track_id]["last_seen"] = now
                # Throttle DB writes: only update last_seen in PostgreSQL periodically
                if do_last_seen_db_update:
                    update_person_last_seen(track_id, now)

        if do_last_seen_db_update:
            self._frames_since_last_seen_update = 0

            # Associate PPE with this person
            ppe_status = self._associate_ppe_to_person(person_bbox, detections)
            tracked_person = {
                "track_id": track_id,
                "bbox": person_bbox,
                "hardhat": ppe_status["hardhat"],
                "vest": ppe_status["vest"],
                "mask": ppe_status["mask"],
                "timestamp": now,
            }
            tracked_persons.append(tracked_person)

            # --- State-Change Recording ---
            current_state = (
                ppe_status["hardhat"],
                ppe_status["vest"],
                ppe_status["mask"],
            )
            last_state = self.person_last_state.get(track_id)

            if last_state is None or last_state != current_state:
                self.person_observations.append(
                    {
                        "track_id": track_id,
                        "timestamp": now,
                        "hardhat": ppe_status["hardhat"],
                        "vest": ppe_status["vest"],
                        "mask": ppe_status["mask"],
                        "bbox": person_bbox,
                    }
                )
                insert_observation(
                    track_id=track_id,
                    timestamp=now,
                    hardhat=ppe_status["hardhat"],
                    vest=ppe_status["vest"],
                    mask=ppe_status["mask"],
                )
                self.person_last_state[track_id] = current_state

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
