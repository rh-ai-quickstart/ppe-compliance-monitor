from flask import Flask, Response, request, jsonify, Blueprint
from flask_cors import CORS
import cv2
import numpy as np
from multimodel import MultiModalAIDemo
from llm import LLMChat
import os
import tempfile
from minio_client import download_file
from logger import get_logger

log = get_logger(__name__)


app = Flask(__name__)
api = Blueprint("api", __name__, url_prefix="/api")

cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000")
if cors_origins.strip() == "*":
    cors_allowed_origins = "*"
else:
    cors_allowed_origins = [
        origin.strip() for origin in cors_origins.split(",") if origin.strip()
    ]

CORS(app, resources={r"/*": {"origins": cors_allowed_origins}})


def get_video_path():
    """
    Get video path, downloading from MinIO if enabled.

    When MINIO_ENABLED=true: Downloads video from MinIO to a temp directory.
        - Used for local development with podman-compose
    When MINIO_ENABLED=false: Uses VIDEO_PATH environment variable.
        - Used for Kubernetes/OpenShift where files are pre-downloaded to PVC
    """
    bucket = os.getenv("MINIO_VIDEO_BUCKET", "data")
    object_name = os.getenv("MINIO_VIDEO_KEY", "combined-video-no-gap-rooftop.mp4")
    local_path = os.path.join(
        tempfile.gettempdir(), "minio_cache", "video", object_name
    )
    log.info(f"Downloading video from MinIO to {local_path}")
    return download_file(bucket, object_name, local_path)


video_path = get_video_path()
demo = MultiModalAIDemo(video_path)
demo.setup_components()
log.info("MultiModalAIDemo initialized and components ready")

llm_chat = LLMChat()

latest_description = "Initializing..."
latest_summary = "Processing video..."


def generate_frames():
    """Stream processed video frames with annotated detections."""
    global latest_description, latest_summary
    while True:
        frame, detections = demo.capture_and_update(resize_to=(1920, 1080))
        if frame is None:
            continue

        annotated_frame = frame.copy()
        for detection in detections:
            x1, y1, x2, y2 = detection["bbox"]
            conf = detection["confidence"]
            currentClass = detection["class_name"]

            if conf > 0.5:
                if currentClass == "Person":
                    color = (0, 255, 255)  # Cyan for person
                elif currentClass in ["NO-Hardhat", "NO-Safety Vest", "NO-Mask"]:
                    color = (0, 0, 255)  # Red for non-compliance
                elif currentClass in ["Hardhat", "Safety Vest", "Mask"]:
                    color = (0, 255, 0)  # Green for compliance
                else:
                    color = (255, 255, 0)  # Yellow for other objects

                # Draw futuristic box
                cv2.line(annotated_frame, (x1, y1), (x1 + 20, y1), color, 2)
                cv2.line(annotated_frame, (x1, y1), (x1, y1 + 20), color, 2)
                cv2.line(annotated_frame, (x2, y1), (x2 - 20, y1), color, 2)
                cv2.line(annotated_frame, (x2, y1), (x2, y1 + 20), color, 2)
                cv2.line(annotated_frame, (x1, y2), (x1 + 20, y2), color, 2)
                cv2.line(annotated_frame, (x1, y2), (x1, y2 - 20), color, 2)
                cv2.line(annotated_frame, (x2, y2), (x2 - 20, y2), color, 2)
                cv2.line(annotated_frame, (x2, y2), (x2, y2 - 20), color, 2)

                # Add glow effect
                glow = np.zeros(annotated_frame.shape, dtype=np.uint8)
                cv2.rectangle(glow, (x1, y1), (x2, y2), color, 5)
                glow = cv2.GaussianBlur(glow, (21, 21), 0)
                annotated_frame = cv2.addWeighted(annotated_frame, 1, glow, 0.5, 0)

                # Add text with futuristic style (include track_id for Person)
                label = f"{currentClass} {conf:.2f}"
                if currentClass == "Person" and detection.get("track_id") is not None:
                    label = f"Person #{detection['track_id']} {conf:.2f}"
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
                cv2.rectangle(
                    annotated_frame,
                    (x1, y1 - text_size[1] - 10),
                    (x1 + text_size[0], y1),
                    color,
                    -1,
                )
                cv2.putText(
                    annotated_frame,
                    label,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 0),
                    2,
                )

        if demo.description_buffer:
            latest_description = demo.description_buffer[-1]
        latest_summary = demo.latest_summary or latest_summary

        ret, buffer = cv2.imencode(".jpg", annotated_frame)
        frame = buffer.tobytes()
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


@api.route("/video_feed")
def video_feed():
    """Video streaming route."""
    return Response(
        generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@api.route("/")
def api_root():
    """Simple health response for the API root."""
    return jsonify({"status": "ok"})


@api.route("/latest_info")
def latest_info():
    """Return the latest description and summary."""
    global latest_description, latest_summary
    demo.capture_and_update()
    if demo.description_buffer:
        latest_description = demo.description_buffer[-1]
    latest_summary = demo.latest_summary or latest_summary
    return jsonify({"description": latest_description, "summary": latest_summary})


@api.route("/chat", methods=["POST"])
def chat():
    """Answer a question based on latest description and summary.

    Supports streaming via Server-Sent Events when ``stream=true`` is sent in
    the JSON body.  An optional ``session_id`` field enables per-session
    conversation memory (defaults to ``"default"``).
    """
    global latest_description, latest_summary
    data = request.get_json(silent=True) or {}
    question = (data.get("question") or "").strip()
    if not question:
        return jsonify({"error": "Field 'question' is required."}), 400

    session_id = (data.get("session_id") or "default").strip()
    context = latest_description + " " + latest_summary

    answer = llm_chat.ask_question(
        question=question,
        context=context,
        session_id=session_id,
    )
    return jsonify({"answer": answer})


app.register_blueprint(api)

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8888"))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=port, debug=debug)
