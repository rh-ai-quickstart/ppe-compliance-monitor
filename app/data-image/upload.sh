#!/bin/sh
set -e

echo "=== PPE Compliance Monitor Data Uploader ==="

# Set mc config directory to writable location (for OpenShift compatibility)
export MC_CONFIG_DIR=/tmp/.mc

# Wait for MinIO to be ready
echo "Waiting for MinIO to be ready..."
until mc alias set myminio "${MINIO_ENDPOINT}" "${MINIO_ACCESS_KEY}" "${MINIO_SECRET_KEY}" 2>/dev/null; do
	echo "MinIO not ready, retrying in 2 seconds..."
	sleep 2
done
echo "MinIO connection established"

# Create buckets (ignore if already exists)
echo "Creating buckets..."
mc mb --ignore-existing myminio/models
mc mb --ignore-existing myminio/data
echo "Buckets ready"

# Upload model (only if not exists)
echo "Checking model file..."
if ! mc stat myminio/models/ppe.pt >/dev/null 2>&1; then
	echo "Uploading model (ppe.pt)..."
	mc cp /upload/models/ppe.pt myminio/models/
	echo "Model uploaded successfully"
else
	echo "Model already exists, skipping"
fi

# Upload OpenVINO model (only if not exists)
echo "Checking OpenVINO model files..."
if ! mc stat myminio/models/ppe/1/ppe.xml >/dev/null 2>&1; then
	echo "Uploading OpenVINO model (ppe/1/)..."
	mc cp --recursive /upload/models/ppe/ myminio/models/ppe/
	echo "OpenVINO model uploaded successfully"
else
	echo "OpenVINO model already exists, skipping"
fi

# Upload video (only if not exists)
echo "Checking video file..."
if ! mc stat myminio/data/combined-video-no-gap-rooftop.mp4 >/dev/null 2>&1; then
	echo "Uploading video (combined-video-no-gap-rooftop.mp4)..."
	mc cp /upload/data/combined-video-no-gap-rooftop.mp4 myminio/data/
	echo "Video uploaded successfully"
else
	echo "Video already exists, skipping"
fi

echo "=== Data upload complete ==="

# List uploaded files
echo ""
echo "Files in MinIO:"
echo "--- models bucket ---"
mc ls myminio/models/
echo "--- data bucket ---"
mc ls myminio/data/
