# PPE Compliance Monitor Demo

This repository contains a Flask backend that performs PPE detection on a video
stream and a React frontend that visualizes the results and provides a chat UI.

## Overview

The application uses a trained model to detect objects from a live video feed.
The feed is sent to an endpoint where the backend detects objects and reports
compliance. For example, a model trained to identify workers wearing vests and
helmets in a boiler room will mark any worker without a helmet as non-compliant
and include that in the reported safety summary.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              MinIO Object Storage                            │
│  ┌─────────────────────┐           ┌─────────────────────────────────────┐  │
│  │  models bucket      │           │  data bucket                        │  │
│  │  └── ppe.pt (84MB)  │           │  └── combined-video-no-gap-...mp4   │  │
│  └─────────────────────┘           └─────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                    ▲                              │
                    │ upload (init)                │ download (init)
                    │                              ▼
┌───────────────────┴─────────┐    ┌──────────────────────────────────────────┐
│     data-loader container   │    │           Backend Container               │
│  - Uploads model to MinIO   │    │  - Reads model/video from PVC (K8s)      │
│  - Uploads video to MinIO   │    │  - Or downloads from MinIO (local)       │
│  - Idempotent               │    │  - Flask API on port 8888                │
└─────────────────────────────┘    └──────────────────────────────────────────┘
                                                    │
                                                    ▼
                                   ┌──────────────────────────────────────────┐
                                   │         Frontend Container                │
                                   │  - React App on port 3000                │
                                   │  - Video feed, dashboard, chat           │
                                   └──────────────────────────────────────────┘
```

### Components

- **Backend** (Flask, OpenCV, Ultralytics): video processing, detection, summaries
- **Frontend** (React): UI for live feed, summaries, and chat
- **MinIO**: S3-compatible object storage for model and video files
- **Data Loader**: Init container that uploads files to MinIO

### Storage Strategy

Model (`ppe.pt`) and video files are stored in MinIO rather than baked into container images:

| Deployment | Storage Method |
|------------|----------------|
| OpenShift/K8s | Files downloaded from MinIO to PVC by init container |
| Local (Podman) | Files downloaded from MinIO at runtime via Python client |

## Prerequisites

- Podman + `podman-compose` for local container runs
- Docker (optional alternative)
- Helm (for Kubernetes/OpenShift deployment)

## Configuration

Backend environment variables:
- `PORT`: backend port (default `8888`)
- `FLASK_DEBUG`: set to `true` to enable debug mode
- `CORS_ORIGINS`: allowed origins, comma-separated or `*`

Frontend runtime config (`app/frontend/public/env.js` or mounted in containers):
- `API_URL`: backend base URL (example: `http://localhost:8888`)

## Local Development (Podman Compose)

### Build and Run

```bash
make local-build-up
```

This starts:
1. **MinIO** - Object storage (ports 9000, 9001)
2. **data-loader** - Uploads model/video to MinIO (runs once)
3. **backend** - Flask API with `MINIO_ENABLED=true` (port 8888)
4. **frontend** - React app (port 3000)

### Run Without Rebuild

```bash
make local-up
```

### Stop

```bash
make local-down
```

### Access

- Frontend: http://localhost:3000
- Backend API: http://localhost:8888/api/
- MinIO Console: http://localhost:9001 (login: `minioadmin` / `minioadmin`)

## Local Development (No Containers)

### Backend

```bash
make dev-backend
```

Note: Requires model and video files in `app/models/` and `app/data/` directories.

### Frontend

```bash
make dev-frontend
```

## OpenShift/Kubernetes Deployment

### Build and Push Images

```bash
# Build backend and frontend images
make build

# Push to registry
make push

# Build and push data loader image (contains model/video for MinIO upload)
make build-push-data
```

### Deploy

```bash
make deploy NAMESPACE=<your-namespace>
```

### Undeploy

```bash
make undeploy NAMESPACE=<your-namespace>
```

### Deployment Workflow

1. **MinIO** starts (from `ai-architecture-charts` dependency)
2. **Backend Pod Init Container 1** (`upload-data`): Uploads model/video to MinIO
3. **Backend Pod Init Container 2** (`download-data`): Downloads files from MinIO to PVC
4. **Backend** starts with `MINIO_ENABLED=false`, reads from PVC paths
5. **Frontend** connects to backend API

### Helm Values

Override settings:

```bash
helm upgrade ppe-compliance-monitor deploy/helm/ppe-compliance-monitor \
  --set frontend.apiUrl=/api \
  --set backend.corsOrigins=http://your-frontend-host \
  --set storage.size=2Gi
```

OpenShift-specific options are included in the chart:
- Frontend Route: `openshift.route.enabled` and optional `openshift.route.host`
- Backend Route: `openshift.backendRoute.enabled` and optional `openshift.backendRoute.host`
- Shared Route host (same host for frontend + backend): `openshift.sharedHost`
- NetworkPolicy: `openshift.networkPolicy.enabled`
- SCC/RoleBinding: `openshift.scc.enabled`, `openshift.scc.name`, `openshift.roleBinding.*`

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/` | GET | Health check |
| `/api/video_feed` | GET | MJPEG video stream |
| `/api/latest_info` | GET | Latest description and summary |
| `/api/ask_question` | POST | Question answering based on context |
| `/api/chat` | POST | Rule-based response using detections |

### Example request

```
curl -X POST http://localhost:8888/ask_question \
  -H 'Content-Type: application/json' \
  -d '{"question": "How many people are detected?"}'
```
