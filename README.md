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

- Backend (Flask, OpenCV, Ultralytics): video processing, detection, summaries.
- Frontend (React): UI for live feed, summaries, and chat.
- Container build: single image that contains both backend and frontend builds.

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

## Local (Podman Compose)

Build and run:

```
podman-compose -f deploy/local/podman-compose.yaml up --build
```

Access:
- Frontend: `http://localhost:3000`
- Backend: `http://localhost:8888`

## Helm

Install the chart (set your image repo/tag):

```
helm install demo deploy/helm/ppe-compliance-monitor \
  --set image.repository=<your-repo> \
  --set image.tag=<your-tag>
```

Override API URLs or CORS:

```
helm upgrade demo deploy/helm/ppe-compliance-monitor \
  --set frontend.apiUrl=http://demo-backend:8888 \
  --set backend.corsOrigins=http://demo-frontend:3000
```

OpenShift-specific options are included in the chart:
- Frontend Route: `openshift.route.enabled` and optional `openshift.route.host`
- Backend Route: `openshift.backendRoute.enabled` and optional `openshift.backendRoute.host`
- Shared Route host (same host for frontend + backend): `openshift.sharedHost`
- NetworkPolicy: `openshift.networkPolicy.enabled`
- SCC/RoleBinding: `openshift.scc.enabled`, `openshift.scc.name`, `openshift.roleBinding.*`

## API Endpoints

- `GET /video_feed`: MJPEG video stream
- `GET /latest_info`: latest description and summary
- `POST /ask_question`: question answering based on latest context
- `POST /chat`: rule-based response using latest detections and summary

### Example request

```
curl -X POST http://localhost:8888/ask_question \
  -H 'Content-Type: application/json' \
  -d '{"question": "How many people are detected?"}'
```
