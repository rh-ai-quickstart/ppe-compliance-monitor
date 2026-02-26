.DEFAULT_GOAL := help
.PHONY: help local-up local-build-up local-down build push deploy undeploy dev-backend dev-frontend local-build build-push-data kill-ports check-openai-env
help:
	@echo "Available targets:"
	@echo "  local-up   - Start local stack with Podman Compose"
	@echo "  local-build-up - Build and start local stack"
	@echo "  local-down - Stop local stack"
	@echo "  build      - Build container image"
	@echo "  push       - Push container image"
	@echo "  build-push-data - Build and push data container image (video + models)"
	@echo "  deploy     - Deploy manifests to OpenShift"
	@echo "  undeploy   - Remove manifests from OpenShift"
	@echo "  dev-backend - Create venv, install deps, run backend"
	@echo "  dev-frontend - Install deps and run frontend"
	@echo "  kill-ports   - Kill processes using required ports (3000, 8888, 8080, 8081, 9000, 9001)"


# Load .env file if it exists
ifneq (,$(wildcard .env))
  include .env
  export OPENAI_API_TOKEN OPENAI_API_ENDPOINT OPENAI_MODEL OPENAI_TEMPERATURE
endif

COMPOSE_FILE ?= $(CURDIR)/deploy/local/podman-compose.yaml
NAMESPACE ?= ppe-compliance-monitor-demo
PLATFORM_RELEASE ?= linux/amd64
PLATFORM_LOCAL ?= $(shell uname -m | sed -e 's/x86_64/linux\/amd64/' -e 's/arm64/linux\/arm64/' -e 's/aarch64/linux\/arm64/')

IMAGE_NAME ?= ppe-compliance-monitor
IMAGE_TAG ?= latest
IMAGE_REGISTRY ?= quay.io/rh-ai-quickstart
IMAGE_REPOSITORY := $(if $(IMAGE_REGISTRY),$(IMAGE_REGISTRY)/,)$(IMAGE_NAME)
BACKEND_IMAGE := $(IMAGE_REPOSITORY)-backend:$(IMAGE_TAG)
FRONTEND_IMAGE := $(IMAGE_REPOSITORY)-frontend:$(IMAGE_TAG)
DATA_IMAGE := $(IMAGE_REPOSITORY)-data:$(IMAGE_TAG)
LOCAL_BACKEND_IMAGE ?= ppe-compliance-monitor-backend:local
LOCAL_FRONTEND_IMAGE ?= ppe-compliance-monitor-frontend:local
LOCAL_DATA_IMAGE ?= ppe-compliance-monitor-data:local
PYTHON ?= python3
VENV_DIR ?= .venv
BACKEND_DIR ?= app/backend
FRONTEND_DIR ?= app/frontend
HELM_RELEASE ?= ppe-compliance-monitor
HELM_CHART ?= deploy/helm/ppe-compliance-monitor

check-openai-env:
	@token="$(OPENAI_API_TOKEN)"; \
	endpoint="$(OPENAI_API_ENDPOINT)"; \
	model="$(OPENAI_MODEL)"; \
	temp="$(OPENAI_TEMPERATURE)"; \
	if [ -z "$$token" ] || [ -z "$$endpoint" ] || [ -z "$$model" ]; then \
		echo "==> Missing required OpenAI environment variables."; \
		if [ -z "$$token" ]; then printf "  OPENAI_API_TOKEN: "; read token; fi; \
		if [ -z "$$endpoint" ]; then printf "  OPENAI_API_ENDPOINT: "; read endpoint; fi; \
		if [ -z "$$model" ]; then printf "  OPENAI_MODEL: "; read model; fi; \
		if [ -z "$$temp" ]; then temp="0.7"; fi; \
		printf 'OPENAI_API_TOKEN=%s\nOPENAI_API_ENDPOINT=%s\nOPENAI_MODEL=%s\nOPENAI_TEMPERATURE=%s\n' \
			"$$token" "$$endpoint" "$$model" "$$temp" > .env; \
		echo "==> Saved to .env"; \
	else \
		echo "==> OpenAI environment variables loaded from .env"; \
	fi

local-up: check-openai-env local-build kill-ports
	PODMAN_DEFAULT_PLATFORM=$(PLATFORM_LOCAL) podman-compose -f $(COMPOSE_FILE) up

local-build-up: check-openai-env kill-ports
	PODMAN_DEFAULT_PLATFORM=$(PLATFORM_LOCAL) podman-compose -f $(COMPOSE_FILE) up --build

local-build:
	@should_build=0; \
	if ! podman image exists $(LOCAL_BACKEND_IMAGE); then should_build=1; fi; \
	if ! podman image exists $(LOCAL_FRONTEND_IMAGE); then should_build=1; fi; \
	if [ $$should_build -eq 1 ]; then \
		echo "Building local image..."; \
		PODMAN_DEFAULT_PLATFORM=$(PLATFORM_LOCAL) podman-compose -f $(COMPOSE_FILE) build; \
	else \
		echo "Local image is up-to-date; skipping build."; \
	fi

local-down:
	podman-compose -f $(COMPOSE_FILE) down

build:
	podman build --platform $(PLATFORM_RELEASE) -t $(BACKEND_IMAGE) -f app/backend/Dockerfile app
	podman build --platform $(PLATFORM_RELEASE) -t $(FRONTEND_IMAGE) -f app/frontend/Dockerfile app/frontend

push:
	@if podman image exists $(BACKEND_IMAGE) && podman image exists $(FRONTEND_IMAGE); then \
		podman push $(BACKEND_IMAGE); \
		podman push $(FRONTEND_IMAGE); \
	else \
		echo "Images not found. Run 'make build' first."; \
		exit 1; \
	fi

build-push-data:
	podman build --platform $(PLATFORM_RELEASE) -t $(DATA_IMAGE) -f app/data-image/Dockerfile app
	podman push $(DATA_IMAGE)

deploy: check-openai-env
	@domain=$$(oc get ingresses.config/cluster -o jsonpath='{.spec.domain}' 2>/dev/null || true); \
	if [ -n "$(NAMESPACE)" ]; then oc new-project "$(NAMESPACE)" --display-name="$(NAMESPACE)" >/dev/null 2>&1 || oc project "$(NAMESPACE)"; fi; \
	if [ -n "$$domain" ]; then \
		host="$(HELM_RELEASE)-$(NAMESPACE).$$domain"; \
	else \
		host=""; \
	fi; \
	helm upgrade --install $(HELM_RELEASE) $(HELM_CHART) \
		--namespace $(NAMESPACE) --create-namespace \
		--set backend.image.repository=$(IMAGE_REPOSITORY)-backend \
		--set backend.image.tag=$(IMAGE_TAG) \
		--set frontend.image.repository=$(IMAGE_REPOSITORY)-frontend \
		--set frontend.image.tag=$(IMAGE_TAG) \
		--set data.image.repository=$(IMAGE_REPOSITORY)-data \
		--set data.image.tag=$(IMAGE_TAG) \
		$${host:+--set openshift.sharedHost=$$host}

undeploy:
	@if [ -n "$(NAMESPACE)" ]; then oc project "$(NAMESPACE)"; fi
	helm uninstall $(HELM_RELEASE) --namespace $(NAMESPACE)

dev-backend:
	$(PYTHON) -m venv $(VENV_DIR)
	. $(VENV_DIR)/bin/activate && pip install -r $(BACKEND_DIR)/requirements.txt
	. $(VENV_DIR)/bin/activate && $(PYTHON) $(BACKEND_DIR)/app.py

dev-frontend:
	cd $(FRONTEND_DIR) && npm install
	cd $(FRONTEND_DIR) && npm start

kill-ports: ## Kill processes using required ports
	@echo "Killing processes on application ports..."
	@if [ "$$(uname)" = "Darwin" ]; then \
		lsof -ti :3000 | xargs kill -9 2>/dev/null || true; \
	else \
		fuser -k 3000/tcp 2>/dev/null || true; \
	fi
	@echo "   ✓ Frontend 3000 killed"
	@if [ "$$(uname)" = "Darwin" ]; then \
		lsof -ti :8888 | xargs kill -9 2>/dev/null || true; \
	else \
		fuser -k 8888/tcp 2>/dev/null || true; \
	fi
	@echo "   ✓ Backend 8888 killed"
	@if [ "$$(uname)" = "Darwin" ]; then \
		lsof -ti :8080 | xargs kill -9 2>/dev/null || true; \
	else \
		fuser -k 8080/tcp 2>/dev/null || true; \
	fi
	@echo "   ✓ OVMS REST 8080 killed"
	@if [ "$$(uname)" = "Darwin" ]; then \
		lsof -ti :8081 | xargs kill -9 2>/dev/null || true; \
	else \
		fuser -k 8081/tcp 2>/dev/null || true; \
	fi
	@echo "   ✓ OVMS gRPC 8081 killed"
	@if [ "$$(uname)" = "Darwin" ]; then \
		lsof -ti :9000 | xargs kill -9 2>/dev/null || true; \
	else \
		fuser -k 9000/tcp 2>/dev/null || true; \
	fi
	@echo "   ✓ MinIO 9000 killed"
	@if [ "$$(uname)" = "Darwin" ]; then \
		lsof -ti :9001 | xargs kill -9 2>/dev/null || true; \
	else \
		fuser -k 9001/tcp 2>/dev/null || true; \
	fi
	@echo "   ✓ MinIO Console 9001 killed"
	@echo "All ports cleared."
