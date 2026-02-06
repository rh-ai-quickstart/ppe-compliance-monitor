.DEFAULT_GOAL := help
.PHONY: help local-up local-build-up local-down build push deploy undeploy dev-backend dev-frontend ensure-assets local-build
help:
	@echo "Available targets:"
	@echo "  local-up   - Start local stack with Podman Compose"
	@echo "  local-build-up - Build and start local stack"
	@echo "  local-down - Stop local stack"
	@echo "  build      - Build container image"
	@echo "  push       - Push container image"
	@echo "  deploy     - Deploy manifests to OpenShift"
	@echo "  undeploy   - Remove manifests from OpenShift"
	@echo "  dev-backend - Create venv, install deps, run backend"
	@echo "  dev-frontend - Install deps and run frontend"
	@echo "  ensure-assets - Unpack model/video assets if missing"


COMPOSE_FILE ?= $(CURDIR)/deploy/local/podman-compose.yaml
NAMESPACE ?= ppe-compliance-monitor-demo
PLATFORM_RELEASE ?= linux/amd64
PLATFORM_LOCAL ?= $(shell uname -m | sed -e 's/x86_64/linux\/amd64/' -e 's/arm64/linux\/arm64/' -e 's/aarch64/linux\/arm64/')

IMAGE_NAME ?= ppe-compliance-monitor
IMAGE_TAG ?= latest
IMAGE_REGISTRY ?= quay.io/rh-ai-quickstart
IMAGE_REPOSITORY := $(if $(IMAGE_REGISTRY),$(IMAGE_REGISTRY)/,)$(IMAGE_NAME)
IMAGE := $(IMAGE_REPOSITORY):$(IMAGE_TAG)
LOCAL_IMAGE ?= ppe-compliance-monitor:local
PYTHON ?= python3
VENV_DIR ?= .venv
BACKEND_DIR ?= app/backend
FRONTEND_DIR ?= app/frontend
HELM_RELEASE ?= ppe-compliance-monitor
HELM_CHART ?= deploy/helm/ppe-compliance-monitor

local-up: ensure-assets local-build
	PODMAN_DEFAULT_PLATFORM=$(PLATFORM_LOCAL) podman-compose -f $(COMPOSE_FILE) up

local-build-up: ensure-assets
	PODMAN_DEFAULT_PLATFORM=$(PLATFORM_LOCAL) podman-compose -f $(COMPOSE_FILE) up --build

local-build:
	@should_build=0; \
	if ! podman image exists $(LOCAL_IMAGE); then should_build=1; fi; \
	if [ $$should_build -eq 1 ]; then \
		echo "Building local image..."; \
		PODMAN_DEFAULT_PLATFORM=$(PLATFORM_LOCAL) podman-compose -f $(COMPOSE_FILE) build; \
	else \
		echo "Local image is up-to-date; skipping build."; \
	fi

local-down:
	podman-compose -f $(COMPOSE_FILE) down

build: ensure-assets
	podman build --platform $(PLATFORM_RELEASE) -t $(IMAGE) -f Dockerfile .

push:
	@if podman image exists $(IMAGE); then \
		podman push $(IMAGE); \
	else \
		echo "Image $(IMAGE) not found. Run 'make build' first."; \
		exit 1; \
	fi

deploy:
	@domain=$$(oc get ingresses.config/cluster -o jsonpath='{.spec.domain}' 2>/dev/null || true); \
	if [ -n "$(NAMESPACE)" ]; then oc new-project "$(NAMESPACE)" --display-name="$(NAMESPACE)" >/dev/null 2>&1 || oc project "$(NAMESPACE)"; fi; \
	if [ -n "$$domain" ]; then \
		host="$(HELM_RELEASE)-$(NAMESPACE).$$domain"; \
	else \
		host=""; \
	fi; \
	helm upgrade --install $(HELM_RELEASE) $(HELM_CHART) \
		--namespace $(NAMESPACE) --create-namespace \
		--set image.repository=$(IMAGE_REPOSITORY) \
		--set image.tag=$(IMAGE_TAG) \
		$${host:+--set openshift.sharedHost=$$host}

undeploy:
	@if [ -n "$(NAMESPACE)" ]; then oc project "$(NAMESPACE)"; fi
	helm uninstall $(HELM_RELEASE) --namespace $(NAMESPACE)

dev-backend: ensure-assets
	$(PYTHON) -m venv $(VENV_DIR)
	. $(VENV_DIR)/bin/activate && pip install -r $(BACKEND_DIR)/requirements.txt
	. $(VENV_DIR)/bin/activate && $(PYTHON) $(BACKEND_DIR)/app.py

ensure-assets:
	@if [ ! -f app/models/custome_ppe.pt ]; then \
		echo "Missing app/models/custome_ppe.pt; running app/models/unzip.sh"; \
		cd app/models && sh ./unzip.sh; \
	fi
	@if [ ! -f app/data/combined-video-no-gap-rooftop.mp4 ]; then \
		echo "Missing app/data/combined-video-no-gap-rooftop.mp4; running app/data/unzip.sh"; \
		cd app/data && sh ./unzip.sh; \
	fi

dev-frontend:
	cd $(FRONTEND_DIR) && npm install
	cd $(FRONTEND_DIR) && npm start
