# Makefile – MDC Challenge 2025 (Compose-Spec edition)
.PHONY: help build build-no-cache up down start stop restart logs \
        clean dev-shell api-shell status ps \
        build-up build-up-no-cache \
        logs-api \
        test-api test-unit test \
        dev prod-setup clean-images clean-all

# ---------- variables ----------
COMPOSE=docker compose                      # canonical Compose v2 CLI
SERVICE_API=mdc-api
SERVICE_DEV=mdc-dev
SERVICE_TEST=mdc-test

# ---------- colours ----------
YELLOW := \033[1;33m
GREEN  := \033[1;32m
BLUE   := \033[1;34m
NC     := \033[0m  # No colour

.DEFAULT_GOAL := help

# ---------- help ----------
help: ## Show this help message
	@echo "$(YELLOW)MDC Challenge 2025 – Docker Management$(NC)"
	@echo "$(BLUE)Available commands:$(NC)"
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make $(GREEN)<target>$(NC)\n\nTargets:\n"} /^[a-zA-Z_-]+:.*##/ { printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

# ---------- build ----------
build: ## Build all images (parallel)
	@echo "$(YELLOW)Building all services…$(NC)"
	$(COMPOSE) build --parallel $(SERVICE_PARSE) $(SERVICE_API)

build-no-cache: ## Build all images without cache
	@echo "$(YELLOW)Building all services (no cache)…$(NC)"
	$(COMPOSE) build --no-cache $(SERVICE_PARSE) $(SERVICE_API)

# ---------- up / down ----------
up: ## Start both services (detached)
	@echo "$(YELLOW)Starting services…$(NC)"
	$(COMPOSE) up -d $(SERVICE_PARSE) $(SERVICE_API)
	@make status

down: ## Stop & remove all services
	@echo "$(YELLOW)Stopping all services…$(NC)"
	$(COMPOSE) down
	@echo "$(GREEN)Services stopped successfully!$(NC)"

start: ## Start existing containers
	$(COMPOSE) start $(SERVICE_PARSE) $(SERVICE_API)

stop: ## Stop running containers
	$(COMPOSE) stop $(SERVICE_PARSE) $(SERVICE_API)

restart: ## Restart both services
	@make stop
	@make start

# ---------- combo ----------
build-up: ## Build & start services
	@make build
	@make up

build-up-no-cache: ## Build (no cache) & start
	@make build-no-cache
	@make up

# ---------- logs ----------
logs: ## Follow logs for both services
	$(COMPOSE) logs -f $(SERVICE_API)

logs-api: ## Logs for chunk-and-embed API
	$(COMPOSE) logs -f $(SERVICE_API)

# ---------- status ----------
status: ## Show container status
	@echo "$(BLUE)Service status:$(NC)"
	@$(COMPOSE) ps --format "table {{.Name}}\t{{.Status}}\t{{.Ports}}"

ps: status ## alias

# ---------- shells ----------
dev-shell: ## Interactive shell in dev container (profile dev)
	$(COMPOSE) --profile dev run --rm $(SERVICE_DEV) /bin/bash

api-shell: ## Shell in API container
	$(COMPOSE) exec $(SERVICE_API) /bin/bash

# ---------- clean ----------
clean: ## Stop & remove containers/vols/orphans
	$(COMPOSE) down --volumes --remove-orphans
	$(COMPOSE) --profile dev down --volumes --remove-orphans

clean-images: ## Remove project images
	-docker image rm -f  mdc-chunk-api:latest || true

clean-all: clean clean-images ## Full clean

# ---------- dev / prod shortcuts ----------
dev: ## Start dev profile containers
	$(COMPOSE) --profile dev up -d
	@make status

prod-setup: build-no-cache up ## Prod-like build & start

# ---------- tests ----------
test-unit: ## Run unit tests inside API container
	$(COMPOSE) run --rm $(SERVICE_API) \
	  pytest tests/test_chunking_and_embedding_services.py -q --disable-warnings --maxfail=1

test-api: ## Quick health check on chunk API
	curl -f http://localhost:8000/health 2>/dev/null || \
	  echo "$(YELLOW)Chunk API not responding$(NC)"

test: test-unit test-main test-api ## Run tests & health checks