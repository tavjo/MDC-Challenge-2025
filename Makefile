# Makefile for MDC Challenge 2025 Docker Services
.PHONY: help build build-no-cache up down start stop restart logs clean dev-shell api-shell status ps

# Default target
.DEFAULT_GOAL := help

# Colors for output
YELLOW := \033[1;33m
GREEN := \033[1;32m
BLUE := \033[1;34m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "$(YELLOW)MDC Challenge 2025 - Docker Management$(NC)"
	@echo "$(BLUE)Available commands:$(NC)"
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make $(GREEN)<target>$(NC)\n\nTargets:\n"} /^[a-zA-Z_-]+:.*##/ { printf "  $(GREEN)%-15s$(NC) %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

build: ## Build both services
	@echo "$(YELLOW)Building all services...$(NC)"
	docker compose build mdc-parse
	docker compose build mdc-api

build-no-cache: ## Build both services without cache
	@echo "$(YELLOW)Building all services (no cache)...$(NC)"
	docker compose build --no-cache mdc-parse
	docker compose build --no-cache mdc-api

up: ## Start both services in detached mode
	@echo "$(YELLOW)Starting both services...$(NC)"
	docker compose up -d mdc-parse
	docker compose up -d mdc-api
	@echo "$(GREEN)Services started successfully!$(NC)"
	@make status

down: ## Stop and remove both services
	@echo "$(YELLOW)Stopping all services...$(NC)"
	docker compose down
	@echo "$(GREEN)Services stopped successfully!$(NC)"

start: ## Start existing containers
	@echo "$(YELLOW)Starting existing containers...$(NC)"
	docker compose start mdc-parse
	docker compose start mdc-api

stop: ## Stop running containers without removing them
	@echo "$(YELLOW)Stopping containers...$(NC)"
	docker compose stop mdc-parse
	docker compose stop mdc-api

restart: ## Restart both services
	@echo "$(YELLOW)Restarting services...$(NC)"
	@make stop
	@make start

build-up: ## Build and start both services
	@echo "$(YELLOW)Building and starting services...$(NC)"
	@make build
	@make up

build-up-no-cache: ## Build (no cache) and start both services
	@echo "$(YELLOW)Building (no cache) and starting services...$(NC)"
	@make build-no-cache
	@make up

logs: ## Show logs for both services
	@echo "$(YELLOW)Showing logs for both services...$(NC)"
	docker compose logs -f mdc-parse mdc-api

logs-main: ## Show logs for main service only
	docker compose logs -f mdc-parse

logs-api: ## Show logs for API service only
	docker compose logs -f mdc-api

status: ## Show status of all services
	@echo "$(BLUE)Service Status:$(NC)"
	@docker compose ps --format "table {{.Name}}\t{{.Status}}\t{{.Ports}}"

ps: status ## Alias for status

dev-shell: ## Open shell in development container
	@echo "$(YELLOW)Opening shell in development container...$(NC)"
	docker compose --profile dev run --rm mdc-dev /bin/bash

api-shell: ## Open shell in API container
	@echo "$(YELLOW)Opening shell in API container...$(NC)"
	docker compose exec mdc-api /bin/bash

main-shell: ## Open shell in main container
	@echo "$(YELLOW)Opening shell in main container...$(NC)"
	docker compose exec mdc-parse /bin/bash

clean: ## Stop and remove all containers, networks, and volumes
	@echo "$(YELLOW)Cleaning up Docker resources...$(NC)"
	docker compose down --volumes --remove-orphans
	docker compose --profile dev down --volumes --remove-orphans
	@echo "$(GREEN)Cleanup completed!$(NC)"

clean-images: ## Remove all built images for this project
	@echo "$(YELLOW)Removing project images...$(NC)"
	docker image rm -f mdc-parse-2025 || true
	docker image rm -f mdc-parse-dev || true
	docker image rm -f mdc-parse-api || true
	@echo "$(GREEN)Images removed!$(NC)"

clean-all: clean clean-images ## Full cleanup: containers, networks, volumes, and images

# Development shortcuts
dev: ## Start development environment
	@echo "$(YELLOW)Starting development environment...$(NC)"
	docker compose --profile dev up -d
	@make status

test-main: ## Test main API endpoints (assuming it runs on port 3000)
	@echo "$(YELLOW)Testing main API...$(NC)"
	curl -f http://localhost:3000/health || curl -f http://localhost:3000/ || echo "$(YELLOW)Main API not responding - check if it's running$(NC)"

test-api: ## Test chunking API endpoints (assuming it runs on port 8000)
	@echo "$(YELLOW)Testing chunking API...$(NC)"
	curl -f http://localhost:8000/health || curl -f http://localhost:8000/ || echo "$(YELLOW)API not responding - check if it's running$(NC)"

test-unit: ## Run unit tests inside API container
	@echo "$(YELLOW)Running unit tests...$(NC)"
	docker compose run --rm mdc-api pytest tests/test_chunking_and_embedding_services.py -q --disable-warnings --maxfail=1

# Modify test target to include unit tests
test: ## Test both API endpoints and unit tests
	@make test-unit
	@make test-main
	@make test-api

# Quick development workflow
dev-setup: build-up ## Complete setup for development (build and start)
	@echo "$(GREEN)Development environment is ready!$(NC)"
	@echo "$(BLUE)Main API:$(NC) http://localhost:3000"
	@echo "$(BLUE)Chunking API:$(NC) http://localhost:8000"

# Production-like workflow
prod-setup: build-no-cache up ## Production setup (clean build and start)
	@echo "$(GREEN)Production environment is ready!$(NC)"
	@make test 