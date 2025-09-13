# Real-time LSTM Prediction System - Makefile
# Minimal Makefile for Docker operations and development tasks

# Variables
DOCKER_IMAGE = realtime-lstm
DOCKER_CONTAINER = realtime-lstm-app
PYTHON_CMD = python

# Colors for output
GREEN = \033[0;32m
YELLOW = \033[1;33m
BLUE = \033[0;34m
NC = \033[0m # No Color

# Default target
.DEFAULT_GOAL := help

##@ General

.PHONY: help
help: ## Display this help message
	@echo "$(BLUE)Real-time LSTM Prediction System - Available Commands$(NC)"
	@echo ""
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Docker Operations

.PHONY: build
build: ## Build the Docker image
	@echo "$(GREEN)Building Docker image...$(NC)"
	docker build -t $(DOCKER_IMAGE) .

.PHONY: build-no-cache
build-no-cache: ## Build the Docker image without cache
	@echo "$(GREEN)Building Docker image without cache...$(NC)"
	docker build --no-cache -t $(DOCKER_IMAGE) .

.PHONY: run
run: ## Run the container with CLI help
	@echo "$(GREEN)Running realtime system CLI...$(NC)"
	docker run --rm $(DOCKER_IMAGE)

.PHONY: run-interactive
run-interactive: ## Run the container in interactive mode
	@echo "$(GREEN)Running container in interactive mode...$(NC)"
	docker run -it --rm -v "$(PWD)":/app $(DOCKER_IMAGE) /bin/bash

.PHONY: run-start
run-start: ## Start the realtime system
	@echo "$(GREEN)Starting realtime system...$(NC)"
	docker run --rm -v "$(PWD)/data":/app/data -v "$(PWD)/logs":/app/logs \
		$(DOCKER_IMAGE) $(PYTHON_CMD) -m core.cli start

.PHONY: run-status
run-status: ## Show system status
	@echo "$(GREEN)Showing system status...$(NC)"
	docker run --rm -v "$(PWD)/data":/app/data -v "$(PWD)/logs":/app/logs \
		$(DOCKER_IMAGE) $(PYTHON_CMD) -m core.cli status

.PHONY: run-api
run-api: ## Start the API server
	@echo "$(GREEN)Starting API server...$(NC)"
	docker run --rm -p 5000:5000 -v "$(PWD)/data":/app/data -v "$(PWD)/logs":/app/logs \
		$(DOCKER_IMAGE) $(PYTHON_CMD) -m core.api_server

##@ Development

.PHONY: dev-setup
dev-setup: build ## Development setup: build image
	@echo "$(GREEN)Development setup completed!$(NC)"

.PHONY: test
test: ## Run tests
	@echo "$(GREEN)Running tests...$(NC)"
	docker run --rm -v "$(PWD)":/app $(DOCKER_IMAGE) $(PYTHON_CMD) tests/test_comprehensive.py

.PHONY: test-pytest
test-pytest: ## Run tests with pytest
	@echo "$(GREEN)Running tests with pytest...$(NC)"
	docker run --rm -v "$(PWD)":/app $(DOCKER_IMAGE) $(PYTHON_CMD) -m pytest tests/

##@ Cleanup

.PHONY: clean
clean: ## Clean up Docker resources
	@echo "$(GREEN)Cleaning up Docker resources...$(NC)"
	docker system prune -f

.PHONY: clean-all
clean-all: ## Clean up all Docker resources including images
	@echo "$(GREEN)Cleaning up all Docker resources...$(NC)"
	docker system prune -af
	docker volume prune -f

##@ Utilities

.PHONY: status
status: ## Show Docker container status
	@echo "$(GREEN)Docker container status:$(NC)"
	docker ps -a | grep $(DOCKER_IMAGE) || echo "No $(DOCKER_IMAGE) containers running"

.PHONY: images
images: ## Show Docker images
	@echo "$(GREEN)Docker images:$(NC)"
	docker images | grep $(DOCKER_IMAGE) || echo "No $(DOCKER_IMAGE) images found"

.PHONY: shell
shell: ## Open shell in running container
	@echo "$(GREEN)Opening shell in container...$(NC)"
	docker exec -it $(DOCKER_CONTAINER) /bin/bash

##@ Quick Start

.PHONY: quick-start
quick-start: build run-status ## Quick start: build and show status
	@echo "$(GREEN)Quick start completed!$(NC)"

# Example usage:
# make build                    # Build Docker image
# make run-start               # Start the realtime system
# make run-status              # Show system status
# make run-api                 # Start API server
# make run-interactive         # Interactive shell
# make clean                   # Clean up resources
