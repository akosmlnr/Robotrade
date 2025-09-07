# Stock Prediction Deep Learning - Makefile
# This Makefile provides convenient commands for Docker operations and project tasks

# Variables
DOCKER_IMAGE = stock-prediction
DOCKER_CONTAINER = stock-prediction-app
COMPOSE_FILE = docker-compose.yml
PYTHON_CMD = python

# Default target
.DEFAULT_GOAL := help

# Colors for output
RED = \033[0;31m
GREEN = \033[0;32m
YELLOW = \033[1;33m
BLUE = \033[0;34m
NC = \033[0m # No Color

##@ General

.PHONY: help
help: ## Display this help message
	@echo "$(BLUE)Stock Prediction Deep Learning - Available Commands$(NC)"
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
run: ## Run the container with default training
	@echo "$(GREEN)Running stock prediction training...$(NC)"
	docker run --rm -v $(PWD)/outputs:/app/outputs -v $(PWD)/models:/app/models -v $(PWD)/data:/app/data $(DOCKER_IMAGE)

.PHONY: run-interactive
run-interactive: ## Run the container in interactive mode
	@echo "$(GREEN)Running container in interactive mode...$(NC)"
	docker run -it --rm -v $(PWD):/app -v $(PWD)/outputs:/app/outputs -v $(PWD)/models:/app/models -v $(PWD)/data:/app/data $(DOCKER_IMAGE) /bin/bash

.PHONY: run-custom
run-custom: ## Run with custom parameters (usage: make run-custom TICKER=AAPL EPOCHS=50)
	@echo "$(GREEN)Running with custom parameters: TICKER=$(TICKER), EPOCHS=$(EPOCHS)$(NC)"
	docker run --rm -v $(PWD)/outputs:/app/outputs -v $(PWD)/models:/app/models -v $(PWD)/data:/app/data \
		$(DOCKER_IMAGE) $(PYTHON_CMD) stock_prediction_deep_learning.py \
		--ticker=$(TICKER) --epochs=$(EPOCHS) --start_date=2017-11-01 --validation_date=2022-09-01

.PHONY: run-realtime
run-realtime: ## Run real-time prediction (requires trained model)
	@echo "$(GREEN)Running real-time prediction...$(NC)"
	docker run --rm -v $(PWD)/outputs:/app/outputs -v $(PWD)/models:/app/models -v $(PWD)/data:/app/data \
		$(DOCKER_IMAGE) $(PYTHON_CMD) stock_prediction_realtime.py \
		--ticker=GOOG --model=./models/model_weights.h5 --time_steps=3 --interval=300

##@ Docker Compose Operations

.PHONY: up
up: ## Start services with docker-compose
	@echo "$(GREEN)Starting services with docker-compose...$(NC)"
	docker-compose -f $(COMPOSE_FILE) up --build

.PHONY: up-detached
up-detached: ## Start services in detached mode
	@echo "$(GREEN)Starting services in detached mode...$(NC)"
	docker-compose -f $(COMPOSE_FILE) up -d --build

.PHONY: up-jupyter
up-jupyter: ## Start Jupyter notebook service
	@echo "$(GREEN)Starting Jupyter notebook...$(NC)"
	docker-compose -f $(COMPOSE_FILE) --profile jupyter up --build

.PHONY: up-realtime
up-realtime: ## Start real-time prediction service
	@echo "$(GREEN)Starting real-time prediction service...$(NC)"
	docker-compose -f $(COMPOSE_FILE) --profile realtime up --build

.PHONY: down
down: ## Stop and remove containers
	@echo "$(GREEN)Stopping and removing containers...$(NC)"
	docker-compose -f $(COMPOSE_FILE) down

.PHONY: logs
logs: ## Show logs from running containers
	@echo "$(GREEN)Showing container logs...$(NC)"
	docker-compose -f $(COMPOSE_FILE) logs -f

.PHONY: exec
exec: ## Execute bash in running container
	@echo "$(GREEN)Executing bash in container...$(NC)"
	docker-compose -f $(COMPOSE_FILE) exec stock-prediction /bin/bash

##@ Development

.PHONY: train
train: ## Train model with default parameters
	@echo "$(GREEN)Training model with default parameters...$(NC)"
	docker run --rm -v $(PWD)/outputs:/app/outputs -v $(PWD)/models:/app/models -v $(PWD)/data:/app/data \
		$(DOCKER_IMAGE) $(PYTHON_CMD) stock_prediction_deep_learning.py \
		--ticker=GOOG --start_date=2017-11-01 --validation_date=2022-09-01 --epochs=100 --batch_size=32 --time_steps=3

.PHONY: train-aapl
train-aapl: ## Train model for AAPL stock
	@echo "$(GREEN)Training model for AAPL...$(NC)"
	docker run --rm -v $(PWD)/outputs:/app/outputs -v $(PWD)/models:/app/models -v $(PWD)/data:/app/data \
		$(DOCKER_IMAGE) $(PYTHON_CMD) stock_prediction_deep_learning.py \
		--ticker=AAPL --start_date=2017-11-01 --validation_date=2022-09-01 --epochs=100 --batch_size=32 --time_steps=3

.PHONY: train-tsla
train-tsla: ## Train model for TSLA stock
	@echo "$(GREEN)Training model for TSLA...$(NC)"
	docker run --rm -v $(PWD)/outputs:/app/outputs -v $(PWD)/models:/app/models -v $(PWD)/data:/app/data \
		$(DOCKER_IMAGE) $(PYTHON_CMD) stock_prediction_deep_learning.py \
		--ticker=TSLA --start_date=2017-11-01 --validation_date=2022-09-01 --epochs=100 --batch_size=32 --time_steps=3

.PHONY: predict
predict: ## Make single prediction
	@echo "$(GREEN)Making single prediction...$(NC)"
	docker run --rm -v $(PWD)/outputs:/app/outputs -v $(PWD)/models:/app/models -v $(PWD)/data:/app/data \
		$(DOCKER_IMAGE) $(PYTHON_CMD) stock_prediction_realtime.py \
		--ticker=GOOG --model=./models/model_weights.h5 --time_steps=3 --single

.PHONY: download-data
download-data: ## Download market data
	@echo "$(GREEN)Downloading market data...$(NC)"
	docker run --rm -v $(PWD)/outputs:/app/outputs -v $(PWD)/models:/app/models -v $(PWD)/data:/app/data \
		$(DOCKER_IMAGE) $(PYTHON_CMD) stock_prediction_download_market_data.py

##@ Cleanup

.PHONY: clean
clean: ## Clean up Docker resources
	@echo "$(GREEN)Cleaning up Docker resources...$(NC)"
	docker-compose -f $(COMPOSE_FILE) down --volumes --remove-orphans
	docker system prune -f

.PHONY: clean-all
clean-all: ## Clean up all Docker resources including images
	@echo "$(GREEN)Cleaning up all Docker resources...$(NC)"
	docker-compose -f $(COMPOSE_FILE) down --volumes --remove-orphans
	docker system prune -af
	docker volume prune -f

.PHONY: clean-outputs
clean-outputs: ## Clean up output files
	@echo "$(GREEN)Cleaning up output files...$(NC)"
	rm -rf outputs/* models/* data/*

##@ Utilities

.PHONY: status
status: ## Show Docker container status
	@echo "$(GREEN)Docker container status:$(NC)"
	docker ps -a | grep stock-prediction || echo "No stock-prediction containers running"

.PHONY: images
images: ## Show Docker images
	@echo "$(GREEN)Docker images:$(NC)"
	docker images | grep stock-prediction || echo "No stock-prediction images found"

.PHONY: shell
shell: ## Open shell in running container
	@echo "$(GREEN)Opening shell in container...$(NC)"
	docker exec -it $(DOCKER_CONTAINER) /bin/bash

.PHONY: logs-container
logs-container: ## Show logs from specific container
	@echo "$(GREEN)Showing container logs...$(NC)"
	docker logs -f $(DOCKER_CONTAINER)

##@ Quick Start

.PHONY: quick-start
quick-start: build train ## Quick start: build and train with default parameters
	@echo "$(GREEN)Quick start completed!$(NC)"

.PHONY: dev-setup
dev-setup: build up-jupyter ## Development setup: build and start Jupyter
	@echo "$(GREEN)Development setup completed! Jupyter available at http://localhost:8888$(NC)"

# Example usage:
# make build                    # Build Docker image
# make train                    # Train with default parameters
# make train-aapl              # Train for AAPL stock
# make run-custom TICKER=TSLA EPOCHS=50  # Custom training
# make up-jupyter              # Start Jupyter notebook
# make up-realtime             # Start real-time prediction
# make clean                   # Clean up resources
