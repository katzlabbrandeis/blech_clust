.PHONY: all base emg neurec blechrnn clean params dev optional test prefect update make_env docker-build docker-run docker-shell docker-stop docker-clean

# Consolidated pip-based installation - no sudo required for base packages

SCRIPT_DIR = $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST)))) # This will return blech_clust base dir
INSTALL_PATH = $(strip $(SCRIPT_DIR))/requirements/BaSAR_1.3.tar.gz
INSTALL_STR = install.packages('$(INSTALL_PATH)', repos=NULL)

# Default target
all: update make_env base emg neurec blechrnn prefect dev optional
	@echo "All setup tasks completed successfully!"

core: update make_env base emg neurec params
	@echo "Core setup tasks completed successfully!"

# Create and setup base environment

update:
	@echo "Updating conda..."
	conda update -n base -c conda-forge conda -y

make_env: params
	@echo "Setting up base blech_clust environment..."
	@echo "Deactivating any active conda environment..."
	conda deactivate || true
	@echo "Cleaning conda cache..."
	conda clean --all -y
	@echo "Creating blech_clust environment with Python 3.8..."
	conda create --name blech_clust python=3.8 -y

base:
	@echo "Installing Python dependencies from requirements.txt..."
	conda run -n blech_clust pip install --no-cache-dir -r requirements/requirements.txt
	@echo "Installing blech_clust package..."
	conda run -n blech_clust pip install --no-cache-dir -e .
	@echo "Base environment setup complete!"
	@echo "Note: GNU parallel should be installed system-wide if needed"

# Install EMG (BSA) requirements
emg:
	@echo "Installing EMG (BSA) requirements..."
	@echo "Configuring conda channel priority..."
	conda run -n blech_clust conda config --set channel_priority strict
	@echo "Installing R and R packages..."
	conda run -n blech_clust conda install -c conda-forge r-base=3.6 r-polynom r-orthopolynom -y
	@echo "Installing libxcrypt (dependency for rpy2)..."
	conda run -n blech_clust conda install --channel=conda-forge libxcrypt
	@echo "Installing rpy2 (building against current R installation)..."
	conda run -n blech_clust pip install rpy2==3.5.12 --no-cache-dir
	@echo "Installing BaSAR from local archive..."
	conda run -n blech_clust Rscript -e "${INSTALL_STR}"
	@echo "EMG requirements installation complete!"

# Install neuRecommend classifier
neurec:
	@echo "Installing neuRecommend classifier..."
	@if [ ! -d ~/Desktop/neuRecommend ]; then \
		echo "Cloning neuRecommend repository..."; \
		cd ~/Desktop && \
		git clone https://github.com/abuzarmahmood/neuRecommend.git; \
	else \
		echo "neuRecommend already exists, skipping clone"; \
	fi
	@echo "Installing neuRecommend dependencies..."
	cd ~/Desktop && \
	conda run -n blech_clust pip install --no-cache-dir -r neuRecommend/requirements.txt
	@echo "neuRecommend installation complete!"

# Install BlechRNN (optional)
blechrnn:
	@echo "Installing BlechRNN (optional)..."
	@if [ ! -d ~/Desktop/blechRNN ]; then \
		echo "Cloning blechRNN repository..."; \
		cd ~/Desktop && \
		git clone https://github.com/abuzarmahmood/blechRNN.git; \
	else \
		echo "blechRNN already exists, skipping clone"; \
	fi
	@echo "Installing PyTorch dependencies for blechRNN..."
	cd ~/Desktop && \
	cd blechRNN && \
	conda run -n blech_clust pip install --no-cache-dir $$(cat requirements.txt | egrep "torch")
	@echo "BlechRNN installation complete!"

# Copy parameter templates
# If more than 1 json file exists in params, don't copy templates and print warning
# This is to prevent overwriting existing parameter files
# If no json files exist, copy templates
params:
	@echo "Checking parameter files..."
	@if [ $$(ls params/*.json 2>/dev/null | wc -l) -gt 1 ]; then \
		echo "Warning: Multiple params files detected in params dir. Not copying templates."; \
	elif [ $$(ls params/*.json 2>/dev/null | wc -l) -eq 1 ]; then \
		echo "Copying parameter templates to params directory..."; \
	else \
		echo "No parameter files found. Templates should be copied if available."; \
	fi

dev:
	@echo "Installing development dependencies..."
	conda run -n blech_clust pip install --no-cache-dir -e .[dev]
	@echo "Development environment setup complete!"

optional:
	@echo "Installing optional dependencies..."
	conda run -n blech_clust pip install --no-cache-dir -e .[optional]
	@echo "Optional dependencies installation complete!"

test:
	@echo "Installing test dependencies..."
	conda run -n blech_clust pip install --no-cache-dir -e .[test]
	@echo "Test dependencies installation complete!"

# Install Prefect
prefect:
	@echo "Installing Prefect workflow management..."
	conda run -n blech_clust pip install --no-cache-dir -U prefect
	@echo "Prefect installation complete!"

# Clean up environments
clean:
	@echo "Cleaning up blech_clust environment..."
	conda env remove -n blech_clust -y
	@echo "Environment cleanup complete!"

# ============================================================================
# Docker Commands
# ============================================================================

# Build Docker image
docker-build:
	@echo "Building blech_clust Docker image..."
	docker build -t blech_clust:latest .
	@echo "Docker image built successfully!"
	@echo "Run 'make docker-shell' to start an interactive shell"

# Run container with docker-compose (maintains file permissions)
docker-run:
	@echo "Starting blech_clust container with docker-compose..."
	@echo "Current user: $(shell id -u):$(shell id -g)"
	UID=$(shell id -u) GID=$(shell id -g) docker-compose up -d
	@echo "Container started! Run 'make docker-shell' to enter the container"

# Open an interactive shell in the running container
docker-shell:
	@echo "Opening shell in blech_clust container..."
	@if [ "$$(docker ps -q -f name=blech_clust)" ]; then \
		docker exec -it blech_clust bash; \
	else \
		echo "Container not running. Starting container..."; \
		UID=$(shell id -u) GID=$(shell id -g) docker-compose up -d; \
		sleep 2; \
		docker exec -it blech_clust bash; \
	fi

# Run a command in the container (usage: make docker-exec CMD="python script.py")
docker-exec:
	@if [ -z "$(CMD)" ]; then \
		echo "Error: Please provide a command using CMD variable"; \
		echo "Example: make docker-exec CMD='python blech_exp_info.py'"; \
		exit 1; \
	fi
	@if [ "$$(docker ps -q -f name=blech_clust)" ]; then \
		docker exec -it blech_clust bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate blech_clust && $(CMD)"; \
	else \
		echo "Container not running. Starting container..."; \
		UID=$(shell id -u) GID=$(shell id -g) docker-compose up -d; \
		sleep 2; \
		docker exec -it blech_clust bash -c "source /opt/conda/etc/profile.d/conda.sh && conda activate blech_clust && $(CMD)"; \
	fi

# Stop the running container
docker-stop:
	@echo "Stopping blech_clust container..."
	docker-compose down
	@echo "Container stopped!"

# Clean up Docker resources (remove containers and images)
docker-clean:
	@echo "Cleaning up Docker resources..."
	docker-compose down -v
	docker rmi blech_clust:latest || true
	@echo "Docker cleanup complete!"

# Quick start guide for Docker
docker-help:
	@echo "Blech_clust Docker Commands:"
	@echo ""
	@echo "  make docker-build     - Build the Docker image"
	@echo "  make docker-run       - Start the container in the background"
	@echo "  make docker-shell     - Open an interactive shell in the container"
	@echo "  make docker-exec CMD='<command>' - Execute a command in the container"
	@echo "  make docker-stop      - Stop the running container"
	@echo "  make docker-clean     - Remove containers and images"
	@echo ""
	@echo "Quick Start:"
	@echo "  1. Build: make docker-build"
	@echo "  2. Run:   make docker-shell"
	@echo "  3. Use blech_clust commands as usual inside the container"
	@echo ""
	@echo "The container has read-write-execute access to:"
	@echo "  - Current directory (mounted at /workspace)"
	@echo "  - ./data directory (mounted at /data)"
	@echo "  - ./output directory (mounted at /output)"
	@echo ""
	@echo "All files created will maintain your user permissions!"
