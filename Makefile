.PHONY: all base emg neurec blechrnn clean params dev optional test prefect update make_env

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
	conda run --no-capture-output -n blech_clust pip install --no-cache-dir -r requirements/requirements.txt
	@echo "Installing blech_clust package..."
	conda run --no-capture-output -n blech_clust pip install --no-cache-dir -e .
	@echo "Base environment setup complete!"
	@echo "Note: GNU parallel should be installed system-wide if needed"

# Install EMG (BSA) requirements
emg:
	@echo "Installing EMG (BSA) requirements..."
	@echo "Configuring conda channel priority..."
	conda run --no-capture-output -n blech_clust conda config --set channel_priority strict
	@echo "Installing R and R packages..."
	conda run --no-capture-output -n blech_clust conda install -c conda-forge r-base=3.6 r-polynom r-orthopolynom -y
	@echo "Installing libxcrypt (dependency for rpy2)..."
	conda run --no-capture-output -n blech_clust conda install --channel=conda-forge libxcrypt -y
	@echo "Installing rpy2 (building against current R installation)..."
	conda run --no-capture-output -n blech_clust pip install rpy2==3.5.12 --no-cache-dir
	@echo "Installing BaSAR from local archive..."
	conda run --no-capture-output -n blech_clust Rscript -e "${INSTALL_STR}"
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
	conda run --no-capture-output -n blech_clust pip install --no-cache-dir -r neuRecommend/requirements.txt
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
	conda run --no-capture-output -n blech_clust pip install --no-cache-dir $$(cat requirements.txt | egrep "torch")
	@echo "BlechRNN installation complete!"

# Copy parameter templates
# If json files exist in params, list them and ask user if they want to overwrite
# If no json files exist, copy templates
params:
	@echo "Checking parameter files..."
	@if [ $$(ls params/*.json 2>/dev/null | wc -l) -gt 0 ]; then \
		echo "Existing parameter files found in params directory:"; \
		ls -la params/*.json; \
		echo ""; \
		read -p "Do you want to continue copying templates and potentially overwrite these files? (y/N): " confirm; \
		if [ "$$confirm" = "y" ] || [ "$$confirm" = "Y" ]; then \
			echo "Copying parameter templates to params directory..."; \
			cp -v blech_clust/params/templates/*.json blech_clust/params/; \
		else \
			echo "Skipping parameter template copying."; \
		fi \
	else \
		echo "No parameter files found. Copying parameter templates to params directory..."; \
		cp -v blech_clust/params/templates/*.json blech_clust/params/; \
	fi

dev:
	@echo "Installing development dependencies..."
	conda run --no-capture-output -n blech_clust pip install --no-cache-dir -e .[dev]
	@echo "Development environment setup complete!"

optional:
	@echo "Installing optional dependencies..."
	conda run --no-capture-output -n blech_clust pip install --no-cache-dir -e .[optional]
	@echo "Optional dependencies installation complete!"

test:
	@echo "Installing test dependencies..."
	conda run --no-capture-output -n blech_clust pip install --no-cache-dir -e .[test]
	@echo "Test dependencies installation complete!"

# Install Prefect
prefect:
	@echo "Installing Prefect workflow management..."
	conda run --no-capture-output -n blech_clust pip install --no-cache-dir -U prefect
	@echo "Prefect installation complete!"

# Clean up environments
clean:
	@echo "Cleaning up blech_clust environment..."
	conda env remove -n blech_clust -y
	@echo "Environment cleanup complete!"
