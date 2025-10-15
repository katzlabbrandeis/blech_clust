.PHONY: all base emg neurec blechrnn clean params patch

# Consolidated pip-based installation - no sudo required for base packages

SCRIPT_DIR = $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST)))) # This will return blech_clust base dir
INSTALL_PATH = $(strip $(SCRIPT_DIR))/requirements/BaSAR_1.3.tar.gz
INSTALL_STR = install.packages('$(INSTALL_PATH)', repos=NULL)

# Default target
all: update make_env base emg neurec blechrnn prefect patch

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
	@echo "Base environment setup complete!"
	@echo "Note: GNU parallel should be installed system-wide if needed"

# Install EMG (BSA) requirements
emg:
	@echo "Installing EMG (BSA) requirements..."
	@echo "Configuring conda channel priority..."
	conda run -n blech_clust conda config --set channel_priority strict
	@echo "Installing R and R packages..."
	conda run -n blech_clust conda install -c conda-forge r-base=3.6 r-polynom r-orthopolynom -y
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

# Patch dependencies
patch:
	@echo "Applying dependency patches..."
	conda run -n blech_clust bash requirements/patch_dependencies.sh
	@echo "Dependency patching complete!"

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
