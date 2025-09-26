.PHONY: all base emg neurec blechrnn clean params patch

# Consolidated pip-based installation - no sudo required for base packages

SCRIPT_DIR = $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))# This will return blech_clust base dir
INSTALL_PATH = $(SCRIPT_DIR)/requirements/BaSAR_1.3.tar.gz
INSTALL_STR = install.packages('$(INSTALL_PATH)', repos=NULL)

# Default target
all: base emg neurec blechrnn prefect patch

# Create and setup base environment
base: params
	conda deactivate || true
	conda update -n base -c conda-forge conda -y
	conda clean --all -y
	conda create --name blech_clust python=3.8 -y
	# Install all dependencies via pip from consolidated requirements file
	conda run -n blech_clust pip install --no-cache-dir -r requirements/requirements.txt
	# Install GNU parallel separately as it's a system package
	@echo "Note: GNU parallel should be installed system-wide if needed"

# Install EMG (BSA) requirements
emg:
	# R packages still need conda as they're not available via pip
	conda run -n blech_clust conda config --set channel_priority strict
	conda run -n blech_clust conda install -c conda-forge r-base=3.6 r-polynom r-orthopolynom -y
	# rpy2 has to be built against current R...caching messes with that
	conda run -n blech_clust pip install rpy2 --no-cache-dir
	# BaSAR is archived on CRAN, so we need to install it from a local file
	conda run -n blech_clust Rscript -e "${INSTALL_STR}"

# Install neuRecommend classifier
neurec:
	@if [ ! -d ~/Desktop/neuRecommend ]; then \
		cd ~/Desktop && \
		git clone https://github.com/abuzarmahmood/neuRecommend.git; \
	else \
		echo "neuRecommend already exists"; \
	fi
	cd ~/Desktop && \
	conda run -n blech_clust pip install --no-cache-dir -r neuRecommend/requirements.txt

# Install BlechRNN (optional)
blechrnn:
	@if [ ! -d ~/Desktop/blechRNN ]; then \
		cd ~/Desktop && \
		git clone https://github.com/abuzarmahmood/blechRNN.git; \
	else \
		echo "blechRNN already exists"; \
	fi
	cd ~/Desktop && \
	cd blechRNN && \
	conda run -n blech_clust pip install --no-cache-dir $$(cat requirements.txt | egrep "torch")

# Patch dependencies
patch:
	conda run -n blech_clust bash requirements/patch_dependencies.sh

# Copy parameter templates
# If more than 1 json file exists in params, don't copy templates and print warning
# This is to prevent overwriting existing parameter files
# If no json files exist, copy templates
params:
	@if [ $$(ls params/*.json 2>/dev/null | wc -l) -gt 1 ]; then \
		echo "Warning: Params files detected in params dir. Not copying templates."; \
	elif [ $$(ls params/*.json 2>/dev/null | wc -l) -eq 1 ]; then \
		echo "Copying parameter templates to params directory"; \
	fi

# Install Prefect
prefect:
	conda run -n blech_clust pip install --no-cache-dir -U prefect

# Clean up environments
clean:
	conda env remove -n blech_clust -y
