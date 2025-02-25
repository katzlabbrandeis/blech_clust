.PHONY: all base emg neurec blechrnn clean params 

# Store sudo password
define get_sudo_password
$(eval SUDO_PASS := $(shell bash -c 'read -s -p "Enter sudo password: " pwd; echo $$pwd'))
@echo
endef

SCRIPT_DIR = $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))# This will return blech_clust base dir
INSTALL_PATH = $(SCRIPT_DIR)/requirements/BaSAR_1.3.tar.gz
INSTALL_STR = install.packages('$(INSTALL_PATH)', repos=NULL)

# Default target
all: base emg neurec blechrnn prefect

# Create and setup base environment
base: params
	$(call get_sudo_password)
	conda deactivate || true
	conda update -n base -c conda-forge conda -y
	conda clean --all -y
	conda create --name blech_clust python=3.8.13 -y
	conda run -n blech_clust conda install -c conda-forge -y --file requirements/conda_requirements_base.txt
	bash requirements/install_gnu_parallel.sh "$(SUDO_PASS)"
	conda run -n blech_clust pip install -r requirements/pip_requirements_base.txt
	conda run -n blech_clust bash requirements/patch_dependencies.sh

# Install EMG (BSA) requirements
emg:
	conda run -n blech_clust conda config --set channel_priority strict
	conda run -n blech_clust conda install -c conda-forge r-base=4.3.1 -y
	conda run -n blech_clust pip install rpy2
	# BaSAR is archived on CRAN, so we need to install it from a local file
	conda run -n blech_clust conda install -c r r-polynom r-orthopolynom -y
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
	conda run -n blech_clust pip install -r neuRecommend/requirements.txt

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
	conda run -n blech_clust pip install $$(cat requirements.txt | egrep "torch")

# Copy parameter templates
params:
	mkdir -p params
	cp params/_templates/* params/

# Install Prefect
prefect:
	conda run -n blech_clust pip install -U prefect

# Clean up environments 
clean:
	conda env remove -n blech_clust -y
