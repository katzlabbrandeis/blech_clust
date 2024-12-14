.PHONY: all base emg neurec blechrnn clean

# Default target
all: base emg neurec blechrnn prefect

# Create and setup base environment
base: params
	conda deactivate || true
	conda update -n base conda -y
	conda clean --all -y
	conda create --name blech_clust python=3.8.13 -y
	conda run -n blech_clust conda install -c conda-forge -y --file requirements/conda_requirements_base.txt
	bash requirements/install_gnu_parallel.sh
	conda run -n blech_clust pip install -r requirements/pip_requirements_base.txt
	conda run -n blech_clust bash requirements/patch_dependencies.sh

# Install EMG (BSA) requirements
emg:
	@if ! conda run -n blech_clust Rscript -e "library(BaSAR)" 2>/dev/null; then \
		conda run -n blech_clust conda config --set channel_priority strict && \
		conda run -n blech_clust bash requirements/emg_install.sh; \
	else \
		echo "EMG dependencies already installed"; \
	fi

# Install neuRecommend classifier
neurec:
	@if [ ! -d ~/Desktop/neuRecommend ]; then \
		cd ~/Desktop && \
		git clone https://github.com/abuzarmahmood/neuRecommend.git && \
		conda run -n blech_clust pip install -r neuRecommend/requirements.txt; \
	else \
		echo "neuRecommend already installed"; \
	fi

# Install BlechRNN (optional) 
blechrnn:
	@if [ ! -d ~/Desktop/blechRNN ]; then \
		cd ~/Desktop && \
		git clone https://github.com/abuzarmahmood/blechRNN.git && \
		cd blechRNN && \
		conda run -n blech_clust pip install $$(cat requirements.txt | egrep "torch"); \
	else \
		echo "blechRNN already installed"; \
	fi

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
