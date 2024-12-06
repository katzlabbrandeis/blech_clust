.PHONY: all base emg neurec blechrnn clean

# Default target
all: base emg neurec blechrnn

# Create and setup base environment
base:
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
	conda run -n blech_clust conda config --set channel_priority strict
	conda run -n blech_clust bash requirements/emg_install.sh

# Install neuRecommend classifier
neurec:
	cd ~/Desktop && \
	git clone https://github.com/abuzarmahmood/neuRecommend.git && \
	conda run -n blech_clust pip install -r neuRecommend/requirements.txt

# Install BlechRNN (optional)
blechrnn:
	cd ~/Desktop && \
	git clone https://github.com/abuzarmahmood/blechRNN.git && \
	cd blechRNN && \
	conda run -n blech_clust pip install $$(cat requirements.txt | egrep "torch")

# Clean up environments
clean:
	conda env remove -n blech_clust -y
