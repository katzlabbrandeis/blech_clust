# Multi-stage Dockerfile for blech_clust
# Supports both spike sorting and EMG analysis workflows
# Based on Python 3.8 with R 3.6 for EMG/BSA analysis

FROM continuumio/miniconda3:latest AS base

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    CONDA_AUTO_UPDATE_CONDA=false \
    PATH=/opt/conda/bin:$PATH

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    parallel \
    curl \
    ca-certificates \
    libxcrypt-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user for running the container
RUN useradd -m -s /bin/bash -u 1000 blechuser

# Set working directory
WORKDIR /workspace

# Copy requirements first for better caching
COPY requirements/ /workspace/requirements/
COPY pyproject.toml /workspace/
COPY README.md /workspace/

# Create conda environment with Python 3.8
RUN conda create --name blech_clust python=3.8 -y && \
    conda clean --all -y

# Install Python dependencies
RUN /opt/conda/envs/blech_clust/bin/pip install --no-cache-dir -r requirements/requirements.txt

# Install R and R packages for EMG analysis
RUN conda install -n blech_clust -c conda-forge \
    r-base=3.6 \
    r-polynom \
    r-orthopolynom \
    libxcrypt \
    -y && \
    conda clean --all -y

# Install rpy2 for Python-R interface
RUN /opt/conda/envs/blech_clust/bin/pip install --no-cache-dir rpy2==3.5.12

# Copy the entire project
COPY . /workspace/

# Install blech_clust package in editable mode
RUN /opt/conda/envs/blech_clust/bin/pip install --no-cache-dir -e .

# Install BaSAR R package for EMG analysis
RUN /opt/conda/envs/blech_clust/bin/Rscript -e \
    "install.packages('/workspace/requirements/BaSAR_1.3.tar.gz', repos=NULL, type='source')"

# Set up conda activation in bash
RUN echo "source /opt/conda/etc/profile.d/conda.sh" >> /root/.bashrc && \
    echo "conda activate blech_clust" >> /root/.bashrc && \
    echo "source /opt/conda/etc/profile.d/conda.sh" >> /home/blechuser/.bashrc && \
    echo "conda activate blech_clust" >> /home/blechuser/.bashrc

# Change ownership of workspace to blechuser
RUN chown -R blechuser:blechuser /workspace

# Create data directory with proper permissions
RUN mkdir -p /data && chown -R blechuser:blechuser /data

# Switch to non-root user
USER blechuser

# Set up environment to use conda by default
ENV PATH=/opt/conda/envs/blech_clust/bin:$PATH \
    CONDA_DEFAULT_ENV=blech_clust

# Default command opens a bash shell with conda environment activated
CMD ["/bin/bash", "-c", "source /opt/conda/etc/profile.d/conda.sh && conda activate blech_clust && exec bash"]
