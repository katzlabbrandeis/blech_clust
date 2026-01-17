"""
Interactive unit explorer for blech_clust data

This utility provides an interactive visualization tool for exploring neural units
in blech_clust datasets. It supports two main modes:

1. Unsorted waveforms: Visualize all detected waveforms from a specific electrode
2. Sorted units: Visualize any number of sorted units from the dataset

Features:
- UMAP embedding of waveform means for overview visualization
- Interactive point selection to view detailed waveforms
- Support for both unsorted and sorted data
- Waveform statistics (mean ± std) with individual waveform overlay
- Flexible data loading from blech_clust HDF5 files

Usage:
    python blech_unit_explorer.py <data_dir> --mode <unsorted|sorted> [options]

Examples:
    # Explore unsorted waveforms from electrode 5
    python blech_unit_explorer.py /path/to/data --mode unsorted --electrode 5

    # Explore specific sorted units
    python blech_unit_explorer.py /path/to/data --mode sorted --units 0 1 2 5

    # Explore all sorted units
    python blech_unit_explorer.py /path/to/data --mode sorted --all-units
"""

import os
import sys
import argparse
import tables
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import umap
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.decomposition import PCA
from scipy import stats
from scipy.stats import gaussian_kde
import warnings
import hashlib
import pickle

# Add blech_clust utils to path
from blech_clust.utils.blech_utils import imp_metadata


class BllechUnitExplorer:
    def __init__(self, data_dir, mode='sorted', electrode=None, units=None, all_units=False,
                 umap_mode='subsample', max_waveforms=5000, kmeans_k=1000,
                 use_pca=True, pca_variance=0.95, kde_bandwidth=0.1, flip_positive=False):
        """
        Initialize the unit explorer

        Parameters:
        -----------
        data_dir : str
            Path to the blech_clust data directory
        mode : str
            'unsorted' for raw waveforms from electrode, 'sorted' for sorted units
        electrode : int or list
            Electrode number(s) (required for unsorted mode). Use -1 for all electrodes,
            or provide a list of specific electrode numbers.
        units : list
            List of unit numbers to visualize (for sorted mode)
        all_units : bool
            Whether to visualize all available sorted units
        umap_mode : str
            'subsample' or 'kmeans' - method for handling large datasets
        max_waveforms : int
            Maximum number of waveforms to use for UMAP (subsample mode)
        kmeans_k : int
            Number of K-means clusters to use (kmeans mode)
        use_pca : bool
            Whether to apply PCA before UMAP (default: True)
        pca_variance : float
            Amount of variance to retain with PCA (0.0-1.0)
        kde_bandwidth : float or None
            Bandwidth for KDE. If None, uses automatic bandwidth selection (default: 0.1)
        flip_positive : bool
            Whether to flip units with positive deflections to negative
        """
        # Assign input parameters to attributes
        self.data_dir = data_dir
        self.mode = mode
        self.electrode = electrode
        self.units = units
        self.all_units = all_units
        self.umap_mode = umap_mode
        self.max_waveforms = max_waveforms
        self.kmeans_k = kmeans_k
        self.use_pca = use_pca
        self.pca_variance = pca_variance
        self.kde_bandwidth = kde_bandwidth
        self.flip_positive = flip_positive

    def load_metadata(self):
        # Load metadata
        self.metadata_handler = imp_metadata([[], self.data_dir])
        self.hdf5_path = os.path.join(
            self.data_dir, self.metadata_handler.hdf5_name)
        self.params_dict = self.metadata_handler.params_dict

    def initialize(self):
        """
        Initialize the unit explorer by loading data and setting up visualization
        """

        # Open HDF5 file
        self.h5 = tables.open_file(self.hdf5_path, mode='r')

        # Load data based on mode
        if self.mode == 'unsorted':
            self._load_unsorted_data()
        elif self.mode == 'sorted':
            self._load_sorted_data()
        else:
            raise ValueError("Mode must be 'unsorted' or 'sorted'")

        # Calculate UMAP embedding (with caching)
        self._calculate_umap()

        # Print parameters being used
        self._print_parameters()

        # Set up the interactive plot
        self.setup_plot()

    def _load_unsorted_data(self):
        """Load unsorted waveforms from specified electrode(s)"""
        print("Loading unsorted waveforms...")

        if self.electrode is None:
            raise ValueError("Electrode number required for unsorted mode")

        if self.electrode == -1:
            # Load from all available electrodes
            self._load_all_electrodes()
        elif isinstance(self.electrode, list):
            # Load from specific list of electrodes
            self._load_electrode_list(self.electrode)
        else:
            # Load from single electrode
            self._load_single_electrode(self.electrode)

        # Prepare data for UMAP based on the selected mode
        self._prepare_umap_data()

        if self.electrode == -1:
            print(
                f"Loaded {len(self.waveform_data)} waveforms from {len(self.electrode_info)} electrodes (all)")
        elif isinstance(self.electrode, list):
            print(
                f"Loaded {len(self.waveform_data)} waveforms from {len(self.electrode_info)} electrodes {self.electrode}")
        else:
            print(
                f"Loaded {len(self.waveform_data)} waveforms from electrode {self.electrode}")
        print(
            f"Using {len(self.umap_data)} data points for UMAP visualization ({self.umap_mode} mode)")

    def _load_single_electrode(self, electrode_num):
        """Load waveforms from a single electrode"""
        waveforms_file = os.path.join(
            self.data_dir,
            'spike_waveforms',
            f'electrode{electrode_num:02d}',
            'spike_waveforms.npy'
        )

        if not os.path.exists(waveforms_file):
            raise ValueError(
                f"No waveforms found for electrode {electrode_num} at {waveforms_file}")

        self.waveform_data = np.load(waveforms_file)
        self.data_labels = [f'Electrode_{electrode_num}_waveform_{i}'
                            for i in range(len(self.waveform_data))]
        self.electrode_info = [(electrode_num, len(self.waveform_data))]

        # Apply polarity flipping if requested
        if self.flip_positive:
            self._flip_positive_waveforms()

    def _load_all_electrodes(self):
        """Load waveforms from all available electrodes"""
        spike_waveforms_dir = os.path.join(self.data_dir, 'spike_waveforms')

        if not os.path.exists(spike_waveforms_dir):
            raise ValueError(
                f"Spike waveforms directory not found: {spike_waveforms_dir}")

        # Find all electrode directories
        electrode_dirs = []
        for item in os.listdir(spike_waveforms_dir):
            if item.startswith('electrode') and os.path.isdir(os.path.join(spike_waveforms_dir, item)):
                try:
                    electrode_num = int(item.replace('electrode', ''))
                    electrode_dirs.append((electrode_num, item))
                except ValueError:
                    continue

        if not electrode_dirs:
            raise ValueError(
                "No electrode directories found in spike_waveforms")

        # Sort by electrode number
        electrode_dirs.sort(key=lambda x: x[0])

        self._load_electrode_dirs(electrode_dirs)

    def _load_electrode_list(self, electrode_list):
        """Load waveforms from a specific list of electrodes"""
        spike_waveforms_dir = os.path.join(self.data_dir, 'spike_waveforms')

        if not os.path.exists(spike_waveforms_dir):
            raise ValueError(
                f"Spike waveforms directory not found: {spike_waveforms_dir}")

        # Create electrode directories list for the specified electrodes
        electrode_dirs = []
        for electrode_num in electrode_list:
            electrode_dir = f'electrode{electrode_num:02d}'
            electrode_path = os.path.join(spike_waveforms_dir, electrode_dir)
            if os.path.isdir(electrode_path):
                electrode_dirs.append((electrode_num, electrode_dir))
            else:
                print(
                    f"Warning: Electrode {electrode_num} directory not found")

        if not electrode_dirs:
            raise ValueError(
                f"No valid electrode directories found for electrodes {electrode_list}")

        # Sort by electrode number
        electrode_dirs.sort(key=lambda x: x[0])

        self._load_electrode_dirs(electrode_dirs)

    def _load_electrode_dirs(self, electrode_dirs):
        """Load waveforms from a list of electrode directories"""
        spike_waveforms_dir = os.path.join(self.data_dir, 'spike_waveforms')

        # Load waveforms from specified electrodes
        all_waveforms = []
        all_labels = []
        self.electrode_info = []

        for electrode_num, electrode_dir in electrode_dirs:
            waveforms_file = os.path.join(
                spike_waveforms_dir, electrode_dir, 'spike_waveforms.npy')

            if os.path.exists(waveforms_file):
                try:
                    electrode_waveforms = np.load(waveforms_file)
                    if len(electrode_waveforms) > 0:
                        all_waveforms.append(electrode_waveforms)
                        electrode_labels = [f'Electrode_{electrode_num}_waveform_{i}'
                                            for i in range(len(electrode_waveforms))]
                        all_labels.extend(electrode_labels)
                        self.electrode_info.append(
                            (electrode_num, len(electrode_waveforms)))
                        print(
                            f"Loaded {len(electrode_waveforms)} waveforms from electrode {electrode_num}")
                except Exception as e:
                    print(
                        f"Warning: Could not load waveforms from electrode {electrode_num}: {e}")

        if not all_waveforms:
            raise ValueError("No valid waveform data found in any electrode")

        # Concatenate all waveforms
        self.waveform_data = np.vstack(all_waveforms)
        self.data_labels = all_labels

        # Apply polarity flipping if requested
        if self.flip_positive:
            self._flip_positive_waveforms()

    def _load_sorted_data(self):
        """Load sorted units from HDF5 file"""

        print("Loading sorted units...")

        if '/sorted_units' not in self.h5:
            raise ValueError("No sorted units found in HDF5 file")

        sorted_units_group = self.h5.get_node('/sorted_units')
        available_units = [
            node._v_name for node in sorted_units_group._f_iter_nodes()]

        if self.all_units:
            selected_units = available_units
        elif self.units is not None:
            # Convert unit numbers to unit names
            selected_units = []
            for unit_num in self.units:
                unit_name = f'unit{unit_num:03d}'
                if unit_name in available_units:
                    selected_units.append(unit_name)
                else:
                    print(f"Warning: Unit {unit_num} not found")
        else:
            raise ValueError(
                "Must specify either --units or --all-units for sorted mode")

        if not selected_units:
            raise ValueError("No valid units found")

        # Load waveforms for selected units
        self.unit_data = {}
        self.unit_means = []
        self.data_labels = []

        for unit_name in selected_units:
            unit_path = f'/sorted_units/{unit_name}/waveforms'
            if unit_path in self.h5:
                unit_waveforms = self.h5.get_node(unit_path)[:]
                self.unit_data[unit_name] = unit_waveforms
                self.unit_means.append(np.mean(unit_waveforms, axis=0))
                self.data_labels.append(unit_name)

        self.unit_means = np.array(self.unit_means)

        # Apply polarity flipping if requested
        if self.flip_positive:
            self._flip_positive_units()

        self.umap_data = self.unit_means
        self.umap_labels = self.data_labels

        print(f"Loaded {len(selected_units)} sorted units")

    def _flip_positive_waveforms(self):
        """Flip waveforms that have positive deflections to negative"""
        if not hasattr(self, 'waveform_data') or len(self.waveform_data) == 0:
            return

        # Extract spike window parameters from blech_clust params
        spike_snapshot_before = self.params_dict['spike_snapshot_before']
        sampling_rate = self.params_dict['sampling_rate']

        # Convert time to samples - the aligning point is at the end of spike_snapshot_before
        spike_align_idx = int(spike_snapshot_before * sampling_rate / 1000)

        flipped_count = 0
        for i in range(len(self.waveform_data)):
            # Check the value at the spike aligning point
            if spike_align_idx < self.waveform_data.shape[1]:
                spike_value = self.waveform_data[i, spike_align_idx]

                # If the spike value is positive, flip the waveform
                if spike_value > 0:
                    self.waveform_data[i] = -self.waveform_data[i]
                    flipped_count += 1

        if flipped_count > 0:
            print(f"Flipped {flipped_count} positive waveforms to negative")

    def _flip_positive_units(self):
        """Flip sorted units that have positive deflections to negative"""
        if not hasattr(self, 'unit_data') or len(self.unit_data) == 0:
            return

        # Extract spike window parameters from blech_clust params
        spike_snapshot_before = self.params_dict['spike_snapshot_before']
        sampling_rate = self.params_dict['sampling_rate']

        # Convert time to samples - the aligning point is at the end of spike_snapshot_before
        spike_align_idx = int(spike_snapshot_before * sampling_rate / 1000)

        flipped_units = []
        for unit_name in self.unit_data.keys():
            unit_waveforms = self.unit_data[unit_name]
            unit_mean = np.mean(unit_waveforms, axis=0)

            # Check the value at the spike aligning point
            if spike_align_idx < len(unit_mean):
                spike_value = unit_mean[spike_align_idx]

                # If the spike value is positive, flip all waveforms for this unit
                if spike_value > 0:
                    self.unit_data[unit_name] = -unit_waveforms
                    # Update the unit mean in our list
                    unit_idx = self.data_labels.index(unit_name)
                    self.unit_means[unit_idx] = -unit_mean
                    flipped_units.append(unit_name)

        if flipped_units:
            print(
                f"Flipped {len(flipped_units)} positive units to negative: {flipped_units}")

    def _prepare_umap_data(self):
        """Prepare data for UMAP based on the selected mode"""
        if self.mode == 'sorted':
            # For sorted data, always use unit means
            self.umap_data = self.unit_means
            self.umap_labels = self.data_labels
            self.umap_indices = np.arange(len(self.unit_means))
            return

        # For unsorted data, apply the selected UMAP mode
        # Use fixed random seed for reproducible results
        np.random.seed(42)

        if self.umap_mode == 'subsample':
            if len(self.waveform_data) > self.max_waveforms:
                indices = np.random.choice(len(self.waveform_data),
                                           self.max_waveforms, replace=False)
                self.umap_data = self.waveform_data[indices]
                self.umap_labels = [self.data_labels[i] for i in indices]
                self.umap_indices = indices
            else:
                self.umap_data = self.waveform_data
                self.umap_labels = self.data_labels
                self.umap_indices = np.arange(len(self.waveform_data))

        elif self.umap_mode == 'kmeans':
            if len(self.waveform_data) > self.kmeans_k:
                print(
                    f"Applying K-means with k={self.kmeans_k} for UMAP visualization...")

                # Apply PCA before K-means if requested to reduce computational cost
                kmeans_data = self.waveform_data
                pca_for_kmeans = None
                if self.use_pca:
                    print(
                        f"Applying PCA before K-means to retain {float(self.pca_variance):.1%} variance...")
                    pca_for_kmeans = PCA(n_components=float(
                        self.pca_variance), random_state=42)
                    kmeans_data = pca_for_kmeans.fit_transform(
                        self.waveform_data)
                    print(
                        f"PCA reduced dimensionality from {self.waveform_data.shape[1]} to {kmeans_data.shape[1]} components for K-means")

                # Use MiniBatchKMeans for large datasets (>10,000 points)
                if len(self.waveform_data) > 10000:
                    print(
                        f"Using MiniBatchKMeans for {len(self.waveform_data)} data points (>10,000)")
                    kmeans = MiniBatchKMeans(
                        n_clusters=int(self.kmeans_k),
                        random_state=42,
                        init='k-means++',
                    )
                else:
                    kmeans = KMeans(n_clusters=int(self.kmeans_k),
                                    random_state=42, n_init=10)

                kmeans.fit(kmeans_data)

                # Get centroids in original space if PCA was applied
                if pca_for_kmeans is not None:
                    # Transform centroids back to original space
                    centroids_original = pca_for_kmeans.inverse_transform(
                        kmeans.cluster_centers_)
                    self.umap_data = centroids_original
                else:
                    self.umap_data = kmeans.cluster_centers_

                self.umap_labels = [
                    f'KMeans_centroid_{i}' for i in range(self.kmeans_k)]

                # Store cluster assignments to map back to original data
                self.kmeans_labels = kmeans.labels_
                self.kmeans_centroids = self.umap_data  # Centroids in original space
                self.umap_indices = np.arange(
                    self.kmeans_k)  # Indices into centroids

                # Track electrode information for each cluster
                self.cluster_electrode_info = {}
                for cluster_id in range(self.kmeans_k):
                    cluster_indices = np.where(
                        self.kmeans_labels == cluster_id)[0]
                    electrode_counts = {}
                    for idx in cluster_indices:
                        label = self.data_labels[idx]
                        if label.startswith('Electrode_'):
                            electrode_num = int(label.split('_')[1])
                            electrode_counts[electrode_num] = electrode_counts.get(
                                electrode_num, 0) + 1
                    # Assign cluster to electrode with most waveforms
                    if electrode_counts:
                        dominant_electrode = max(
                            electrode_counts.items(), key=lambda x: x[1])[0]
                        self.cluster_electrode_info[cluster_id] = dominant_electrode
                    else:
                        self.cluster_electrode_info[cluster_id] = list(self.electrode_info)[
                            0][0]  # Fallback
            else:
                # Not enough data for K-means, fall back to using all data
                self.umap_data = self.waveform_data
                self.umap_labels = self.data_labels
                self.umap_indices = np.arange(len(self.waveform_data))
                self.kmeans_labels = None
        else:
            raise ValueError(f"Unknown umap_mode: {self.umap_mode}")

    def _get_cache_key(self):
        """Generate a cache key based on embedding parameters and data"""
        # For unsorted data, we need to use the original data for consistent hashing
        # since umap_data might be different due to random sampling/kmeans
        if self.mode == 'unsorted':
            data_for_hash = self.waveform_data
        else:
            data_for_hash = self.umap_data

        # Create a hash based on parameters that affect the embedding
        params = {
            'mode': self.mode,
            'electrode': self.electrode,
            'units': sorted(self.units) if self.units else None,
            'all_units': self.all_units,
            'umap_mode': self.umap_mode,
            'max_waveforms': self.max_waveforms,
            'kmeans_k': self.kmeans_k,
            'use_pca': self.use_pca,
            'pca_variance': self.pca_variance,
            'flip_positive': self.flip_positive,
            'data_shape': data_for_hash.shape,
            # Short hash of data
            'data_hash': hashlib.md5(data_for_hash.tobytes()).hexdigest()[:16],
            'electrode_info': self.electrode_info if hasattr(self, 'electrode_info') else None
        }

        # Convert to string and hash
        params_str = str(sorted(params.items()))
        cache_key = hashlib.md5(params_str.encode()).hexdigest()
        return cache_key

    def _ensure_cache_dir(self):
        """Ensure cache directory exists and return its path"""
        cache_dir = os.path.join(self.data_dir, '.umap_cache')
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        return cache_dir

    def _get_cache_path(self, cache_key):
        """Get the cache file path for a given cache key"""
        cache_dir = self._ensure_cache_dir()
        return os.path.join(cache_dir, f'umap_embedding_{cache_key}.pkl')

    def _save_embedding_to_cache(self, cache_key, embedding_data):
        """Save UMAP embedding and related data to cache"""
        cache_path = self._get_cache_path(cache_key)
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(embedding_data, f)
            print(
                f"Saved UMAP embedding to cache: {os.path.basename(cache_path)}")
        except Exception as e:
            print(f"Warning: Could not save embedding to cache: {e}")

    def _load_embedding_from_cache(self, cache_key):
        """Load UMAP embedding and related data from cache"""
        cache_path = self._get_cache_path(cache_key)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    embedding_data = pickle.load(f)
                print(
                    f"Loaded UMAP embedding from cache: {os.path.basename(cache_path)}")
                return embedding_data
            except Exception as e:
                print(f"Warning: Could not load embedding from cache: {e}")
                # Remove corrupted cache file
                try:
                    os.remove(cache_path)
                except:
                    pass
        return None

    def _calculate_umap(self):
        """Calculate UMAP embedding of the data with caching"""

        print("Calculating UMAP embedding...")

        # Generate cache key
        cache_key = self._get_cache_key()

        # Try to load from cache first
        cached_data = self._load_embedding_from_cache(cache_key)
        if cached_data is not None:
            self.umap_embedding = cached_data['umap_embedding']
            self.umap_reducer = cached_data.get('umap_reducer')
            if 'pca_reducer' in cached_data:
                self.pca_reducer = cached_data['pca_reducer']
            return

        print("Computing UMAP embedding...")

        # Standardize the data for UMAP
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(self.umap_data)

        # Apply PCA if requested, but only if not already applied in K-means mode
        pca_reducer = None
        pca_already_applied = (self.mode == 'unsorted' and
                               self.umap_mode == 'kmeans' and
                               self.use_pca and
                               hasattr(self, 'kmeans_labels') and
                               self.kmeans_labels is not None)

        if self.use_pca and not pca_already_applied:
            print(
                f"Applying PCA to retain {float(self.pca_variance):.1%} variance...")
            pca = PCA(n_components=float(self.pca_variance), random_state=42)
            data_scaled = pca.fit_transform(data_scaled)
            pca_reducer = pca
            self.pca_reducer = pca
            print(
                f"PCA reduced dimensionality from {self.umap_data.shape[1]} to {data_scaled.shape[1]} components")
        elif pca_already_applied:
            print("PCA already applied during K-means clustering, skipping PCA for UMAP")

        # Compute UMAP embedding
        n_neighbors = min(15, len(data_scaled) - 1)
        umap_reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=0.1,
            random_state=42
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            umap_embedding = umap_reducer.fit_transform(data_scaled)

        self.umap_embedding = umap_embedding
        self.umap_reducer = umap_reducer

        # Save to cache
        embedding_data = {
            'umap_embedding': umap_embedding,
            'umap_reducer': umap_reducer,
        }
        if pca_reducer is not None:
            embedding_data['pca_reducer'] = pca_reducer

        self._save_embedding_to_cache(cache_key, embedding_data)

    def _print_parameters(self):
        """Print the parameters being used for visualization"""
        print("\n" + "="*60)
        print("VISUALIZATION PARAMETERS")
        print("="*60)
        print(f"Mode: {self.mode}")

        if self.mode == 'unsorted':
            if self.electrode == -1:
                electrode_nums = [info[0] for info in self.electrode_info]
                print(
                    f"Electrodes: All ({min(electrode_nums)}-{max(electrode_nums)})")
            elif isinstance(self.electrode, list):
                print(f"Electrodes: {self.electrode}")
            else:
                print(f"Electrode: {self.electrode}")
            print(f"UMAP mode: {self.umap_mode}")
            if self.umap_mode == 'subsample':
                print(f"Max waveforms for UMAP: {self.max_waveforms}")
            elif self.umap_mode == 'kmeans':
                print(f"K-means clusters: {self.kmeans_k}")
        else:
            if self.all_units:
                print("Units: All available units")
            else:
                print(f"Units: {self.units}")

        print(f"PCA enabled: {self.use_pca}")
        if self.use_pca:
            print(f"PCA variance retained: {self.pca_variance:.1%}")

        print(f"KDE bandwidth: {self.kde_bandwidth}")
        print(f"Flip positive units: {self.flip_positive}")

        if hasattr(self, 'electrode_info') and len(self.electrode_info) > 1:
            print("\nElectrode waveform counts:")
            for electrode_num, count in self.electrode_info:
                print(f"  Electrode {electrode_num}: {count} waveforms")

        if hasattr(self, 'waveform_data'):
            print(f"\nTotal waveforms loaded: {len(self.waveform_data)}")
        elif hasattr(self, 'unit_means'):
            print(f"\nTotal units loaded: {len(self.unit_means)}")

        print(f"Data points used for UMAP: {len(self.umap_data)}")
        print("="*60 + "\n")

    def setup_plot(self):
        """Set up the interactive matplotlib plot"""
        self.fig, (self.ax_umap, self.ax_waveform) = plt.subplots(
            1, 2, figsize=(15, 6))

        # Create KDE background
        self._plot_kde_background()

        # Plot UMAP embedding on top of KDE with electrode-specific colors
        self._plot_umap_points()

        title = f'UMAP of {self.mode.title()} Data\n(Click on points to explore)'
        if self.mode == 'unsorted':
            if self.electrode == -1:
                electrode_nums = [info[0] for info in self.electrode_info]
                title += f' - All Electrodes ({min(electrode_nums)}-{max(electrode_nums)}) ({self.umap_mode} mode)'
            elif isinstance(self.electrode, list):
                electrode_nums = [info[0] for info in self.electrode_info]
                title += f' - Electrodes {electrode_nums} ({self.umap_mode} mode)'
            else:
                title += f' - Electrode {self.electrode} ({self.umap_mode} mode)'
        if self.use_pca:
            title += f'\nPCA: {self.pca_variance:.1%} variance'
        if self.flip_positive:
            title += f'\nPositive units flipped'

        self.ax_umap.set_title(title)
        self.ax_umap.set_xlabel('UMAP 1')
        self.ax_umap.set_ylabel('UMAP 2')

        # Initialize waveform plot
        self.ax_waveform.set_title('Select a point to view waveforms')
        self.ax_waveform.set_xlabel('Time Point')
        self.ax_waveform.set_ylabel('Amplitude')

        # Connect click event
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

        # Add selected point marker
        self.selected_point = None

        plt.tight_layout()

    def _plot_umap_points(self):
        """Plot UMAP points with electrode-specific colors if multiple electrodes"""
        if (self.mode == 'unsorted' and
            (self.electrode == -1 or isinstance(self.electrode, list)) and
                len(self.electrode_info) > 1):

            # Create color map for electrodes
            electrode_nums = [info[0] for info in self.electrode_info]
            colors = plt.cm.tab10(np.linspace(0, 1, len(electrode_nums)))
            electrode_color_map = dict(zip(electrode_nums, colors))

            # Create color array for each point
            point_colors = []
            for i, label in enumerate(self.umap_labels):
                # Handle different label formats (Electrode_X_waveform_Y vs KMeans_centroid_X)
                if label.startswith('Electrode_'):
                    electrode_num = int(label.split('_')[1])
                elif label.startswith('KMeans_centroid_'):
                    # For K-means centroids, use the tracked electrode information
                    cluster_id = int(label.split('_')[2])
                    electrode_num = self.cluster_electrode_info.get(
                        cluster_id, list(electrode_color_map.keys())[0])
                else:
                    electrode_num = list(electrode_color_map.keys())[
                        0]  # Fallback
                point_colors.append(electrode_color_map[electrode_num])

            # Plot points with electrode-specific colors
            self.scatter = self.ax_umap.scatter(
                self.umap_embedding[:, 0],
                self.umap_embedding[:, 1],
                c=point_colors,
                alpha=0.7,
                s=30,
                edgecolors='white',
                linewidth=0.5
            )

            # Add legend
            legend_elements = []
            for electrode_num, color in electrode_color_map.items():
                count = next(
                    info[1] for info in self.electrode_info if info[0] == electrode_num)
                legend_elements.append(plt.Line2D([0], [0], marker='o', color='w',
                                                  markerfacecolor=color, markersize=8,
                                                  label=f'Electrode {electrode_num} ({count})'))

            self.ax_umap.legend(handles=legend_elements, loc='upper right',
                                bbox_to_anchor=(1.0, 1.0), framealpha=0.9)
        else:
            # Single electrode or sorted mode - use single color
            self.scatter = self.ax_umap.scatter(
                self.umap_embedding[:, 0],
                self.umap_embedding[:, 1],
                c='red',
                alpha=0.7,
                s=30,
                edgecolors='white',
                linewidth=0.5
            )

    def _plot_kde_background(self):
        """Plot KDE density background for UMAP points"""
        if len(self.umap_embedding) < 3:
            # Need at least 3 points for KDE
            return

        try:
            # Create KDE from UMAP embedding
            kde = gaussian_kde(self.umap_embedding.T)

            # Set custom bandwidth if provided
            if self.kde_bandwidth is not None:
                kde.set_bandwidth(float(self.kde_bandwidth))

            # Create a grid for plotting the KDE
            x_min, x_max = self.umap_embedding[:, 0].min(
            ), self.umap_embedding[:, 0].max()
            y_min, y_max = self.umap_embedding[:, 1].min(
            ), self.umap_embedding[:, 1].max()

            # Add some padding
            x_range = x_max - x_min
            y_range = y_max - y_min
            x_pad = x_range * 0.1
            y_pad = y_range * 0.1

            x_min -= x_pad
            x_max += x_pad
            y_min -= y_pad
            y_max += y_pad

            # Create grid
            xx, yy = np.meshgrid(
                np.linspace(x_min, x_max, 100),
                np.linspace(y_min, y_max, 100)
            )

            # Evaluate KDE on grid
            grid_coords = np.vstack([xx.ravel(), yy.ravel()])
            density = kde(grid_coords).reshape(xx.shape)

            # Plot KDE as contour/contourf
            self.ax_umap.contourf(
                xx, yy, density, levels=20, alpha=0.3, cmap='jet')
            self.ax_umap.contour(xx, yy, density, levels=10,
                                 alpha=0.5, colors='navy', linewidths=0.5)

        except Exception as e:
            print(f"Warning: Could not create KDE background: {e}")
            # Continue without KDE if it fails
            pass

    def on_click(self, event):
        """Handle click events on the UMAP plot"""
        if event.inaxes != self.ax_umap:
            return

        # Find the closest point to the click
        click_point = np.array([event.xdata, event.ydata])
        distances = np.sqrt(
            np.sum((self.umap_embedding - click_point)**2, axis=1))
        closest_idx = np.argmin(distances)

        # Update selected point visualization
        if self.selected_point is not None:
            self.selected_point.remove()

        self.selected_point = self.ax_umap.scatter(
            self.umap_embedding[closest_idx, 0],
            self.umap_embedding[closest_idx, 1],
            c='black',
            s=200,
            marker='x',
            linewidths=4
        )

        # Plot the selected data
        self.plot_waveforms(closest_idx)

        # Refresh the plot
        self.fig.canvas.draw()

    def _plot_background_waveforms(self, time_points, waveforms, max_plot=200):
        """Plot background waveforms with consistent styling"""
        if len(waveforms) > max_plot:
            plot_indices = np.random.choice(
                len(waveforms), max_plot, replace=False)
            plot_waveforms = waveforms[plot_indices]
        else:
            plot_waveforms = waveforms

        for wf in plot_waveforms:
            self.ax_waveform.plot(time_points, wf, 'gray',
                                  alpha=0.1, linewidth=0.5)

    def _plot_mean_std_envelope(self, time_points, waveforms, color='blue', label='Mean ± SD'):
        """Plot mean ± std envelope for waveforms"""
        mean_wf = np.mean(waveforms, axis=0)
        std_wf = np.std(waveforms, axis=0)

        self.ax_waveform.fill_between(
            time_points,
            mean_wf - std_wf,
            mean_wf + std_wf,
            alpha=0.2,
            color=color,
            label=label
        )
        return mean_wf, std_wf

    def _finalize_waveform_plot(self, title):
        """Apply consistent formatting to waveform plot"""
        self.ax_waveform.set_title(title)
        self.ax_waveform.set_xlabel('Time Point')
        self.ax_waveform.set_ylabel('Amplitude')
        self.ax_waveform.legend()
        self.ax_waveform.grid(True, alpha=0.3)

    def plot_waveforms(self, data_idx):
        """Plot waveforms for the selected data point"""
        # Clear previous waveform plot
        self.ax_waveform.clear()

        if self.mode == 'unsorted':
            if self.umap_mode == 'kmeans' and hasattr(self, 'kmeans_labels'):
                # For K-means mode, show all waveforms from the selected cluster
                centroid_idx = self.umap_indices[data_idx]
                cluster_waveforms = self.waveform_data[self.kmeans_labels == centroid_idx]
                selected_centroid = self.kmeans_centroids[centroid_idx]

                time_points = np.arange(len(selected_centroid))

                # Plot centroid
                self.ax_waveform.plot(time_points, selected_centroid, 'r-',
                                      linewidth=3, label='K-means Centroid')

                # Plot background waveforms
                self._plot_background_waveforms(time_points, cluster_waveforms)

                # Plot mean ± std envelope
                self._plot_mean_std_envelope(time_points, cluster_waveforms,
                                             color='blue', label='Cluster Mean ± SD')

                if self.electrode == -1 or isinstance(self.electrode, list):
                    # Count waveforms per electrode in this cluster
                    electrode_counts = {}
                    cluster_indices = np.where(
                        self.kmeans_labels == centroid_idx)[0]
                    for idx in cluster_indices:
                        label = self.data_labels[idx]
                        # Handle different label formats
                        if label.startswith('Electrode_'):
                            electrode_num = int(label.split('_')[1])
                            electrode_counts[electrode_num] = electrode_counts.get(
                                electrode_num, 0) + 1

                    electrode_info = ', '.join(
                        [f'E{e}:{c}' for e, c in sorted(electrode_counts.items())])
                    title = f'K-means Cluster {centroid_idx}: {len(cluster_waveforms)} waveforms ({electrode_info})'
                else:
                    title = f'K-means Cluster {centroid_idx}: {len(cluster_waveforms)} waveforms'

            else:
                # For subsample mode, show the specific waveform and nearby waveforms
                actual_idx = self.umap_indices[data_idx]
                selected_waveform = self.waveform_data[actual_idx]

                # Get a window of waveforms around the selected one for context
                window_size = 100
                start_idx = max(0, actual_idx - window_size // 2)
                end_idx = min(len(self.waveform_data),
                              actual_idx + window_size // 2)
                context_waveforms = self.waveform_data[start_idx:end_idx]

                time_points = np.arange(len(selected_waveform))

                # Plot context waveforms
                for i, wf in enumerate(context_waveforms):
                    alpha = 0.3 if start_idx + i != actual_idx else 1.0
                    color = 'gray' if start_idx + i != actual_idx else 'red'
                    linewidth = 0.5 if start_idx + i != actual_idx else 2
                    self.ax_waveform.plot(time_points, wf, color=color,
                                          alpha=alpha, linewidth=linewidth)

                # Plot mean ± std envelope
                self._plot_mean_std_envelope(time_points, context_waveforms,
                                             color='blue', label='Local Mean ± SD')

                if self.electrode == -1 or isinstance(self.electrode, list):
                    # Extract electrode info from label
                    selected_label = self.data_labels[actual_idx]
                    if selected_label.startswith('Electrode_'):
                        electrode_num = int(selected_label.split('_')[1])
                        waveform_num = int(selected_label.split('_')[3])
                        title = f'Electrode {electrode_num}, Waveform {waveform_num} (red) with {len(context_waveforms)} neighbors'
                    else:
                        title = f'Waveform {actual_idx} (red) with {len(context_waveforms)} neighbors'
                else:
                    title = f'Waveform {actual_idx} (red) with {len(context_waveforms)} neighbors'

        else:
            # For sorted data, show all waveforms from the selected unit
            unit_name = self.umap_labels[data_idx]
            unit_waveforms = self.unit_data[unit_name]

            time_points = np.arange(unit_waveforms.shape[1])

            # Calculate and plot mean waveform
            unit_mean = np.mean(unit_waveforms, axis=0)
            self.ax_waveform.plot(time_points, unit_mean, 'b-',
                                  linewidth=2, label='Mean')

            # Plot mean ± std envelope
            self._plot_mean_std_envelope(time_points, unit_waveforms,
                                         color='blue', label='Mean ± SD')

            # Plot background waveforms
            self._plot_background_waveforms(
                time_points, unit_waveforms, max_plot=50)

            title = f'{unit_name}: {unit_waveforms.shape[0]} waveforms'

        # Apply consistent formatting
        self._finalize_waveform_plot(title)

    def show(self):
        """Display the interactive plot"""
        plt.show()

    def close(self):
        """Close the HDF5 file"""
        self.h5.close()

    def clear_cache(self):
        """Clear all cached UMAP embeddings for this dataset"""
        cache_dir = self._ensure_cache_dir()
        if os.path.exists(cache_dir):
            import shutil
            shutil.rmtree(cache_dir)
            print("Cleared UMAP embedding cache")
        else:
            print("No cache directory found")


def main():
    """Main function to run the unit explorer"""

    test_bool = False
    if test_bool:
        args = argparse.Namespace(
            data_dir='/home/abuzarmahmood/Desktop/test_data/AC5_D4_odors_tastes_251102_090233',
            mode='unsorted',
            electrode=23,
            units=None,
            all_units=False,
            umap_mode='kmeans',
            max_waveforms=5000,
            kmeans_k=1000,
            no_pca=False,
            pca_variance=0.95,
            kde_bandwidth=0.1,
            flip_positive=True,
            clear_cache=False
        )
    else:
        parser = argparse.ArgumentParser(
            description='Interactive unit explorer for blech_clust data',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
    Examples:
      # Explore unsorted waveforms from electrode 5 (subsample mode)
      python blech_unit_explorer.py /path/to/data --mode unsorted --electrode 5

      # Explore unsorted waveforms from all electrodes
      python blech_unit_explorer.py /path/to/data --mode unsorted --electrode -1

      # Explore unsorted waveforms from specific electrodes
      python blech_unit_explorer.py /path/to/data --mode unsorted --electrode 1,5,23

      # Explore unsorted waveforms using K-means mode
      python blech_unit_explorer.py /path/to/data --mode unsorted --electrode 5 --umap-mode kmeans --kmeans-k 500

      # Explore multiple electrodes with K-means (PCA enabled by default)
      python blech_unit_explorer.py /path/to/data --mode unsorted --electrode 1,5,23 --umap-mode kmeans

      # Explore all electrodes with K-means, disable PCA
      python blech_unit_explorer.py /path/to/data --mode unsorted --electrode -1 --umap-mode kmeans --no-pca

      # Explore unsorted waveforms with custom PCA variance
      python blech_unit_explorer.py /path/to/data --mode unsorted --electrode 5 --pca-variance 0.9

      # Explore specific sorted units
      python blech_unit_explorer.py /path/to/data --mode sorted --units 0 1 2 5

      # Explore all sorted units with custom PCA variance
      python blech_unit_explorer.py /path/to/data --mode sorted --all-units --pca-variance 0.95

      # Use custom KDE bandwidth for smoother/sharper density visualization
      python blech_unit_explorer.py /path/to/data --mode unsorted --electrode 5 --kde-bandwidth 0.5

      # Flip positive units to focus on shape rather than polarity
      python blech_unit_explorer.py /path/to/data --mode sorted --all-units --flip-positive

      # Clear UMAP embedding cache
      python blech_unit_explorer.py /path/to/data --clear-cache
            """
        )

        parser.add_argument(
            'data_dir', help='Path to blech_clust data directory')
        parser.add_argument('--mode', choices=['unsorted', 'sorted'],
                            default='sorted', help='Visualization mode')

        def parse_electrode_arg(electrode_str):
            """Parse electrode argument - can be single int, -1, or comma-separated list"""
            if str(electrode_str) == '-1':
                return -1
            elif ',' in str(electrode_str):
                try:
                    return [int(x.strip()) for x in str(electrode_str).split(',')]
                except ValueError:
                    raise argparse.ArgumentTypeError(
                        f"Invalid electrode list: {electrode_str}")
            else:
                try:
                    return int(electrode_str)
                except ValueError:
                    raise argparse.ArgumentTypeError(
                        f"Invalid electrode number: {electrode_str}")

        parser.add_argument('--electrode', type=parse_electrode_arg,
                            help='Electrode number (required for unsorted mode). Use -1 for all electrodes, or comma-separated list (e.g., "1,5,23").')
        parser.add_argument('--units', type=int, nargs='+',
                            help='Unit numbers to visualize (for sorted mode)')
        parser.add_argument('--all-units', action='store_true',
                            help='Visualize all available sorted units')
        parser.add_argument('--umap-mode', choices=['subsample', 'kmeans'],
                            default='subsample', help='Method for handling large datasets')
        parser.add_argument('--max-waveforms', type=int, default=5000,
                            help='Maximum waveforms for subsample mode')
        parser.add_argument('--kmeans-k', type=int, default=1000,
                            help='Number of K-means clusters for kmeans mode')
        parser.add_argument('--no-pca', action='store_true',
                            help='Disable PCA before UMAP (PCA is enabled by default)')
        parser.add_argument('--pca-variance', type=float, default=0.95,
                            help='Variance to retain with PCA (0.0-1.0)')
        parser.add_argument('--kde-bandwidth', type=float, default=0.1,
                            help='KDE bandwidth (None for automatic selection)')
        parser.add_argument('--flip-positive', action='store_true',
                            help='Flip units with positive deflections to negative')
        parser.add_argument('--clear-cache', action='store_true',
                            help='Clear UMAP embedding cache and exit')

    args = parser.parse_args()

    # Validate arguments
    if args.mode == 'unsorted' and args.electrode is None:
        parser.error("--electrode is required for unsorted mode")

    if args.mode == 'sorted' and not args.all_units and args.units is None:
        parser.error(
            "Either --units or --all-units is required for sorted mode")

    if float(args.pca_variance) <= 0 or args.pca_variance > 1:
        parser.error("--pca-variance must be between 0 and 1")

    # Handle cache clearing
    if args.clear_cache:
        # Create a temporary explorer instance just to access the cache clearing method
        try:
            temp_explorer = BllechUnitExplorer.__new__(BllechUnitExplorer)
            temp_explorer.data_dir = args.data_dir
            temp_explorer.clear_cache()
        except Exception as e:
            print(f"Error clearing cache: {e}")
            return 1
        return 0

    try:
        explorer = BllechUnitExplorer(
            args.data_dir,
            mode=args.mode,
            electrode=args.electrode,
            units=args.units,
            all_units=args.all_units,
            umap_mode=args.umap_mode,
            max_waveforms=args.max_waveforms,
            kmeans_k=args.kmeans_k,
            use_pca=not args.no_pca,
            pca_variance=args.pca_variance,
            kde_bandwidth=args.kde_bandwidth,
            flip_positive=args.flip_positive
        )
        explorer.load_metadata()
        explorer.initialize()

        print("Click on points in the UMAP plot to explore waveforms")
        print("Close the plot window to exit")
        explorer.show()

    except Exception as e:
        print(f"Error: {e}")
        return 1
    finally:
        if 'explorer' in locals():
            explorer.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
