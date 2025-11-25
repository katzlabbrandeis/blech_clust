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
from sklearn.cluster import KMeans
from scipy import stats
import warnings

# Add blech_clust utils to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from blech_utils import imp_metadata

class BllechUnitExplorer:
    def __init__(self, data_dir, mode='sorted', electrode=None, units=None, all_units=False,
                 umap_mode='subsample', max_waveforms=5000, kmeans_k=1000):
        """
        Initialize the unit explorer
        
        Parameters:
        -----------
        data_dir : str
            Path to the blech_clust data directory
        mode : str
            'unsorted' for raw waveforms from electrode, 'sorted' for sorted units
        electrode : int
            Electrode number (required for unsorted mode)
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
        """
        self.data_dir = data_dir
        self.mode = mode
        self.electrode = electrode
        self.units = units
        self.all_units = all_units
        self.umap_mode = umap_mode
        self.max_waveforms = max_waveforms
        self.kmeans_k = kmeans_k
        
        # Load metadata
        self.metadata_handler = imp_metadata([[], data_dir])
        self.hdf5_path = os.path.join(data_dir, self.metadata_handler.hdf5_name)
        
        # Open HDF5 file
        self.h5 = tables.open_file(self.hdf5_path, mode='r')
        
        # Load data based on mode
        if mode == 'unsorted':
            self._load_unsorted_data()
        elif mode == 'sorted':
            self._load_sorted_data()
        else:
            raise ValueError("Mode must be 'unsorted' or 'sorted'")
        
        # Calculate UMAP embedding
        self._calculate_umap()
        
        # Set up the interactive plot
        self.setup_plot()
    
    def _load_unsorted_data(self):
        """Load unsorted waveforms from specified electrode"""
        if self.electrode is None:
            raise ValueError("Electrode number required for unsorted mode")
        
        # Load waveforms from spike_waveforms directory (saved as .npy files)
        waveforms_file = os.path.join(
            self.data_dir, 
            'spike_waveforms', 
            f'electrode{self.electrode:02d}', 
            'spike_waveforms.npy'
        )
        
        if not os.path.exists(waveforms_file):
            raise ValueError(f"No waveforms found for electrode {self.electrode} at {waveforms_file}")
        
        self.waveform_data = np.load(waveforms_file)
        self.data_labels = [f'Electrode_{self.electrode}_waveform_{i}' 
                           for i in range(len(self.waveform_data))]
        
        # Prepare data for UMAP based on the selected mode
        self._prepare_umap_data()
        
        print(f"Loaded {len(self.waveform_data)} waveforms from electrode {self.electrode}")
        print(f"Using {len(self.umap_data)} data points for UMAP visualization ({self.umap_mode} mode)")
    
    def _load_sorted_data(self):
        """Load sorted units from HDF5 file"""
        if '/sorted_units' not in self.h5:
            raise ValueError("No sorted units found in HDF5 file")
        
        sorted_units_group = self.h5.get_node('/sorted_units')
        available_units = [node._v_name for node in sorted_units_group._f_iter_nodes()]
        
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
            raise ValueError("Must specify either --units or --all-units for sorted mode")
        
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
        self.umap_data = self.unit_means
        self.umap_labels = self.data_labels
        
        print(f"Loaded {len(selected_units)} sorted units")
    
    def _prepare_umap_data(self):
        """Prepare data for UMAP based on the selected mode"""
        if self.mode == 'sorted':
            # For sorted data, always use unit means
            self.umap_data = self.unit_means
            self.umap_labels = self.data_labels
            self.umap_indices = np.arange(len(self.unit_means))
            return
        
        # For unsorted data, apply the selected UMAP mode
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
                print(f"Applying K-means with k={self.kmeans_k} for UMAP visualization...")
                kmeans = KMeans(n_clusters=self.kmeans_k, random_state=42, n_init=10)
                kmeans.fit(self.waveform_data)
                
                # Use centroids for UMAP
                self.umap_data = kmeans.cluster_centers_
                self.umap_labels = [f'KMeans_centroid_{i}' for i in range(self.kmeans_k)]
                
                # Store cluster assignments to map back to original data
                self.kmeans_labels = kmeans.labels_
                self.kmeans_centroids = kmeans.cluster_centers_
                self.umap_indices = np.arange(self.kmeans_k)  # Indices into centroids
            else:
                # Not enough data for K-means, fall back to using all data
                self.umap_data = self.waveform_data
                self.umap_labels = self.data_labels
                self.umap_indices = np.arange(len(self.waveform_data))
                self.kmeans_labels = None
        else:
            raise ValueError(f"Unknown umap_mode: {self.umap_mode}")
    
    def _calculate_umap(self):
        """Calculate UMAP embedding of the data"""
        print("Computing UMAP embedding...")
        
        # Standardize the data for UMAP
        scaler = StandardScaler()
        if self.mode == 'unsorted':
            # For unsorted data, use the waveforms directly
            data_scaled = scaler.fit_transform(self.umap_data)
        else:
            # For sorted data, use the unit means
            data_scaled = scaler.fit_transform(self.umap_data)
        
        # Compute UMAP embedding
        n_neighbors = min(15, len(data_scaled) - 1)
        self.umap_reducer = umap.UMAP(
            n_neighbors=n_neighbors, 
            min_dist=0.1, 
            random_state=42
        )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.umap_embedding = self.umap_reducer.fit_transform(data_scaled)
    
    def setup_plot(self):
        """Set up the interactive matplotlib plot"""
        self.fig, (self.ax_umap, self.ax_waveform) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot UMAP embedding
        self.scatter = self.ax_umap.scatter(
            self.umap_embedding[:, 0], 
            self.umap_embedding[:, 1],
            c='blue', 
            alpha=0.6, 
            s=30
        )
        
        title = f'UMAP of {self.mode.title()} Data\n(Click on points to explore)'
        if self.mode == 'unsorted':
            title += f' - Electrode {self.electrode} ({self.umap_mode} mode)'
        
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
    
    def on_click(self, event):
        """Handle click events on the UMAP plot"""
        if event.inaxes != self.ax_umap:
            return
        
        # Find the closest point to the click
        click_point = np.array([event.xdata, event.ydata])
        distances = np.sqrt(np.sum((self.umap_embedding - click_point)**2, axis=1))
        closest_idx = np.argmin(distances)
        
        # Update selected point visualization
        if self.selected_point is not None:
            self.selected_point.remove()
        
        self.selected_point = self.ax_umap.scatter(
            self.umap_embedding[closest_idx, 0],
            self.umap_embedding[closest_idx, 1],
            c='red',
            s=100,
            marker='x',
            linewidths=3
        )
        
        # Plot the selected data
        self.plot_waveforms(closest_idx)
        
        # Refresh the plot
        self.fig.canvas.draw()
    
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
                
                # Time points
                time_points = np.arange(len(selected_centroid))
                
                # Plot centroid
                self.ax_waveform.plot(time_points, selected_centroid, 'r-', 
                                    linewidth=3, label='K-means Centroid')
                
                # Plot a subset of cluster waveforms
                max_plot_waveforms = 200
                if len(cluster_waveforms) > max_plot_waveforms:
                    plot_indices = np.random.choice(len(cluster_waveforms), 
                                                  max_plot_waveforms, replace=False)
                    plot_waveforms = cluster_waveforms[plot_indices]
                else:
                    plot_waveforms = cluster_waveforms
                
                for wf in plot_waveforms:
                    self.ax_waveform.plot(time_points, wf, 'gray', 
                                        alpha=0.1, linewidth=0.5)
                
                # Calculate statistics for the cluster
                cluster_mean = np.mean(cluster_waveforms, axis=0)
                cluster_std = np.std(cluster_waveforms, axis=0)
                
                # Plot mean ± std as filled area
                self.ax_waveform.fill_between(
                    time_points,
                    cluster_mean - cluster_std,
                    cluster_mean + cluster_std,
                    alpha=0.2,
                    color='blue',
                    label='Cluster Mean ± SD'
                )
                
                title = f'K-means Cluster {centroid_idx}: {len(cluster_waveforms)} waveforms'
                
            else:
                # For subsample mode, show the specific waveform and nearby waveforms
                actual_idx = self.umap_indices[data_idx]
                selected_waveform = self.waveform_data[actual_idx]
                
                # Get a window of waveforms around the selected one for context
                window_size = 100
                start_idx = max(0, actual_idx - window_size // 2)
                end_idx = min(len(self.waveform_data), actual_idx + window_size // 2)
                context_waveforms = self.waveform_data[start_idx:end_idx]
                
                # Time points
                time_points = np.arange(len(selected_waveform))
                
                # Plot context waveforms
                for i, wf in enumerate(context_waveforms):
                    alpha = 0.3 if start_idx + i != actual_idx else 1.0
                    color = 'gray' if start_idx + i != actual_idx else 'red'
                    linewidth = 0.5 if start_idx + i != actual_idx else 2
                    self.ax_waveform.plot(time_points, wf, color=color, 
                                        alpha=alpha, linewidth=linewidth)
                
                # Calculate statistics for the context window
                context_mean = np.mean(context_waveforms, axis=0)
                context_std = np.std(context_waveforms, axis=0)
                
                # Plot mean ± std as filled area
                self.ax_waveform.fill_between(
                    time_points,
                    context_mean - context_std,
                    context_mean + context_std,
                    alpha=0.2,
                    color='blue',
                    label='Local Mean ± SD'
                )
                
                title = f'Waveform {actual_idx} (red) with {len(context_waveforms)} neighbors'
            
        else:
            # For sorted data, show all waveforms from the selected unit
            unit_name = self.umap_labels[data_idx]
            unit_waveforms = self.unit_data[unit_name]
            
            # Calculate statistics
            unit_mean = np.mean(unit_waveforms, axis=0)
            unit_std = np.std(unit_waveforms, axis=0)
            
            # Time points
            time_points = np.arange(len(unit_mean))
            
            # Plot mean waveform
            self.ax_waveform.plot(time_points, unit_mean, 'b-', 
                                linewidth=2, label='Mean')
            
            # Plot mean ± std as filled area
            self.ax_waveform.fill_between(
                time_points,
                unit_mean - unit_std,
                unit_mean + unit_std,
                alpha=0.3,
                color='blue',
                label='Mean ± SD'
            )
            
            # Plot a subset of individual waveforms for context
            n_plot_waveforms = min(50, unit_waveforms.shape[0])
            plot_indices = np.random.choice(unit_waveforms.shape[0], 
                                          n_plot_waveforms, replace=False)
            for i in plot_indices:
                self.ax_waveform.plot(time_points, unit_waveforms[i], 
                                    'gray', alpha=0.1, linewidth=0.5)
            
            title = f'{unit_name}: {unit_waveforms.shape[0]} waveforms'
        
        # Formatting
        self.ax_waveform.set_title(title)
        self.ax_waveform.set_xlabel('Time Point')
        self.ax_waveform.set_ylabel('Amplitude')
        self.ax_waveform.legend()
        self.ax_waveform.grid(True, alpha=0.3)
    
    def show(self):
        """Display the interactive plot"""
        plt.show()
    
    def close(self):
        """Close the HDF5 file"""
        self.h5.close()

def main():
    """Main function to run the unit explorer"""
    parser = argparse.ArgumentParser(
        description='Interactive unit explorer for blech_clust data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Explore unsorted waveforms from electrode 5 (subsample mode)
  python blech_unit_explorer.py /path/to/data --mode unsorted --electrode 5
  
  # Explore unsorted waveforms using K-means mode
  python blech_unit_explorer.py /path/to/data --mode unsorted --electrode 5 --umap-mode kmeans --kmeans-k 500
  
  # Explore specific sorted units
  python blech_unit_explorer.py /path/to/data --mode sorted --units 0 1 2 5
  
  # Explore all sorted units
  python blech_unit_explorer.py /path/to/data --mode sorted --all-units
        """
    )
    
    parser.add_argument('data_dir', help='Path to blech_clust data directory')
    parser.add_argument('--mode', choices=['unsorted', 'sorted'], 
                       default='sorted', help='Visualization mode')
    parser.add_argument('--electrode', type=int, 
                       help='Electrode number (required for unsorted mode)')
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
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.mode == 'unsorted' and args.electrode is None:
        parser.error("--electrode is required for unsorted mode")
    
    if args.mode == 'sorted' and not args.all_units and args.units is None:
        parser.error("Either --units or --all-units is required for sorted mode")
    
    try:
        explorer = BllechUnitExplorer(
            args.data_dir, 
            mode=args.mode,
            electrode=args.electrode,
            units=args.units,
            all_units=args.all_units,
            umap_mode=args.umap_mode,
            max_waveforms=args.max_waveforms,
            kmeans_k=args.kmeans_k
        )
        
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
