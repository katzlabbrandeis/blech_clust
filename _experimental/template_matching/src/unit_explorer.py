"""
Interactive plot to explore database of positive units
"""

import os
import tables
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import umap
from sklearn.preprocessing import StandardScaler
from scipy import stats

# Set up paths
base_dir = '/home/abuzarmahmood/projects/blech_clust/_experimental/template_matching/'
data_dir = os.path.join(base_dir, 'data')
waveform_h5_path = os.path.join(data_dir, 'final_dataset.h5')

class UnitExplorer:
    def __init__(self, h5_path):
        self.h5 = tables.open_file(h5_path, mode='r')
        self.pos_units = list(self.h5.root.sorted.pos._v_children.keys())
        self.n_pos_units = len(self.pos_units)
        
        # Calculate means for all positive units
        print("Calculating unit means...")
        self.unit_means = []
        self.unit_nodes = []
        for unit_name in self.pos_units:
            unit_node = self.h5.get_node('/sorted/pos/' + unit_name)
            unit_data = unit_node[:]
            unit_mean = np.mean(unit_data, axis=0)
            self.unit_means.append(unit_mean)
            self.unit_nodes.append(unit_node)
        
        self.unit_means = np.array(self.unit_means)
        
        # Standardize the means for UMAP
        scaler = StandardScaler()
        self.unit_means_scaled = scaler.fit_transform(self.unit_means)
        
        # Compute UMAP embedding
        print("Computing UMAP embedding...")
        self.umap_reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)
        self.umap_embedding = self.umap_reducer.fit_transform(self.unit_means_scaled)
        
        # Set up the interactive plot
        self.setup_plot()
        
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
        self.ax_umap.set_title('UMAP of Positive Unit Means\n(Click on points to explore)')
        self.ax_umap.set_xlabel('UMAP 1')
        self.ax_umap.set_ylabel('UMAP 2')
        
        # Initialize waveform plot
        self.ax_waveform.set_title('Select a unit to view waveforms')
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
        
        # Plot the selected unit's waveforms
        self.plot_unit_waveforms(closest_idx)
        
        # Refresh the plot
        self.fig.canvas.draw()
        
    def plot_unit_waveforms(self, unit_idx):
        """Plot waveforms for the selected unit with mean ± std"""
        # Clear previous waveform plot
        self.ax_waveform.clear()
        
        # Get unit data
        unit_node = self.unit_nodes[unit_idx]
        unit_data = unit_node[:]
        unit_name = self.pos_units[unit_idx]
        
        # Calculate statistics
        unit_mean = np.mean(unit_data, axis=0)
        unit_std = np.std(unit_data, axis=0)
        
        # Time points
        time_points = np.arange(len(unit_mean))
        
        # Plot mean waveform
        self.ax_waveform.plot(time_points, unit_mean, 'b-', linewidth=2, label='Mean')
        
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
        n_plot_waveforms = min(50, unit_data.shape[0])
        plot_indices = np.random.choice(unit_data.shape[0], n_plot_waveforms, replace=False)
        for i in plot_indices:
            self.ax_waveform.plot(time_points, unit_data[i], 'gray', alpha=0.1, linewidth=0.5)
        
        # Formatting
        self.ax_waveform.set_title(f'Unit: {unit_name}\n({unit_data.shape[0]} waveforms)')
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
    try:
        explorer = UnitExplorer(waveform_h5_path)
        print(f"Loaded {explorer.n_pos_units} positive units")
        print("Click on points in the UMAP plot to explore unit waveforms")
        explorer.show()
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure the HDF5 file exists and contains the expected structure")
    finally:
        if 'explorer' in locals():
            explorer.close()

if __name__ == "__main__":
    main()
