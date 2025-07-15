"""
Code to handle electrode and cluster handling operations

High-level implementation:

    1. `ElectrodeHandler` class:
        - Load data for all putative waveforms for an electroe
        - Handle I/O of all clusters present on the electrode
        - Container for all clusters on an electrode
    2. `ClusterHandler` class:
        - Container for waveforms and spiketimes of a single cluster
        - Attribute to be a `unit`
        - Methods to merge or split clusters
            - Can overload `__add__` and `__div__` to merge or split clusters
        - Methods to handle visualization of waveforms, spiketimes, latent space representation
"""


class ElectrodeHandler:
    def __init__(self, electrode_data):
        # Initialize with electrode data
        self.electrode_data = electrode_data
        self.clusters = []

    def load_data(self):
        # Load data for all putative waveforms for an electrode
        pass

    def handle_io(self):
        # Handle I/O of all clusters present on the electrode
        pass

    def add_cluster(self, cluster):
        # Add a cluster to the electrode
        self.clusters.append(cluster)


class ClusterHandler:
    def __init__(self, waveforms, spiketimes):
        # Initialize with waveforms and spiketimes
        self.waveforms = waveforms
        self.spiketimes = spiketimes
        self.unit = None

    def merge(self, other):
        # Merge with another cluster
        pass

    def split(self):
        # Split the cluster
        pass

    def visualize(self):
        # Visualize waveforms, spiketimes, and latent space representation
        pass

    def __add__(self, other):
        # Overload + operator to merge clusters
        return self.merge(other)

    def __div__(self, other):
        # Overload / operator to split clusters
        return self.split()
