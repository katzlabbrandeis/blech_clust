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
