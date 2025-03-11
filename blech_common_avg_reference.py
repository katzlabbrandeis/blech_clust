# Run through all the raw electrode data,
# and subtract a common average reference from every electrode's recording
# The user specifies the electrodes to be used as a common average group

# Import stuff!
import tables
import numpy as np
import os
import easygui
import sys
from tqdm import tqdm
import glob
import json
import matplotlib.pyplot as plt
from sklearn.mixture import BayesianGaussianMixture
from sklearn.decomposition import PCA
from utils.blech_utils import imp_metadata


def get_electrode_by_name(raw_electrodes, name):
    """
    Get the electrode data from the list of raw electrodes
    by the name of the electrode
    """
    str_name = f"electrode{name:02}"
    wanted_electrode_ind = [
        x for x in raw_electrodes if str_name in x._v_pathname][0]
    return wanted_electrode_ind

def cluster_electrodes(electrode_data, n_components=10, n_iter=100, threshold=1e-3):
    """
    Perform Bayesian Gaussian Mixture clustering on electrode data
    
    Parameters:
    -----------
    electrode_data : numpy.ndarray
        Array of electrode data features for clustering
    n_components : int
        Maximum number of mixture components
    n_iter : int
        Number of iterations for the optimization
    threshold : float
        Convergence threshold
        
    Returns:
    --------
    predictions : numpy.ndarray
        Cluster assignments for each electrode
    model : BayesianGaussianMixture
        Fitted clustering model
    """
    # Perform Bayesian Gaussian Mixture clustering
    model = BayesianGaussianMixture(
        n_components=n_components, 
        covariance_type='full', 
        tol=threshold, 
        max_iter=n_iter,
        random_state=42
    )
    model.fit(electrode_data)
    predictions = model.predict(electrode_data)
    
    return predictions, model

def extract_electrode_features(raw_electrodes, electrode_indices, n_samples=10000):
    """
    Extract features from electrode data for clustering
    
    Parameters:
    -----------
    raw_electrodes : list
        List of electrode nodes from HDF5 file
    electrode_indices : list
        List of electrode indices to process
    n_samples : int
        Number of samples to use for feature extraction
        
    Returns:
    --------
    features : numpy.ndarray
        Array of features for each electrode
    """
    # Extract statistical features from each electrode
    n_electrodes = len(electrode_indices)
    features = np.zeros((n_electrodes, 4))
    
    for i, electrode_num in enumerate(electrode_indices):
        electrode_data = get_electrode_by_name(raw_electrodes, electrode_num)[:]
        
        # Use a subset of samples for efficiency
        if len(electrode_data) > n_samples:
            indices = np.random.choice(len(electrode_data), n_samples, replace=False)
            electrode_data = electrode_data[indices]
        
        # Extract statistical features
        features[i, 0] = np.mean(electrode_data)
        features[i, 1] = np.std(electrode_data)
        features[i, 2] = np.median(electrode_data)
        features[i, 3] = np.percentile(electrode_data, 75) - np.percentile(electrode_data, 25)
    
    # Normalize features
    features = (features - np.mean(features, axis=0)) / np.std(features, axis=0)
    
    return features

def plot_clustered_electrodes(features, predictions, electrode_indices, output_dir):
    """
    Plot the clustered electrodes
    
    Parameters:
    -----------
    features : numpy.ndarray
        Array of electrode features
    predictions : numpy.ndarray
        Cluster assignments for each electrode
    electrode_indices : list
        List of electrode indices
    output_dir : str
        Directory to save the plot
    """
    # Reduce dimensionality for visualization if needed
    if features.shape[1] > 2:
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(features)
    else:
        features_2d = features
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    # Plot points colored by cluster
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], 
                          c=predictions, cmap='viridis', s=100, alpha=0.8)
    
    # Add electrode labels
    for i, electrode_num in enumerate(electrode_indices):
        plt.annotate(str(electrode_num), (features_2d[i, 0], features_2d[i, 1]),
                     fontsize=8, ha='center', va='center')
    
    # Add colorbar and labels
    plt.colorbar(scatter, label='Cluster')
    plt.title('Electrode Clustering Results')
    plt.xlabel('Feature Dimension 1')
    plt.ylabel('Feature Dimension 2')
    
    # Save the plot
    plot_path = os.path.join(output_dir, 'electrode_clusters.png')
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Clustering plot saved to: {plot_path}")
    return plot_path

def cluster_electrodes(electrode_data, n_components=10, n_iter=100, threshold=1e-3):
    """
    Perform Bayesian Gaussian Mixture clustering on electrode data
    
    Parameters:
    -----------
    electrode_data : numpy.ndarray
        Array of electrode data features for clustering
    n_components : int
        Maximum number of mixture components
    n_iter : int
        Number of iterations for the optimization
    threshold : float
        Convergence threshold
        
    Returns:
    --------
    predictions : numpy.ndarray
        Cluster assignments for each electrode
    model : BayesianGaussianMixture
        Fitted clustering model
    """
    # Perform Bayesian Gaussian Mixture clustering
    model = BayesianGaussianMixture(
        n_components=n_components, 
        covariance_type='full', 
        tol=threshold, 
        max_iter=n_iter,
        random_state=42
    )
    model.fit(electrode_data)
    predictions = model.predict(electrode_data)
    
    return predictions, model

def extract_electrode_features(raw_electrodes, electrode_indices, n_samples=10000):
    """
    Extract features from electrode data for clustering
    
    Parameters:
    -----------
    raw_electrodes : list
        List of electrode nodes from HDF5 file
    electrode_indices : list
        List of electrode indices to process
    n_samples : int
        Number of samples to use for feature extraction
        
    Returns:
    --------
    features : numpy.ndarray
        Array of features for each electrode
    """
    # Extract statistical features from each electrode
    n_electrodes = len(electrode_indices)
    features = np.zeros((n_electrodes, 4))
    
    for i, electrode_num in enumerate(electrode_indices):
        electrode_data = get_electrode_by_name(raw_electrodes, electrode_num)[:]
        
        # Use a subset of samples for efficiency
        if len(electrode_data) > n_samples:
            indices = np.random.choice(len(electrode_data), n_samples, replace=False)
            electrode_data = electrode_data[indices]
        
        # Extract statistical features
        features[i, 0] = np.mean(electrode_data)
        features[i, 1] = np.std(electrode_data)
        features[i, 2] = np.median(electrode_data)
        features[i, 3] = np.percentile(electrode_data, 75) - np.percentile(electrode_data, 25)
    
    # Normalize features
    features = (features - np.mean(features, axis=0)) / np.std(features, axis=0)
    
    return features

def plot_clustered_electrodes(features, predictions, electrode_indices, output_dir):
    """
    Plot the clustered electrodes
    
    Parameters:
    -----------
    features : numpy.ndarray
        Array of electrode features
    predictions : numpy.ndarray
        Cluster assignments for each electrode
    electrode_indices : list
        List of electrode indices
    output_dir : str
        Directory to save the plot
    """
    # Reduce dimensionality for visualization if needed
    if features.shape[1] > 2:
        pca = PCA(n_components=2)
        features_2d = pca.fit_transform(features)
    else:
        features_2d = features
    
    # Create plot
    plt.figure(figsize=(12, 8))
    
    # Plot points colored by cluster
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], 
                          c=predictions, cmap='viridis', s=100, alpha=0.8)
    
    # Add electrode labels
    for i, electrode_num in enumerate(electrode_indices):
        plt.annotate(str(electrode_num), (features_2d[i, 0], features_2d[i, 1]),
                     fontsize=8, ha='center', va='center')
    
    # Add colorbar and labels
    plt.colorbar(scatter, label='Cluster')
    plt.title('Electrode Clustering Results')
    plt.xlabel('Feature Dimension 1')
    plt.ylabel('Feature Dimension 2')
    
    # Save the plot
    plot_path = os.path.join(output_dir, 'electrode_clusters.png')
    plt.savefig(plot_path)
    plt.close()
    
    print(f"Clustering plot saved to: {plot_path}")
    return plot_path

############################################################
############################################################


# Get name of directory with the data files
metadata_handler = imp_metadata(sys.argv)
dir_name = metadata_handler.dir_name
os.chdir(dir_name)
print(f'Processing : {dir_name}')

# Open the hdf5 file
hf5 = tables.open_file(metadata_handler.hdf5_name, 'r+')

# Read CAR groups from info file
# Every region is a separate group, multiple ports under single region is a separate group,
# emg is a separate group
info_dict = metadata_handler.info_dict
electrode_layout_frame = metadata_handler.layout
# Remove emg and none channels from the electrode layout frame
emg_bool = ~electrode_layout_frame.CAR_group.str.contains('emg')
none_bool = ~electrode_layout_frame.CAR_group.str.contains('none')
fin_bool = np.logical_and(emg_bool, none_bool)
electrode_layout_frame = electrode_layout_frame[fin_bool]

# Since electrodes are already in monotonic numbers (A : 0-31, B: 32-63)
# we can directly pull them
grouped_layout = list(electrode_layout_frame.groupby('CAR_group'))
all_car_group_names = [x[0] for x in grouped_layout]
# Note: electrodes in HDF5 are also saved according to inds
# specified in the layout file
all_car_group_vals = [x[1].electrode_ind.values for x in grouped_layout]

CAR_electrodes = all_car_group_vals
num_groups = len(CAR_electrodes)
print(f" Number of groups : {num_groups}")
for region, vals in zip(all_car_group_names, all_car_group_vals):
    print(f" {region} :: {vals}")

# Pull out the raw electrode nodes of the HDF5 file
raw_electrodes = hf5.list_nodes('/raw')

# First get the common average references by averaging across
# the electrodes picked for each group
print(
    "Calculating common average reference for {:d} groups".format(num_groups))
common_average_reference = np.zeros(
    (num_groups, raw_electrodes[0][:].shape[0]))
print('Calculating mean values')
for group in range(num_groups):
    print('Processing Group {}'.format(group))
    # First add up the voltage values from each electrode to the same array
    # then divide by number of electrodes to get the average
    # This is more memory efficient than loading all the electrode data into
    # a single array and then averaging
    for electrode_name in tqdm(CAR_electrodes[group]):
        common_average_reference[group, :] += \
            get_electrode_by_name(raw_electrodes, electrode_name)[:]
    common_average_reference[group, :] /= float(len(CAR_electrodes[group]))

print("Common average reference for {:d} groups calculated".format(num_groups))

# Now run through the raw electrode data and
# subtract the common average reference from each of them
print('Performing background subtraction')
for group_num, group_name in tqdm(enumerate(all_car_group_names)):
    print(f"Processing group {group_name}")
    for electrode_num in tqdm(all_car_group_vals[group_num]):
        # Subtract the common average reference for that group from the
        # voltage data of the electrode
        wanted_electrode = get_electrode_by_name(raw_electrodes, electrode_num)
        referenced_data = wanted_electrode[:] - common_average_reference[group_num]
        # Overwrite the electrode data with the referenced data
        wanted_electrode[:] = referenced_data
        hf5.flush()
        del referenced_data

# Check if auto CAR inference is enabled in the parameters
auto_car_inference = metadata_handler.params_dict.get('auto_car_inference', False)

if auto_car_inference:
    print("\nPerforming automatic CAR group inference using Bayesian clustering...")
    
    # Create a directory for clustering results if it doesn't exist
    cluster_dir = os.path.join(dir_name, 'clustering_results')
    os.makedirs(cluster_dir, exist_ok=True)
    
    # Process each CAR group
    new_car_groups = []
    
    for group_num, group_name in enumerate(all_car_group_names):
        print(f"\nProcessing group {group_name} for clustering")
        
        # Extract features for electrodes in this group
        electrode_indices = all_car_group_vals[group_num]
        features = extract_electrode_features(raw_electrodes, electrode_indices)
        
        # Perform clustering
        predictions, model = cluster_electrodes(features)
        
        # Get the effective number of components (some may have zero weight)
        effective_components = np.sum(model.weights_ > 0.01)
        print(f"Identified {effective_components} effective clusters in group {group_name}")
        
        # Plot the clustering results
        plot_path = plot_clustered_electrodes(features, predictions, electrode_indices, cluster_dir)
        
        # Create new CAR groups based on clustering
        unique_clusters = np.unique(predictions)
        for cluster_id in unique_clusters:
            # Get electrodes in this cluster
            cluster_electrodes = [electrode_indices[i] for i, pred in enumerate(predictions) if pred == cluster_id]
            
            if len(cluster_electrodes) > 0:
                new_group_name = f"{group_name}-cluster{cluster_id}"
                new_car_groups.append((new_group_name, cluster_electrodes))
                print(f"  New CAR group: {new_group_name} with {len(cluster_electrodes)} electrodes")
    
    # Save the new CAR groups to a JSON file
    new_car_groups_dict = {name: electrodes for name, electrodes in new_car_groups}
    car_groups_path = os.path.join(cluster_dir, 'inferred_car_groups.json')
    with open(car_groups_path, 'w') as f:
        json.dump(new_car_groups_dict, f, indent=4)
    
    print(f"\nInferred CAR groups saved to: {car_groups_path}")
    print("You can update your electrode layout file with these new groups for future processing.")

# Check if auto CAR inference is enabled in the parameters
auto_car_inference = metadata_handler.params_dict.get('auto_car_inference', False)

if auto_car_inference:
    print("\nPerforming automatic CAR group inference using Bayesian clustering...")
    
    # Create a directory for clustering results if it doesn't exist
    cluster_dir = os.path.join(dir_name, 'clustering_results')
    os.makedirs(cluster_dir, exist_ok=True)
    
    # Process each CAR group
    new_car_groups = []
    
    for group_num, group_name in enumerate(all_car_group_names):
        print(f"\nProcessing group {group_name} for clustering")
        
        # Extract features for electrodes in this group
        electrode_indices = all_car_group_vals[group_num]
        features = extract_electrode_features(raw_electrodes, electrode_indices)
        
        # Perform clustering
        predictions, model = cluster_electrodes(features)
        
        # Get the effective number of components (some may have zero weight)
        effective_components = np.sum(model.weights_ > 0.01)
        print(f"Identified {effective_components} effective clusters in group {group_name}")
        
        # Plot the clustering results
        plot_path = plot_clustered_electrodes(features, predictions, electrode_indices, cluster_dir)
        
        # Create new CAR groups based on clustering
        unique_clusters = np.unique(predictions)
        for cluster_id in unique_clusters:
            # Get electrodes in this cluster
            cluster_electrodes = [electrode_indices[i] for i, pred in enumerate(predictions) if pred == cluster_id]
            
            if len(cluster_electrodes) > 0:
                new_group_name = f"{group_name}-cluster{cluster_id}"
                new_car_groups.append((new_group_name, cluster_electrodes))
                print(f"  New CAR group: {new_group_name} with {len(cluster_electrodes)} electrodes")
    
    # Save the new CAR groups to a JSON file
    new_car_groups_dict = {name: electrodes for name, electrodes in new_car_groups}
    car_groups_path = os.path.join(cluster_dir, 'inferred_car_groups.json')
    with open(car_groups_path, 'w') as f:
        json.dump(new_car_groups_dict, f, indent=4)
    
    print(f"\nInferred CAR groups saved to: {car_groups_path}")
    print("You can update your electrode layout file with these new groups for future processing.")

hf5.close()
print("Modified electrode arrays written to HDF5 file after "
      "subtracting the common average reference")
