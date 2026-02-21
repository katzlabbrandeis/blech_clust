from blech_clust.utils.makeRaisedCosBasis import gen_raised_cosine_basis
import os
import tables
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.neighbors import NeighborhoodComponentsAnalysis
from numpy.linalg import norm
from scipy.optimize import minimize
from pprint import pprint as pp
from itertools import product
import json
import pandas as pd

base_dir = '/home/abuzarmahmood/projects/blech_clust/_experimental/template_matching/'
data_dir = os.path.join(base_dir, 'data')
plot_dir = os.path.join(base_dir, 'plots')
artifacts_dir = os.path.join(base_dir, 'artifacts')

for directory in [data_dir, plot_dir, artifacts_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)

waveform_h5_path = os.path.join(data_dir, 'final_dataset.h5')

h5 = tables.open_file(waveform_h5_path, mode='r')

##############################
# Visualize some positive and negative waveforms

# h5 structure
# root
#  - sorted
#     - pos
#     - neg

pos_units = list(h5.root.sorted.pos._v_children.keys())
neg_units = list(h5.root.sorted.neg._v_children.keys())

n_pos_units = len(pos_units)
n_neg_units = len(neg_units)

n_rand_units = 5
rand_pos_inds = np.random.choice(
    len(pos_units), size=n_rand_units, replace=False)
rand_neg_inds = np.random.choice(
    len(neg_units), size=n_rand_units, replace=False)

rand_pos_units = [h5.get_node('/sorted/pos/' + pos_units[i])
                  for i in rand_pos_inds]
rand_neg_units = [h5.get_node('/sorted/neg/' + neg_units[i])
                  for i in rand_neg_inds]

downsample_factor = 10
fig, axs = plt.subplots(2, n_rand_units, figsize=(15, 6))
for i in range(n_rand_units):
    # Extract all waveforms for this positive unit
    pos_waveforms = rand_pos_units[i][:]
    # Extract all waveforms for this negative unit
    neg_waveforms = rand_neg_units[i][:]

    # Plot positive waveforms
    axs[0, i].plot(pos_waveforms[::downsample_factor].T,
                   color='blue', alpha=0.1)
    axs[0, i].set_title(f'Pos Unit {rand_pos_inds[i]}')
    axs[0, i].set_xlabel('Time')
    axs[0, i].set_ylabel('Amplitude')

    # Plot negative waveforms
    axs[1, i].plot(neg_waveforms[::downsample_factor].T,
                   color='red', alpha=0.1)
    axs[1, i].set_title(f'Neg Unit {rand_neg_inds[i]}')
    axs[1, i].set_xlabel('Time')
    axs[1, i].set_ylabel('Amplitude')
plt.tight_layout()
plt.show()

##############################
# Fit a Logistic classifier between pairs of positive and negative units

n_fits = 1000
all_combinations = [(i, j) for i in range(n_pos_units)
                    for j in range(n_neg_units)]
selected_comb_inds = np.random.choice(
    len(all_combinations), size=n_fits, replace=False)
selected_combinations = [all_combinations[i] for i in selected_comb_inds]

logistic_models = []
model_accuracies = []
X_transformed_list = []
y_list = []
max_waveforms = 1000
for pos_ind, neg_ind in tqdm(selected_combinations):
    pos_unit = h5.get_node('/sorted/pos/' + pos_units[pos_ind])
    neg_unit = h5.get_node('/sorted/neg/' + neg_units[neg_ind])

    pos_waveforms = pos_unit[:]
    neg_waveforms = neg_unit[:]

    n_pos_waveforms = pos_waveforms.shape[0]
    n_neg_waveforms = neg_waveforms.shape[0]

    if n_pos_waveforms > max_waveforms:
        pos_waveforms = pos_waveforms[np.random.choice(
            n_pos_waveforms, size=max_waveforms, replace=False)]
    if n_neg_waveforms > max_waveforms:
        neg_waveforms = neg_waveforms[np.random.choice(
            n_neg_waveforms, size=max_waveforms, replace=False)]

    X = np.vstack((pos_waveforms, neg_waveforms))
    y = np.hstack(
        (np.ones(pos_waveforms.shape[0]), np.zeros(neg_waveforms.shape[0])))

    # Zscore data to focus on shape differences
    X = stats.zscore(X, axis=1)

    clf = LogisticRegression(max_iter=5000)
    clf.fit(X, y)
    accuracy = clf.score(X, y)
    model_accuracies.append(accuracy)

    logistic_models.append({
        'model': clf,
        'pos_unit': pos_units[pos_ind],
        'neg_unit': neg_units[neg_ind]
    })

    # Extract coefficients and intercept
    coef = clf.coef_.flatten()
    intercept = clf.intercept_[0]

    # Transform data and plot
    X_transformed = X @ coef + intercept

    # Check that transformation matches
    y_pred = clf.predict(X)
    assert np.array_equal(y_pred, (X_transformed > 0).astype(int))

    X_transformed_list.append(X_transformed)
    y_list.append(y)

    #
    # cmap = plt.get_cmap('bwr')
    # fig, ax = plt.subplots(2, 2, figsize=(15,15))
    # for label in [0, 1]:
    #     ax[0,label].plot(X[y == label].T, color=cmap(label), alpha=0.1)
    #     ax[0,label].set_title(f'Class {label} Waveforms, Total: {np.sum(y == label)}')
    # # Plot misclassified waveforms
    # y_pred = clf.predict(X)
    # for label in [0, 1]:
    #     misclassified = X[(y == label) & (y != y_pred)]
    #     if misclassified.shape[0] > 0:
    #         ax[1,label].plot(misclassified.T, color='red', alpha=0.3)
    #         ax[1,label].set_title(f'Class {label} Misclassified Waveforms, Total: {misclassified.shape[0]}')
    #     else:
    #         ax[1,label].set_title(f'Class {label} Misclassified Waveforms, Total: 0')
    # fig.suptitle(f'Logistic Regression between Pos Unit {pos_units[pos_ind]} and Neg Unit {neg_units[neg_ind]}\nAccuracy: {accuracy:.2f}')
    # plt.show()

# Plot X and y for some models
n_plot_models = 5
plot_inds = np.random.choice(n_fits, size=n_plot_models, replace=False)
fig, axs = plt.subplots(n_plot_models, 1, figsize=(10, 4 * n_plot_models))
for i, dat_ind in enumerate(plot_inds):
    this_X_transformed = X_transformed_list[dat_ind]
    this_y = y_list[dat_ind]
    for this_label in [0, 1]:
        axs[i].hist(
            this_X_transformed[this_y == this_label], bins=30, alpha=0.5,
            label=f'Class {this_label}', density=True
        )
    axs[i].set_title(
        f'Model {dat_ind} - Pos Unit: {logistic_models[dat_ind]["pos_unit"]}, Neg Unit: {logistic_models[dat_ind]["neg_unit"]}, Accuracy: {model_accuracies[dat_ind]:.2f}')
plt.tight_layout()
plt.show()

# Extract coefficients from all models
all_coefs = np.array([model['model'].coef_.flatten()
                     for model in logistic_models])

plt.imshow(all_coefs, aspect='auto', cmap='bwr')
plt.colorbar(label='Coefficient Value')
plt.title('Logistic Regression Coefficients for All Models')
plt.xlabel('Time Point')
plt.ylabel('Model Index')
plt.show()

# Perform PCA on coefficients to extract main modes of variation
pca_obj = PCA(n_components=10)
pca_obj.fit(all_coefs)
explained_variance = pca_obj.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

# Plot components and explained variance
fig, axs = plt.subplots(2, 1, figsize=(10, 8))
# for i in range(len(explained_variance)):
for i in range(5):
    axs[0].plot(pca_obj.components_[i], label=f'PC {i+1}')
axs[0].set_title('Top 3 PCA Components of Logistic Regression Coefficients')
axs[0].set_xlabel('Time Point')
axs[0].set_ylabel('Coefficient Value')
axs[0].legend()
axs[1].plot(cumulative_variance, marker='o')
axs[1].set_title('Cumulative Explained Variance by PCA Components')
axs[1].set_xlabel('Number of Components')
axs[1].set_ylabel('Cumulative Explained Variance')
plt.tight_layout()
plt.show()

##############################
# Test if dot product with first 5 PCA components separates classes
n_components_to_use = 5
top_pca_components = pca_obj.components_[:n_components_to_use]
X_transformed_pca = X @ top_pca_components.T

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(
    *(X_transformed_pca[y == 1, :3].T),
    c='blue', label='Positive', alpha=0.5
)
ax.scatter(
    *(X_transformed_pca[y == 0, :3].T),
    c='red', label='Negative', alpha=0.5
)
ax.set_title('PCA Transformed Waveforms (Top 3 Components)')
ax.set_xlabel('PC 1')
ax.set_ylabel('PC 2')
ax.set_zlabel('PC 3')
ax.legend()
plt.show()

##############################
# Try Neighborhood Components Analysis (NCA) as an alternative

n_pos_waveforms = [len(h5.get_node('/sorted/pos/' + unit))
                   for unit in pos_units]
n_neg_waveforms = [len(h5.get_node('/sorted/neg/' + unit))
                   for unit in neg_units]

total_wanted_pos = 10000
total_wanted_neg = 10000

pos_waveforms_per_unit = total_wanted_pos // n_pos_units
neg_waveforms_per_unit = total_wanted_neg // n_neg_units

pos_waveform_data = [h5.get_node(
    '/sorted/pos/' + unit)[:pos_waveforms_per_unit] for unit in pos_units]
neg_waveform_data = [h5.get_node(
    '/sorted/neg/' + unit)[:neg_waveforms_per_unit] for unit in neg_units]

X_pos = np.vstack(pos_waveform_data)
X_neg = np.vstack(neg_waveform_data)

# Zscore
X_pos = stats.zscore(X_pos, axis=1)
X_neg = stats.zscore(X_neg, axis=1)


##############################
# Pick pos units using active learning
# In each loop, add the unit that is worst classified by a logistic regression model
worst_accuracy_list = []
zscored_pos_waveform_data = [stats.zscore(
    unit, axis=1) for unit in pos_waveform_data]
selected_pos_units = [zscored_pos_waveform_data[0]]  # Start with first unit
remaining_pos_units = zscored_pos_waveform_data[1:]
n_active_learning_iters = 100
for iter in tqdm(range(n_active_learning_iters)):
    # Prepare data
    X_current = np.vstack((np.vstack(selected_pos_units), X_neg))
    y_current = np.hstack((
        np.ones(np.vstack(selected_pos_units).shape[0]),
        np.zeros(X_neg.shape[0])
    ))

    # Fit logistic regression
    clf = LogisticRegression(max_iter=5000)
    clf.fit(X_current, y_current)

    # Evaluate remaining pos units
    worst_accuracy = 1.0
    worst_unit = None
    for unit_ind, unit in enumerate(remaining_pos_units):
        X_unit = unit
        y_unit = np.ones(X_unit.shape[0])
        accuracy = clf.score(X_unit, y_unit)
        if accuracy < worst_accuracy:
            worst_accuracy = accuracy
            worst_unit = unit
    worst_accuracy_list.append(worst_accuracy)

    # Add worst unit to selected units
    selected_pos_units.append(worst_unit)
    # Remove worst unit from remaining units
    _ = remaining_pos_units.pop(unit_ind)

plt.plot(worst_accuracy_list)
plt.show()

plt.imshow(np.vstack(selected_pos_units), aspect='auto',
           cmap='viridis', interpolation='none')
plt.show()


##############################

fig, ax = plt.subplots(1, 2, figsize=(10, 6))
ax[0].imshow(X_pos, aspect='auto', cmap='viridis', interpolation='none')
ax[0].set_title('Positive Waveforms')
ax[0].set_xlabel('Time Point')
ax[0].set_ylabel('Waveform Index')
ax[1].imshow(X_neg, aspect='auto', cmap='viridis', interpolation='none')
ax[1].set_title('Negative Waveforms')
ax[1].set_xlabel('Time Point')
ax[1].set_ylabel('Waveform Index')
plt.tight_layout()
plt.show()

# Plot random subset of waveforms from each class
n_plot = 500
rand_pos_inds = np.random.choice(X_pos.shape[0], size=n_plot, replace=False)
rand_neg_inds = np.random.choice(X_neg.shape[0], size=n_plot, replace=False)
X_plot_pos = X_pos[rand_pos_inds]
X_plot_neg = X_neg[rand_neg_inds]

fig, axs = plt.subplots(1, 2, figsize=(10, 8))
axs[0].plot(X_plot_pos.T, color='blue', alpha=0.1)
axs[0].set_title('Random Positive Waveforms')
axs[0].set_xlabel('Time Point')
axs[0].set_ylabel('Amplitude')
axs[1].plot(X_plot_neg.T, color='red', alpha=0.1)
axs[1].set_title('Random Negative Waveforms')
axs[1].set_xlabel('Time Point')
axs[1].set_ylabel('Amplitude')
plt.tight_layout()
plt.show()

X = np.vstack((X_pos, X_neg))
y = np.hstack((np.ones(X_pos.shape[0]), np.zeros(X_neg.shape[0])))

# Plot
fig, axs = plt.subplots(1, 2, figsize=(10, 8))
axs[0].imshow(X_pos, aspect='auto', cmap='viridis', interpolation='none')
axs[0].set_title('Positive Waveforms')
axs[0].set_xlabel('Time Point')
axs[0].set_ylabel('Waveform Index')
axs[1].imshow(X_neg, aspect='auto', cmap='viridis', interpolation='none')
axs[1].set_title('Negative Waveforms')
axs[1].set_xlabel('Time Point')
axs[1].set_ylabel('Waveform Index')
plt.tight_layout()
plt.show()

##############################

nca = NeighborhoodComponentsAnalysis(
    n_components=3, random_state=42, max_iter=1000)
nca.fit(X, y)
X_nca = nca.transform(X)

# Plot NCA transformed data
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_nca[y == 1, 0], X_nca[y == 1, 1],
           X_nca[y == 1, 2], c='blue', label='Positive', alpha=0.5)
ax.scatter(X_nca[y == 0, 0], X_nca[y == 0, 1],
           X_nca[y == 0, 2], c='red', label='Negative', alpha=0.5)
ax.set_title('NCA Transformed Waveforms')
ax.set_xlabel('NCA Component 1')
ax.set_ylabel('NCA Component 2')
ax.set_zlabel('NCA Component 3')
ax.legend()
plt.show()

# Extract NCA components
nca_components = nca.components_

plt.figure(figsize=(10, 6))
for i in range(nca_components.shape[0]):
    plt.plot(nca_components[i], label=f'NCA Component {i+1}')
plt.title('NCA Components')
plt.xlabel('Time Point')
plt.ylabel('Component Value')
plt.legend()
plt.show()

############################################################
############################################################

# Find orthonormal templates for which dot product with data
# 1) maximizes separation between positive and negative classes

# Generate filters using basis functions

# linear_basis_funcs = gen_raised_cosine_basis(X.shape[1], 5, spread='log')
n_basis_funcs = 20
forward_basis_funcs = gen_raised_cosine_basis(
    75-30, n_basis_funcs//2, spread='log')
backward_basis_funcs = gen_raised_cosine_basis(
    30, n_basis_funcs//2, spread='log')[:, ::-1]
mirrored_basis_funcs = np.zeros(
    (forward_basis_funcs.shape[0] + backward_basis_funcs.shape[0],
     X.shape[1])
)
for i, this_func in enumerate(forward_basis_funcs):
    mirrored_basis_funcs[i, 30:] = this_func
for i, this_func in enumerate(backward_basis_funcs):
    mirrored_basis_funcs[i + forward_basis_funcs.shape[0], :30] = this_func

# mirrored_basis_funcs = np.concatenate([
#     backward_basis_funcs,
#     forward_basis_funcs
#     ], axis=1)

plt.plot(mirrored_basis_funcs.T)
plt.show()

plt.imshow(mirrored_basis_funcs, aspect='auto',
           cmap='bwr', interpolation='none')
plt.title('Mirrored Raised Cosine Basis Functions')
plt.xlabel('Time Point')
plt.ylabel('Basis Function Index')
plt.show()

# rand_weights = np.random.randn(10, linear_basis_funcs.shape[0])
# rand_filters = rand_weights @ linear_basis_funcs
n_templates = 10
rand_weights = np.random.randn(n_templates, mirrored_basis_funcs.shape[0])
rand_filters = rand_weights @ mirrored_basis_funcs
plt.imshow(rand_filters, aspect='auto', cmap='bwr')
plt.title('Random Filters Generated from Basis Functions')
plt.xlabel('Time Point')
plt.ylabel('Filter Index')
plt.show()


def loss_function_basis(
        weights,
        basis_funcs,
        X_norm,
        y,
        orthogonality_weight=1.0
):
    filters = weights.reshape((-1, basis_funcs.shape[0])) @ basis_funcs

    # Normalize filters
    filters -= np.mean(filters, axis=1, keepdims=True)
    filters /= norm(filters, axis=1, keepdims=True)

    # plt.imshow(filters, aspect='auto', cmap='bwr', interpolation='none')
    # plt.show()

    # Norm X
    X_transformed = X_norm @ filters.T
    # Calculate norm of abs dot product
    norm_score = norm(np.abs(X_transformed), axis=1)
    # Calculate separation between classes
    class_0_mean = np.mean(norm_score[y == 0])
    class_1_mean = np.mean(norm_score[y == 1])
    # class_delta = class_1_mean - class_0_mean
    # We want to project such that class 0 has close to 0 score, class 1 has high score
    class_0_delta = class_0_mean
    class_1_delta = 1 - class_1_mean

    # We also want to encourage orthogonality between filters
    filt_dot_products = filters @ filters.T
    # Get off-diagonal elements
    tril_indices = np.tril_indices(filters.shape[0], k=-1)
    off_diag_elements = filt_dot_products[tril_indices]
    # orthogonality_penalty = np.mean(np.abs(off_diag_elements))
    # Use L2 penalty for orthogonality
    orthogonality_penalty = np.mean(off_diag_elements ** 2)
    # Final loss is negative class delta plus orthogonality penalty
    # loss = -class_delta + (orthogonality_penalty * orthogonality_weight)
    loss = class_0_delta + class_1_delta + \
        (orthogonality_penalty * orthogonality_weight)

    return loss

# def loss_function(weights, X, y, orthogonality_weight=1.0):
#     # Reshape filters
#     n_filters = filters.shape[0] // X.shape[1]
#     filters = filters.reshape((n_filters, X.shape[1]))
#
#     # Normalize filters
#     filters -= np.mean(filters, axis=1, keepdims=True)
#     filters /= norm(filters, axis=1, keepdims=True)
#
#     # Norm X
#     X_transformed = X_norm @ filters.T
#     # Calculate norm of abs dot product
#     norm_score = norm(np.abs(X_transformed), axis=1)
#     # Calculate separation between classes
#     class_0_mean = np.mean(norm_score[y == 0])
#     class_1_mean = np.mean(norm_score[y == 1])
#     # class_delta = class_1_mean - class_0_mean
#     # We want to project such that class 0 has close to 0 score, class 1 has high score
#     class_0_delta = class_0_mean
#     class_1_delta = 1 - class_1_mean
#
#     # We also want to encourage orthogonality between filters
#     orthogonality_penalty = 0
#     filt_dot_products = filters @ filters.T
#     # Get off-diagonal elements
#     tril_indices = np.tril_indices(n_filters, k=-1)
#     off_diag_elements = filt_dot_products[tril_indices]
#     orthogonality_penalty = np.sum(off_diag_elements ** 2)
#     # Final loss is negative class delta plus orthogonality penalty
#     # loss = -class_delta + (orthogonality_penalty * orthogonality_weight)
#     loss = class_0_delta + class_1_delta + (orthogonality_penalty * orthogonality_weight)
#
#     return loss


X_norm = (X - np.mean(X, axis=1, keepdims=True))
X_norm /= norm(X_norm, axis=1, keepdims=True)
# initial_filters = rand_filters.flatten()

# result = minimize(
#         loss_function,
#         initial_filters,
#         args=(X_norm, y, 0.05),
#         method='L-BFGS-B',
#         # options={'maxiter': 500, 'disp': True, 'gtol': 1e-6}
#         options={'maxfun': 100000, 'disp': True, 'gtol': 1e-6}
#         )
# optimized_filters = result.x.reshape((10, X.shape[1]))

# Test run
loss_function_basis(
    rand_weights.flatten(),
    mirrored_basis_funcs,
    X_norm,
    y,
    orthogonality_weight=1.0
)

result = minimize(
    loss_function_basis,
    rand_weights.flatten(),
    args=(mirrored_basis_funcs, X_norm, y, 1),
    method='L-BFGS-B',
    options={'maxfun': 100000, 'disp': True, 'gtol': 1e-6}
)

# optimized_filters = (result.x.reshape((-1, linear_basis_funcs.shape[0])) @ linear_basis_funcs)
optimized_filters = (result.x.reshape(
    (-1, mirrored_basis_funcs.shape[0])) @ mirrored_basis_funcs)

# Normalize optimized filters
optimized_filters -= np.mean(optimized_filters, axis=1, keepdims=True)
optimized_filters /= norm(optimized_filters, axis=1, keepdims=True)

# Check orthogonality
filter_dot_products = np.abs(optimized_filters @ optimized_filters.T)
plt.matshow(filter_dot_products, cmap='bwr', vmin=0, vmax=1)
plt.colorbar(label='Dot Product')
plt.title('Dot Products Between Optimized Filters')
plt.xlabel('Filter Index')
plt.ylabel('Filter Index')
plt.show()

# Plot optimized filters
plt.plot(optimized_filters.T)
plt.show()

plt.figure(figsize=(10, 6))
plt.imshow(optimized_filters, aspect='auto', cmap='bwr', interpolation='none')
plt.title('Optimized Filters for Class Separation')
plt.xlabel('Time Point')
plt.ylabel('Filter Value')
plt.legend()
plt.show()

# Project data onto optimized filters and plot
X_optimized = X_norm @ optimized_filters.T

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(
    *(X_optimized[y == 1, :3].T),
    c='blue', label='Positive', alpha=0.5
)
ax.scatter(
    *(X_optimized[y == 0, :3].T),
    c='red', label='Negative', alpha=0.5
)
ax.set_title('Optimized Filter Transformed Waveforms (Top 3 Filters)')
ax.set_xlabel('Filter 1')
ax.set_ylabel('Filter 2')
ax.set_zlabel('Filter 3')
ax.legend()
plt.show()

# Perform PCA on the transformed data
pca_optimized = PCA()
X_optimized_pca = pca_optimized.fit_transform(X_optimized)
explained_variance_optimize = pca_optimized.explained_variance_ratio_

# Plot X_optimized_pca colored by class
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(
    *(X_optimized_pca[y == 1, :3].T),
    c='blue', label='Positive', alpha=0.05,
)
ax.scatter(
    *(X_optimized_pca[y == 0, :3].T),
    c='red', label='Negative', alpha=0.05,
)
ax.set_title('PCA of Optimized Filter Transformed Waveforms (Top 3 PCs)')
ax.set_xlabel('PC 1')
ax.set_ylabel('PC 2')
ax.set_zlabel('PC 3')
ax.legend()
plt.show()

# Perform PCA on the optimized filters
pca_filters = PCA(n_components=0.95)
filters_pca = pca_filters.fit_transform(optimized_filters)
explained_variance_filters = pca_filters.explained_variance_ratio_

filter_pca_components = pca_filters.components_
# Calculate score of each filter as mean absolute projection onto PCA components
X_norm_pca_filters = X_norm @ filter_pca_components.T
filter_scores = np.mean(np.abs(X_norm_pca_filters), axis=1)

plt.imshow(X_norm_pca_filters, aspect='auto', cmap='bwr', interpolation='none')
plt.title('Projection of Data onto PCA Components of Optimized Filters')
plt.xlabel('Time Point')
plt.ylabel('PCA Component Index')
plt.show()

# Plot optimized filter and pca filters
fig, axs = plt.subplots(2, 1, figsize=(10, 8))
axs[0].plot(optimized_filters.T)
axs[0].set_title('Optimized Filters')
axs[1].plot(filter_pca_components.T)
axs[1].set_title('PCA Components of Optimized Filters')
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(2, 1, figsize=(10, 8))
ax[0].plot(np.cumsum(explained_variance_optimize), marker='o')
ax[0].set_title(
    'Cumulative Explained Variance - Optimized Filter Transformed Data')
ax[0].set_xlabel('Number of PCA Components')
ax[0].set_ylabel('Cumulative Explained Variance')
ax[1].plot(np.cumsum(explained_variance_filters), marker='o')
ax[1].set_title('Cumulative Explained Variance - Optimized Filters')
ax[1].set_xlabel('Number of PCA Components')
ax[1].set_ylabel('Cumulative Explained Variance')
ax[0].set_ylim([0, 1])
ax[1].set_ylim([0, 1])
plt.tight_layout()
plt.show()

# Histogram of score of optimized filters
X_optimized_score = np.mean(np.abs(X_optimized), axis=1)
fig, ax = plt.subplots(figsize=(10, 6))
for this_label in [0, 1]:
    ax.hist(
        X_optimized_score[y == this_label], bins=30, alpha=0.5,
        label=f'Class {this_label}', density=True
    )
ax.set_title('Histogram of Optimized Filter Scores')
ax.set_xlabel('Optimized Filter Score')
ax.set_ylabel('Density')
ax.legend()
plt.show()

# Plot imshow of X_optimized and X_optimized_pca
fig, ax = plt.subplots(1, 2, figsize=(10, 6))
ax[0].imshow(X_optimized, aspect='auto', cmap='viridis', interpolation='none')
ax[1].imshow(X_optimized_pca, aspect='auto',
             cmap='viridis', interpolation='none')
plt.show()

# Plot X_optimized_score against filter scores
fig, ax = plt.subplots(figsize=(10, 6))
ax.scatter(
    filter_scores,
    X_optimized_score,
    c=['blue' if label == 1 else 'red' for label in y],
    alpha=0.05
)
ax.set_title('Optimized Filter Score vs. Filter PCA Score')
ax.set_xlabel('Filter PCA Score')
ax.set_ylabel('Optimized Filter Score')
plt.show()

# Also plot max filter score
max_filter_scores = np.max(np.abs(X_optimized), axis=1)
fig, ax = plt.subplots(figsize=(10, 6))
for this_label in [0, 1]:
    ax.hist(
        max_filter_scores[y == this_label], bins=30, alpha=0.5,
        label=f'Class {this_label}', density=True
    )
ax.set_title('Histogram of Max Optimized Filter Scores')
ax.set_xlabel('Max Optimized Filter Score')
ax.set_ylabel('Density')
ax.legend()
# plt.show()
fig.savefig(os.path.join(plot_dir, 'max_optimized_filter_scores_histogram.png'))
plt.close()

# Write out optimized filters and pca filters
np.savez(
    os.path.join(artifacts_dir, 'optimized_filters.npz'),
    optimized_filters=optimized_filters,
    filter_pca_components=filter_pca_components
)

##############################
# Run grid over:
# 1) number of basis functions
# 2) orthogonality weight
# 3) number of templates
# Generate filters using basis functions

# n_basis_funcs = 20
# n_templates = 10

n_basis_vec = np.arange(5, 25, 5)
n_templates_vec = np.arange(4, 21, 4)
orthogonality_weight_vec = np.logspace(-2, 2, 5)

all_combinations = list(product(
    n_basis_vec,
    n_templates_vec,
    orthogonality_weight_vec
))

all_losses = []
for comb in tqdm(all_combinations):
    n_basis_funcs = comb[0]
    n_templates = comb[1]
    orthogonality_weight = comb[2]

    forward_basis_funcs = gen_raised_cosine_basis(
        75-30, n_basis_funcs//2, spread='log')
    backward_basis_funcs = gen_raised_cosine_basis(
        30, n_basis_funcs//2, spread='log')[:, ::-1]
    mirrored_basis_funcs = np.zeros(
        (forward_basis_funcs.shape[0] + backward_basis_funcs.shape[0],
         X.shape[1])
    )
    for i, this_func in enumerate(forward_basis_funcs):
        mirrored_basis_funcs[i, 30:] = this_func
    for i, this_func in enumerate(backward_basis_funcs):
        mirrored_basis_funcs[i + forward_basis_funcs.shape[0], :30] = this_func

    rand_weights = np.random.randn(n_templates, mirrored_basis_funcs.shape[0])

    result = minimize(
        loss_function_basis,
        rand_weights.flatten(),
        args=(mirrored_basis_funcs, X_norm, y, orthogonality_weight),
        method='L-BFGS-B',
        options={
            'maxfun': 100000,
            # 'disp': True,
            'gtol': 1e-6,
        }
    )
    # final_loss = result.fun
    # Get basis and test using no orthogonality weight
    final_loss = loss_function_basis(
        result.x,
        mirrored_basis_funcs,
        X_norm,
        y,
        orthogonality_weight=0.0
    )

    all_losses.append({
        'n_basis_funcs': int(n_basis_funcs),
        'n_templates': int(n_templates),
        'orthogonality_weight': orthogonality_weight,
        'final_loss': final_loss
    })
    # Append results to a json file
    with open(os.path.join(artifacts_dir, 'grid_search_losses.json'), 'w') as f:
        json.dump(all_losses, f, indent=4)

# Plot heatmaps of all combinations
loss_df = pd.DataFrame(all_losses)

fig, axs = plt.subplots(
    1, len(orthogonality_weight_vec),
    figsize=(20, 5),
    sharex=True,
    sharey=True
)
vmin = loss_df['final_loss'].min()
vmax = loss_df['final_loss'].max()
for i, orthogonality_weight in enumerate(orthogonality_weight_vec):
    subset_df = loss_df[loss_df['orthogonality_weight']
                        == orthogonality_weight]
    pivot_table = subset_df.pivot('n_basis_funcs', 'n_templates', 'final_loss')
    im = axs[i].imshow(
        pivot_table,
        aspect='auto',
        origin='lower',
        cmap='viridis',
        vmin=vmin,
        vmax=vmax
    )
    axs[i].set_title(f'Orthogonality Weight: {orthogonality_weight:.2f}')
    axs[i].set_xlabel('Number of Templates')
    axs[i].set_ylabel('Number of Basis Functions')
    fig.colorbar(im, ax=axs[i], label='Final Loss')
plt.tight_layout()
plt.show()

best_params = loss_df.loc[loss_df['final_loss'].idxmin()]

# Generate filters using best params
n_basis_funcs = int(best_params['n_basis_funcs'])
n_templates = int(best_params['n_templates'])
orthogonality_weight = best_params['orthogonality_weight']

forward_basis_funcs = gen_raised_cosine_basis(
    75-30, n_basis_funcs//2, spread='log')
backward_basis_funcs = gen_raised_cosine_basis(
    30, n_basis_funcs//2, spread='log')[:, ::-1]
mirrored_basis_funcs = np.zeros(
    (forward_basis_funcs.shape[0] + backward_basis_funcs.shape[0],
     X.shape[1])
)
for i, this_func in enumerate(forward_basis_funcs):
    mirrored_basis_funcs[i, 30:] = this_func
for i, this_func in enumerate(backward_basis_funcs):
    mirrored_basis_funcs[i + forward_basis_funcs.shape[0], :30] = this_func

rand_weights = np.random.randn(n_templates, mirrored_basis_funcs.shape[0])

result = minimize(
    loss_function_basis,
    rand_weights.flatten(),
    args=(mirrored_basis_funcs, X_norm, y, orthogonality_weight),
    method='L-BFGS-B',
    options={
        'maxfun': 100000,
        'disp': True,
        'gtol': 1e-6,
    }
)

# Value without orthogonality weight
final_loss = loss_function_basis(
    result.x,
    mirrored_basis_funcs,
    X_norm,
    y,
    orthogonality_weight=0.0
)
