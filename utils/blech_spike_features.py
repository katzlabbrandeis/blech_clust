"""
This module provides functionality for creating a feature extraction pipeline using scikit-learn, specifically for processing and transforming data with PCA, energy, and amplitude features. It includes custom transformers and utility functions for handling data paths and metadata.

- `EnergyFeature` class: A custom transformer that calculates the energy of input data.
- `AmpFeature` class: A custom transformer that extracts the amplitude at a specified index from input data.
- `zscore_custom(x)`: A function that applies z-score normalization along the last axis of the input data.
- `return_feature_pipeline(data_dir_name)`: A function that constructs and returns a scikit-learn pipeline for feature extraction, including PCA, energy, and amplitude transformations, followed by feature scaling. It uses metadata to configure the pipeline based on the data directory provided.
"""
import os
import sys
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats import zscore
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.decomposition import PCA
from blech_clust.utils.blech_process_utils import path_handler

############################################################
# Figure out paths
path_handler = path_handler()
blech_clust_dir = path_handler.blech_clust_dir
# data_dir_name = path_handler.data_dir
############################################################

sys.path.append(blech_clust_dir)
from blech_clust.utils.blech_utils import imp_metadata  # noqa: E402


class EnergyFeature(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        energy = np.sqrt(np.sum(X**2, axis=-1))/X.shape[-1]
        energy = energy[:, np.newaxis]
        return energy

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)


class AmpFeature(BaseEstimator, TransformerMixin):
    def __init__(self, zero_ind):
        self.zero_ind = zero_ind

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        amplitude = X[:, self.zero_ind]
        amplitude = amplitude[:, np.newaxis]
        return amplitude

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)


def zscore_custom(x):
    return zscore(x, axis=-1)


zscore_transform = FunctionTransformer(zscore_custom)
log_transform = FunctionTransformer(np.log, validate=True)
arcsinh_transform = FunctionTransformer(np.arcsinh, validate=True)

pca_components = 3
feature_names = ['pca_{}'.format(i) for i in range(pca_components)] + \
                ['energy', 'amplitude']


def return_feature_pipeline(data_dir_name):
    metadata_handler = imp_metadata([[], data_dir_name])
    params_dict = metadata_handler.params_dict
    sampling_rate = params_dict['sampling_rate']
    zero_ind = int(params_dict['spike_snapshot_before']*sampling_rate/1000)

    pca_pipeline = Pipeline(
        steps=[
            ('zscore', zscore_transform),
            ('pca', PCA(n_components=pca_components)),
        ]
    )

    energy_pipeline = Pipeline(
        steps=[
            ('energy', EnergyFeature()),
            ('log', log_transform),
        ]
    )

    # use arcsinh_transform for amplitude due to negative values
    amplitude_pipeline = Pipeline(
        steps=[
            ('amplitude', AmpFeature(zero_ind)),
            ('arcsinh', arcsinh_transform),
        ]
    )

    collect_feature_pipeline = FeatureUnion(
        n_jobs=1,
        transformer_list=[
            ('pca_features', pca_pipeline),
            ('energy', energy_pipeline),
            ('amplitude', amplitude_pipeline),
        ]
    )

    feature_pipeline = Pipeline(
        steps=[
            ('get_features', collect_feature_pipeline),
            ('scale_features', StandardScaler())
        ]
    )

    return feature_pipeline
