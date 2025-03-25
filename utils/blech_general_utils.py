"""
General utility functions for blech_clust that are used across multiple modules.
"""

import os
import shutil
import numpy as np


def ifisdir_rmdir(dir_name):
    """Remove directory if it exists"""
    if os.path.isdir(dir_name):
        shutil.rmtree(dir_name)


def return_cutoff_values(
    filt_el,
    sampling_rate,
    voltage_cutoff,
    max_breach_rate,
    max_secs_above_cutoff,
    max_mean_breach_rate_persec
):
    """
    Return the cutoff values for the electrode recording

    Inputs:
        filt_el: numpy array (in microvolts)
        sampling_rate: int
        voltage_cutoff: float
        max_breach_rate: float
        max_secs_above_cutoff: float
        max_mean_breach_rate_persec: float

    Outputs:
        breach_rate: float
        breaches_per_sec: numpy array
        secs_above_cutoff: int
        mean_breach_rate_persec: float
        recording_cutoff: int
    """

    breach_rate = float(len(np.where(filt_el > voltage_cutoff)[0])
                        * int(sampling_rate))/len(filt_el)
    test_el = np.reshape(filt_el, (-1, sampling_rate))
    breaches_per_sec = (test_el > voltage_cutoff).sum(axis=-1)
    secs_above_cutoff = (breaches_per_sec > 0).sum()
    if secs_above_cutoff == 0:
        mean_breach_rate_persec = 0
    else:
        mean_breach_rate_persec = np.mean(breaches_per_sec[
            breaches_per_sec > 0])

    # And if they all exceed the cutoffs,
    # assume that the headstage fell off mid-experiment
    recording_cutoff = int(len(filt_el)/sampling_rate)
    if breach_rate >= max_breach_rate and \
            secs_above_cutoff >= max_secs_above_cutoff and \
            mean_breach_rate_persec >= max_mean_breach_rate_persec:
        # Find the first 1 second epoch where the number of cutoff breaches
        # is higher than the maximum allowed mean breach rate
        recording_cutoff = np.where(breaches_per_sec >
                                    max_mean_breach_rate_persec)[0][0]

    return (breach_rate, breaches_per_sec, secs_above_cutoff,
            mean_breach_rate_persec, recording_cutoff)
