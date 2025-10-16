"""
This module provides a Python implementation of the Bayesian Adaptive Kernel Smoother (BAKS) as described in the paper with DOI: 10.1371/journal.pone.0206794. It is used to estimate the firing rate from spike times.

- `BAKS(SpikeTimes, Time)`: Computes the firing rate given spike times and a time vector. It calculates a smoothing parameter `h` using Bayesian methods and applies a Gaussian kernel to estimate the firing rate over the specified time points. Returns an array representing the estimated firing rate.

EXAMPLE WORKFLOWS:

This module provides a Bayesian method for estimating firing rates from spike times.
Here's how to use it:

Workflow: Estimating Firing Rates from Spike Times
-----------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from blech_clust.utils.ephys_data.BAKS import BAKS

# Create sample spike times (in seconds)
spike_times = np.array([0.1, 0.15, 0.17, 0.32, 0.4, 0.55, 0.6, 0.61, 0.7])

# Create time vector for rate estimation (in seconds)
time_vector = np.linspace(0, 1, 1000)  # 1000 time points from 0 to 1 second

# Estimate firing rate using BAKS
firing_rate = BAKS(spike_times, time_vector)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(time_vector, firing_rate, 'b-', linewidth=2)
plt.stem(spike_times, np.ones_like(spike_times) * np.max(firing_rate) * 0.1,
         'r', markerfmt='ro', basefmt=' ', label='Spikes')
plt.xlabel('Time (s)')
plt.ylabel('Firing Rate (Hz)')
plt.title('BAKS Firing Rate Estimation')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Compare with different spike patterns
spike_times2 = np.array([0.2, 0.21, 0.22, 0.23, 0.5, 0.8, 0.81, 0.82])
firing_rate2 = BAKS(spike_times2, time_vector)

plt.figure(figsize=(10, 6))
plt.plot(time_vector, firing_rate, 'b-', linewidth=2, label='Pattern 1')
plt.plot(time_vector, firing_rate2, 'g-', linewidth=2, label='Pattern 2')
plt.xlabel('Time (s)')
plt.ylabel('Firing Rate (Hz)')
plt.title('Comparison of Different Spike Patterns')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
"""

import numpy as np
import scipy.special


def BAKS(SpikeTimes, Time):

    N = len(SpikeTimes)
    a = float(4)
    b = float(N**0.8)
    sumnum = float(0)
    sumdenum = float(0)

    for i in range(N):
        numerator = (((Time-SpikeTimes[i])**2)/2 + 1/b)**(-a)
        denumerator = (((Time-SpikeTimes[i])**2)/2 + 1/b)**(-a-0.5)
        sumnum = sumnum + numerator
        sumdenum = sumdenum + denumerator

    if len(SpikeTimes) > 0:  # Catch for no firing trials
        h = (scipy.special.gamma(a)/scipy.special.gamma(a+0.5))*(sumnum/sumdenum)

        FiringRate = np.zeros((len(Time)))
        for j in range(N):
            K = (1/(np.sqrt(2*np.pi)*h)) * \
                np.exp(-((Time-SpikeTimes[j])**2)/((2*h)**2))
            FiringRate = FiringRate + K

    else:
        FiringRate = np.zeros((len(Time)))

    return FiringRate
