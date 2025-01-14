"""
Python implementation of BAKS from 10.1371/journal.pone.0206794
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
