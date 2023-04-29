#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 12:38:15 2019

@author: abuzarmahmood

Generates oscillations at a particular frequency
Uses oscillations as rate for inhomogenous Poisson process to generate spikes
"""

import numpy as np
import pylab as plt

class LockingSpikes():
    
    def __init__(self,FrequencyList, PowerList, dt, duration,trials):
        self.FrequencyList = FrequencyList
        self.PowerList = PowerList
        self.dt = dt
        self.duration = duration
        self.trials = trials
        
    def CreateLFP(self):
        self.t_vec = np.linspace(0,self.duration,int(self.duration/self.dt))
        LFP = np.zeros((1,int(self.duration/self.dt)))
        for freq, power in zip(self.FrequencyList,self.PowerList):
            LFP = LFP + np.sin(2*np.pi*freq*self.t_vec)*power
        self.LFP = LFP + np.random.normal(size=LFP.size)*0.1
    
    def CreateSpikes(self):
        self.spikes = np.asarray([self.InhomogeneousPoisson(self.LFP, self.dt) for\
                       trial in range(self.trials)])[:,0,:]
    
    @staticmethod
    def InhomogeneousPoisson(rate,dt):
        rand_vec = np.random.rand(rate.size)
        spike_train = (rand_vec < rate*dt)*1
        return spike_train
    
    @staticmethod
    def raster(axes, data, time_vector, markersize = 5):
        # data : trials x time
        # Plot spikes 
        for unit in range(data.shape[0]):
            for time in range(data.shape[1]):
                if data[unit, time] > 0:
                    axes.plot(time_vector[time], unit, 'ko',markersize=markersize)
        axes.set_xlim(0,np.max(time_vector))
        return axes
                    
    def ShowData(self, type = 'psth'):
        ax1 = plt.subplot(211)
        ax1.plot(self.t_vec,locking_spikes.LFP.T)
        ax2 = plt.subplot(212)
        if type in 'PSTH' or 'psth':
            plt.plot(self.t_vec,np.sum(self.spikes,axis=0))
        else:
            self.raster(ax2, self.spikes, self.t_vec)
