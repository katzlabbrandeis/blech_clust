#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  5 15:27:24 2019

@author: abuzarmahmood
"""
from LFP_Spike_Locking_Test import LockingSpikes

locking_spikes = LockingSpikes([2,7,50],
                               [1,0.5,0.1],
                               0.001,5,1000)
locking_spikes.CreateLFP()
locking_spikes.CreateSpikes()
locking_spikes.ShowData()