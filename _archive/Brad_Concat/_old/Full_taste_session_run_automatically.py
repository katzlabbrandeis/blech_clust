#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 19:16:01 2019

@author: bradly
"""

import subprocess

filename = 'Full_taste_session.py'
while True:
    """However, you should be careful with the '.wait()'"""
    p = subprocess.Popen('python '+filename, shell=True).wait()

    """#if your there is an error from running 'Full_taste_session.py',
    the while loop will be repeated,
    otherwise the program will break from the loop"""
    if p != 0:
        continue
    else:
        break
