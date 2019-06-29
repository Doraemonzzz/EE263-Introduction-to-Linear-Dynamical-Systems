# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 11:51:58 2019

@author: qinzhen
"""

from scipy.linalg import expm
import numpy as np

A = np.array([
        [0.5, 1.4],
        [-0.7, 0.5]])

print(expm(A))

print(expm(2 * A))