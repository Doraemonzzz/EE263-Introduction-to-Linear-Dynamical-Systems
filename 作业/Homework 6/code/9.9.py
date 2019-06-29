# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 11:19:22 2019

@author: qinzhen
"""

import numpy as np

A = np.array([
        [0, 0.2, 0.1],
        [0.05, 0, 0.05],
        [0.1, 1/30, 0]])
res = np.linalg.eigvals(A)

#gamma=3
print(res * 1.2 * 3)

#gamma=5
print(res * 1.2 * 5)