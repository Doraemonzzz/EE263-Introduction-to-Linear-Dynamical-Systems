# -*- coding: utf-8 -*-
"""
Created on Fri Jun 28 10:53:41 2019

@author: qinzhen
"""

import numpy as np

A = np.array([
        [0, 0, 1, 0, 1],
        [1, 1, 0, 1, 0],
        [1, 0, 0, 0, 1],
        [0, 0, 0, 1, 0],
        [0, 1, 0, 1, 0]], dtype=np.float64)

res = np.linalg.eigvals(A)
print(np.log2(np.max(np.abs(res))))

print(np.log2(5))

