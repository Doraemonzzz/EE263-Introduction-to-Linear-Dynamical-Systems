# -*- coding: utf-8 -*-
"""
Created on Thu May 23 08:24:17 2019

@author: qinzhen
"""

import numpy as np

A = np.array([
        [-1, 0, 0, -1, 1],
        [0, 1, 1, 0, 0],
        [1, 0, 0, 1, 0]
        ])
n, m = A.shape

#### (a)
A1 = np.c_[A[:, 0], A[:, 2:]]
#QR分解
Q, R = np.linalg.qr(A1.T)
#计算右逆
B1 = Q.dot(np.linalg.inv(R.T))
#计算最终结果
B = np.r_[B1[0, :], np.zeros(n)].reshape(2, n)
B = np.r_[B, B1[1:, :]]
print(B)
#验证结果
print(A.dot(B))

#### (d)
B = np.array([
        [0, 0, 0.5],
        [0, 0.5, 0],
        [0, 0.5, 0],
        [0, 0, 0.5],
        [1, 0, 1]
        ])
print(A.dot(B))

#### (f)
B = np.array([
        [0, 0, 0],
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 1]
        ])
print(A.dot(B))