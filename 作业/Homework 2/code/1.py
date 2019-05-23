# -*- coding: utf-8 -*-
"""
Created on Mon May 20 10:26:46 2019

@author: qinzhen
"""

import numpy as np

#数据
P_1 = np.array([10, 10, 10])
P_2 = np.array([100, 10, 10])
P_3 = np.array([10, 100, 10])
P_4 = np.array([10, 10, 100])
T_1 = np.array([27, 29])
T_2 = np.array([45, 37])
T_3 = np.array([41, 49])
T_4 = np.array([35, 55])

#计算
P = np.c_[P_2-P_1, P_3-P_1, P_4-P_1]
T = np.c_[T_2-T_1, T_3-T_1, T_4-T_1]
A = T.dot(np.linalg.inv(P))
b = T_1 - A.dot(P_1)

print("A =", A)
print("b =", b)

T0 = 70
tmp = (T0 - b) / np.sum(A, axis=1)
pmin = np.min(tmp)
print("p_min =", pmin)