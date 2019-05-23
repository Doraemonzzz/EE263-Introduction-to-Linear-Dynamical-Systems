# -*- coding: utf-8 -*-
"""
Created on Mon May 20 11:07:33 2019

@author: qinzhen
"""

import numpy as np

eta_ab = 0.1
a1 = 1 - eta_ab ** 2
b1 = 2
c1 = 1
b2 = -2

def solve(a, b, c):
    delta = b ** 2 - 4 * a * c
    x1 = (-b - np.sqrt(delta)) / (2 * a)
    x2 = (-b + np.sqrt(delta)) / (2 * a)
    
    return x1, x2

#求解方程
t1, t2 = solve(a1, b1, c1)
t3, t4 = solve(a1, b2, c1)
print(t1, t2)
print(t3, t4)

#计算最小值
tmin = t3
tmax = t4
eta_bamin = eta_ab * tmin
eta_bamax = eta_ab * tmax
print("eta_ba的最小值为{}".format(eta_bamin))
print("eta_ba的最大值为{}".format(eta_bamax))

#求角度
theta_min = 0
theta_max = np.arccos(np.sqrt(1-eta_ab**2))
print("theta的最小值为{}".format(theta_min))
print("theta的最大值为{}".format(theta_max))