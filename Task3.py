# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 10:26:23 2026

@author: regnerk
"""

###Task 3
import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt


#Density as in workbook task
def f_task3(s, theta):
    return 1/(120*theta**7)*s**6*np.exp(-s/theta)

#Area as in workbook task
for theta in [0.5, 1, 2, 5]:
    a = quad(f_task3, 0, np.inf, args=(theta,))[0]
    print(f"theta = {theta}: Area as in workbook task = {a:.4f}")
    

# Comparison: 720 = 6! instead of 120 = 5!
def f_normalized(s, theta):
    return 1/(720*theta**7)*s**6*np.exp(-s/theta)

#Area normalized
for theta in [0.5, 1, 2, 5]:
    n = quad(f_normalized, 0, np.inf, args=(theta,))[0]
    print(f"theta = {theta}: area with normalized density function = {n:.4f}")
    

#Plot comparison
theta = 1
s = np.linspace(0, 20, 500)

area_task = quad(f_task3, 0, np.inf, args=(theta,))[0]
area_normalized = quad(f_normalized, 0, np.inf, args=(theta,))[0]

plt.plot(s, f_task3(s, theta), label=f"1/(120θ⁷)·s⁶·e^(−s/θ), area = {area_task:.2f}")
plt.fill_between(s, f_task3(s, theta), alpha=0.2)
plt.plot(s, f_normalized(s, theta), label=f"1/(720θ⁷)·s⁶·e^(−s/θ), area = {area_normalized:.2f}")
plt.fill_between(s, f_normalized(s, theta), alpha=0.2)
plt.title(f"Density from the workbook vs. normalized density (θ = {theta})")
plt.xlabel("s (terabytes)")
plt.ylabel("f(s)")
plt.legend()
plt.show()