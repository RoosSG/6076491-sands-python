# -*- coding: utf-8 -*-
"""
Created on Mon Sep 29 16:18:56 2025

@author: roosg
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.signal import unit_impulse

#sinusoidal wave 
Fs = 8000
f = 5
sample = 8000
x = np.arange(sample)
y = np.sin(2 * np.pi * f * x / Fs)
plt.plot(x, y)
plt.xlabel('sample(n)')
plt.ylabel('voltage(V)')
plt.show()

#square wave
t = np.linspace(0, 1, 500, endpoint=False)
square_wave = signal.square(2 * np.pi * 5 * t)
plt.plot(t, square_wave)
plt.title("Square Wave")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.ylim(-2, 2)
plt.grid()
plt.show()

#traingle wave 
t = np.linspace(0, 1, 500) 
frequency = 5 
triangle_wave = signal.sawtooth(2 * np.pi * frequency * t, width=0.5)
plt.plot(t, triangle_wave)
plt.title("Triangle Wave")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.grid()
plt.show()

# unit impulse 
n_samples = 10  
impulse_index = 4 
impulse = unit_impulse(n_samples, idx=impulse_index)
plt.stem(np.arange(n_samples), impulse, use_line_collection=True)
plt.title("Unit Impulse Signal")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.grid()
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# unit step
def unit_step(x):
    return np.where(x >= 0, 1, 0)
x = np.linspace(-10, 10, 500)  
y = unit_step(x)
plt.figure(figsize=(8, 4))
plt.plot(x, y, label="Unit Step Function", color="blue")
plt.title("Unit Step Function")
plt.xlabel("x")
plt.ylabel("u(x)")
plt.grid(True)
plt.legend()
plt.show()
