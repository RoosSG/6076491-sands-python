# -*- coding: utf-8 -*-
"""
Created on Fri Oct  3 17:32:56 2025

@author: roosg
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.signal import unit_impulse

# Common parameters 
Fs = 8000       
f = 5            
samples = 8000   
t = np.arange(samples) / Fs

# Sinusoidal wave 
sin_wave = np.sin(2 * np.pi * f * t)
plt.plot(t, sin_wave)
plt.title("Sinusoidal Wave")
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid()
plt.show()

# Square wave
square_wave = signal.square(2 * np.pi * f * t)
plt.plot(t, square_wave)
plt.title("Square Wave")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.ylim(-2, 2)
plt.grid()
plt.show()

# Triangle wave 
triangle_wave = signal.sawtooth(2 * np.pi * f * t, width=0.5)
plt.plot(t, triangle_wave)
plt.title("Triangle Wave")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")
plt.grid()
plt.show()

# Unit impulse 
n_samples = 20  
impulse_index = 5 
impulse = unit_impulse(n_samples, idx=impulse_index)
plt.stem(np.arange(n_samples), impulse, use_line_collection=True)
plt.title("Unit Impulse Signal")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.grid()
plt.show()

# Unit step
def unit_step(x):
    return np.where(x >= 0, 1, 0)

x_step = np.linspace(-1, 1, 500)  
step_wave = unit_step(x_step)
plt.plot(x_step, step_wave, label="Unit Step Function", color="blue")
plt.title("Unit Step Function")
plt.xlabel("x")
plt.ylabel("u(x)")
plt.grid(True)
plt.legend()
plt.show()

# Operations on sinusoidal

# 1. Time shifting
shift = 0.2
sin_shifted = np.sin(2 * np.pi * f * (t - shift))

# 2. Time scaling
scale = 2
sin_scaled = np.sin(2 * np.pi * f * (scale * t))

# 3. Addition
add_signal = sin_wave + square_wave

# 4. Multiplication
mult_signal = sin_wave * triangle_wave

# Plot operations
plt.figure(figsize=(12, 8))

plt.subplot(2,2,1)
plt.plot(t, sin_wave, label="Original")
plt.plot(t, sin_shifted, 'r', alpha=0.7, label="Shifted")
plt.title("Time Shifting")
plt.legend()
plt.grid()

plt.subplot(2,2,2)
plt.plot(t, sin_wave, label="Original")
plt.plot(t, sin_scaled, 'g', alpha=0.7, label="Scaled (×2)")
plt.title("Time Scaling")
plt.legend()
plt.grid()

plt.subplot(2,2,3)
plt.plot(t, add_signal)
plt.title("Addition: Sine + Square")
plt.grid()

plt.subplot(2,2,4)
plt.plot(t, mult_signal)
plt.title("Multiplication: Sine × Triangle")
plt.grid()

plt.tight_layout()
plt.show()
