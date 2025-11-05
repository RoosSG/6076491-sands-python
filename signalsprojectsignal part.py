# -*- coding: utf-8 -*-
"""
Created on Wed Nov  5 14:10:34 2025

@author: roosg
"""

import numpy as np

def sinusoidal(frequency, phase, amplitude, duration, Fs):
    """
    Create a sinusoidal wave signal.
    Parameters:
        frequency: frequency of the wave in Hz
        phase: phase of the wave in radians
        amplitude: amplitude of the sine wave
        Fs: samples per second (sampling frequency)
        duration: total signal duration in seconds

    Returns:
        t : time vector, with the np.arrange(start, stop, step) 
        x : the sinusoidal signal 
    """
    t = np.arange(0, duration, 1/Fs)          
    x = amplitude * np.sin(2 * np.pi * frequency * t + phase)
    return t, x

from scipy import signal
def triangular(frequency, amplitude, duration, Fs):
    """
    Create a triangular wave signal.
    Parameters:
        frequency: frequency of the wave in Hz
        amplitude: amplitude of the triangle wave
        Fs: Samples per second (sampling frequency)
        duration: total signal duration in seconds

    Returns:
        t : time vector, with the np.arrange(start, stop, step)
        x : the triangular wave signal
    """
    t = np.arange(0, duration, 1/Fs)
    x = amplitude * signal.sawtooth(2 * np.pi * frequency * t, width=0.5)
    return t, x

def unit_impulse(n_samples, impulse_index=0):
    """
    Create a discrete unit impulse signal.
    Parameters:
        n_samples: total number of samples
        impulse_index: index where the impulse occurs, the direc delta is 0 everywhere
    except at the impulse index

    Returns:
        impulse: array with 1 at impulse_index and 0 elsewhere
    """
    impulse = np.zeros(n_samples)
    if 0 <= impulse_index < n_samples:
        impulse[impulse_index] = 1.0
    return impulse
