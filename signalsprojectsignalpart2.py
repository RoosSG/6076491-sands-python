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

from scipy.interpolate import interp1d

def time_scaling(t, x, scale_factor):
    """
    Perform time scaling on a continuous signal.
    Parameters:
        t: original time vector
        x: original signal
        scale_factor: factor by which time is scaled
                      >1 compresses time, <1 stretches time
    Returns:
        t_scaled: new time vector after scaling
        x_scaled: signal evaluated at the new time vector
    """
    
    t_scaled = t / scale_factor
    interpolator = interp1d(t, signal, kind='linear', fill_value="extrapolate")
    x_scaled = interpolator(t_scaled)
    
    return t_scaled, x_scaled


def multiply_signals(signal1, signal2):
    """
    Multiply two signals element-wise.
    Parameters:
        signal1: first input signal
        signal2: second input signal, which schould be the same length as signal 1
    Returns:
        multiplied_signal: product of the two signals
    """
    if len(signal1) != len(signal2):
        raise ValueError("Signals must have the same length for multiplication.")
    multiplied_signal = signal1 * signal2
    return multiplied_signal
