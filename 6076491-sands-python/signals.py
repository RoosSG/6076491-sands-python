import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.interpolate import interp1d

def sinusoidal(frequency=10, phase=0, amplitude=1, duration=2, Fs=1000):
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
        
    Meking sure the 
    """
    if duration < 0:
         return np.array([]), np.array([])
    t = np.linspace(0, duration, int(Fs * duration), endpoint=False)
    x = amplitude * np.sin(2 * np.pi * frequency * t)
    return t, x
  
def triangular(frequency=10, amplitude=1, duration=2, Fs=1000):
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
    if duration < 0:
         return np.array([]), np.array([])
    t = np.linspace(0,duration,int(Fs*duration), endpoint=False)
    x = amplitude * signal.sawtooth(2 * np.pi * frequency * t, width=0.5)
    return t, x

def unit_impulse(n_samples=20, impulse_index=5):
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
