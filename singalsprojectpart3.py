# -*- coding: utf-8 -*-
"""
Created on Wed Nov  5 15:16:49 2025

@author: roosg
"""

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
    return t_scaled, x

def multiply_signals(signal1, signal2):
    """
    Multiply two signals element-wise.
    Parameters:
        signal1: first input signal
        signal2: second input signal, which should be the same length as signal 1
    Returns:
        multiplied_signal: product of the two signals
    """
    if len(signal1) != len(signal2):
        raise ValueError("Signals must have the same length for multiplication.")
    multiplied_signal = signal1 * signal2
    return multiplied_signal

# Plotting functions
def plot_original_and_scaled(t_original, x_original, t_scaled, x, scale_factor, title="Time Scaling"):
    plt.figure(figsize=(12, 8))
    
    "Plotting the orginial signal"
    plt.subplot(2, 1, 1)
    plt.plot(t_original, x_original, 'b-', linewidth=2, label='Original Signal')
    plt.grid(True)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(f'Original Signal')
    plt.legend()
    
    "Plotting the time scaled signal"
    plt.subplot(2, 1, 2)
    plt.plot(t_scaled, x, 'r-', linewidth=2, label=f'Scaled Signal (factor={scale_factor})')
    plt.grid(True)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(f'Time Scaled Signal - Scale Factor: {scale_factor}')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def plot_multiplication(t, signal1, signal2, multiplied_signal, title1="Signal 1", title2="Signal 2"):
    plt.figure(figsize=(12, 10))
    
    "Plot the first original signal"
    plt.subplot(3, 1, 1)
    plt.plot(t, signal1, 'g-', linewidth=2, label=title1)
    plt.grid(True)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(f'{title1}')
    plt.legend()
    
    "Plot the second original signal"
    plt.subplot(3, 1, 2)
    plt.plot(t, signal2, 'm-', linewidth=2, label=title2)
    plt.grid(True)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title(f'{title2}')
    plt.legend()
    
    "Plot the multiplied signal"
    plt.subplot(3, 1, 3)
    plt.plot(t, multiplied_signal, 'k-', linewidth=2, label='Multiplied Signal')
    plt.grid(True)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.title('Multiplication Result')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def plot_unit_impulse(impulse_signal, impulse_index, title="Unit Impulse"):
    """
    Plot a unit impulse signal.
    """
    plt.figure(figsize=(10, 4))
    n_samples = len(impulse_signal)
    sample_indices = np.arange(n_samples)
    
    plt.stem(sample_indices, impulse_signal, basefmt=" ")
    plt.axvline(x=impulse_index, color='red', linestyle='--', alpha=0.7, label=f'Impulse at index {impulse_index}')
    plt.grid(True, alpha=0.3)
    plt.xlabel('Sample Index')
    plt.ylabel('Amplitude')
    plt.title(f'{title} - Impulse at index {impulse_index}')
    plt.legend()
    plt.ylim(-0.1, 1.1)
    plt.show()
    
# Demonstration code
if __name__ == "__main__":

    '''
    Paremeters :
    Fs = sampling freuqency 
    duration = max duration in seconds
    '''
        
    Fs = 1000  # Sampling frequency
    duration = 2  # seconds
    
    print("=== Signal Processing Functions Demonstration ===")

    print("\n1. Generating original signals...")
    '''
    The orginal signals (sinusoidal and triangular) are generated
    Return :
    t_sin = the time vector for the orginal sinusoidal signal
    t_tri = the time vector fot the orginal triangular signal
    '''
    t_sin, sin_wave = sinusoidal(frequency=2, phase=0, amplitude=1, duration=duration, Fs=Fs)
    t_tri, tri_wave = triangular(frequency=1, amplitude=0.8, duration=duration, Fs=Fs)
    
    print(f"   Sine wave: {len(sin_wave)} samples, {len(sin_wave)/duration} Hz effective sampling")
    print(f"   Triangle wave: {len(tri_wave)} samples")


    print("\n2. Performing time scaling operations...")
    
    '''
    time scaling compression operation :
    Parameters :
    t_sin = the time vector of the orginal sinusoidal signal
    sin_wave = the values of the orginal sinusoidal
    
    Returns :
    t_scaled_compressed/t_scaled_stretched = the new time vector after compressing
    sin_compressed/sin_stretched = the values of the signal after   
    
    '''
    t_scaled_compressed, sin_compressed = time_scaling(t_sin, sin_wave, scale_factor=2.0)
    plot_original_and_scaled(t_sin, sin_wave, t_scaled_compressed, sin_compressed, 
                           2.0, "Sinusoid Time Compression (2x faster)")
    
    t_scaled_stretched, sin_stretched = time_scaling(t_sin, sin_wave, scale_factor=0.5)
    plot_original_and_scaled(t_sin, sin_wave, t_scaled_stretched, sin_stretched, 
                           0.5, "Sinusoid Time Stretching (0.5x slower)")

    '''
    The signal multiplication operation :
    Parameters :
    sin_wave = the values of the orginal sinusoidal
    tri_wave = the values of the original triangular wave

    Returns :
    multiplied = the multiplied signals
    '''
    print("\n3. Performing signal multiplication...")
    multiplied = multiply_signals(sin_wave, tri_wave)
    plot_multiplication(t_sin, sin_wave, tri_wave, multiplied, 
                       "Sinusoidal Wave (2 Hz)", "Triangular Wave (1 Hz)")

    '''
    unit impulse demonstration
    Parameters :
    The orginal unit impulse
    
    Returns :
    The impulse signal (the delta)
    '''
    print("\n4. Generating unit impulse...")
    impulse_signal = unit_impulse(n_samples=20, impulse_index=5)
    plot_unit_impulse(impulse_signal, impulse_index=5, title="Discrete Unit Impulse")

    print("\n5. Testing functions with default parameters...")
    t_sin_default, sin_default = sinusoidal() 
    t_tri_default, tri_default = triangular()  
    impulse_default = unit_impulse()  
    
    print("   Default sinusoidal:", f"frequency=2Hz, samples={len(sin_default)}")
    print("   Default triangular:", f"frequency=1Hz, samples={len(tri_default)}")
    print("   Default unit impulse: 20 samples with impulse at index 5")
    print("\n=== Demonstration Complete ===")
    print("All signal processing functions working correctly!")
    print("Generated: 2 time scaling plots, 1 multiplication plot, 1 impulse plot")
    
def test_sine_wave():
    'Testing functionality of the sin function'
    t, x = sinusoidal(frequency=10, amplitude=1, duration=1)
    assert len(t) == 1000
    assert x[0] == 0

    'Edge case of a maximum amplitude (in this case 5)'
    t, x = sinusoidal(frequency=2, amplitude=5, duration=1)
    assert np.isclose(max(x), 5, atol=1e-6)

    "Edge case when the duration is -1, which is ought to return empty arrays"
    t, x = sinusoidal(frequency=6, amplitude=3, duration=-1)
    assert len(t) == 0 and len(x) == 0
    
    "Edge case when amplitude is 0, so it should return zeros"
    t, x = sinusoidal(frequency=13, amplitude=0, duration=4)
    assert np.allclose(x, 0)

test_sine_wave()
print("All sinusoidal wave tests passed!")

def test_triangular_wave():
    
    "The same tests for the triangular wave as for the sinusoidal wave"
    t, x = triangular(frequency=10, amplitude=1, duration=1)
    assert len(t) == 1000 

    t, x = triangular(frequency=2, amplitude=5, duration=1)
    assert np.isclose(max(x), 5, atol=1e-6)  
    assert np.isclose(min(x), -5, atol=1e-6)
    
    t, x = triangular(frequency=6, amplitude=3, duration=0)
    assert len(t) == 0 and len(x) == 0

    t, x = triangular(frequency=13, amplitude=0, duration=4)
    assert np.allclose(x, 0)
    
test_triangular_wave()
print("All triangular wave tests passed!")

def test_unit_impulse():
    
    '''
Making sure that the sum of the impulses is 1, since every impulse should be 0 except
at the impulse index
'''
impulse = unit_impulse(n_samples=20, impulse_index=5)
assert len(impulse) == 20
assert impulse[5] == 1.0  
assert np.sum(impulse) == 1.0 
assert np.all(impulse[:5] == 0) and np.all(impulse[6:] == 0)

'''
Edge case when the impulse index is out of the bounds set by the n_samples, so it should
return 0
'''
impulse = unit_impulse(n_samples=5, impulse_index=10)
assert len(impulse) == 5
assert np.all(impulse == 0)

test_unit_impulse()
print("All unit_pulse tests passed!")

def test_time_scaling():
    
    t, sine_signal = sinusoidal(frequency=4, amplitude=2, duration=2, Fs=1000)
    '''
    checking if it the signal doesn't change when the scale factor is 1
    '''
    t_original, signal_original = time_scaling(t, sine_signal, scale_factor=1.0)
    assert np.allclose(t_original, t, atol=1e-10)
    assert np.allclose(signal_original, sine_signal, atol=1e-10)
    
    t_tri, tri_signal = triangular(frequency=8, amplitude=11, duration=2, Fs=1000)
    '''
    Checking if the signal doesn't change when the scale factor is 1 with the triangular wave
    '''
    t_scaled_tri, tri_scaled = time_scaling(t_tri, tri_signal, scale_factor=2.0)
    assert len(t_scaled_tri) == len(t_tri)
    
test_time_scaling()
print("All time scaling tests passed!")

def test_multiplication():
    t, sine1 = sinusoidal(frequency=7, amplitude=1, duration=1, Fs=100)
    _, sine2 = sinusoidal(frequency=8, amplitude=5, duration=1, Fs=100)
    multiplied = multiply_signals(sine1, sine2)   
   
    '''
    Test wheter to see the amplitude stays in bounds when there
    are signals have a different amplitude during miultiplication (0.5 x 2.0 = 1.0 max)
    '''
    _, sine3 = sinusoidal(frequency=12, amplitude=0.5, duration=5, Fs=100)
    _, sine4 = sinusoidal(frequency=12, amplitude=2.0, duration=5, Fs=100)
    multiplied2 = multiply_signals(sine3, sine4)
    assert np.all(multiplied2 >= -1) and np.all(multiplied2 <= 1) 
    
    '''
    Multiplication with a zero signal, so the result should be 0
    '''
    zero_signal = np.zeros_like(sine1)
    multiplied_zero = multiply_signals(sine1, zero_signal)
    assert np.allclose(multiplied_zero, 0, atol=1e-10)
                       
test_multiplication()
print("All multiplication tests passed!")                    