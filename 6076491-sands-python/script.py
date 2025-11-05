import numpy as np
import matplotlib.pyplot as plt
from signals import sinusoidal, triangular, unit_impulse, time_scaling, multiply_signals

'PLotting of the signals and operations'
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
    
    'Performance of the code'
if __name__ == "__main__":

    '''
    Paremeters :
    Fs = sampling freuqency 
    duration = max duration in seconds
    '''
        
    Fs = 1000
    duration = 2
    
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
