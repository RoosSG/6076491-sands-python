import pytest
import numpy as np
from signals import sinusoidal, triangular, unit_impulse, time_scaling, multiply_signals

    
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
