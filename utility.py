'''
A collection of useful functions
Author: Tristhal Parasram
Date: 2020-05-01
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

def FWHM(dist, x, samples=10, padding=1, plotting=False):
    '''
    Find the FWHM of a single peak
    
    Parameters
    ----------
    dist: numpy array
        the curve to find fwhm
    x: numpy array
        x values corresponding to dist
    samples: integer
        number of values to interpolate
    plotting: boolean
        show the fwhm
        
    Output
    ------
    int
        FWHM
    
    Author: Tristhal Parasram
    '''
    # Pad with zeroes to ensure a zero
    distribution = np.pad(dist, 1, 'constant')
    # Matching the x values by extending the edges linearly from where they were
    Gaussian_x = np.pad(x, 1, 'edge')
    Gaussian_x[-1] += Gaussian_x[-2]-Gaussian_x[-3]
    Gaussian_x[0] -= Gaussian_x[2]-Gaussian_x[1]
    
    max_spacing = np.max(np.abs(distribution[:-1]-distribution[1:]))
    # Getting the interpolated points for the peak
    # interpolate.interp1d returns a function and the x values are passed
    x_values=np.linspace(Gaussian_x[0], Gaussian_x[-1], len(dist)*samples)
    peak = interpolate.interp1d(Gaussian_x, distribution)(x_values)
    # Capture window scales based of samples 1/2*max(dist)/(samples-1) is the max distance from 0 to a point
    window = padding/2*max_spacing/(samples-1)
    
    # Find values near zero based on the window
    query = np.transpose(abs(np.max(peak)/2 - np.transpose(peak)) < window)
    
    # Take the distance between
    # Faster Method: fwhm = np.abs((np.nonzero(query)[0][-1] - np.nonzero(query)[0][0]))/samples*np.abs(Gaussian_x[0]-Gaussian_x[-1])/len(dist)
    
    fwhm = np.array([(x_values[np.nonzero(query)[0][-1]] - x_values[np.nonzero(query)[0][0]])])
    
    # Plotting
    if plotting:
        plt.plot(x_values, query*np.max(peak))
        plt.plot(x_values, peak)
    return fwhm

def gaussian(amplitude, position, width, x):
    """
    Calculates a gaussian function at x

    Parameters
    ----------
    amplitude: float
        amplitude of the curve
    position: float
        log of the position
    width: float
        width
    x: numpy array
        array of x values to compute

    Return
    ------
    array floats
        array of y values
        
    Author: Tristhal Parasram
    """
    return amplitude*np.exp(-1/2 * (((x - position) / width)**2))