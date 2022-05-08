# -*- coding: utf-8 -*-
"""
Simulating FID signal

Original Author: Layale Bazzi, University of Windsor. 2019.
Features: 
    -Simulates FID signal for any T1/T2 combination
    -B0 inhomogeneity incorporated (see mripy.py for details)
    -Uses frequency offsets in M instead of isochromats
    
Please make sure this file is in the same directory as mripy.py

Last updated: May 8, 2022

"""
import mripy as mp
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.fftpack import fft,ifft,fftshift,ifftshift, fftfreq
plt.rcParams.update({'font.size': 40})
plt.rcParams.update({'figure.max_open_warning': 0})
    
"""
##############################################################################
# =============================================================================
# Defining variables
# =============================================================================
##############################################################################
"""
# for i in range(1,21):
# 1 timestep is 1 us
FWHM=10
spacing=.8 # spacing used in frequency distribution (see mripy)
shift=0 # 0 means on-resonance (initial shift)

# Constants
T1 = 200*1e3 # T1 relaxation time (us)
T2 = 100*1e3  # T2 relaxation time (us)
g  = 42.57e6*2*np.pi # Gyromagnetic ratio in rad*Hz/T

A1=90

# Pulse params
TxDuration  = 1 # Excitation/Refocusing pulse length
FlipAngle   = np.array([np.deg2rad(A1)]) # Excitation
phase90     = 0 # Phase of excitation pulse

BW   = (1/TxDuration)*1e6 # Excitation bandwidth in Hz

# The echo time (TE) is defined as the period of time between middle/70% point
# of the 90 and the formation of an echo. 
# Further on (e.g.: between refocusing pulses), the echo time is the time between subsequent echoes.
TE = int(3*1e3) # Echo time

print('Pulse Duration:',TxDuration, 'us')

"""
##############################################################################
# =============================================================================
# Creating initial magnetization vectors
# =============================================================================
##############################################################################
"""
# Lorentzian distribution of isochromats
offset,BW0,distribution=mp.distShape(spacing,TxDuration,shift,FWHM,CalculateBW=True,plotVar=True)

isochromats=len(offset)

# M is three dimensional for matmul to work, third dimension is not used
M = np.zeros([isochromats,3,1]) # Magnetization in three dimensions
# First dim: all isochromats, second dim: 0=x,1=y,2=z, third dim: not used

M[:,2,0] = np.ones(isochromats) # Thermal equilibrium (in z)

print(0,"|",end="")

print("B0 bandwidth:",BW0,"Hz\n")
print("Isochromats:",isochromats,"\n")
"""
##############################################################################
# =============================================================================
# Creating free precession + relaxation matrices (time evolution)
# =============================================================================
##############################################################################
"""
# Acquiring relaxation matrices: AA is transverse, BB is longitudinal
(AA,BB) = mp.relaxation(1,T1,T2)

# Updating magnetization at equilibrium
BB = M*BB

# Time evolution matrix for 1us (free precession with offset)
rotMatrix=np.zeros([isochromats,3,3]) # Allocating memory
rotMatrix=mp.update_rotMatrix(isochromats,rotMatrix,offset,AA)

# Pre-allocating memory for xy-magnetization vector for 10000000 timesteps
XY = np.zeros([10000000,2])
# x=0, y=1

"""
##############################################################################
# =============================================================================
# Pulse Sequence
# =============================================================================
##############################################################################
"""
# Start clock at time=0
time=0
StartTime=datetime.now()

print("Start.\n")

(M,XY[0:TxDuration,0],XY[0:TxDuration,1],pulse) = mp.rectPulse(FlipAngle, M, 
                                                      rotMatrix, BB, phase90, TxDuration)

# Updating time step
time=TxDuration

# Adding in effects of distribution
for i in range(3):
    M[:,i,0]=M[:,i,0]*distribution

# Free precession
duration=15*TE
M,XY,time=mp.free_precess(duration,rotMatrix,M,BB,XY,time)

print("\n\nDone.\n")
RunTime=datetime.now()-StartTime

#%%
"""
##############################################################################
# =============================================================================
# Results: Total signal in both channels
# =============================================================================
##############################################################################
"""
timeArr=np.linspace(0,len(XY[TxDuration:time,0]),len(XY[TxDuration:time,0]))

title_str="Real and Imaginary Signal"

# Plotting signal over time in both channels
plt.figure(figsize=(15,10))
plt.plot(timeArr,XY[TxDuration:time,0],label='x-channel',linewidth=4)
plt.plot(timeArr,XY[TxDuration:time,1],label='y-channel',linewidth=4)
plt.legend(loc=1)
# plt.xticks([],[])
# plt.yticks([],[])
plt.title(title_str)

print("B0 bandwidth:",BW0,"Hz\n")
print("Pulse Duration (us):", TxDuration, "\n", "Actual T2 (ms):", 
        T2*1e-3, "\n", "Number of isochromats:", isochromats, "\n", 
        "Total Runtime:", RunTime, "\n")
print("\n _______________________________________________________")
