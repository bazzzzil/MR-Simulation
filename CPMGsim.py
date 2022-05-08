# -*- coding: utf-8 -*-
"""
Simulating CPMG signal

Original Author: Mark Armstrong, University of Windsor. 2018.
Comments & Modifications: Layale Bazzi, University of Windsor. 2019.
Features: 
    -Simulates signal of CPMG pulse sequence for any T1/T2 combination
    -B0 inhomogeneity incorporated (see mripy.py for details)
    -Uses frequency offsets in M instead of isochromats
    
Please make sure this file is in the same directory as mripy.py

Last updated: May 6, 2020

"""
import pickle
import mripy as mp
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as optimization
from datetime import datetime
from scipy.fft import fft,ifft,fftshift,ifftshift, fftfreq
plt.rcParams.update({'font.size': 40})

"""
##############################################################################
# =============================================================================
# Defining variables
# =============================================================================
##############################################################################
"""
# 1 timestep is 1 us
FWHM=150 # Desired FWHM of distribution
spacing=10 # spacing used in frequency distribution (see mripy)
shift=0 # 0 means on-resonance (initial shift)
TotalB0Drift=0 # Hz

# Constants
T1 = 75*1e3 # T1 relaxation time (us)
T2 = 50*1e3  # T2 relaxation time (us)
g  = 42.57e6*2*np.pi # Gyromagnetic ratio in rad*Hz/T

# Pulse params
TxDuration  = 100 # Excitation/Refocusing pulse length
phase90     = 0 # Phase of excitation pulse
phase180    = np.pi/2 # Phase of refocusing pulse
# Can change CPMG to CP by modifying phase180 to 0

BW   = (1/TxDuration)*1e6 # Excitation bandwidth in Hz
ETL  = 50 # Echo train length
DriftIncrement=TotalB0Drift/ETL

# The echo time (TE) is defined as the period of time between middle/70% point
# of the 90 and the formation of an echo. 
# Further on (e.g.: between refocusing pulses), the echo time is the time between subsequent echoes.
TE = int(3*1e3) # Echo time

print('Pulse Duration:',TxDuration, 'us')

# Lorentzian distribution of isochromats
offset,BW0,distribution=mp.distShape(spacing,TxDuration,shift,FWHM,CalculateBW=True,plotVar=False)
isochromats=len(offset)

A2=180
A1=A2/2

FlipAngle   = np.array([np.deg2rad(A1)]) # Excitation
Refocusing  = np.array([np.deg2rad(A2)]) # Refocusing


"""
##############################################################################
# =============================================================================
# Creating initial magnetization vectors
# =============================================================================
##############################################################################
"""

# M is three dimensional for matmul to work, third dimension is not used
M = np.zeros([isochromats,3,1]) # Magnetization in three dimensions
# First dim: all isochromats, second dim: 0=x,1=y,2=z, third dim: not used

M[:,2,0] = np.ones(isochromats) # Thermal equilibrium (in z)

print(0,"|",end="")

print("B0 bandwidth:",BW0,"Hz\n")

# Setting all offsets to zero
# offset=np.zeros(isochromats)

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

# Pre-allocating memory for echo amplitude and time arrays
echoAmp=np.zeros(ETL)
timeVec=np.linspace(TE,TE*ETL,ETL)*1e-3

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

# Applying pi/2 pulse 
(M,XY[0:TxDuration,0],XY[0:TxDuration,1],pulse) = mp.rectPulse(FlipAngle, M, 
                                                     rotMatrix, BB, phase90, TxDuration)

# Updating time step
time=TxDuration
    
total_signal=sum(M[:,1,0])
print("Total signal:",np.round(total_signal,3))
# print(np.sqrt(total_signal/(g*TxDuration)))
#%% 180 echo loop below

# B0 drift, shift of DriftIncrement between 90 and 180 pulse
print("After Excitation ","|",end="")
shift=round(shift+DriftIncrement,2)
offset,remp,distribution=mp.distShape(spacing,TxDuration,shift,FWHM,CalculateBW=False,plotVar=False)
rotMatrix=mp.update_rotMatrix(isochromats,rotMatrix,offset,AA)

# TE/2 is at the edge of the 90 - b
# b is some adjustable time parameter

b=0 # d3
# b=round(TxDuration*(1-2/np.pi)) # d2
# b=int(TxDuration/2) # d1

# Adding in effects of distribution
for i in range(3):
    M[:,i,0]=M[:,i,0]*distribution

# Free precession
duration=int(TE/2)-int(TxDuration/2)-b
M,XY,time=mp.free_precess(duration,rotMatrix,M,BB,XY,time)

print("Echo # (total=", ETL,"):",sep="")

# Echo train
for i in range(ETL):
    print(i+1,"|",end="")
    
    # Applying pi pulse 
    (M,XY[time:time+TxDuration,0],
     XY[time:time+TxDuration,1],temp) = mp.rectPulse(Refocusing, M, 
                                            rotMatrix, BB, phase180, TxDuration)
        
    # Updating timestep
    time=time+TxDuration

    # Recording pre-echo formation timestep
    preEcho=time
    
    # Relaxation and free precession up to echo formation
    duration=int(TE/2)-int(TxDuration/2)
    M,XY,time=mp.free_precess(duration,rotMatrix,M,BB,XY,time)
    
    time=time-1
    
    # Recording echo based off mid point of acquisition
    echoAmp[i]=np.sqrt(XY[time,0]**2 + XY[time,1]**2)
    
    time=time+1
    
    # Relaxation and free precession past echo formation up to start of next pulse
    duration=int(TE/2)-int(TxDuration/2)
    M,XY,time=mp.free_precess(duration,rotMatrix,M,BB,XY,time)
      
    # Recording post-echo formation timestep
    postEcho=time
    
    # Recording echo based off max echo of acquisition
    # echoAmp[i]=max(np.sqrt(XY[preEcho:postEcho,0]**2 + XY[preEcho:postEcho,1]**2))
    
    # B0 drift
    shift=round(shift+DriftIncrement,2)
    offset,temp,distribution=mp.distShape(spacing,TxDuration,shift,FWHM,CalculateBW=False,plotVar=False)
    rotMatrix=mp.update_rotMatrix(isochromats,rotMatrix,offset,AA)
    
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
plt.xticks([],[])
plt.yticks([],[])
plt.title(title_str)

#%%
"""
##############################################################################
# =============================================================================
# Results: Decay curve + Fitting
# =============================================================================
##############################################################################
"""

def func(x,S,T):
    return S*np.exp(-x/T)

x0=[max(echoAmp),T2*1e-3]

fitResult=optimization.curve_fit(func, timeVec, echoAmp, x0)
decayParams=fitResult[0]

print("B0 bandwidth:",BW0,"Hz\n")
print(" Echo Train Length:", ETL, "\n", "Echo Spacing (ms):", TE*1e-3, "\n",
      "Pulse Duration (us):", TxDuration, "\n", "Actual T2 (ms):", 
        T2*1e-3, "\n", "Number of isochromats:", isochromats, "\n", 
        "Predicted isochromats:", round(decayParams[0]),"\n", 
        "Predicted T2 (ms):", round(decayParams[1],1),"\n",
        "Total Runtime:", RunTime, "\n")
print("\n _______________________________________________________")

# FitDecay=decayParams[0]*np.exp(-timeVec/(decayParams[1]))

plt.figure(figsize=(20,15))
plt.plot(timeVec,echoAmp,'b',linewidth=5,label='Simulated Decay')
# plt.plot(timeVec,FitDecay,'--',color='salmon',linewidth=5,label='Fit')
plt.xlabel('Time (ms)')
plt.ylabel('Echo Amplitude')
plt.title('Decay Curve')
plt.gca()
plt.legend()
plt.xticks([0,50,100,150],[0,50,100,150])
plt.show()

#%%
"""
##############################################################################
# =============================================================================
# Results: Echoes Spectra
# =============================================================================
##############################################################################
"""

EchoStart=int(TE/2)+b
EchoEnd=3*int(TE/2)

step=3

firstEcho=XY[EchoStart:EchoEnd,0] + 1j*XY[EchoStart:EchoEnd,1]
secondEcho=XY[step*TE+EchoStart:step*TE+EchoEnd,0] + 1j*XY[step*TE+EchoStart:step*TE+EchoEnd,1]
freq=fftfreq(firstEcho.size)
idx = np.argsort(freq)

FTecho=fftshift(fft(fftshift(firstEcho)))

plt.figure(figsize=(15,10))
plt.xticks([],[])
plt.yticks([],[])
plt.plot(np.abs(FTecho),label='abs')
plt.plot(np.imag(FTecho),label='imag')
plt.plot(np.real(FTecho),label='real')
plt.legend()


#%% Saving Data

# with open('test.pickle', 'wb') as sim:
#     pickle.dump([echoAmp, pulse], sim)




