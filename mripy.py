"""
Original Author: Mark Armstrong (2018)
Comments & Modifications: Layale Bazzi (2019)

Module for Bloch simulation script
Please make sure this file is in the same directory as CPMGsim.py
Uses FWHM code by Tristhal Parasram: utility.py

Last updated: May 8, 2022

"""
import numpy as np
import utility as util
import matplotlib.pyplot as plt

###############################################################################
def rotZ(a):
    """
    Returns rotation matrix for a left handed 
    rotation about the Z-axis.

    Parameters
    ----------
    a : float
        Rotation in radians

    Returns
    -------
    3x3 array
        Rotation matrix
    """
    return np.array([[np.cos(a),np.sin(a),0],[-np.sin(a),np.cos(a),0],[0,0,1]])

###############################################################################
def rotY(a):
    """
    Returns rotation matrix for a left handed 
    rotation about the Y-axis.

    Parameters
    ----------
    a : float
        Rotation in radians

    Returns
    -------
    3x3 array
        Rotation matrix
    """
    return np.array([[np.cos(a),0,-np.sin(a)],[0,1,0],[np.sin(a),0,np.cos(a)]])

###############################################################################
def rotX(a):
    """
    Returns rotation matrix for a left handed 
    rotation about the X-axis.

    Parameters
    ----------
    a : float
        Rotation in radians

    Returns
    -------
    3x3 array
        Rotation matrix
    """
    return np.array([[1,0,0],[0,np.cos(a),np.sin(a)],[0,-np.sin(a),np.cos(a)]])

############################################################################
def lorentzian(x,x0=0,a=100,c=1000):
    """
    Returns lorentzian distrbution over domain of x.

    Parameters
    ----------
    x : int array
        Grid points
        
    a : float
        Height of lorentzian
        
    x0 : float
        Centre location of lorentzian
        
    c : float
        Width of Lorentzian   

    Returns
    -------
    array
        Lorentzian distribution float array
    """
    y = a/(((x-x0)/c)**2+1)
    return y

###############################################################################
def sinc(x):
    """
    Returns sinc function over domain of x.
    Note: care is taken to avoid division by 0.

    Parameters
    ----------
    x : int array
        Grid points 

    Returns
    -------
    array
        Sinc function float array
    """
    y = np.zeros(len(x))
    y[x==0]=1
    y[x!=0]=np.sin(x[x!=0])/(x[x!=0])
    return y

###############################################################################
def gauss(x,s,A):
    """
    Returns gauss function over domain of x with width s^2 and height A

    Parameters
    ----------
    x : int
        Grid points 
        
    s : float
        Width of Gaussian
        
    A : float
        Height of Gaussian
    Returns
    -------
    float array
        Gauss function float array
    """
    y = np.exp(-(x**2)/(s**2))
    y=y/max(y)
    y=A*y
    
    return y

###############################################################################
def rotAlpha(alpha, theta=0, phi=0):  
    """
    Returns rotation matrix for a rotation by alpha radians about 
    an axis defined by theta and phi in right-handed coordinate system.
    See Euler angles for more information.

    Parameters
    ----------
    alpha : float
        Rotation in radians 
        
    theta : float
        Angle between the z-axis and the xy-plane 
        
    phi : float
        Angle in the xy-plane starting counterclockwise from x-axis 
        
    Returns
    -------
    3x3 array
        Arbitrary rotation matrix
    """
    return rotZ(-phi).dot(rotY(theta).dot(rotX(alpha).dot(rotY(-theta).dot(rotZ(phi)))))

###############################################################################
def rotXY(alpha,phi=0):
    """
    Returns rotation matrix for a rotation by alpha radians 
    about an axis defined by phi in the xy-plane.

    Parameters
    ----------
    alpha : float
        Rotation in radians  (flip angle)
        
    phi : float
        Angle in the xy-plane starting counterclockwise from x-axis 
        
    Returns
    -------
    3x3 array
        xy-plane rotation matrix
    """
    cA = np.cos(alpha)
    sA = np.sin(alpha)
    cP = np.cos(phi)
    sP = np.sin(phi)
    return np.array([[sP**2*cA+cP**2,-sP*cA*cP+sP*cP,-sA*sP],
                     [-sP*cA*cP+sP*cP,sP**2+cA*cP**2,sA*cP],
                     [sA*sP,-sA*cP,cA]])

###############################################################################
def relaxation(t,T1=100000,T2=10000):
    """
    Returns transverse (AA) and longitudinal (BB) relaxation parameters 
    via T1 and T2 input relaxation times. All time units are in microseconds.

    Parameters
    ----------
    t : int
      Time over which relaxation occurs  
      
    T1 : float
      Longitudinal relaxation time
      
     T2 : float
      Transverse relaxation time
        
    Returns
    -------
    AA : 3x3 float array
        Transverse relaxation matrix
        
     BB : float
        Longitudinal relaxation constant
    """
    decay1=np.exp(-t/T1);
    decay2=np.exp(-t/T2);
    
    AA = np.array([[decay2,0,0],
                   [0,decay2,0],
                   [0,0,decay1]])
    BB = 1-decay1
    return (AA,BB)

###############################################################################
def free_precess(duration,rotMatrix,M,BB,XY,time):
    """
    Free precession of magnetization calculated with rotation matrix offset.

    Parameters
    ----------
    duration : int
      Duration of free precession  
      
     rotMatrix : float 3D array
      Rotation matrix that contains relaxation & free precess params
      
     M : float 3D array
      Magnetization vector
      
     BB : float
      Longitudinal relaxation 
      
     XY : float array
      x & y channels of receiver over simulation duration
      
     time : int
      Current timestep in simulation 

    Returns
    -------
    M : float 3D array
        Updated magnetization vector
        
     XY : float 2D array
        Updated x & y channels of receiver over simulation duration
        
     time : int
        Updated time step
    """
    for j in range(duration):
        M = np.matmul(rotMatrix,M)+BB
        XY[time,0] = np.sum(M[:,0,0])
        XY[time,1] = np.sum(M[:,1,0])
        time = time + 1
    return M,XY,time

###############################################################################
# Updates rotation matrix based on offset array being used
def update_rotMatrix(isochromats,rotMatrix,offset,AA):
    """
    Updates rotation matrix based on offset array being used.

    Parameters
    ----------
    isochromats : int
      Number of isochromats in simulation
      
     rotMatrix : float 3D array
      Rotation matrix that contains relaxation & free precess params
      
    offset : float array
      Isochromat frequencies array
      
     AA : 3x3 float array
        Transverse relaxation matrix

    Returns
    -------
    rotMatrix : float 3D array
        Updated rotation matrix
    """    
    for p in range(isochromats): # Setting up matrix for each isochromat
        rotMatrix[p,:,:]=rotZ(offset[p]*2*np.pi/10**6)
        
    # Updating rotation matrix to contain relaxation
    rotMatrix = np.matmul(rotMatrix,AA)
    
    return rotMatrix

###############################################################################
def sincPulse(alpha, M, rotMatrix, BB, phi=0, TxDuration=50):
    """
    Returns updated magnetization vector M and bulk transverse signal
    components x and y after application of a sinc pulse. All time units are in microseconds.

    Parameters
    ----------
    alpha : float array (one element)
      Flip angle in radians     
      
     M : float 3D array
      Magnetization vector
      
     rotMatrix : float 3D array
      Rotation matrix that contains relaxation & free precess params
      
     BB : float
      Longitudinal relaxation 
      
     phi : int
      Phase of RF pulse
      
     TxDuration : int
      Duration of pulse      

    Returns
    -------
    M : float 3D array
        Updated magnetization vector
        
     x : float array
        x channel of transverse magnetization
        
     y : float array
        y channel of transverse magnetization
        
     parts : float array
        Pulse shape used for plotting 
    """
    alpha=alpha[0]
    # n and TxDuration both control the number of lobes in the sinc
    # n=0.28 and TxDuration=700 give a sinc with one lobe
    n=0.28
    
    pulse_length = np.arange(-TxDuration/2+1,TxDuration/2+1)
    a = alpha/sum(sinc(n*pulse_length/10))
    parts = a*sinc(n*pulse_length/10)
    
    x=np.zeros(TxDuration)
    y=np.zeros(TxDuration)

    for i in range(TxDuration):
        newRotMatrix= np.matmul(rotMatrix,rotXY(parts[i],phi=phi)) # New Matrix
        M = np.matmul(newRotMatrix,M)+BB # RF + Precession + relaxation
        x[i] = np.sum(M[:,0,0])
        y[i] = np.sum(M[:,1,0])
    return(M,x,y,parts)

###############################################################################
def gaussPulse(alpha, M, rotMatrix, BB, phi=0, TxDuration=50):
    """
    Returns updated magnetization vector M and bulk transverse signal
    components x and y after application of a Gaussian pulse. All time units are in microseconds.

    Parameters
    ----------
    alpha : float array (one element)
      Flip angle in radians
      
     M : float 3D array
      Magnetization vector
      
     rotMatrix : float 3D array
      Rotation matrix that contains relaxation & free precess params
      
     BB : float
      Longitudinal relaxation 
      
     phi : int
      Phase of RF pulse
      
     TxDuration : int
      Duration of pulse      

    Returns
    -------
    M : float 3D array
        Updated magnetization vector
        
     x : float array
        x channel of transverse magnetization
        
     y : float array
        y channel of transverse magnetization
        
     parts : float array
        Pulse shape used for plotting 
    """
    alpha=alpha[0]
    # c and TxDuration both control the width of the Gaussian
    # c=100 and TxDuration=600 give a medium-sized Gaussian
    c=100
    pulse_length = np.arange(-TxDuration/2+1,TxDuration/2+1)
    a = alpha/sum(np.exp(-((pulse_length)**2)/(c**2)))
    parts = a*np.exp(-((pulse_length)**2)/(c**2))
    
    x=np.zeros(TxDuration)
    y=np.zeros(TxDuration)

    for i in range(TxDuration):
        newRotMatrix= np.matmul(rotMatrix,rotXY(parts[i],phi=phi)) # New Matrix
        M = np.matmul(newRotMatrix,M)+BB # RF + Precession + relaxation
        x[i] = np.sum(M[:,0,0])
        y[i] = np.sum(M[:,1,0])
    return(M,x,y,parts)

###############################################################################
def rectPulse(alpha, M, rotMatrix, BB, phi=0, TxDuration=50):
    """
    Returns updated magnetization vector M and bulk transverse signal
    components x and y after application of a Gaussian pulse. All time units are in microseconds.

    Parameters
    ----------
    alpha : float array (one element)
      Flip angle in radians
      
     M : float 3D array
      Magnetization vector
      
     rotMatrix : float 3D array
      Rotation matrix that contains relaxation & free precess params
      
     BB : float
      Longitudinal relaxation 
      
     phi : int
      Phase of RF pulse
      
     TxDuration : int
      Duration of pulse      

    Returns
    -------
    M : float 3D array
        Updated magnetization vector
        
     x : float array
        x channel of transverse magnetization
        
     y : float array
        y channel of transverse magnetization
        
     parts : float array
        Pulse shape used for plotting 
    """
    alpha=alpha[0]
    TxDuration = int(TxDuration)
    newRotMatrix = np.matmul(rotMatrix,rotXY(alpha/TxDuration,phi=phi))
    
    x=np.zeros(TxDuration)
    y=np.zeros(TxDuration)
    
    parts=np.zeros(TxDuration+200)
    parts[0+100:-100]=alpha/TxDuration

    for i in range(TxDuration):
        M = np.matmul(newRotMatrix,M)+BB # Precession + relaxation
        x[i] = np.sum(M[:,0,0])
        y[i] = np.sum(M[:,1,0])
    return (M,x,y,parts)

###############################################################################
def distShape(spacing,TxDuration,shift,FWHM,CalculateBW=False,plotVar=False):
    """
    Defines Lorentzian (or Gaussian) distribution of isochromat offsets. 
    Simulates presumed distribution imparted by static magnetic field inhomogeneities.

    Parameters
    ----------    
    spacing : int
      Space between points in distribution. Affects computational speed
      
    TxDuration : float
      Pulse duration in us
      
    shift : int
      Off-resonance frequency shift in isochromat distribution in Hz
      
    CalculateBW : Bool
      Used so that the echo loop doesn't run this snippet all the time
        
    plotVar : Bool
      Used so that the echo loop doesn't run this snippet all the time
        
    FWHM : float
      Desired FWHM of distribution  
      
    Returns
    -------
        
    freeOffset : float array
        Freqeuncies array
        
     BW : float 
        True FWHM of distribution
        
    distribution : float array
        B0 bandwidth of isochromats
    """
    
    xrange=np.arange(-FWHM*4,FWHM*4+spacing,spacing) # Plotting active bandwidth
    
    freeOffset=xrange
    
    # Choose Gaussian or Lorentzian
    distribution=lorentzian(xrange,x0=shift,a=100,c=FWHM/2)
    # distribution=gauss(xrange,s=FWHM/2,A=100)
    BW=0
    
    # Calculating FWHM  
    if CalculateBW:
         if plotVar:
            plt.figure(figsize=(15,10))
            plt.title('B0 BW')
            plt.xlabel('Off-Resonance Shift (Hz)')
            
         BW=util.FWHM(distribution,xrange,plotting=plotVar)
    
    # Normalizing distribution to number of isochromats
    norm=len(freeOffset)/np.sum(distribution)
    distribution=np.array(distribution*norm,dtype=float)
    
    if plotVar:
        
        plt.figure(figsize=(15,10))
        plt.plot(xrange,distribution)
        plt.title('B0 BW')
        plt.xlabel('Off-Resonance Shift (Hz)')
        
        plt.figure(figsize=(12,10))
        plt.plot(freeOffset,'.')
        plt.title('Isochromats across B0 BW')
        plt.xlabel('Isochromat')
        plt.ylabel('BW (Hz)')
    
    print(" Shift:",shift, "Hz \n")
    # print(freeOffset)

    return freeOffset,BW,distribution   
    
    
    
    
    
    
    
    
    