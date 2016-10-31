# non-uniform discrete fourier transform
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import periodogram

def ndft_matrix(time,freq):
  return np.exp(-2*np.pi*1j*freq[:,None]*time[None,:])
  
def ndft(u,time,T=None):   
  ''' 
  computes discrete fourier transform from non-uniformly spaced observations
  '''
  N = len(time)
  if T is None:
    T = N*time.ptp()/(N-1)

  freq = np.fft.fftfreq(N,T/N)
  F = ndft_matrix(time,freq)
  uhat = F.dot(u)  
  return freq,uhat

def psd(u,time,T=None):
  N = len(time)
  if T is None:
    T = N*time.ptp()/(N-1)

  freq = np.fft.fftfreq(N,T/N)
  df = freq[1] - freq[0]
  F = ndft_matrix(time,freq)
  uhat = F.dot(u)/N
  pow = (uhat.conj()*uhat).real
  freq = freq[:N//2]
  pow = 2*pow[:N//2]/df
  return freq[1:],pow[1:]
  

if __name__ == '__main__':
  N = 10000
  T = 2*np.pi*10.0
  time = np.arange(N)*T/N
  #time = np.random.uniform(0.0,T,N)
  time = np.sort(time)
  #freq = np.fft.fftfreq(N,T/N)
  #time_itp = np.arange(N)*T/N
  #F = ndft_matrix(time,freq)  
  u = np.random.normal(0.0,10.0,N)
  #plt.plot(time,u)
  #nplt.show()
  u = np.cumsum(u)
  plt.plot(time,u,'o')
  plt.show()
  freq1,pow1 = psd(u,time)
  freq2,pow2 = periodogram(u,N/T,scaling='density')
  
  plt.figure(1)
  plt.loglog(freq1[1:],pow1[1:],'k.')
  plt.figure(2)
  plt.loglog(freq2[1:],pow2[1:],'k.')
  plt.show()
  #uhat_diff = 2*np.pi*1j*freq*uhat
  #udiff = np.linalg.solve(F,uhat_diff)
  #plt.figure(1)
  #nplt.plot(time,udiff,'b.')
  #plt.show()
