#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import rbf.fd
import sympy
import scipy.linalg
import scipy.signal
np.random.seed(4)

def psd(signal,times):
  dt = times[1] - times[0]
  fs = 1.0/dt
  freq,pow = scipy.signal.periodogram(signal,fs,detrend='constant',scaling='density',return_onesided=True)
  return freq[1:],pow[1:]


def spectral_diff_matrix(N,dt,diff):
  ''' 
  generates a periodic sinc differentation matrix. This is equivalent 
  
  Parameters
  ----------
    N : number of observations 
    dt : sample spacing
    diff : derivative order (max=2)
       
  '''
  scale = dt*N/(2*np.pi)
  dt = 2*np.pi/N
  t,h = sympy.symbols('t,h')
  sinc = sympy.sin(sympy.pi*t/h)/((2*sympy.pi/h)*sympy.tan(t/2))
  if diff == 0:
    sinc_diff = sinc
  else:  
    sinc_diff = sinc.diff(*(t,)*diff)

  func = sympy.lambdify((t,h),sinc_diff,'numpy')
  times = dt*np.arange(N)
  val = func(times,dt)
  if diff == 0:
    val[0] = 1.0
  elif diff == 1:
    val[0] = 0.0
  elif diff == 2:
    val[0] = -(np.pi**2/(3*dt**2)) - 1.0/6.0        

  D = scipy.linalg.circulant(val)/scale**diff
  return D

# XXXXXXXXXXXX
R = 100
w = np.random.normal(0.0,2.0,R)
t = np.linspace(0.0,100.0,R)
dt = t[1] - t[0]
b = np.cumsum(w)*dt
D1 = spectral_diff_matrix(R,dt,1)
print(np.linalg.inv(D1))
print(D1)
D2 = rbf.fd.diff_matrix_1d(t[:,None],(1,)).toarray()
bdiff = D2.dot(b)
print(np.std(bdiff))
plt.plot(t,bdiff)
plt.show()
# XXXXXXXXXXXX

N = 2
P = 100
cutoff = 2.0
signal_freq = 1.0
min_time = 0.0
max_time = 5.0
var = 1.0*np.ones(P)
var[20:40] = 10.0
var_bar = 1.0/np.mean(1.0/var)
lamb_square = (2*np.pi*cutoff)**(2*N)*var_bar

time = np.linspace(min_time,max_time,P)


signal = 1*np.sin(2*np.pi*time*signal_freq)
data = signal + 1*np.random.normal(0.0,np.sqrt(var))

# compute differentiation matrix
D = spectral_diff_matrix(P,time[1]-time[0],N)

# perform inversion
Cobs_inv = np.diag(1.0/var)
Cprior_inv = 1.0/lamb_square*D.T.dot(D)
Cpost = np.linalg.inv(Cobs_inv + Cprior_inv)
upost = Cpost.dot(Cobs_inv).dot(data)
stdpost = np.sqrt(np.diag(Cpost))

# plot one of the data and filter realizations 
fig1,ax1 = plt.subplots(figsize=(6,5))
ax1.set_xlabel(r'time [yr]')
ax1.set_ylabel(r'displacement [mm]')
ax1.errorbar(time,data,np.sqrt(var),fmt='k.',label=r'$u_\mathrm{obs}$',capsize=0,zorder=2)
ax1.plot(time,upost,'b-',lw=2,label=r'$u_\mathrm{post}$',zorder=3)
ax1.plot(time,signal,'r-',lw=2,label=r'$u_\mathrm{true}$',zorder=1)
ax1.set_ylim((-13,13))
ax1.grid()
ax1.legend(frameon=False)
ax1.fill_between(time,upost+stdpost,upost-stdpost,color='b',alpha=0.3,edgecolor='none')
fig1.tight_layout()

# plot freqency content
def true_filter(freq):
  return 1.0/(1.0 + (freq/cutoff)**(2*N))

fig2,ax2 = plt.subplots(figsize=(6,5))
ax2.set_xlabel(r'frequency [1/yr]')
ax2.set_ylabel(r'power spectral density [mm**2 yr]')
# plot frequency content of observation
freq,pow = psd(data,time)
ax2.loglog(freq,pow,'k',lw=2,label=r'$u_\mathrm{obs}$',zorder=2)
# plot frequency content of posterior
freq,pow = psd(upost,time)
ax2.loglog(freq,pow,'b',lw=2,label=r'$u_\mathrm{post}$',zorder=3)
ax2.loglog(freq,true_filter(freq)**2,'k--',lw=2)
ax2.set_xlim((10**(-0.8),10**(1.1)))
ax2.set_ylim((10**(-9.0),10**(3.5)))
ax2.vlines(signal_freq,10**-9,10**3.5,color='r',label=r'$u_\mathrm{true}$',lw=2,zorder=1)
ax2.legend(frameon=False)
ax2.grid()
fig2.tight_layout()
plt.show()
quit()
