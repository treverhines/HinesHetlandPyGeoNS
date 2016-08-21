#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import rbf.fd
import sympy
import scipy.linalg
np.random.seed(4)

def psd(signals,times):
  signals = np.asarray(signals)
  times = np.asarray(times)
  Nt = times.shape[0]
  dt = times[1] - times[0]
  freq = np.fft.fftfreq(Nt,dt)
  # normalize the coefficients by 1/sqrt(Nt)
  coeff = np.array([np.fft.fft(i) for i in signals])
  # get the complex modulus
  pow = coeff*coeff.conj()/Nt
  # get the expected value
  pow = np.mean(pow,axis=0)
  return freq,pow
                        

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


N = 2
P = 100
cutoff = 1.0
signal_freq = 0.5*cutoff
min_time = 0.0
max_time = 10/cutoff
var = 1.0

time = np.linspace(min_time,max_time,P)
var = var*np.ones(P)
var[20:40] = 10.0
var_bar = 1.0/np.mean(1.0/var)
std = np.sqrt(var)
signal = np.sin(2*np.pi*time*signal_freq)
data = signal[None,:] + np.random.normal(0.0,std[None,:].repeat(10000,axis=0)) 

D = spectral_diff_matrix(P,time[1]-time[0],N)
#D = rbf.fd.poly_diff_matrix(time[:,None],(N,)).toarray()
Cobs_inv = np.diag(1.0/var)
lamb_square = (2*np.pi*cutoff)**(2*N)*var_bar

Cpost = np.linalg.inv(Cobs_inv + (1.0/lamb_square)*D.T.dot(D))
upost = Cpost.dot(Cobs_inv).dot(data.T).T
stdpost = np.sqrt(np.diag(Cpost))

# plot one of the data and filter realizations 
fig,ax = plt.subplots(figsize=(6,5))
ax.set_xlabel(r'time [ $1/\omega_c$ ]')
ax.plot(time*cutoff,signal,'r-')
ax.plot(time*cutoff,upost[0],'b-',label=r'$u_\mathrm{post}$')
ax.errorbar(time*cutoff,data[0],std,fmt='k.',label=r'$u_\mathrm{obs}$',capsize=0)
ax.set_ylim((-13,13))
ax.grid()
ax.legend(frameon=False)
ax.fill_between(time*cutoff,upost[0]+stdpost,upost[0]-stdpost,color='b',alpha=0.2,edgecolor='none')
fig.tight_layout()

# plot freqency content
def true_filter(freq):
  return 1.0/(1.0 + (freq/cutoff)**(2*N))

fig,ax = plt.subplots(figsize=(6,5))
ax.set_xlabel(r'frequency [ $\omega_c$ ]')
freq,pow = psd(upost,time)
ax.loglog(freq/cutoff,pow,'b',label=r'$\mathbf{E}\left[|\hat{u}_\mathrm{post}|^2\right]$')
ax.loglog(freq/cutoff,true_filter(freq)**2,'k',label=r'$\left|\frac{1}{1 + \left(\frac{\omega}{\omega_c}\right)^4}\right|^2$')
ax.set_xlim(10**(-1.1),10**(0.75))
ax.legend(frameon=False)
ax.grid()
fig.tight_layout()
plt.show()



