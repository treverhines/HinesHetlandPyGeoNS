#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import rbf.fd
import rbf.basis
import rbf.stencil
from rbf.interpolate import RBFInterpolant
import sympy
import scipy.linalg
import scipy.signal
from scipy.interpolate import interp1d
from modules.ndft import psd
import matplotlib.gridspec as gridspec
np.random.seed(6)

def make_time(low,high,N1,N2):
  mid = (low + high)/2.0
  l1 = np.linspace(low,mid,N1)[:-1]
  l2 = np.linspace(mid,high,N2)
  return np.concatenate((l1,l2))

N = 2
P1 = 100
P2 = 400
cutoff = 2.0
signal_freq = 1.0
min_time = 0.0
max_time = 10.0

time = make_time(min_time,max_time,P1,P2)
P = len(time)

var = 1.0*np.ones(P)

var_bar = 1.0/np.mean(1.0/var)
lamb_square = (2*np.pi*cutoff)**(2*N)*var_bar

  
signal = 1*np.sin(2*np.pi*time*signal_freq)
data = signal + np.random.normal(0.0,np.sqrt(var))

# compute differentiation matrix
D = rbf.fd.diff_matrix_1d(time[:,None],(N,)).toarray()

# perform inversion
Cobs_inv = np.diag(1.0/var)
Cprior_inv = 1.0/lamb_square*D.T.dot(D)
Cpost = np.linalg.inv(Cobs_inv + Cprior_inv)
upost = Cpost.dot(Cobs_inv).dot(data)
#upost = RBFInterpolant(time[:,None],data,basis=rbf.basis.phs2,order=1,penalty=0.1)(time[:,None])

stdpost = np.sqrt(np.diag(Cpost))


# plot one of the data and filter realizations 
fig = plt.figure(figsize=(10,8))
gs = gridspec.GridSpec(2,2)
gs.update(left=0.1,right=0.95,top=0.95,bottom=0.1,hspace=0.25, wspace=0.25)
ax1 = plt.subplot(gs[0,:])
ax2 = plt.subplot(gs[1,0])
ax3 = plt.subplot(gs[1,1])

ax1.set_xlabel(r'time [yr]')
ax1.set_ylabel(r'displacement [mm]')
#ax1.errorbar(time,data,np.sqrt(var),fmt='k.',label=r'$u_\mathrm{obs}$',markersize=1,capsize=0,zorder=2)
ax1.plot(time,data,'k.',label=r'$u_\mathrm{obs}$',zorder=0)
ax1.plot(time,upost,'b-',lw=1,label=r'$u_\mathrm{post}$',zorder=3)
ax1.plot(time,signal,'r-',lw=1,label=r'$u_\mathrm{true}$',zorder=1)
ax1.set_ylim((-5,8))
ax1.grid()
ax1.legend(frameon=False)
ax1.fill_between(time,upost+stdpost,upost-stdpost,color='b',alpha=0.3,edgecolor='none')
ax1.text(0.025,0.9,'A',transform=ax1.transAxes,fontsize=16)

# plot freqency content
def true_filter(freq):
  return 1.0/(1.0 + (freq/cutoff)**(2*N))

ax2.set_xlabel(r'frequency [1/yr]')
ax2.set_ylabel(r'power spectral density [mm**2 yr]')
# plot frequency content of observation
freq,pow = psd(data[:P1],time[:P1])
ax2.loglog(freq,pow,'k-',lw=1,label=r'$u_\mathrm{obs}$',zorder=2)
freq,pow = psd(upost[:P1],time[:P1])
ax2.loglog(freq,pow,'b-',lw=1,label=r'$u_\mathrm{post}$',zorder=3)
ax2.loglog(freq,true_filter(freq)**2,'k--',lw=1)
ax2.set_xlim((10**(-0.8),10**(1.1)))
ax2.set_ylim((10**(-6.5),10**(3.0)))
ax2.vlines(signal_freq,10**-6.5,10**3.0,color='r',label=r'$u_\mathrm{true}$',lw=1,zorder=1)
ax2.legend(frameon=False)
ax2.text(0.05,0.9,'B',transform=ax2.transAxes,fontsize=16)
ax2.grid()

ax3.set_xlabel(r'frequency [1/yr]')
ax3.set_ylabel(r'power spectral density [mm**2 yr]')
# plot frequency content of observation
freq,pow = psd(data[P1:],time[P1:])
ax3.loglog(freq,pow,'k-',lw=1,label=r'$u_\mathrm{obs}$',zorder=2)
freq,pow = psd(upost[P1:],time[P1:])
ax3.loglog(freq,pow,'b-',lw=1,label=r'$u_\mathrm{post}$',zorder=3)
ax3.loglog(freq,true_filter(freq)**2,'k--',lw=1)
ax3.set_xlim((10**(-0.8),10**(1.1)))
ax3.set_ylim((10**(-6.5),10**(2.0)))
ax3.vlines(signal_freq,10**-6.5,10**2,color='r',label=r'$u_\mathrm{true}$',lw=1,zorder=1)
ax3.legend(frameon=False)
ax3.text(0.05,0.9,'C',transform=ax3.transAxes,fontsize=16)
ax3.grid()
plt.show()

