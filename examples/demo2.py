#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import rbf.fd
import sympy
import scipy.linalg
import scipy.signal
np.random.seed(6)


N = 2
P = 100
cutoff = 2.0
signal_freq = 1.0
min_time = 0.0
max_time = 5.0
var = 1.0*np.ones(P)
#var[20:40] = 10.0
var_bar = 1.0/np.mean(1.0/var)
lamb_square = (2*np.pi*cutoff)**(2*N)*var_bar
time = np.sort(np.random.uniform(min_time,max_time,P))
  
signal = 1*np.sin(2*np.pi*time*signal_freq)
data = signal + np.random.normal(0.0,np.sqrt(var))

# compute differentiation matrix
D = rbf.fd.poly_diff_matrix(time[:,None],(N,)).toarray()
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
ax1.set_ylim((-5,7))
ax1.grid()
ax1.legend(frameon=False)
ax1.fill_between(time,upost+stdpost,upost-stdpost,color='b',alpha=0.3,edgecolor='none')
fig1.tight_layout()

plt.show()

