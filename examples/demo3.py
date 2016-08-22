#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import rbf.fd
import rbf.halton
import sympy
import scipy.linalg
import scipy.signal
np.random.seed(6)
H = rbf.halton.Halton(1)


def extend_data(time,data,var):
  R = len(time)
  print(R)
  new_time = max_time*H(R)[:,0]
  new_data = np.zeros(R)
  new_var = np.inf*np.ones(R)
  time = np.concatenate((time,new_time))
  data = np.concatenate((data,new_data))
  var = np.concatenate((var,new_var))
  sort_idx = np.argsort(time)
  time  = time[sort_idx]
  data = data[sort_idx]
  var = var[sort_idx]
  return time,data,var
  
N = 2
P = 10
cutoff = 0.5
signal_freq = 1.0
min_time = 0.0
max_time = 5.0
var = 1.0*np.ones(P)
time = np.sort(max_time*H(P)[:,0])
  
signal = 1*np.sin(2*np.pi*time*signal_freq)
data = signal + np.random.normal(0.0,np.sqrt(var))


for i in range(6):
  data = np.nan_to_num(data)
  # compute differentiation matrix
  
  D = rbf.fd.poly_diff_matrix(time[:,None],(N,)).toarray()
  var_bar = 1.0/np.mean(1.0/var)
  lamb_square = (2*np.pi*cutoff)**(2*N)*var_bar
  Cobs_inv = np.diag(1.0/var)
  Cprior_inv = 1.0/lamb_square*D.T.dot(D)
  Cpost = np.linalg.inv(Cobs_inv + Cprior_inv)
  upost = Cpost.dot(Cobs_inv).dot(data)
  stdpost = np.sqrt(np.diag(Cpost))

  data[np.isinf(var)] = np.nan
  # plot one of the data and filter realizations 
  fig1,ax1 = plt.subplots(figsize=(6,5))
  ax1.set_xlabel(r'time [yr]')
  ax1.set_ylabel(r'displacement [mm]')
  ax1.errorbar(time,data,np.sqrt(var),fmt='k.',label=r'$u_\mathrm{obs}$',capsize=0,zorder=2)
  ax1.plot(time,upost,'b-',lw=2,label=r'$u_\mathrm{post}$',zorder=3)
  #ax1.plot(time,signal,'r-',lw=2,label=r'$u_\mathrm{true}$',zorder=1)
  ax1.set_ylim((-5,7))
  ax1.grid()
  ax1.legend(frameon=False)
  ax1.fill_between(time,upost+stdpost,upost-stdpost,color='b',alpha=0.3,edgecolor='none')
  fig1.tight_layout()

  time,data,var = extend_data(time,data,var)


plt.show()

