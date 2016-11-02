''' 
In this script I illustrate the frequency content of the filtered 
solution when idealized conditions are not met
'''
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import rbf.filter
import rbf.fd
import rbf.halton
from scipy.signal import periodogram
import sympy
from scipy.linalg import circulant
from numpy.linalg import inv
np.random.seed(2)

def significant_eigenvectors(G):
  ''' 
  returns eigenvectors with a eigenvalue greater than 0.5
  '''
  val,vec = np.linalg.eig(G)
  idx = np.argsort(val)[::-1]
  val = val[idx]
  vec = vec[:,idx]
  idx = val > 0.5
  vec = vec[:,idx]
  return vec,val

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

  D = circulant(val)/scale**diff
  return D

fig,axs = plt.subplots(4,3,figsize=(15,10),sharex="col",
                       gridspec_kw={'bottom':0.05,
                                    'left':0.05,
                                    'top':0.95,
                                    'right':0.95,
                                    'wspace':0.15,
                                    'hspace':0.10})
N = 100
w = 5.0 # cutoff frequency
#####################################################################
# idealized conditions:
#   periodic spectral differentiation matrix
#   constant uncertainty
#   constant spacing
x = np.linspace(0.0,1.0,N)
s = 0.1*np.ones(N) # uncertainties
t = np.sqrt(x) + 0.2*np.sin(4*np.pi*x) 
u = t + np.random.normal(0.0,s)
D = spectral_diff_matrix(N,x[1]-x[0],2)
C = np.diag(1.0/s**2) # inverse covariance
# generalized inverse
b = np.sqrt(N/np.sum(1.0/s**2)) # mean uncertainty
p = np.sqrt((2*np.pi*w)**4*b**2)
G = inv(C + 1.0/p**2*D.T.dot(D)).dot(C)
z = G.dot(u)
y = np.sqrt(np.diag(inv(C + 1.0/p**2*D.T.dot(D))))
# get eigenvalues and vectors
vec,val = significant_eigenvectors(G)

ax = axs[0][0]
ax.minorticks_on()
ax.errorbar(x,u,s,fmt='k.',capsize=0,label='observed')
ax.plot(x,t,'r-',label='true signal')
ax.plot(x,z,'b-',label='filtered')
ax.fill_between(x,z-y,z+y,color='b',alpha=0.4,edgecolor='none')
#ax.set_xlabel('x',fontsize=12)
ax.legend(loc=4,frameon=False,fontsize=12)
ax.grid()

ax = axs[0][1]
ax.minorticks_on()
ax.plot(x,vec,'-',color=(0.8,0.8,0.8))
ax.plot(x,vec[:,-1],'b-',lw=2)

ax = axs[0][2]
ax.loglog(np.arange(1,N+1),val,'ko',mec='none',markersize=5)
ax.loglog(np.arange(1,N+1)[val>0.5][-1],val[val>0.5][-1],'bo',mec='none',markersize=10)
ax.set_ylim(3e-5,3e0)
ax.set_xlim(5e-1,1.7e2)
ax.grid()
#ax.set_xlabel('x',fontsize=12)

#####################################################################
# conditions:
#   constant uncertainty
#   constant spacing
x = np.linspace(0.0,1.0,N)
s = 0.1*np.ones(N) # uncertainties
t = np.sqrt(x) + 0.2*np.sin(4*np.pi*x) 
u = t + np.random.normal(0.0,s)
D = rbf.fd.weight_matrix(x[:,None],x[:,None],diffs=(2,)).toarray()
C = np.diag(1.0/s**2) # inverse covariance
# generalized inverse
b = np.sqrt(N/np.sum(1.0/s**2)) # mean uncertainty
p = np.sqrt((2*np.pi*w)**4*b**2)
G = inv(C + 1.0/p**2*D.T.dot(D)).dot(C)
z = G.dot(u)
y = np.sqrt(np.diag(inv(C + 1.0/p**2*D.T.dot(D))))
# get eigenvalues and vectors
vec,val = significant_eigenvectors(G)

ax = axs[1][0]
ax.minorticks_on()
ax.errorbar(x,u,s,fmt='k.',capsize=0,label='observed')
ax.plot(x,t,'r-',label='true signal')
ax.plot(x,z,'b-',label='filtered')
ax.fill_between(x,z-y,z+y,color='b',alpha=0.4,edgecolor='none')
#ax.set_xlabel('x',fontsize=12)
ax.legend(loc=4,frameon=False,fontsize=12)
ax.grid()

ax = axs[1][1]
ax.minorticks_on()
ax.plot(x,vec,'-',color=(0.8,0.8,0.8))
ax.plot(x,vec[:,-1],'b-',lw=2)
#ax.set_xlabel('x',fontsize=12)

ax = axs[1][2]
ax.loglog(np.arange(1,N+1),val,'ko',mec='none',markersize=5)
ax.loglog(np.arange(1,N+1)[val>0.5][-1],val[val>0.5][-1],'bo',mec='none',markersize=10)
ax.set_ylim(3e-5,3e0)
ax.set_xlim(5e-1,1.7e2)
ax.grid()

#####################################################################
# conditions:
#   constant uncertainty
#   constant spacing
x = np.linspace(0.0,1.0,N)
s = 0.1*np.ones(N) # uncertainties
s[(x>0.2) & (x < 0.5)] = 0.3
t = np.sqrt(x) + 0.2*np.sin(4*np.pi*x) 
u = t + np.random.normal(0.0,s)
D = rbf.fd.weight_matrix(x[:,None],x[:,None],diffs=(2,)).toarray()
C = np.diag(1.0/s**2) # inverse covariance
# generalized inverse
b = np.sqrt(N/np.sum(1.0/s**2)) # mean uncertainty
print('characteristic sigma %s' % b)
p = np.sqrt((2*np.pi*w)**4*b**2)
G = inv(C + 1.0/p**2*D.T.dot(D)).dot(C)
z = G.dot(u)
y = np.sqrt(np.diag(inv(C + 1.0/p**2*D.T.dot(D))))
# get eigenvalues and vectors
vec,val = significant_eigenvectors(G)

ax = axs[2][0]
ax.minorticks_on()
ax.errorbar(x,u,s,fmt='k.',capsize=0,label='observed')
ax.plot(x,t,'r-',label='true signal')
ax.plot(x,z,'b-',label='filtered')
ax.fill_between(x,z-y,z+y,color='b',alpha=0.4,edgecolor='none')
#ax.set_xlabel('x',fontsize=12)
ax.legend(loc=4,frameon=False,fontsize=12)
ax.grid()

ax = axs[2][1]
ax.minorticks_on()
ax.plot(x,vec,'-',color=(0.8,0.8,0.8))
ax.plot(x,vec[:,-1],'b-',lw=2)
#ax.set_xlabel('x',fontsize=12)

ax = axs[2][2]
ax.loglog(np.arange(1,N+1),val,'ko',mec='none',markersize=5)
ax.loglog(np.arange(1,N+1)[val>0.5][-1],val[val>0.5][-1],'bo',mec='none',markersize=10)
ax.set_ylim(3e-5,3e0)
ax.set_xlim(5e-1,1.7e2)
ax.grid()

#####################################################################
# conditions:
#   constant uncertainty
#   constant spacing
x = np.linspace(0.05,1.0,N)**2
s = 0.1*np.ones(N) # uncertainties
t = np.sqrt(x) + 0.2*np.sin(4*np.pi*x) 
u = t + np.random.normal(0.0,s)
D = rbf.fd.weight_matrix(x[:,None],x[:,None],diffs=(2,)).toarray()
C = np.diag(1.0/s**2) # inverse covariance
# generalized inverse
b = np.sqrt(N/np.sum(1.0/s**2)) # mean uncertainty
print('characteristic sigma %s' % b)
p = np.sqrt((2*np.pi*w)**4*b**2)
G = inv(C + 1.0/p**2*D.T.dot(D)).dot(C)
z = G.dot(u)
y = np.sqrt(np.diag(inv(C + 1.0/p**2*D.T.dot(D))))
# get eigenvalues and vectors
vec,val = significant_eigenvectors(G)

ax = axs[3][0]
ax.minorticks_on()
ax.errorbar(x,u,s,fmt='k.',capsize=0,label='observed')
ax.plot(x,t,'r-',label='true signal')
ax.plot(x,z,'b-',label='filtered')
ax.fill_between(x,z-y,z+y,color='b',alpha=0.4,edgecolor='none')
ax.set_xlabel('x',fontsize=12)
ax.legend(loc=4,frameon=False,fontsize=12)
ax.grid()

ax = axs[3][1]
ax.minorticks_on()
ax.plot(x,vec,'-',color=(0.8,0.8,0.8))
ax.plot(x,vec[:,-1],'b-',lw=2)
ax.set_xlabel('x',fontsize=12)

ax = axs[3][2]
ax.loglog(np.arange(1,N+1),val,'ko',mec='none',markersize=5)
ax.loglog(np.arange(1,N+1)[val>0.5][-1],val[val>0.5][-1],'bo',mec='none',markersize=10)
ax.set_ylim(3e-5,3e0)
ax.set_xlim(5e-1,1.7e2)
ax.grid()
ax.set_xlabel('eigenvalue number',fontsize=12)


plt.show()
