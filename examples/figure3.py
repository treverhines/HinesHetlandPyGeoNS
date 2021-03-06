''' 
In this script I demonstrate smoothing synthetic data for a creeping fault

ax1: synthetic and smoothed data
ax2: min eigenvector
ax3: strain
ax4: true strain?

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
from slippy.okada import dislocation
from myplot.quiver import Quiver
np.random.seed(5)

def significant_eigenvectors(G,target=0.5):
  ''' 
  returns eigenvectors with a eigenvalue greater than 0.5
  '''
  val,vec = np.linalg.eig(G)
  idx = np.argsort(val)[::-1]
  val = val[idx]
  vec = vec[:,idx]
  idx = np.argmin(np.abs(val - target))
  vec = vec[:,idx]
  val = val[idx]
  return vec,val

fig,axs = plt.subplots(1,2,figsize=(7,3.5),
                       gridspec_kw={'bottom':0.075,
                                    'left':0.075,
                                    'top':0.975,
                                    'right':0.975,
                                    'wspace':0.2,
                                    'hspace':0.2})


axs[0].set_aspect('equal')
axs[0].tick_params(labelsize=10)
axs[0].minorticks_on()
axs[0].set_xlabel(r'$\mathregular{x_1 \cdot \omega_c}$',labelpad=-1,fontsize=10)
axs[0].set_ylabel(r'$\mathregular{x_2 \cdot \omega_c}$',labelpad=-4,fontsize=10)
#axs[0].get_xaxis().set_visible(False)
#axs[0].get_yaxis().set_visible(False)
axs[1].set_aspect('equal')
axs[1].tick_params(labelsize=10)
axs[1].minorticks_on()
axs[1].set_xlabel(r'$\mathregular{x_1 \cdot \omega_c}$',labelpad=-1,fontsize=10)
axs[1].set_ylabel(r'$\mathregular{x_2 \cdot \omega_c}$',labelpad=-4,fontsize=10)
#axs[1].get_xaxis().set_visible(False)
#axs[1].get_yaxis().set_visible(False)
axs[0].text(0.05,0.05,'A',transform=axs[0].transAxes,
            fontsize=12, fontweight='bold', va='bottom')
axs[1].text(0.05,0.05,'B',transform=axs[1].transAxes,
            fontsize=12, fontweight='bold', va='bottom')
                
# Generate synthetic data
N = 80 # number of stations
w = 0.2 # cutoff frequency
D = 1.0 # locking depth
L = 6.0 # creeping segment length

# initiate halton sequence
H = rbf.halton.Halton(2)
x = (H(N) - 0.5)*10
coeff = 1.0/(2*np.sqrt(2))
# define fault geometry
vert = np.array([[-coeff*L,-coeff*L],
                 [coeff*L,coeff*L]])
smp = np.array([[0,1]])                 
x_ext = np.concatenate((x,np.zeros((N,1))),axis=1)
disp_cos,derr = dislocation(x_ext,[1.0,0.0,0.0],[0.0,0.0,0.0],L,D,45.0,90.0)
disp_int,derr = dislocation(x_ext,[1.0,0.0,0.0],[0.0,0.0,-D],200*L,100*D,45.0,90.0)
disp = disp_int + disp_cos
u_true,v_true = disp[:,0],disp[:,1]
s = 0.1*np.ones(N)
u = u_true + np.random.normal(0.0,s)
v = v_true + np.random.normal(0.0,s)

# smooth the data
D = rbf.fd.weight_matrix(x,x,diffs=[[2,0],[0,2]],vert=vert,smp=smp,n=30).toarray()
C = np.diag(1.0/s**2) # inverse covariance
b = np.sqrt(N/np.sum(1.0/s**2)) # mean uncertainty
p = np.sqrt((2*np.pi*w)**4*b**2)
G = inv(C + 1.0/p**2*D.T.dot(D)).dot(C)
M = inv(C + 1.0/p**2*D.T.dot(D))
s_smooth = np.sqrt(np.diag(M))
u_smooth = G.dot(u)
v_smooth = G.dot(v)

# plot the results
q = Quiver(axs[0],x[:,0]*w,x[:,1]*w,u,v,sigma=(s,s,0.0*s),
           scale=2.0,width=0.005,color=(0.2,0.2,1.0),zorder=1,
           ellipse_kwargs={'edgecolors':(0.2,0.2,1.0)})
axs[0].add_collection(q)
q = Quiver(axs[0],x[:,0]*w,x[:,1]*w,u_true,v_true,scale_units='xy',angles='xy',
           scale=2.0,width=0.005,color='k',zorder=0)
axs[0].add_collection(q)

q = Quiver(axs[1],x[:,0]*w,x[:,1]*w,u_smooth,v_smooth,sigma=(s_smooth,s_smooth,0.0*s),
           scale=2.0,width=0.005,color=(0.2,0.2,1.0),zorder=1,
           ellipse_kwargs={'edgecolors':(0.2,0.2,1.0)})
axs[1].add_collection(q)
q = Quiver(axs[1],x[:,0]*w,x[:,1]*w,u_true,v_true,scale_units='xy',angles='xy',
           scale=2.0,width=0.005,color='k',zorder=0)
axs[1].add_collection(q)


axs[0].set_xlim((-1.2,1.2))
axs[1].set_xlim((-1.2,1.2))
axs[0].set_ylim((-1.2,1.2))
axs[1].set_ylim((-1.2,1.2))
for s in smp: axs[0].plot(vert[s,0]*w,vert[s,1]*w,'r--',lw=1,zorder=0)
for s in smp: axs[1].plot(vert[s,0]*w,vert[s,1]*w,'r--',lw=1,zorder=0)

#plt.tight_layout()
plt.show()

