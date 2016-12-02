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
from scipy.spatial import Delaunay
import sympy
from scipy.linalg import circulant
from numpy.linalg import inv
from slippy.okada import dislocation
from myplot.quiver import Quiver
from pygeons.strain import strain_glyph
np.random.seed(5)

# Generate synthetic data
N = 20 # number of stations
d = 2.0 # locking depth

# initiate halton sequence
H = rbf.halton.Halton(2,start=1)
x = (H(N) - 0.5)*(5*d)
coeff = 1.0/(2*np.sqrt(2))

# define fault geometry
x_ext = np.concatenate((x,np.zeros((N,1))),axis=1)
disp,derr = dislocation(x_ext,[1.0,0.0,0.0],[0.0,0.0,-d],200*d,100*d,45.0,90.0)
u,v = disp[:,0],disp[:,1]

# compute strain with delaunay triangulation
deltri = Delaunay(x)
stencil = deltri.simplices
centers = np.mean(x[stencil],axis=1)

M = len(centers)
Dx1 = np.zeros((M,N))
for i,s in enumerate(stencil):
  Dx1[i,s] = rbf.fd.weights(centers[i],x[s],(1,0),order=1)

Dy1 = np.zeros((M,N))
for i,s in enumerate(stencil):
  Dy1[i,s] = rbf.fd.weights(centers[i],x[s],(0,1),order=1)

centers_ext = np.concatenate((centers,np.zeros((M,1))),axis=1)
disp,derr = dislocation(centers_ext,[1.0,0.0,0.0],[0.0,0.0,-d],200*d,100*d,45.0,90.0)
dudx4 = derr[:,0,0]
dudy4 = derr[:,0,1]
dvdx4 = derr[:,1,0]
dvdy4 = derr[:,1,1]


dudx1 = Dx1.dot(u)
dudy1 = Dy1.dot(u)
dvdx1 = Dx1.dot(v)
dvdy1 = Dy1.dot(v)

Dx2 = rbf.fd.weight_matrix(centers,x,(1,0),order=1,n=5)
Dy2 = rbf.fd.weight_matrix(centers,x,(0,1),order=1,n=5)
dudx2 = Dx2.dot(u)
dudy2 = Dy2.dot(u)
dvdx2 = Dx2.dot(v)
dvdy2 = Dy2.dot(v)

Dx3 = rbf.fd.weight_matrix(centers,x,(1,0),order=1,n=20)
Dy3 = rbf.fd.weight_matrix(centers,x,(0,1),order=1,n=20)
dudx3 = Dx3.dot(u)
dudy3 = Dy3.dot(u)
dvdx3 = Dx3.dot(v)
dvdy3 = Dy3.dot(v)


# plot the results
fig,axs = plt.subplots(2,2,figsize=(6.25,6),
                       gridspec_kw={'bottom':0.075,
                                    'left':0.075,
                                    'top':0.975,
                                    'right':0.975,
                                    'wspace':0.15,
                                    'hspace':0.15})
axs[0][0].set_aspect('equal')
axs[0][0].tick_params(labelsize=10)
axs[0][0].minorticks_on()
axs[0][0].set_xlabel(r'$\mathregular{x_1/h}$',labelpad=-1,fontsize=10)
axs[0][0].set_ylabel(r'$\mathregular{x_2/h}$',labelpad=-4,fontsize=10)
axs[0][1].set_aspect('equal')
axs[0][1].tick_params(labelsize=10)
axs[0][1].minorticks_on()
axs[0][1].set_xlabel(r'$\mathregular{x_1/h}$',labelpad=-1,fontsize=10)
axs[0][1].set_ylabel(r'$\mathregular{x_2/h}$',labelpad=-4,fontsize=10)
axs[0][0].text(0.05,0.05,'A',transform=axs[0][0].transAxes,
            fontsize=12, fontweight='bold', va='bottom')
axs[0][1].text(0.05,0.05,'B',transform=axs[0][1].transAxes,
            fontsize=12, fontweight='bold', va='bottom')                
axs[0][0].text(0.6,0.85,'triangulated\nstrain',transform=axs[0][0].transAxes,
               fontsize=10, va='bottom')
axs[0][1].text(0.6,0.85,'RBF-FD strain\nn = 5',transform=axs[0][1].transAxes,
               fontsize=10, va='bottom')

axs[1][0].set_aspect('equal')
axs[1][0].tick_params(labelsize=10)
axs[1][0].minorticks_on()
axs[1][0].set_xlabel(r'$\mathregular{x_1/h}$',labelpad=-1,fontsize=10)
axs[1][0].set_ylabel(r'$\mathregular{x_2/h}$',labelpad=-4,fontsize=10)
axs[1][1].set_aspect('equal')
axs[1][1].tick_params(labelsize=10)
axs[1][1].minorticks_on()
axs[1][1].set_xlabel(r'$\mathregular{x_1/h}$',labelpad=-1,fontsize=10)
axs[1][1].set_ylabel(r'$\mathregular{x_2/h}$',labelpad=-4,fontsize=10)
axs[1][0].text(0.05,0.05,'C',transform=axs[1][0].transAxes,
            fontsize=12, fontweight='bold', va='bottom')
axs[1][1].text(0.05,0.05,'D',transform=axs[1][1].transAxes,
            fontsize=12, fontweight='bold', va='bottom')                
axs[1][0].text(0.6,0.85,'RBF-FD strain\nn = 20',transform=axs[1][0].transAxes,
               fontsize=10, va='bottom')
axs[1][1].text(0.6,0.9,'true strain',transform=axs[1][1].transAxes,
               fontsize=10, va='bottom')

# plot triangles
for a in deltri.simplices:
  b = np.array([a,np.roll(a,1)]).T
  for c in b: 
    axs[0][0].plot(x[c,0]/d,x[c,1]/d,'-',color=(0.8,0.8,0.8),zorder=0)
  
# plot strain glyphs
for i,c in enumerate(centers):
  artists = strain_glyph(c[0]/d,c[1]/d,[dudx1[i],dudy1[i],dvdx1[i],dvdy1[i]],scale=4.0)
  for a in artists: 
    axs[0][0].add_artist(a) 

# plot strain glyphs
for i,c in enumerate(centers):
  artists = strain_glyph(c[0]/d,c[1]/d,[dudx2[i],dudy2[i],dvdx2[i],dvdy2[i]],scale=4.0)
  for a in artists: 
    axs[0][1].add_artist(a) 

# plot strain glyphs
for i,c in enumerate(centers):
  artists = strain_glyph(c[0]/d,c[1]/d,[dudx3[i],dudy3[i],dvdx3[i],dvdy3[i]],scale=4.0)
  for a in artists: 
    axs[1][0].add_artist(a) 

# plot strain glyphs
for i,c in enumerate(centers):
  artists = strain_glyph(c[0]/d,c[1]/d,[dudx4[i],dudy4[i],dvdx4[i],dvdy4[i]],scale=4.0)
  for a in artists: 
    axs[1][1].add_artist(a) 
  
# plot strain glyphs
#axs[0].plot(centers[:,0]/d,centers[:,1]/d,'ko')

q = Quiver(axs[0][0],x[:,0]/d,x[:,1]/d,u,v,
           color=(0.0,0.0,0.0),zorder=1)
axs[0][0].add_collection(q)
q = Quiver(axs[0][1],x[:,0]/d,x[:,1]/d,u,v,
           color=(0.0,0.0,0.0),zorder=1)
axs[0][1].add_collection(q)
q = Quiver(axs[1][0],x[:,0]/d,x[:,1]/d,u,v,
           color=(0.0,0.0,0.0),zorder=1)
axs[1][0].add_collection(q)
q = Quiver(axs[1][1],x[:,0]/d,x[:,1]/d,u,v,
           color=(0.0,0.0,0.0),zorder=1)
axs[1][1].add_collection(q)

axs[0][0].set_xlim((-2.8,2.8))
axs[0][1].set_xlim((-2.8,2.8))
axs[0][0].set_ylim((-2.8,2.8))
axs[0][1].set_ylim((-2.8,2.8))
axs[1][0].set_xlim((-2.8,2.8))
axs[1][1].set_xlim((-2.8,2.8))
axs[1][0].set_ylim((-2.8,2.8))
axs[1][1].set_ylim((-2.8,2.8))
plt.show()           

