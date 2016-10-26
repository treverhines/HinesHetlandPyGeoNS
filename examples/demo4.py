#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import rbf.fd
import rbf.nodes
import rbf.domain
import scipy.sparse
import matplotlib.cm as cm
np.random.seed(4)

# setup axes
fig = plt.figure(figsize=(10,8))
gs = gridspec.GridSpec(2,2)
gs.update(left=0.1,right=0.95,top=0.95,bottom=0.1,hspace=0.2,wspace=0.2)
ax1 = plt.subplot(gs[0,0])
ax2 = plt.subplot(gs[0,1])
ax3 = plt.subplot(gs[1,0])
ax4 = plt.subplot(gs[1,1])
ax1.set_aspect('equal')
ax2.set_aspect('equal')
ax3.set_aspect('equal')
ax4.set_aspect('equal')
ax1.set_xlim((-11,11))
ax2.set_xlim((-11,11))
ax3.set_xlim((-11,11))
ax4.set_xlim((-11,11))
ax1.set_ylim((-11,11))
ax2.set_ylim((-11,11))
ax3.set_ylim((-11,11))
ax4.set_ylim((-11,11))

# make nodes
P = 500
diff = 2
vert,smp = rbf.domain.circle(5)
vert *= 10.0

fix_nodes = np.array([[0.0,0.0],
                      [-10.0,0.0]])
nodes,sid = rbf.nodes.make_nodes(P-2,vert,smp,itr=1000,delta=0.01,neighbors=5,
                                 fix_nodes=fix_nodes)
nodes = np.vstack((fix_nodes,nodes))


# make synthetic data
u = np.random.normal(0.0,1.0,P)

D = rbf.fd.diff_matrix(nodes,[(diff,0),(0,diff)])
I = scipy.sparse.eye(P)
# smooth
sigma = 1.0
cutoff = 1.0/10.0
penalty = (2*np.pi*cutoff)**(2*diff)*sigma**2

us = scipy.sparse.linalg.spsolve(I + (1.0/penalty)*D.T.dot(D),u)
cov = np.linalg.inv((I + (1.0/penalty)*D.T.dot(D)).toarray())
sigma = np.sqrt(np.diag(cov))
corr = (cov/sigma[None,:])/sigma[:,None]

p = ax1.scatter(nodes[:,0],nodes[:,1],s=40,c=u,edgecolor='k',vmin=-2.0,vmax=2.0,zorder=1)
plt.colorbar(p,ax=ax1)
for s in smp:
  ax1.plot(vert[s,0],vert[s,1],'k-',zorder=0)

p = ax2.scatter(nodes[:,0],nodes[:,1],s=40,c=us,edgecolor='k',zorder=2)
p = ax2.tripcolor(nodes[:,0],nodes[:,1],us,shading='gouraud',
                  vmin=p.get_clim()[0],vmax=p.get_clim()[1],zorder=0)
plt.colorbar(p,ax=ax2)
for s in smp:
  ax2.plot(vert[s,0],vert[s,1],'k-',zorder=1)

# plot corr
p = ax3.scatter(nodes[:,0],nodes[:,1],s=40,c=sigma,edgecolor='k',zorder=2)
p = ax3.tripcolor(nodes[:,0],nodes[:,1],sigma,shading='gouraud',
                  vmin=p.get_clim()[0],vmax=p.get_clim()[1],zorder=0)
cbar = plt.colorbar(p,ax=ax3)
for s in smp:
  ax3.plot(vert[s,0],vert[s,1],'k-',zorder=1)

# plot corr
p = ax4.scatter(nodes[:,0],nodes[:,1],s=40,c=corr[0,:],cmap='seismic',edgecolor='k',vmin=-1.0,vmax=1.0,zorder=2)
p = ax4.tripcolor(nodes[:,0],nodes[:,1],corr[0,:],cmap='seismic',shading='gouraud',
                  vmin=-1.0,vmax=1.0,zorder=0)
cbar = plt.colorbar(p,ax=ax4)
for s in smp:
  ax4.plot(vert[s,0],vert[s,1],'k-',zorder=1)


ax1.grid()
ax2.grid()
ax3.grid()
ax4.grid()
plt.show()
print(len(vert))