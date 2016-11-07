''' 
this script simply plots the RBF-FD filter frequency response
'''
import numpy as np
import matplotlib.pyplot as plt

def freq_response(f,n):
  return 1.0/(1 + f**(2*n))
  

freq = 10**np.linspace(-2,2,1000)  
fig,ax = plt.subplots(figsize=(4,3.2))
ax.minorticks_on()
ax.grid()
ax.set_ylim(1e-5,1e1)
ax.set_xlabel(r'$\mathregular{\omega / \omega_c}$',fontsize=10,labelpad=-1)

u = freq_response(freq,1)
ax.loglog(freq,u,'k-')
u = freq_response(freq,2)
ax.loglog(freq,u,'k-')
u = freq_response(freq,4)
ax.loglog(freq,u,'k-')

ax.tick_params(labelsize=10)
ax.text(2e-2,1e-4,r'$\mathregular{\frac{1}{1 + \left(\frac{\omega}{\omega_c}\right)^{2n}}}$',fontsize=16)
ax.text(4.0e0,3.0e-5,r'$\mathregular{n=4}$',fontsize=10)
ax.text(7.0e0,5.0e-4,r'$\mathregular{n=2}$',fontsize=10)
ax.text(1.1e1,1.1e-2,r'$\mathregular{n=1}$',fontsize=10)

plt.tight_layout()
plt.show()
