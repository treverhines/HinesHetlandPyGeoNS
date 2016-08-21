#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import rbf.fd
np.random.seed(1)

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
    
def freq_response(f,order):
  return 1.0/(1.0 + (f/cutoff)**(2*order))

# number of observations
N = 100
var = 10.0
cutoff = 1.0
print(1.0/cutoff)
time = np.linspace(0.0,10.0,N)
dt = time[1] - time[0]
freq = np.fft.fftfreq(N,dt)
data = np.random.normal(0.0,np.sqrt(var),(10000,N))
filter = freq_response(freq,2)
coeff = [np.fft.fft(d) for d in data]
coeff = [c*filter for c in coeff]
data = [np.fft.ifft(c) for c in coeff]
cov = np.cov(np.array(data).real.T)
plt.figure(1)
plt.imshow(cov)
plt.colorbar()
plt.figure(2)
freq,pow = psd(data,time)
#coeff = np.fft.fft(data[0,:])
#pow = coeff.conj()*coeff/N

cov = np.fft.ifft(filter**2*var)
cov = cov/cov[0]
#cov2 = np.fft.ifft(pow)

plt.plot(time,cov)
#plt.plot(time,cov2)
plt.show()
quit()
# locations of observations
x = np.linspace(-10.0,10.0,P)
freq = np.fft.fftfreq(P,x[1]-x[0])

# power spectral density
pow = var*freq_response(freq,N)**2
autocov = np.roll(np.fft.ifft(pow)*P,P//2)
plt.figure(1)
plt.loglog(freq,pow,'k-')
plt.figure(2)
plt.plot(x,autocov,'k-')
#plt.plot(x,np.sinc(x),'k-')
plt.show()
quit()
# observation variance
#var = np.random.uniform(0.1,10.0,P)
var = np.ones(P)

u_obs = np.random.normal(0.0,np.sqrt(var),(1000,P))

# covariance matrix
Cd = np.diag(var)
Cdinv = np.linalg.inv(Cd)

# finite difference matrix
N = 1
D = rbf.fd.poly_diff_matrix(x[:,None],(N,)).toarray()

# determine penalty parameter
cutoff = 5.0
var_mean = 1.0/np.mean(1.0/var)
lambda_squared = (2*np.pi*cutoff)**(2*N)*var_mean

# posterior covariance
#Cpost = np.linalg.inv(Cdinv + (1.0/lambda_squared)*D.T.dot(D))
#upost = Cpost.dot(Cdinv).dot(u_obs.T).T


pred_autocov = np.fft.ifft(pred_psd)
plt.plot(x,np.roll(pred_autocov,100))
#plt.loglog(freq,pow)
#plt.loglog(freq,var*freq_response(freq)**2)
plt.show()

quit()
std_post = np.sqrt(np.diag(Cpost))
corr = (Cpost/std_post[:,None])/std_post[None,:]

for i in range(0,P,50):
  plt.plot(x,corr[i,:],'k-')
  ideal_corr = np.sinc(2*cutoff*(x-x[i]))
  #ideal_corr = np.exp(-np.abs(x-x[i])*cutoff*2*np.pi)
  plt.plot(x,ideal_corr,'b-')

plt.show()  
