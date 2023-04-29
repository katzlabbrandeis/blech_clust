from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
from ssqueezepy import cwt

import numpy as np
from ssqueezepy import cwt
from ssqueezepy.visuals import plot, imshow
import pywt

t = np.linspace(-1, 1, 200, endpoint=False)
a = np.cos(2 * np.pi * 3 * t)
b = signal.gausspulse(t - 0.4, fc=10)*5
c = signal.gausspulse(t + 0.4, fc=7)*5
sig  = a + b + c 
#widths = np.arange(1, 31, step = 0.1)
#cwtmatr = signal.cwt(sig, signal.ricker, widths)
Wx_k, scales = cwt(sig, 'morlet')
cwtmatr = Wx_k 

freqs = pywt.scale2frequency('morl', scales) / np.mean(np.diff(t))
max_freq = 6
inds = freqs<max_freq

fig,ax = plt.subplots(3,1, sharex=True)
ax[0].plot(t,a)
ax[0].plot(t,b)
ax[0].plot(t,c)
ax[1].plot(t,sig)
#ax[2].imshow(np.abs(cwtmatr), cmap='jet', aspect='auto')
ax[2].pcolormesh(t, freqs[inds], np.abs(cwtmatr)[inds], cmap='jet')
ax[2].set_ylabel('Freq (Hz)')
#fig.colorbar(im, ax=ax[2])
plt.show()

#plt.imshow(cwtmatr, yticks=scales, abs=1,
#       title="abs(CWT) | Morlet wavelet",
#       ylabel="scales", xlabel="samples")

# https://dsp.stackexchange.com/questions/74032/scalograms-in-python

##############################################################
##############################################################


#%%# Helper fn + params #####################################################
def exp_am(t, offset):
    return np.exp(-pi*((t - offset) / .1)**10)

pi = np.pi
v1, v2, v3 = 64, 128, 32

#%%# Make `x` & plot #########################################################
t = np.linspace(0, 1, 2048, 1)
x = (np.sin(2*pi * v1 * t) * exp_am(t, .2) +
     (np.sin(2*pi * v1 * t) + 2*np.cos(2*pi * v2 * t)) * exp_am(t, .5)  + 
     (2*np.sin(2*pi * v2 * t) - np.cos(2*pi * v3 * t)) * exp_am(t, .8))
plot(x, title="x(t) | t=[0, ..., 1], %s samples" % len(x), show=1)

#%%# Take CWT & plot #########################################################
Wx, scales = cwt(x, 'morlet')
imshow(Wx, yticks=scales, abs=1,
       title="abs(CWT) | Morlet wavelet",
       ylabel="scales", xlabel="samples")

##############################################################
##############################################################
import pywt
import numpy as np
import matplotlib.pyplot as plt

##############################

fs = 100
t = np.arange(0,10,step = 1/fs)
y = np.sin(2*np.pi*t*7)
coef, freqs=pywt.cwt(y,np.arange(1,129),'cmor1.5-1.0')
freqs = freqs*fs
fig,ax = plt.subplots(2,1, sharex=True)
ax[0].plot(t,y)
#ax[1].matshow(np.abs(coef))
ax[1].pcolormesh(t, freqs, np.abs(coef), cmap='jet')
plt.show() 

##############################
fs = 100
t = np.arange(0,10,step = 1/fs)
a = np.cos(2 * np.pi * 3 * t)
b = signal.gausspulse(t - 4, fc=10)*5
c = signal.gausspulse(t - 8, fc=7)*5
y  = a + b + c 

coef, freqs=pywt.cwt(y,np.arange(1,100),'cmor20.0-1.0')
freqs = freqs*fs
max_freq = 20
inds = freqs<max_freq
fig,ax = plt.subplots(2,1, sharex=True)
ax[0].plot(t,y)
#ax[1].matshow(np.abs(coef))
ax[1].pcolormesh(t, freqs[inds], np.abs(coef)[inds], cmap='jet')
plt.show() 

#----------

import pywt
import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(-1, 1, 200, endpoint=False)
sig  = np.cos(2 * np.pi * 7 * t) + np.real(np.exp(-7*(t-0.4)**2)*np.exp(1j*2*np.pi*2*(t-0.4)))
widths = np.arange(1, 31)
cwtmatr, freqs = pywt.cwt(sig, widths, 'mexh')

fig,ax = plt.subplots(2,1)
ax[0].plot(t,sig)
ax[1].imshow(cwtmatr, extent=[-1, 1, 1, 31], cmap='PRGn', aspect='auto',
           vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())  
plt.show() 

############################################################
# Scipy test
############################################################
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

t = np.linspace(-1, 1, 200, endpoint=False)
sig  = np.cos(2 * np.pi * 7 * t) + signal.gausspulse(t - 0.4, fc=2)
widths = np.arange(1, 31)
#cwtmatr = signal.cwt(sig, signal.ricker, widths)
cwtmatr = signal.cwt(sig, signal.morlet2, widths)

cwtmatr_yflip = np.flipud(cwtmatr)

fig,ax = plt.subplots(2,1)
ax[0].plot(sig)
ax[1].imshow(np.abs(cwtmatr_yflip), extent=[-1, 1, 1, 31], cmap='PRGn', aspect='auto',
           vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
plt.show()

############################################################
# Scipy test 2
############################################################
t = np.linspace(-1, 1, 200, endpoint=False)
a = np.cos(2 * np.pi * 3 * t)
b = signal.gausspulse(t - 0.4, fc=10)*5
c = signal.gausspulse(t + 0.4, fc=7)*5
sig  = a + b + c 

widths = np.arange(1, 31)
cwtmatr = signal.cwt(sig, signal.morlet2, widths)
cwtmatr_yflip = np.flipud(cwtmatr)

abs_cwt = np.abs(cwtmatr_yflip)
fig,ax = plt.subplots(2,1)
ax[0].plot(sig)
ax[1].imshow(abs_cwt, extent=[-1, 1, 1, 31], cmap='jet', aspect='auto',
           vmax=abs(cwtmatr).max(), vmin=abs(cwtmatr).min())
plt.show()

############################################################
# Scipy test 2
############################################################
t = np.linspace(0,10,1000) 
sig  = signal.chirp(t, f0=0, t1=10, f1=10, method = 'linear') 

widths = np.arange(1,31)
cwtmatr = signal.cwt(sig, signal.morlet2, widths)
cwtmatr_yflip = np.flipud(cwtmatr)
abs_cwt = np.abs(cwtmatr_yflip)
fig,ax = plt.subplots(2,1)
ax[0].plot(sig)
ax[1].imshow(abs_cwt, extent=[-1, 1, 1, 31], cmap='jet', aspect='auto',
           vmax=abs(cwtmatr).max(), vmin=abs(cwtmatr).min())
plt.show()
