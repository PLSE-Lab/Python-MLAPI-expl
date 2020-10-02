#!/usr/bin/env python
# coding: utf-8

# # Microwave instability simulation
# 
# <a href="http://www.inp.nsk.su/~petrenko/">A. Petrenko</a> (Novosibirsk, 2019)
# 
# This notebook explains the basics of longitudinal particle motion in a storage ring with impedance.

# In[ ]:


import numpy as np
import holoviews as hv

hv.extension('matplotlib')


# In[ ]:


import warnings
warnings.filterwarnings('ignore')


# In[ ]:


c = 299792458 # m/c
mc = 0.511e6 # eV/c
Qe = 1.60217662e-19 # elementary charge in Coulombs

p0 = 400e6 # eV/c


# ### Electron beam definition

# In[ ]:


Ne = 2e10  # Number of electrons/positrons in the beam
N  = 200000 # number of macro-particles in this simulation


# In[ ]:


print("Bunch charge = %.1f nC" % (Ne*Qe/1e-9))


# Electron beam parameters:

# In[ ]:


sigma_z = 0.6 # m
#sigma_z = 1.0e-2 # m -- to test wakefield calculation

sigma_dp = 0.004 # relative momentum spread


# Distribution in $z$ and $\delta p = \frac{\Delta p}{p}$ can be defined easily since they are not correlated:

# In[ ]:


z0  = np.random.normal(scale=sigma_z, size=N)
#z0  = np.random.uniform(low=-sigma_z*2, high=sigma_z*2, size=N)
dp0 = np.random.normal(scale=sigma_dp, size=N)


# In[ ]:


get_ipython().run_line_magic('opts', 'Scatter (alpha=0.01 s=1) [aspect=3 show_grid=True]')

dim_z  = hv.Dimension('z',  unit='m', range=(-12,+12))
dim_dp = hv.Dimension('dp', label='100%*$\Delta p/p$', range=(-1.5,+1.5))

get_ipython().run_line_magic('output', "backend='matplotlib' fig='png' size=200 dpi=100")


# In[ ]:


hv.Scatter((z0,dp0*100), kdims=[dim_z,dim_dp])


# The function to get beam current profile corresponding to particle distribution:

# In[ ]:


def get_I(z, z_bin = 0.05, z_min=-15, z_max=+15):
    # z, z_bin, z_min, z_max in meters
    
    hist, bins = np.histogram( z, range=(z_min, z_max), bins=int((z_max-z_min)/z_bin) )
    Qm = Qe*Ne/N # macroparticle charge in C
    I = hist*Qm/(z_bin/c) # A

    z_centers = (bins[:-1] + bins[1:]) / 2
    
    return z_centers, I


# In[ ]:


get_ipython().run_line_magic('opts', 'Area [show_grid=True aspect=3] (alpha=0.5)')

dim_I = hv.Dimension('I',  unit='A',  range=(0.0,+1.0))

hv.Area(get_I(z0), kdims=[dim_z], vdims=[dim_I])


# ### Effects of RF-resonator

# Longitudinal momentum gain of electron after it has passed through the RF-resonator depends on the electron phase with respect to the RF:

# $$
# \frac{dp_z}{dt} = eE_{\rm{RF}}\cos\phi,
# $$

# where $E_{\rm{RF}}$ is the accelerating electric field and $\phi$ is the electron phase in the RF resonator. The resulting longitudinal momentum change:

# $$
# \delta p_z = e\frac{ V_{\rm{RF}} }{ L_{\rm{RF}}} (\cos\phi) \Delta t = e\frac{ V_{\rm{RF}} }{c} \cos\phi,
# $$

# where $V_{\rm{RF}}$ is the RF-voltage.

# RF-resonator frequency $f_{\rm{RF}}$ is some harmonic $h$ of revolution frequency:
# 
# $$
# f_{\rm{RF}} = \frac{h}{T_s},
# $$
# 
# where $T_s$ is the revolution period.

# Longitudinal coordinate $z$ gives the longitudinal distance from the electron to the reference particle at the moment when the reference particle arrives at the RF-phase $\phi_0$ (which is always the same). So the electron then arrives to the RF-resonator after the time
# 
# $$
# \Delta T = -\frac{z}{c}.
# $$

# Then the electron phase in the RF-resonator is
# 
# $$
# \phi = \phi_0 + 2\pi f_{\rm{RF}}\Delta T = \phi_0 - 2\pi \frac{hz}{T_s c} \approx \phi_0 - 2\pi \frac{hz}{L}.
# $$

# where $L$ is the ring perimeter. If the electron momentum is different from its reference value then the period of revolution $T$ is different from $T_s$:
# $$
# T = \frac{L+\Delta l}{\upsilon_s + \Delta \upsilon} \approx T_s \left ( 1 + \frac{\Delta l}{L} - \frac{\Delta \upsilon}{\upsilon_s} \right ) \approx T_s \left ( 1 + \frac{\Delta l}{L} - \frac{1}{\gamma^2} \frac{\Delta p}{p} \right ).
# $$
# The difference between the length of electron trajectory and the reference orbit length is given by the $M_{56}$ element of the 1-turn transport matrix:
# $$
# \Delta l = M_{56} \frac{\Delta p} {p}.
# $$
# $\Delta T = T - T_s$ in can be written as
# $$
# \Delta T \approx T_s \left ( \frac{M_{56}}{L} - \frac{1}{\gamma^2} \right ) \frac{\Delta p}{p} = T_s \left ( \frac{1}{\gamma_t^2} - \frac{1}{\gamma^2} \right ) \frac{\Delta p}{p} = T_s\eta\frac{\Delta p}{p}.
# $$

# Then the longitudinal position after one turn is
# $$
# z_{n+1} = z_n - L \eta\frac{\Delta p_{n+1}}{p}.
# $$

# ## Multi-turn tracking

# In[ ]:


L = 27.0 # m -- storage ring perimeter
gamma_t = 6.0 # gamma transition in the ring
eta = 1/(gamma_t*gamma_t) - 1/((p0/mc)*(p0/mc))


# In[ ]:


#N_turns = 1000020
N_turns = 2000
N_plots = 11

h = 1
eVrf = 5e3 # eV
#eVrf = 0.0 # eV
phi0 = np.pi/2

t_plots = np.arange(0,N_turns+1,int(N_turns/(N_plots-1)))

data2plot = {}

z = z0; dp = dp0
for turn in range(0,N_turns+1):
    if turn in t_plots:
        print( "\rturn = %g (%g %%)" % (turn, (100*turn/N_turns)), end="")
        data2plot[turn] = (z,dp)
    
    phi = phi0 - 2*np.pi*h*(z/L)  # phase in the resonator
    
    # 1-turn transformation:
    dp  = dp + eVrf*np.cos(phi)/p0
    z = z - L*eta*dp


# In[ ]:


def plot_z_dp(turn):
    z, dp = data2plot[turn]
    z_dp = hv.Scatter((z, dp*100), [dim_z,dim_dp])
    z_I  = hv.Area(get_I(z), kdims=[dim_z], vdims=[dim_I])
    return (z_dp+z_I).cols(1)


# In[ ]:


#plot_z_dp(1000)


# In[ ]:


items = [(turn, plot_z_dp(turn)) for turn in t_plots]

m = hv.HoloMap(items, kdims = ['Turn'])
m.collate()


# ## Longitudinal wakefield

# Now let's introduce the longitiudinal wakefield.
# 
# First define wake-function in terms of $\xi = z - ct$:

# Equation for wake-function can be obtained by performing a Fourier transformation of the impedance
# 
# $$
# Z = \frac{R_s}{1 + iQ\left(\frac{\omega_R}{\omega} -\frac{\omega}{\omega_R}\right)},
# $$
# where $Q$ is the quality factor, $\omega_R$ is the frequency.
# 
# (from A. Chao's book &#147;<a href=http://www.slac.stanford.edu/~achao/wileybook.html>Physics of Collective Beam Instabilities in High Energy Accelerators</a>&#148;. <a href=http://www.slac.stanford.edu/~achao/WileyBook/WileyChapter2.pdf>Chapter 2</a>, p. 73):
# 
# $$
# W(\xi) = \begin{cases}
#  2\alpha R_s e^{\alpha \xi/c}\left(\cos\frac{\overline{\omega} \xi}{c} + \frac{\alpha}{\overline{\omega}}\sin\frac{\overline{\omega} \xi}{c}\right), & \mbox{if } \xi < 0 \\
#  \alpha R_s, & \mbox{if } \xi = 0 \\
#  0, & \mbox{if } \xi > 0,
# \end{cases}
# $$
# 
# where $\alpha = \omega_R / 2Q$ and $\overline\omega = \sqrt{\omega_R^2 -\alpha^2}$.

# In[ ]:


def Wake(xi):
    # of course some other wake can be defined here.
    
    fr = 0.3e9 # Hz
    Rs = 1.0e5 # Ohm
    Q  = 5  # quality factor
   
    wr = 2*np.pi*fr
    alpha = wr/(2*Q)
    wr1 = wr*np.sqrt(1 - 1/(4*Q*Q))
    
    W = 2*alpha*Rs*np.exp(alpha*xi/c)*(np.cos(wr1*xi/c) + (alpha/wr1)*np.sin(wr1*xi/c))
    W[xi==0] = alpha*Rs
    W[xi>0] = 0
    
    return W


# In[ ]:


get_ipython().run_line_magic('opts', 'Curve [show_grid=True aspect=3]')

dim_xi   = hv.Dimension('xi', label=r"$\xi$", unit='m')
dim_Wake = hv.Dimension('W',  label=r"$W$", unit='V/pC')

L_wake = 10 # m
dz = 0.04 # m
xi = np.linspace(-L_wake, 0, int(L_wake/dz)) # m
W = Wake(xi)

hv.Curve((xi, W/1.0e12), kdims=[dim_xi], vdims=[dim_Wake])


# ### Wakefield from e-bunch

# Longitudinal wake-function defines the wakefield amplitude from a point-like charge. Therefore a distribution of charge will produce the wakefield
# 
# $$
# E(z) = -\int\limits_{z}^{+\infty} W(z-z')I(z')dz'/c = -\int\limits_{-\infty}^{0} W(\xi)I(z-\xi)d\xi/c,
# $$

# In[ ]:


zc, I = get_I(z0, z_bin=dz)

V = -np.convolve(W, I)*dz/c # V


# In[ ]:


zV = np.linspace(max(zc)-dz*len(V), max(zc), len(V))


# In[ ]:


dim_V = hv.Dimension('V', unit='kV', range=(-10,+10))

(hv.Curve((zV, V/1e3), kdims=[dim_z], vdims=[dim_V]) +  hv.Area((zc,I), kdims=[dim_z], vdims=[dim_I])).cols(1)


# ## Tracking with impedance

# In[ ]:


data2plot = {}

#eVrf = 0    # V
#eVrf = 3e3 # V

z = z0; dp = dp0
for turn in range(0,N_turns+1):
    if turn in t_plots:
        print( "\rturn = %g (%g %%)" % (turn, (100*turn/N_turns)), end="")
        data2plot[turn] = (z,dp)
    
    phi = phi0 - 2*np.pi*h*(z/L)  # phase in the resonator
    
    # RF-cavity
    dp  = dp + eVrf*np.cos(phi)/p0
    
    # wakefield:
    zc, I = get_I(z, z_bin=dz) # A
    V = -np.convolve(W, I)*dz/c # V    
    V_s = np.interp(z,zV,V)
    dp  = dp + V_s/p0

    # z after one turn:
    z = z - L*eta*dp


# In[ ]:


def plot_z_dp(turn):
    z, dp = data2plot[turn]
    z_dp = hv.Scatter((z, dp*100), [dim_z,dim_dp])
    zc, I = get_I(z, z_bin=dz)
    z_I  = hv.Area((zc,I), kdims=[dim_z], vdims=[dim_I])
    V = -np.convolve(W, I)*dz/c # V
    z_V  = hv.Curve((zV, V/1e3), kdims=[dim_z], vdims=[dim_V])
    return (z_dp+z_I+z_V).cols(1)


# In[ ]:


items = [(turn, plot_z_dp(turn)) for turn in t_plots]

m = hv.HoloMap(items, kdims = ['Turn'])
m.collate()


# In[ ]:


#np.save("plots.npy", data2plot)

