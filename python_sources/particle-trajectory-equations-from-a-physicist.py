#!/usr/bin/env python
# coding: utf-8

# # Particle trajectories
# I will derive particle trajectories wrt time for a particle that is not acted upon by any force and for a particle that is interacting with a constant magnetic field.
# This might be an idealized view of what goes on in the detector, neglecting particle-particle ineractions, particle- detector interactions, inhomogenous magnetic field and other things I cannot think of :). Might be a bit confusing, but hope that you will find use for the final equations at least. They are boxed.
# 
# ## Special relativity
# Since the particles have high energies and thus speeds, one should use special relativity to describe the trajectories.
# As you might be aware of, time is relative in special relativity. This means that, starting off, one parametrizes the trajectories in terms of "proper time" $\tau$ instead of "regular time" $t$.
# Proper time is time measured in a coordinate system moving with the same speed and direction as the particle. I.e. the particle is at rest in this coordinate system.
# Regular time (time in our coordinate frame, where the detector is at rest) is treated in many ways as a spatial coordinate.
# 
# Lets get started.
# 
# I will use $\vec{overhead arrow}$ for regular coordinate vectors and $\bf{Boldface}$ for four-vecors. That is vectors that also contains this new "time dimension" as the first entry.
# 
# Some examples of four-vecors:
# 
# $\bf{f}$ - four force
# 
# $\bf{p}$ - four momentum
# 
# $\bf{u}$ - four velocity
# 
# Regular vectors:
# 
# $\vec{r} = \langle x,y,z\rangle$
# 
# ## No force
# 
# $$\bf{f} = \frac{\partial \bf{p}}{\partial\tau} = 0$$
# 
# This means four-momentum is conserved:
# 
# $$ m\langle \frac{\partial c t}{\partial\tau}, \frac{\partial x}{\partial\tau}, \frac{\partial y}{\partial\tau}, \frac{\partial z}{\partial\tau} \rangle = m\langle \frac{\partial c t}{\partial\tau}, \frac{\partial \vec{r}}{\partial\tau}\rangle =\bf{p_0}$$
# 
# By intagrating we can conclude that:
# $$ \vec{r} = \frac{\vec{p_0}}{m}\tau + \vec{r_0}$$
# 
# From the first "time component " of the four-vector we have:
# $$ mc\frac{\partial t}{\partial \tau} = \frac{E}{c} \rightarrow \tau = \frac{mc^2}{E} t, $$
# assuming $t(0)=0$. Where $E$ is the energy of the particle (The time-component of the four-momentum is energy divided by c). Also note that if the energy is not much higher than the resting energy $mc^2$, time ticks at the same rate in both coordinate systems.
# Lets plug this into our parametrized trajectory:
# 
# $$ \boxed{ \vec{r} = \frac{\vec{p_0} c^2}{E} t+ \vec{r_0} =\frac{\vec{p_0}}{m\sqrt{1+\frac{|\vec{p_0}|^2}{(mc)^2}}} t+ \vec{r_0} }$$
# 
# Note that for small momentums(much smaller than $mc$), we get back the classical result $\vec{r} = \frac{\vec{p}}{m} t + \vec{r_0}$
# 
# ## Constant magnetic field (z-direction)
# The forces acting upon a chared particle by electromagnetism is called the lorentz force. Or four-force in special relativity. For constant magnetic field in z-dir, we have the lorentz four-force:
# $$ \frac{\partial \bf{p}}{\partial \tau} = q \begin{bmatrix}
#     0   & 0 & 0 & 0 \\
#     0  & 0 & -B  & 0 \\ 
#     0  & B & 0  & 0 \\
#     0  & 0 & 0 &0 
# \end{bmatrix}\begin{bmatrix}
#    \frac{\partial t}{\partial \tau}\\
#    \frac{\partial x}{\partial \tau}\\ 
#    \frac{\partial y}{\partial \tau}\\
#    \frac{\partial z}{\partial \tau}  
# \end{bmatrix} = qB\langle 0, -  \frac{\partial y}{\partial \tau},  \frac{\partial x}{\partial \tau},0\rangle,$$
# where q is the particle's charge and B is the magnitude of the magnetic field.
# 
# Lets look at the x and y coordinatesof the particle, since there now is a force acting in the x and y directions.
# 
# We have:
# $$ m\frac{\partial^2 x}{\partial \tau^2} = -qB\frac{\partial y}{\partial \tau} \\
#     m\frac{\partial^2 y}{\partial \tau^2} = qB \frac{\partial x}{\partial \tau} $$
#     
#    Integrating:
#     $$ m\frac{\partial x}{\partial \tau} = -qB y + p_{0,x}\\
#     m\frac{\partial y}{\partial \tau} = qB x + p_{0,y}$$
#     
#   This set of first-order differential equations has the solution:
#   $$ x(\tau)=(x_0 + \frac{p_{0,y}}{qB})\cos[\frac{qB}{m}\tau] + (\frac{p_{0,x}}{qB}-y_0)\sin[\frac{qB}{m}\tau] - \frac{p_{0,y}}{qB}  \\
#          y(\tau)=(x_0 + \frac{p_{0,y}}{qB})\sin[\frac{qB}{m}\tau] - (\frac{p_{0,x}}{qB}-y_0)\cos[\frac{qB}{m}\tau] + \frac{ p_{0,x}}{qB}$$
#  
#        
#   In terms of t:
#      $$ \boxed{x(t)=(x_0 + \frac{p_{0,y}}{qB})\cos[\frac{qB}{m}\gamma t] + (\frac{p_{0,x}}{qB}-y_0)\sin[\frac{qB}{m}\gamma t] - \frac{p_{0,y}}{qB} } \\
#         \boxed{ y(t)=(x_0 + \frac{p_{0,y}}{qB})\sin[\frac{qB}{m}\gamma t] - (\frac{p_{0,x}}{qB}-y_0)\cos[\frac{qB}{m}\gamma t] + \frac{ p_{0,x}}{qB}}  \\
#        \boxed{ z(t) = \frac{p_{0,z}}{m} \gamma t + z_0 }$$
#     With $$ \gamma = \frac{1}{\sqrt{1+\frac{|\vec{p_0}|^2}{(mc)^2}}} $$
#     
#        
#     
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 

# # Shortest distance from a point to a trajectory
# 
# ## No force
# 
# For a point in space $\vec{c}$, the squared distance between this point and our trajectory $\vec{r}(\tau) = \frac{\vec{p_0}}{m}\tau + \vec{r_0}$ is:
# $$ |\vec{r}-\vec{c}|^2 = \frac{|\vec{p_0}|^2}{m^2}\tau^2 + |\vec{r_0}|^2 + |\vec{c}|^2 + 2\frac{\vec{p_0}\cdot(\vec{r_0}-\vec{c})}{m}\tau -2 \vec{r_0}\cdot\vec{c} $$
# 
# This distance is minimized by:
# 
# $$ \tau_{min} = \frac{m \vec{p_0}\cdot(\vec{r_0}-\vec{c})}{|\vec{p_0}|^2}$$
# 
# ## Constant magnetic field (z-dir)
# 
# This is a lot of work to derive an analytical expression. If needed one can try numerical minimization of distance between point and trajectory starting at $\tau = (c_z-z_0)m/p_{0,z}$ first.
# 

# In[ ]:




