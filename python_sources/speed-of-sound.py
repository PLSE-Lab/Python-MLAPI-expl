#!/usr/bin/env python
# coding: utf-8

# # Measuring the speed of sound
# For this lab we use a straw, a ruler to measure the length of the straw and a laptop or smartphone with a HTML5 browser and the build-in microphone.
# 
# The websites to explore the properties of sound like frequency and amplitude can be visualized using a spectrometer. This can be done in the browser. Some examples:
# 
# https://musiclab.chromeexperiments.com/Spectrogram/
# 
# https://webaudiodemos.appspot.com/pitchdetect/
# 
# The relationship is pretty simpe:
# 
# $$c=\frac{\lambda}{T}=\lambda f$$
# 
# In the straw we get a standing wave. Since both ends are open the first node would have a node in the center of the tube and 
# 
# Regrading the length $l$ of the tube we get the relationship
# 
# $$l=\frac{\lambda}{2}$$
# 
# Now measuring the length and the frequency of the pitch we can calculate the speed of sound $c$ in air.

# In[ ]:


l = float(input(" Please enter the length of the straw in meter: "))
f = float(input(" Please enter the frequency of the pitch in Hertz: "))

lambd = 2 * l
c = lambd * f

print(str(c) + " m/s is the speed of sound in the air in Ho Chi Minh")

