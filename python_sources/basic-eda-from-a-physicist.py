#!/usr/bin/env python
# coding: utf-8

# In this notebook, I'm looking at a few things in the small training set to try to understand what is in the data.
# 
# If you want to understand more about how detectors work and how particles interact with them, I would strongly suggest at least going over the relevant review articles from the Particle Data Group (PDG), which are all available for free online. There are also several textbooks on detectors available if you're really interested in knowing more details. Unfortunately, there aren't so many easily accessible resources for learning in detail how things like track reconstruction (the topic of this competition) are implemented in practice.

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# First, let's read in one of the events.

# In[ ]:


def get_event_data(path='../input/train_1/event000001000'):
    cells = pd.read_csv(path+'-cells.csv')
    hits = pd.read_csv(path+'-hits.csv',index_col=0)
    particles = pd.read_csv(path+'-particles.csv',index_col=0)
    truth = pd.read_csv(path+'-truth.csv',index_col=0)
    return cells, hits, particles, truth


# In[ ]:


cells, hits, particles, truth = get_event_data()


# Now, let's print the head of each of the dataframes.

# In[ ]:


truth.head()


# We see that the particle IDs are in the form of a very long integer. My guess is that the number encodes some information about the particle but it may also just be some sort of randomized or hashed number to keep participants from learning any information from that. 
# 
# A couple of the first hits have no corresponding particle ID. The momentum suggests that these would have PeV-scale energies, which is obviously too high to be created at the LHC. These hits have weight 0, so they should be excluded from the data.

# In[ ]:


particles.head()


# This looks more reasonable.

# In[ ]:


hits.head()


# This also looks fine. It gives us some mappings from the hit IDs to positions and detector IDs.

# In[ ]:


cells.head()


# And this gives us channel and energy loss information. A single hit can cover multiple channels. We might want to try to study things like the resolution for different types of channels. It's also not clear exactly how to use the signal. Do we add all the signals for a given hit?

# ## Understanding the Geometry
# 
# First, we want to understand the geometry. We do have a geometry file available, but we can also plot out hte hits. First, let's look at the 2D projections for everything. The volume ID provides some information about what piece of the detctor we hit.

# In[ ]:


fig = plt.figure(figsize=(10,10))
plt.scatter(hits.x,hits.y,marker='.',alpha=0.1,c=hits.volume_id)
plt.xlabel('x')
plt.ylabel('y')
plt.show()


# Here, we see from the colors that there are 3 different volumes included here. We also see that there are a series of concentric rings with some random scattered hits in between the rings. There are gaps between the different volumes. The concentric-ring type geometry is very common in collider physics, as it matches the cylindrical symmetry of collisions well.

# In[ ]:


fig = plt.figure(figsize=(10,10))
plt.scatter(hits.x,hits.z,marker='.',alpha=0.1,c=hits.volume_id)
plt.xlabel('x')
plt.ylabel('z')
plt.show()


# The $xz$ projection shows a bit more about what's going on. The random scattering between the rings in the $xy$ projection are from vertically-aligned layers at both ends of the tracker. (Basically, endcap layers.)
# 
# We can also clearly see the distinction between the inner tracker and the outer tracker. I haven't checked this but it's likely that much of the outer tracker is made of strip detectors while the inner tracker may be made only of pixels.
# 
# In this projection, the beam is going in the vertical dimension. The region around $z=0$ consists of concentric circles around the beam pipe.
# 
# Now we can start zooming in. First, remove the endcap layers.

# In[ ]:


fig = plt.figure(figsize=(10,10))
plt.scatter(hits[np.abs(hits.z)<1000].x,hits[np.abs(hits.z)<1000].y,marker='.',alpha=0.1)
plt.xlabel('x')
plt.ylabel('y')
plt.show()


# As we might expect, we now see the outer tracker as just a series of concentric rings around the beam pipe.

# In[ ]:


fig = plt.figure(figsize=(10,10))
hits_inner = hits[(np.hypot(hits.x,hits.y)<200) & (np.abs(hits.z)<1500)]
plt.scatter(hits_inner.x,hits_inner.y,marker='.',alpha=0.1,c=hits_inner.volume_id)
plt.xlabel('x')
plt.ylabel('y')
plt.show()


# Looking closer at the inner tracker, we see that the innermost layer is only around 2.5 cm from the center. Having detectors so close to the beam is critical to many measurements, as that is what allows for good vertex resolution.
# 
# Some interesting types of events, such as tau lepton and b quark production events can be identified because the tau or b decays a measurable distance from the original collision vertex. This can only be seen with a very good vertex position resolution.
# 
# One unfortunate effect of this is that the harsh environment slowly kills the detectors as they succumb to radiation damage. Modules in trackers typically need to be periodically replaced or there will end up with dead regions of the detector.

# In[ ]:


fig = plt.figure(figsize=(10,10))
plt.scatter(hits_inner.x,hits_inner.z,marker='.',alpha=0.1,c=hits_inner.volume_id)
plt.xlabel('x')
plt.ylabel('z')
plt.show()


# The inner tracker looks a lot like the outer tracker (but smaller obviously). There are a series of concentric rings right near the origin and then a series of endcap layers for very forward/backward-going tracks.

# In[ ]:


fig = plt.figure(figsize=(10,10))
hits_inner2 = hits[(np.hypot(hits.x,hits.y)<200) & (np.abs(hits.z)<500)]
plt.scatter(hits_inner2.x,hits_inner2.y,marker='.',alpha=1,c=hits_inner2.layer_id)
plt.xlabel('x')
plt.ylabel('y')
plt.colorbar()
plt.show()


# Now looking only at the concentric rings, I've colored things by the layer ID. It is fairly simple to select hits only in certain layers using the volume and layer ID fields.
# 
# When zoomed in this far, we also see something interesting. While previously, it might have looked like there was some aliasing in the images, here we clearly see that the concentric circles are not really circles. Instead, they are build from a series of planes. The actual silicon detectors are small flat rectangles, so this shows how the rings are created.

# ## Energy Loss Distribution
# 
# Now, let's select all the hits from a particular layer. 

# In[ ]:


hits_layer3 = hits[(np.abs(hits.z)<500)&(np.hypot(hits.x,hits.y)<190)&(np.hypot(hits.x,hits.y)>150)]
hits_layer3.describe()


# This is the outermost concentric layer of the inner tracker, which has volume_id=8 and layer_id=8. 
# 
# I can merge the hits with the cells data to bring in signal information. Silicon detectors measure induced currents from particle-hole pairs created by ionization in the depletion region of the detector (typically a pn junction). 
# 
# Of course, there is a lot more that goes into taking a raw signal and getting something useful. The total energy lost is miniscule, and what I really might care about is how much energy is lost per amount of stuff the particle traveled through. To do this, I should add in the particle momentum information and use the angle of incidence and the detector thickness to normalize the results.
# 
# The dataset also does not include truth information about the particle type beyond the charge, so particle identification (PID) might not be feasible here.

# In[ ]:


hits_with_energy = cells.merge(hits_layer3,how='inner',left_on='hit_id',right_index=True)
fig = plt.figure(figsize=(12,6))
ax = fig.add_subplot(121)
ax.hist(hits_with_energy.value,bins=100)
ax.set_xlabel('Cell Signal Size')
ax = fig.add_subplot(122)
ax.hist(hits_with_energy.groupby('hit_id').value.sum(),bins=100,range=(0,1.2))
ax.set_xlabel('Summed Hit Signal')
plt.show()


# This is quite interesting. Typically, as relativistic particles travel through thin detectors, the energy loss due to ionization follows a Landau distribution (think a Gaussian with a very long positive tail).
# 
# Here, if we look at the individual cell signals, we see a nice peak with a long positive tail, but we also see a shoulder at low energy loss.
# 
# If we add all the cells together, we lose the shoulder, but also see the signal distribution mostly cut off around 1. Maybe this is related to how cells are combined into hits.

# In[ ]:


hits_with_energy = cells.merge(hits[(hits.volume_id==8)&(hits.layer_id==6)],how='inner',left_on='hit_id',right_index=True)
fig = plt.figure(figsize=(12,6))
ax = fig.add_subplot(121)
ax.hist(hits_with_energy.value,bins=100,range=(0,0.25))
ax.set_xlabel('Cell Signal Size')

ax = fig.add_subplot(122)
ax.hist(hits_with_energy.groupby('hit_id').value.sum(),bins=100,range=(0,1.4))
ax.set_xlabel('Summed Hit Signal')

plt.show()


# We see something similar if we go inward one layer, but now the summed hit signal distribution extends quite a bit farther. This may just be an effect from (1) maybe using different detector or readout hardware and (2) having a higher particle density.

# In[ ]:


hits_with_energy = cells.merge(hits[(hits.volume_id==8)&(hits.layer_id==4)],how='inner',left_on='hit_id',right_index=True)
fig = plt.figure(figsize=(12,6))
ax = fig.add_subplot(121)
ax.hist(hits_with_energy.value,bins=100,range=(0,0.25))
ax = fig.add_subplot(122)
ax.hist(hits_with_energy.groupby('hit_id').value.sum(),bins=100,range=(0,2))
plt.show()


# And we see the same thing as we continue to move closer to the beam.

# ## Particle Truth Information
# 
# The particles file includes truth information from the particles when they are created. We don't have much beyond basic kinematics and a charge. First, let's add in some common kinematic variables used in high energy physics. The are:
# 
#   - $p_{t}$: The transverse momentum (i.e. momentum perpendicular to the beam)
#   - $E$: Energy. Technically, this is just the total momentum since we don't have the mass, but in the relativistic limit, energy and the magnitude of the momentum are identical.
#   - $\eta$: The pseudorapidity. This is an alternative measure of the direction with respect to the beam. High rapidity means very close to the beam direction
#   - $\phi$: Azimuthal angle. Things are very close to cylindrically symmetric here, so we expect that this should not matter when large amounts of data are aggregated. If the data show significant dependence on this, then we'll need to understand why.

# In[ ]:


def add_kinematics(df):
    df['pt'] = np.hypot(df.px,df.py)
    df['E'] = np.hypot(df.pt,df.pz)
    df['eta'] = -0.5 * np.log( (df.E + df.pz)/(df.E-df.pz))
    df['phi'] = np.arctan2(df.vy,df.vx)
add_kinematics(particles)


# In[ ]:


plt.hist(particles.nhits,bins=20,range=(0,20))
plt.xlabel('Number of hits')
plt.show()


# Particles mostly either have very few or around 12 hits. The number 12 is probably just related to how the layers are set up in the detector. Low momentum or very forward particles will tend to miss everything.

# In[ ]:


particles.q.value_counts()


# There are substantially more positively-charged particles than negatively charged ones. I guess we expect somewhat more positively charged particles since the LHC runs proton-proton collisions rather than proton-antiproton like at the Tevatron or the S$\rm p\bar{p}$S collider. I'm not sure if we expect this much of a difference though.

# In[ ]:


plt.hist(particles.pt,bins=100,range=(0,10))
plt.xlabel('Transverse Momentum [GeV]')
plt.show()


# The transverse momentum distribution looks pretty exponential, with few particles having $p_{t}>2$ GeV. One thing we can try to look at here is the momentum balance. We can sort by transverse momentum first and then check the balance by looking at the "missing $p_{t}$" for different subsets of particles.
# 
# Note that this won't really work here since I haven't tried to work out particle ancestry. If one high-$p_{t}$ particle is the parent of another (due to something like a decay process), I could be double counting the momentum.
# 
# But if you want to figure out things like when decays and inelastic interactions happen, or at least select only things originating right near the beam, maybe you will see something. This is complicated by the fact that there are going  to be many interactions per collision and that most collisions just lead to a bunch of junk being sprayed everywhere. You might miss a lot of momentum due to particles staying in the beam pipe. Ideally, you can find the particles associated with each interaction vertex and then look at the momentum balance for each vertex that looks like it might be intersting.

# In[ ]:


particles_pt_sort = particles.sort_values(by='pt')


# In[ ]:


px_summed = particles_pt_sort.py.cumsum()
py_summed =particles_pt_sort.px.cumsum()
pt_summed = np.hypot(px_summed,py_summed)
print(pt_summed.values[-20:])


# This first set shows what happens if I look at the total momentum balance excluding some of the highest $p_{t}$ particles. We can see that the momentum balance is quite far off but actually is reduced significantly by those particles. But again, I really need to do this in a more careful way to really say anything about this.

# In[ ]:


particles_pt_sort = particles.sort_values(by='pt',ascending=False)
px_summed = particles_pt_sort.py.cumsum()
py_summed =particles_pt_sort.px.cumsum()
pt_summed = np.hypot(px_summed,py_summed)
print(pt_summed.values[:20])


# And this shows the momentum balance for only the few highest $p_{t}$ particles. Again, it looks pretty high but I am summing up everything from a number of vertices and may be double-counting momentum from decay and inelastic interaction events.

# In[ ]:


particles_pt_sort.head(20)


# We can see that there are some very high transverse momentum particles, but many of them also have a large longitudinal momentum. The kinds of things that are really interesting to analyses for things like electroweak physics are particles with high transverse momentum and not much longitudinal momentum. A low pseudorapidity and high energy means that a lot of momentum had to have been transferred somewhere.

# ## Particles Near/In the Beam
# 
# Now let's take particles with fairly high $p_t$ (1 GeV or more) and zoom in on the beam.

# In[ ]:


particles_high_pt = particles[particles.pt>1]


# In[ ]:


plt.scatter(particles_high_pt.vx,particles_high_pt.vy,alpha=0.3)
plt.show()


# In[ ]:


plt.scatter(particles_high_pt.vx,particles_high_pt.vy,alpha=0.3)
plt.xlim([-50,50])
plt.ylim([-50,50])
plt.show()


# In[ ]:


plt.scatter(particles_high_pt.vz,particles_high_pt.vx,alpha=0.3)
plt.ylim([-5,5])
plt.xlim([-25,25])
plt.show()


# I didn't put axis labels on everything but the distance scales are in millimeters. The collisions here are all happening within around 1 cm of the origin in the $z$ (beam) direction and a fraction of a cm in the transverse direction.
# 
# ## Invariant Mass
# 
# One way to look for decays of interesting particles is to calculate the invariant mass of sets of particles that you're interested in. Often, a particle will appear as a peak in a invariant mass distribution. For example, if you select a bunch of $Z\rightarrow\mu^+\mu^-$ decays, you should see a nice peak centered at 91.2 GeV (well, it can get a bit more complicated than that due to various effects). The invariant mass is nice because it corrects for things like boosted frames, but it can be hard to interpret if you don't reconstruct all the particles in the final state. Here, we also don't know how particles are selected in the simulation, so we could accidentally look at too many particles. 
# 
# For now, let's look at everything with $p_{t}>1$ GeV.

# In[ ]:


invariant_mass = []
for ids,gp in particles[particles.pt>1].groupby(['vx','vy','vz']):
    if gp.vx.count()==1:
        continue
    px_tot = gp.px.sum()
    py_tot = gp.py.sum()
    pz_tot = gp.pz.sum()
    E_tot = gp.E.sum()
    invariant_mass.append( np.sqrt(E_tot*E_tot-px_tot*px_tot-py_tot*py_tot-pz_tot*pz_tot) )
    


# In[ ]:


plt.hist(invariant_mass,bins=100,range=(1,2000))
plt.xlabel('Invariant Mass [GeV]')
plt.show()


# We don't see much, but that's not surprising. Most of the particles here are probably from uninteresting sprays of hadrons or are secondary particles. It's actually not going to be particularly common to see truly interesting events, and probably we'd need more than just the tracker to tell how interesting something is.
# 
# ## Back to particles near the beam
# 

# In[ ]:


particles_near_center = particles[(np.hypot(particles['vx'],particles['vy'])<0.05)
                                  &(np.abs(particles['vz'])<25)]
plt.scatter(particles_near_center.vz,particles_near_center.vx,alpha=0.3)
plt.ylim([-0.1,0.1])
plt.xlim([-25,25])
plt.show() 


# In[ ]:



plt.hist(particles_near_center.drop_duplicates(subset=['vx','vy','vz']).vz,bins=25)
plt.title('Particle Position Near Beam')
plt.xlabel('Z [mm]')
plt.show()


# We see that if we select only things right near the beam, there is a distribution peaked near 0 but with a width of order 5 mm. It doesn't really look Gaussian, but we probably want to add in more data to get a nicer profile and to define a cleaner selection.
# 
# ## $\eta-\phi$ Distributions
# 
# Finally, let's look at the direction distributions of particles originating near the beam

# In[ ]:


fig = plt.figure(figsize=(8,6))
plt.hist2d(particles_near_center['eta'],particles_near_center['phi'],bins=(50,50))
plt.xlabel(r'$\eta$')
plt.ylabel(r'$\phi$')
plt.colorbar()
plt.show()


# We see a couple potentially worrying things here. (Again, it would be good to get a cleaner selection). First, there is a big peak in the distribution. With just one event, maybe this is due to some correlated particles (like a jet or shower) and low statistics.
# 
# We also see clear horizontal bands, indicating that some azimuthal angles seem to be favored.

# In[ ]:


fig = plt.figure(figsize=(8,6))

plt.hist2d(particles_near_center['eta'],particles_near_center['phi'],bins=(50,50),range=([-4,4],[1,3]))
plt.xlabel(r'$\eta$')
plt.ylabel(r'$\phi$')
plt.colorbar()
plt.show()


# In[ ]:


fig = plt.figure(figsize=(8,6))

particles_tmp = particles_near_center[particles_near_center.E>1]
plt.hist2d(particles_tmp['eta'],particles_tmp['phi'],bins=(50,50),range=([-4,4],[1,3]))
plt.xlabel(r'$\eta$')
plt.ylabel(r'$\phi$')
plt.colorbar()

plt.show()


# We see that these effects persits even as we zoom in or make some minimum cut on the transverse momentum. So, let's bring in more data.

# In[ ]:


def get_particle_data(path='../input/train_1/event0000010{:02}-particles.csv'):
    df = pd.DataFrame()
    for i in range(100):
        data = pd.read_csv(path.format(i),index_col=0)
        df = pd.concat([df,data])
    return df
particle_all  = get_particle_data()


# In[ ]:


add_kinematics(particle_all)    


# In[ ]:


def near_center(df):
    return df[(np.hypot(df['vx'],df['vy'])<0.05)
               &(np.abs(df['vz'])<25)]
center = near_center(particle_all)


# In[ ]:


fig = plt.figure(figsize=(8,6))

plt.hist2d(center['eta'],center['phi'],bins=(50,50))
plt.xlabel(r'$\eta$')
plt.ylabel(r'$\phi$')
plt.colorbar()

plt.show()


# The big peak has gone away clearly see that certain azimuthal angles still appear to be favored. So, it might be important to figure out if this is some weird effect from how I chose my particles (maybe double counting some things?) or if there is some reason for this.

# In[ ]:


fig = plt.figure(figsize=(8,6))

plt.hist2d(center[center.E>1]['eta'],center[center.E>1]['phi'],bins=(50,50))
plt.xlabel(r'$\eta$')
plt.ylabel(r'$\phi$')
plt.colorbar()

plt.show()


# Things still persist as we select on high energy. High energies are dominated by forward-going tracks.

# In[ ]:


fig = plt.figure(figsize=(8,6))

plt.hist2d(center[center.E>5]['eta'],center[center.E>5]['phi'],bins=(50,50))
plt.xlabel(r'$\eta$')
plt.ylabel(r'$\phi$')
plt.colorbar()

plt.show()


# And that's it for now. There's a lot more explore in the data before even attempting any model building. It would definitely be good to start building some diagnostic tools based on the truth information to identify various kinds of events. I also haven't even looked at trying to build tracks, but a proper model will have to somehow account for the fact that tracks mostly move in helical paths due to a magnetic field that is probably in the geometry and also the fact that multiple scattering can lead to significant changes in particle directions (especially for electrons/positrons). So, it won't be enough to just look for straight lines. Additionally, there are various processes that can create new particles far from the beam, and a model will need to try to identify these as well.
