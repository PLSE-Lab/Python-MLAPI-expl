#!/usr/bin/env python
# coding: utf-8

# Hi Kaggle!
# 
# In this notebook I share with you the tool I use to work with PLAsTiCC data. 
# 
# Below, you'll find the definition of the module, which you can move to another file.
# 
# In the next cells, I'll show you what's the module can show you :) Enjoy!

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings

warnings.filterwarnings('ignore')
import itertools


class PlasticcVis:
    def __init__(self, data):
        self.data = data

        # Fix for missing PROJ4 env var https://github.com/conda-forge/basemap-feedstock/issues/30#issuecomment-423512069
        import os
        import conda

        conda_file_dir = conda.__file__
        conda_dir = conda_file_dir.split('lib')[0]
        proj_lib = os.path.join(os.path.join(conda_dir, 'share'), 'proj')
        os.environ["PROJ_LIB"] = proj_lib

    def light_curve(self, object_id):
        """
        Plot light curves of a single object in all passbands
        :param object_id: id of the object for which the curves will be plotted
        :return: ax with drown chart
        """
        ax = sns.lineplot(x="mjd", y="flux", hue="passband", data=self.data[self.data['object_id'] == object_id])
        ax.plot()
        return ax

    def light_curve_split(self, object_id):
        """
        Plot light curves of a single object with separate chart for each passband
        :param object_id: id of the object for which the curves will be plotted
        :return: axes array with plotted charts
        """
        data = self.data[self.data['object_id'] == object_id]
        passbands = data['passband'].sort_values().unique()
        fig, axes = plt.subplots(nrows=len(passbands), ncols=1, sharex=True, sharey=True)
        fig.tight_layout()
        with sns.plotting_context("notebook", rc={"lines.linewidth": 3}):
            for i, passband in enumerate(passbands):
                ax = sns.lineplot(x="mjd", y="flux", hue="object_id", data=data[data['passband'] == passband],
                                  ax=axes[i], legend=False)
                ax.set_title(f"Passband {passband}")

        return axes

    def light_curve_compared(self, object_id, target):
        """
        Plot specified object light curves along with other light curves in specified target class
        :param object_id: id of the object for which the curves will be plotted
        :param target: target class of other objects which will be drawn along specified object
        :return: ax with drown chart
        """
        ax = None
        other_ids = self.data[(self.data['object_id'] != object_id) & (self.data['target'] == target)][
            'object_id'].unique()
        for other_id in other_ids:
            ax = sns.lineplot(x="mjd", y="flux", hue="passband", data=self.data[self.data['object_id'] == other_id],
                              palette=sns.cubehelix_palette(dark=.7, light=.9, as_cmap=True),
                              legend='brief' if ax is None else False)
        with sns.plotting_context("notebook", rc={"lines.linewidth": 3}):
            ax = sns.lineplot(x="mjd", y="flux", hue="passband", data=self.data[self.data['object_id'] == object_id],
                              palette=sns.light_palette("green", as_cmap=True))
        ax.plot()

        return ax

    def light_curve_split_compared(self, object_id, target):
        """
        Plot light curves of a single object with separate chart for each passband
        along with other light curves in specified target class
        :param object_id: id of the object for which the curves will be plotted
        :param target: target class of other objects which will be drawn along specified object
        :return: axes array with plotted charts
        """
        passbands = self.data['passband'].sort_values().unique()
        fig, axes = plt.subplots(nrows=len(passbands), ncols=1, sharex=True, sharey=True)
        fig.tight_layout()
        for i, passband in enumerate(passbands):
            ax = sns.lineplot(x="mjd", y="flux", hue="object_id", data=self.data[
                (self.data['object_id'] != object_id) & (self.data['target'] == target) & (
                            self.data['passband'] == passband)], ax=axes[i])
            with sns.plotting_context("notebook", rc={"lines.linewidth": 3}):
                sns.lineplot(x="mjd", y="flux", hue="object_id", data=self.data[
                    (self.data['object_id'] == object_id) & (self.data['passband'] == passband)],
                             palette=sns.light_palette("green", reverse=True, as_cmap=True), ax=ax)
            ax.set_title(f"Passband {passband}")
            ax.plot()

        return axes

    def light_curve_interpolate_compare(self, object_id, target):
        """
        Plot light curves of a single object along
        with interpolated line representing average light curve of the target class
        :param object_id: id of the object for which the curves will be plotted
        :param target: target class of other objects which will be drawn along specified object
        :return: ax with drown chart
        """
        passbands = self.data['passband'].sort_values().unique()
        fig, ax = plt.subplots()
        for i, passband in enumerate(passbands):
            data = self.data[(self.data['passband'] == passband) & (self.data['target'] == target) & (
                        self.data['object_id'] != object_id)]
            x_pane = np.linspace(min(data['mjd']), max(data['mjd']), 100)
            y_pane = np.poly1d(np.polyfit(data['mjd'], data['flux'], 15, 25))(x_pane)
            sns.lineplot(x=x_pane, y=y_pane, palette=sns.dark_palette("palegreen", as_cmap=True), ax=ax)

            with sns.plotting_context("notebook", rc={"lines.linewidth": 3}):
                sns.lineplot(x="mjd", y="flux", data=self.data[
                    (self.data['object_id'] == object_id) & (self.data['passband'] == passband)], ax=ax,
                             palette=sns.light_palette("green", reverse=True, as_cmap=True))
            ax.plot()

        return ax

    def light_curve_interpolate_split_compare(self, object_id, target):
        """
        Plot light curves of a single object with separate chart for each passband
        along with interpolated line representing average3 light curve of the target class
        :param object_id: id of the object for which the curves will be plotted
        :param target: target class of other objects which interpolation will be calculated
        :return: axes array with plotted charts
        """
        passbands = self.data['passband'].sort_values().unique()
        fig, axes = plt.subplots(nrows=len(passbands), ncols=1, sharex=True, sharey=False)
        for i, passband in enumerate(passbands):
            ax = axes[i]
            data = self.data[(self.data['passband'] == passband) & (self.data['target'] == target) & (
                        self.data['object_id'] != object_id)]
            x_pane = np.linspace(min(data['mjd']), max(data['mjd']), 100)
            y_pane = np.poly1d(np.polyfit(data['mjd'], data['flux'], 15, 25))(x_pane)
            sns.lineplot(x=x_pane, y=y_pane, palette=sns.dark_palette("palegreen", as_cmap=True), ax=ax)

            with sns.plotting_context("notebook", rc={"lines.linewidth": 3}):
                sns.lineplot(x="mjd", y="flux", data=self.data[
                    (self.data['object_id'] == object_id) & (self.data['passband'] == passband)], ax=ax,
                             palette=sns.dark_palette("green", reverse=True, as_cmap=True))
            ax.set_title(f"Passband {passband}")
            ax.plot()

        return axes

    def sky_pos(self, object_id):
        """
        Show position of specified object on Aitoff projection of night sky between other objects from dataset
        :param object_id: id of the object for which the curves will be plotted
        :return: ax with drown chart
        """
        from mpl_toolkits.basemap import Basemap

        fig, ax = plt.subplots()
        data = self.data.copy()
        m = Basemap(projection='hammer', lon_0=0, ax=ax)
        data['x'], data['y'] = m(data['ra'].apply(lambda x: x - 180).tolist(), data['decl'].tolist())

        sns.scatterplot(x='x', y='y', hue="object_id", data=data[data['object_id'] != object_id],
                        palette=sns.cubehelix_palette(dark=.6, light=.9, as_cmap=True), ax=ax, linewidth=0)
        sns.scatterplot(x="x", y="y", hue="object_id", data=data[data['object_id'] == object_id], ax=ax,
                        palette=sns.light_palette("palegreen", as_cmap=True), marker='x', linewidth=4, s=400)

        m.imshow(plt.imread('../input/plasticc-merged/mell_rgb_450.jpg', 0))
        ax.set_title(f'Sky placement of the object {object_id} among other objects')
        ax.legend(loc='center right', bbox_to_anchor=(1.1, 0.9), ncol=1)
        fig.show()

        return ax

    def sky_pos_target(self, object_id, target):
        """
        Show position of specified object on Aitoff projection of night sky between other objects in specified target
        :param object_id: id of the object for which the curves will be plotted
        :return: ax with drown chart
        """
        from mpl_toolkits.basemap import Basemap
        fig, ax = plt.subplots()
        data = self.data.copy()
        m = Basemap(projection='hammer', lon_0=0, ax=ax)
        data['x'], data['y'] = m(data['ra'].apply(lambda x: x - 180).tolist(), data['decl'].tolist())

        sns.scatterplot(x='x', y='y', hue="object_id",
                        data=data[(data['object_id'] != object_id) & (self.data['target'] == target)],
                        palette=sns.cubehelix_palette(dark=.6, light=.9, as_cmap=True), ax=ax, linewidth=0)
        sns.scatterplot(x="x", y="y", hue="object_id", data=data[data['object_id'] == object_id], ax=ax,
                        palette=sns.light_palette("palegreen", as_cmap=True), marker='x', linewidth=4, s=400)

        m.imshow(plt.imread('../input/plasticc-merged/mell_rgb_450.jpg', 0))
        ax.set_title(f'Sky placement of the object {object_id} among other objects in target {target}')
        ax.legend(loc='center right', bbox_to_anchor=(1.1, 0.9), ncol=1)
        fig.show()
        return ax

    def galactic_pos(self, object_id):
        """
        Show position of specified object on Aitoff projection of our galaxy between other objects from dataset
        :param object_id: id of the object for which the curves will be plotted
        :return: ax with drown chart
        """
        from mpl_toolkits.basemap import Basemap
        fig, ax = plt.subplots()
        data = self.data.copy()
        m = Basemap(projection='hammer', lon_0=0, ax=ax)
        data['x'], data['y'] = m(data['gal_l'].apply(lambda x: x - 180).tolist(), data['gal_b'].tolist())

        sns.scatterplot(x='x', y='y', hue="object_id", data=data[data['object_id'] != object_id],
                        palette=sns.cubehelix_palette(dark=.6, light=.9, as_cmap=True), ax=ax, linewidth=0)
        sns.scatterplot(x="x", y="y", hue="object_id", data=data[data['object_id'] == object_id], ax=ax,
                        palette=sns.light_palette("palegreen", as_cmap=True), marker='x', linewidth=4, s=400)

        m.imshow(plt.imread('../input/plasticc-merged/Milky_Way_infrared.jpg', 0))
        ax.set_title(f'Galactic placement of the object {object_id} among other objects')
        ax.legend(loc='center right', bbox_to_anchor=(1.1, 0.9), ncol=1)
        fig.show()
        return ax

    def galactic_pos_target(self, object_id, target):
        """
        Show position of specified object on Aitoff projection of our galaxy between other objects in specified target
        :param object_id: id of the object for which the curves will be plotted
        :return: ax with drown chart
        """
        from mpl_toolkits.basemap import Basemap
        fig, ax = plt.subplots()
        data = self.data.copy()
        m = Basemap(projection='hammer', lon_0=0, ax=ax)
        data['x'], data['y'] = m(data['gal_l'].apply(lambda x: x - 180).tolist(), data['gal_b'].tolist())

        sns.scatterplot(x='x', y='y', hue="object_id",
                        data=data[(data['object_id'] != object_id) & (self.data['target'] == target)],
                        palette=sns.cubehelix_palette(dark=.6, light=.9, as_cmap=True), ax=ax, linewidth=0)
        sns.scatterplot(x="x", y="y", hue="object_id", data=data[data['object_id'] == object_id], ax=ax,
                        palette=sns.light_palette("palegreen", as_cmap=True), marker='x', linewidth=4, s=400)

        m.imshow(plt.imread('../input/plasticc-merged/Milky_Way_infrared.jpg', 0))
        ax.set_title(f'Galactic placement of the object {object_id} among other objects in target {target}')
        ax.legend(loc='center right', bbox_to_anchor=(1.1, 0.9), ncol=1)
        fig.show()
        return ax

    def passband_target_interpolate_matrix(self):
        """
        Calculates interpolated line for every target and every passband, then plots them in separate charts
        :return: axes array with plotted charts
        """
        targets = self.data['target'].sort_values().unique()
        passbands = self.data['passband'].sort_values().unique()
        fig, axes = plt.subplots(nrows=len(targets), ncols=len(passbands), sharex=True, sharey=False)
        axes = axes.flatten()
        fig.tight_layout()
        for i, (target, passband) in enumerate(list(itertools.product(targets, passbands))):
            data = self.data[(self.data['passband'] == passband) & (self.data['target'] == target)]
            x_pane = np.linspace(min(data['mjd']), max(data['mjd']), 100)
            y_pane = np.poly1d(np.polyfit(data['mjd'], data['flux'], 15, 25))(x_pane)
            ax = sns.lineplot(x=x_pane, y=y_pane, palette=sns.dark_palette("palegreen", as_cmap=True), ax=axes[i])
            ax.set_title(f"Passband {passband}, target {target}")

        return axes

    def extragalactic_light_curves_split(self):
        """
        Plot every light curve from extragalactic sources, split into targets and passbands
        :return: axes array with plotted charts
        """
        sources = [15, 42, 52, 62, 64, 67, 88, 90, 95]
        passbands = self.data['passband'].sort_values().unique()
        pairs = list(itertools.product(sources, passbands))
        fig, axes = plt.subplots(nrows=len(sources), ncols=len(passbands), sharex=True, sharey=False)
        axes = axes.flatten()
        for i, (source, passband) in enumerate(pairs):
            ax = sns.lineplot(x="mjd", y="flux", hue="object_id",
                              data=self.data[(self.data['passband'] == passband) & (self.data['target'] == source)],
                              ax=axes[i], legend=False)
            ax.set_title(f"source {source} passband {passband}")

    def galactic_light_curves_split(self):
        """
        Plot every light curve from galactic sources, split into targets and passbands
        :return: axes array with plotted charts
        """
        sources = [6, 16, 53, 65, 92]
        passbands = self.data['passband'].sort_values().unique()
        pairs = list(itertools.product(sources, passbands))
        fig, axes = plt.subplots(nrows=len(sources), ncols=len(passbands), sharex=True, sharey=False)
        axes = axes.flatten()
        for i, (source, passband) in enumerate(pairs):
            ax = sns.lineplot(x="mjd", y="flux", hue="object_id",
                              data=self.data[(self.data['passband'] == passband) & (self.data['target'] == source)],
                              ax=axes[i], legend=False)
            ax.set_title(f"source {source} passband {passband}")


# # How to use PlasticcVis module
# 
# Let's assume we have object of interest with id 80067866 and it's target class is 6.
# We want to find out more about it. 
# This module makes this analysis easy.
# 
# 
# 
# First, let's initialize our module with data. Remember that this has to be combined dataset, containing both flux and meta data

# In[ ]:


plt.rcParams["figure.figsize"] = (20, 10)
import pandas as pd

analysis = PlasticcVis(data=pd.read_csv('../input/plasticc-merged/dataset.csv'))
analysis.data.head()


# In[ ]:


object_id = 80067866
target = 6


# Now, let's plot the light curve of our object

# In[ ]:


ax = analysis.light_curve(object_id=object_id)


# Now, how that compares to other objects in it's target, is it an average object or some outlier?

# In[ ]:


ax = analysis.light_curve_compared(object_id=object_id, target=target)


# Ok, so all we can see here is that our object stands out from the rest, but is is far away from average? Let's find out.

# In[ ]:


ax = analysis.light_curve_interpolate_compare(object_id=object_id, target=target)


# since we have 6 passbands in our light curves, let's make the same analysis for each passband separately, so that we get a better view

# In[ ]:


axes = analysis.light_curve_split(object_id=object_id)


# In[ ]:


axes = analysis.light_curve_split_compared(object_id=object_id, target=target)


# In[ ]:


axes = analysis.light_curve_interpolate_split_compare(object_id=object_id, target=target)


# So, clearly we can see that in each passband our object stands out from the average. 
# 
# Let's check the sky position and galactic position of this object (ra and decl)

# In[ ]:


ax = analysis.sky_pos(object_id=object_id)


# Notice the white 'X' in the bottom right part of the chart, that's where our object lies on the sky map. Let's see where on the sky map are objects from our target

# In[ ]:


ax = analysis.sky_pos_target(object_id=object_id, target=target)


# They seem to be distributed equally. Let's do the same for galactic coordinates (gal_l and gal_b)

# In[ ]:


ax = analysis.galactic_pos(object_id=object_id)


# In[ ]:


ax = analysis.galactic_pos_target(object_id=object_id, target=target)


# As an extra, there are some other metrics, not related to specific objects
# 
# First, plot each target's each passband average light curve

# In[ ]:


ax = analysis.passband_target_interpolate_matrix()


# Plot for each target from outside of our galaxy combined light curves

# In[ ]:


ax = analysis.extragalactic_light_curves_split()


# And do the same for targets from our galaxy

# In[ ]:


ax = analysis.galactic_light_curves_split()


# In[ ]:




