#!/usr/bin/env python
# coding: utf-8

# # Tips and Tricks for Large Datasets
# Here are a couple of tricks that will help you be more efficient when developing you code.
# Feel free to leave any suggestions in the comment section.

# ## Play Sound After Cell Execution
# Instead of going back and forth checking if a cell has been executed, we will play a sound after the execution if it took longer than X seconds. In this case, I chose 5 seconds.
# 

# In[ ]:


# source: https://stackoverflow.com/q/17323336
from time import time, sleep
from IPython import get_ipython
from IPython.display import Audio, display

class InvisibleAudio(Audio):
    def _repr_html_(self):
        audio = super()._repr_html_()
        audio = audio.replace('<audio', f'<audio onended="this.parentNode.removeChild(this)"')
        return f'<div style="display:none">{audio}</div>'

class Beeper:
    def __init__(self, threshold, **audio_kwargs):
        self.threshold = threshold
        self.start_time = None    # time in sec, or None
        self.audio = audio_kwargs
    def pre_execute(self):
        if not self.start_time:
            self.start_time = time()
    def post_execute(self):
        end_time = time()
        if self.start_time and end_time - self.start_time > self.threshold:
            audio = InvisibleAudio(**self.audio, autoplay=True)
            display(audio, f'execution took {round(end_time - self.start_time)} seconds')
        self.start_time = None

# Customize your params here
beeper = Beeper(5, url='https://www.freesoundslibrary.com/wp-content/uploads/2018/03/tada-sound.mp3')

ipython = get_ipython()
ipython.events.register('pre_execute', beeper.pre_execute)
ipython.events.register('post_execute', beeper.post_execute)


# ### Example

# In[ ]:


sleep(6)


# ## Version Control
# Making a commit on a Kernel can take forever.  In order to make commits without having to run the kernel each time, you can click 'Cancel Run' right after clicking 'Commit'
