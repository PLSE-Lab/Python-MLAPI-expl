#!/usr/bin/env python
# coding: utf-8

# # CPU time vs. I/O time
# *%%time* magic command helps to give CPU times (time that CPU is busy) and Wall time (total time for the script execution). So Wall time - CPU times gives the time that the system is busy elsewhere (time.sleep or time for I/O)
# This example we use time.sleep to simulate that the system is busy elsewhere (not for this program).

# In[ ]:


get_ipython().run_cell_magic('time', '', 'import time\nfor i in range(10):\n    the_sum = sum([j*j for j in range(1000000)])\n    time.sleep(1)')


# Another way to accomplish the same thing is to use time.clock() and time.time().
# time.time gives the Wall time and time.clock() gives the CPU busy time.

# In[ ]:


import time
start_time = time.time()
start_cpu_time = time.clock()
for i in range(10):
    the_sum = sum([j*j for j in range(1000000)])
    time.sleep(1)

print(f'CPU time {time.clock() - start_cpu_time}')
print(f'Wall time {time.time() - start_time}')


# To demonstrate the time for CPU vs. time for I/O we use the web downloading example (most of the Wall time is for I/O - network connection for downloading the website).
# (Credit: This example is from RealPython)

# In[ ]:


get_ipython().run_cell_magic('time', '', 'import requests\nimport multiprocessing\nimport time\n\nsession = None\n\n\ndef set_global_session():\n    global session\n    if not session:\n        session = requests.Session()\n\n\ndef download_site(url):\n    with session.get(url) as response:\n        name = multiprocessing.current_process().name\n        print(f"{name}: Read {len(response.content)} from {url}")\n\n\ndef download_all_sites(sites):\n    with multiprocessing.Pool(initializer=set_global_session) as pool:\n        pool.map(download_site, sites)\n\n\nif __name__ == \'__main__\':\n    sites = ["https://www.jython.org", "http://olympus.realpython.org/dice"]*80\n    start_time = time.time()\n    download_all_sites(sites)\n    duration = time.time() - start_time\n    print(f"Downloaded {len(sites)} in {duration} seconds")')


# In[ ]:


get_ipython().run_line_magic('whois', '')


# In[ ]:




