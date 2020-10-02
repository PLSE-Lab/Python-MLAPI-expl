#!/usr/bin/env python
# coding: utf-8

# # Using Libpostal on Windows Through a Virtual Machine with Jupyter Notebook

# ## A powerful library for address parsing and expansion
# 
# [Libpostal](https://github.com/openvenues/libpostal) is an extremely powerful C library built on NLP[](http://) for parsing and normalizing addresses from any country.
# 
# It takes a considerable amount of effort when dealing with address data from multiple sources to either parse them into their components (house number, PO box, street, city, etc.,) or expand them so they follow a consistent format. 
# 
# Having a single library do it with a high degree of accuracy can save you an immense amount of time and in my experience Libpostal by [openvenues](https://github.com/openvenues) has been a godsend.
# 
# #### Parsing Addresses
# 
# ![From the Libpostal GitHub](https://cloud.githubusercontent.com/assets/238455/24703087/acbe35d8-19cf-11e7-8850-77fb1c3446a7.gif)
# 
# #### Expanding Addresses
# 
# ![From the Libpostal GitHub](https://cloud.githubusercontent.com/assets/238455/14115012/52990d14-f5a7-11e5-9797-159dacdf8c5f.gif)

# ## Using it on Windows
# 
# Libpostal was built for Unix environments, so naturally, I ran into several issues when I tried installing it on Windows using MSys2/MinGW. I'm quite new to Unix environments so Googling every issue led me down a wormhole of troubleshooting articles that came with their own set of issues, and the suggested fixes never seemed to end!
# 
# Here's a taste of what that's like:
# * https://github.com/openvenues/libpostal/issues/219
# * https://github.com/openvenues/libpostal/issues/69
# * https://github.com/openvenues/libpostal/issues/64
# * https://github.com/openvenues/libpostal/issues/347
# * https://github.com/openvenues/libpostal/issues/339
# 
# Finally, I figured that the most straightforward thing to do would be to install Libpostal on an Ubuntu Server instance running on a virtual machine, and then access the library through a Jupyter Notebook server running on the same virtual machine.
# 
# This is a step-by-step guide on how to get all of that set up.

# ### Step 1) Install Oracle VM VirtualBox
# 
# Download [Oracle VM VirtualBox](https://www.virtualbox.org/wiki/Downloads) and install it on your computer.

# ### Step 2) Download Ubuntu Server 18.10
# 
# Download an [Ubuntu Server 18.10](https://www.ubuntu.com/download/server) image. I selected Ubuntu Server over an Ubuntu GUI for the lower resource consumption. After all, I'd rather dedicate more resources to the address handling than the hosting of a virtual machine.

# ### Step 3) Create a virtual machine instance
# 
# Start up Oracle VM VirtualBox and click "New" on the main toolbar.
# 
# Give your OS a name and select "Linux" as the Type and "Ubuntu (64-bit)" as the Version.
# 
# ![](https://i.imgur.com/uWkz9O1.png)
# 
# Dedicate at least 4 GB of RAM to your virtual machine.
# 
# ![](https://i.imgur.com/q2e2D54.png)
# 
# Create a virtual hard disk, select "VDI", select "Dynamically Allocated" and give it at least 10 GB of file space (depending on the size of the datasets you wish to analyze).
# 
# ![](https://i.imgur.com/oporLaM.png)
# 
# ![](https://i.imgur.com/X9kPWGA.png)
# 
# ![](https://i.imgur.com/2eX3AG2.png)
# 
# ![](https://i.imgur.com/NUuxtlc.png)

# ### Step 4) Install Ubuntu Server on your virtual machine
# 
# Select your virtual machine from the sidebar on Oracle VM VirtualBox and hit Start. You'll be prompted to select a start-up disk. Select the Ubuntu Server 18.10 image that you downloaded in Step 2.
# 
# ![](https://i.imgur.com/8wDui1T.png)
# 
# Follow the installation prompts, selecting the default (pre-selected) options as shown below.
# 
# ![](https://i.imgur.com/XKjO7wW.png)
# 
# ![](https://i.imgur.com/ie4xPlt.png)
# 
# ![](https://i.imgur.com/RJoXjSV.png)
# 
# ![](https://i.imgur.com/V1wn7eA.png)
# 
# ![](https://i.imgur.com/e94AWn0.png)
# 
# ![](https://i.imgur.com/ov3tqnd.png)
# 
# ![](https://i.imgur.com/fAXDzQv.png)
# 
# ![](https://i.imgur.com/oeAvwXW.png)
# 
# ![](https://i.imgur.com/kROYGYg.png)
# 
# ![](https://i.imgur.com/wotT00Y.png)
# 
# ![](https://i.imgur.com/Ixnn7ww.png)
# 
# ![](https://i.imgur.com/jujdk2H.png)
# 
# Give it some time to install. Once it's done, you'll see a "Reboot Now" option. Hit Enter and wait for it to start up. If you see a message with "Please remove the installation medium, then press ENTER:", just hit Enter. The first start up will take some time. 
# 
# If you see "[  OK  ] Reached target Cloud-init target" and there's been nothing going on for some time, the installation is most likely over and awaiting your input. Hit enter to proceed to login.
# 
# Once you're at the login screen, enter in the username and password you defined during the installation and log in to the system.

# ### Step 5) Follow the installation instructions for Ubuntu/Debian on the Libpostal GitHub page
# 
# Enter in each of commands below sequentially. Wait for the command to run and for the server to return you to the user prompt before entering in the next command.
# 
# `sudo apt-get install curl autoconf automake libtool pkg-config
# git clone https://github.com/openvenues/libpostal
# cd libpostal
# ./bootstrap.sh
# ./configure --datadir="/home/[EnterYourUsernameHere]"
# make -j4
# sudo make install
# sudo ldconfig`
# 
# Each of these may take quite some time to run as there are over 2 GB of files that need to be downloaded and compiled.
# 
# You will be prompted for your password after running the first (sudo) command. Make sure to enter that in.

# ### Step 6) Install Python and PIP
# 
# Since we'll be using the [Libpostal's Python wrapper](https://github.com/openvenues/pypostal) to use the library with Python, we first need to install Python 3 and PIP, python's package management tool.
# 
# Run the following commands:
# 
# `sudo apt-get install python3
# sudo apt-get install python3-pip`
# 
# Afterwards, make sure Python 3 and PIP are installed by typing in:
# 
# `python3 --version
# pip3 --version`
# 
# If you see a version number instead of an error message, you've successfully installed Python 3 and PIP.

# ### Step 7) Install all required Python libraries including Jupyter and the Libpostal wrapper
# 
# Install the Python libraries you require as well as Jupyter and [pypostal](https://github.com/openvenues/pypostal) (the libpostal Python wrapper) using the command below (I've included only numpy and pandas as the other libraries, feel free to add more to the list).
# 
# `pip3 install pandas numpy jupyter postal`

# ### Step 8) (OPTIONAL) Edit the Jupyter configuration file for easier access
# 
# When you try to access a Jupyter server instance, you're often prompted for a token provided by the server. When running Jupyter on a virtual machine, it's quite cumbersome to copy the token since it's a long string of random characters. In order to avoid having to copy and type in a long string each time, we can edit the Jupyter config file to allow us to use a string of our choice.
# 
# Navigate back to the home folder (use `pwd` to check where you are and `cd ..` to move back) and type in the following command:
# 
# `jupyter notebook --generate-config`
# 
# The config file will automtically be saved in `/home/[YourUsernameHere]/.jupyter`. Navigate to this folder using the following command:
# 
# `cd /home/[YourUsernameHere]/.jupyter`
# 
# We'll need to edit the config file stored in this folder using VIM, Ubuntu Server's built-in text editor. Type in the following to enter the file.
# 
# `vim jupyter_notebook_config.py`
# 
# You'll then see something similar to this:
# 
# ![](https://i.imgur.com/RgWqanA.png)
# 
# When you're on this screen, type `?` followed by `c.NotebookApp.token`. This will automatically search for the c.NotebookApp.token parameter in the file. This parameter determines the token you use to access the jupyter server.
# 
# Hit Enter to end your search, hit ESC and then hit `i` to enter edit mode. Then edit whatever is between the single quotes to enter in a token of your choice, similar to what I've done below.
# 
# ![](https://i.imgur.com/86r7zOn.png)
# 
# Once done, hit ESC again and type in `:wq` to save and close the file. Navigate back to the home folder using `cd ..`

# ### Step 9) Change your virtual machine's network from NAT to Host-Only Adapter
# 
# To access a Jupyter server instance running on your virtual machine from a browser on your host machine, you'll first need to adjust your virtual machine's network settings.
# 
# Right click on your virtual machine on the Oracle VM VirtualBox sidebar, select Network from the sidebar and select "Host-only adapter" in the dropdown next to "Attached to:". You do not need to shutdown your virtual machine before doing this.
# 
# ![](https://i.imgur.com/nOLSbZc.png)
# 
# ![](https://i.imgur.com/R6s0TmX.png)
# 
# Please note that while doing this will allow you to access the Jupyter server running on your virtual machine, you will lose access to the internet (on your virtual machine) and will not be able to install any tools or libraries. To do so, switch back to "NAT" in the network settings, install the tools or libraries you require, and switch back to "Host-only Adapter" before you start the Jupyter server.

# ### Step 10) Take note of your virtual machine's IP address
# 
# Type in the following command and note down your virtual machine's IP address:
# 
# `ip addr show`
# 
# The address will be similar to `192.168.56.123/24` and will be between the words `inet` and `brd`. We're only interested in the digits before the `/24`. We're not interested in the 127.0.0.1 address displayed above this.

# ### Step 11) Start a Jupyter server instance
# 
# Type in the following command to start a Jupyter server instance:
# 
# `jupyter notebook --no-browser --ip=0.0.0.0`
# 
# On the browser of your host machine, type in the IP address you noted down earlier followed by the port number provided by the Jupyter server (contained in the messages displayed after running the command above; usually 8888) e.g. `192.168.56.123:8888`.
# 
# You'll be prompted for a token. If you followed Step 8), you can enter in the token you specified in the config file. If not, enter in the token contained in the messages displayed after running the command above.
# 
# ![](https://i.imgur.com/sYfOAjk.png)
# 
# If you've entered in the token correctly, you'll then gain access to the Jupyter server instance running on your virtual machine through a browser on your host machine.
# 
# ![](https://i.imgur.com/eAglGUb.png?1)

# ### Step 12) Enjoy using Libpostal on Windows via a Jupyter notebook server on a virtual machine
# 
# The files and folders you'll see on the Jupyter server are those contained in your virtual machine. If you wish to upload a file from your host machine to your virtual machine, use Jupyter's upload feature on the main toolbar.
# 
# Once you've opened a Jupyter notebook instance, you can use the following commands to import Libpostal's primary functions for address expansion and parsing.
# 
# `from postal.parser import parse_address
# from postal.expand import expand_address`
# 
# If you've followed the instructions correctly, this code will run with no issues. You can then use each of the functions as you wish.
# 
# ![](https://i.imgur.com/FH19emI.png)

# ---

# ***Many, many thanks to openvenues and every single person who contributed to the Libpostal library for making our lives as data analysts so much easier. We truly stand on the shoulders of giants to get our job done.***
