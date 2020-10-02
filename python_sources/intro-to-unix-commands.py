#!/usr/bin/env python
# coding: utf-8

# # Introduction to Unix Commands
# ---
# 
# Computers organize files in a hierarchical structure, composed of folders (which we refer to as **directories**) and files.  For instance, you probably have a **directory** containing documents on your computer, corresponding to **.txt** and **.doc** files, among other file types.  This directory is likely organized in several directories, for example corresponding to different seasons (e.g., **Winter 18**, **Spring 18**, etc) or quarters of the year (e.g., **Q1 2018**, **Q2 2018**, etc).  
# 
# ![file hierarchy](https://i.imgur.com/7JdB8Cq.png)
# 
# This directory of documents is contained in another directory, which is contained inside another directory, and so on.  In fact, *all* of the files on your computer are organized similarly, in a hierarchical file structure.
# 
# In this tutorial, you will learn how to navigate and explore the file structure on a computer through a **[command-line interface](https://en.wikipedia.org/wiki/Command-line_interface) (CLI)**.  You will also learn how to make changes to the file structure directly from the CLI.
# 
# ### Getting started
# ---
# 
# Begin by opening the CLI on your computer.  
# - If you are using a Mac, the CLI is an application called **Terminal**.  You can find it in the **Applications** folder.
# - If you are using a Windows machine, the CLI is a program called **Command Prompt**.  The instructions for accessing it vary with the version of Windows you're using; see [this reference](https://www.lifewire.com/how-to-open-command-prompt-2618089) for more detailed instructions.
# - If you are using a Linux machine with Ubuntu, press **Ctrl-Alt-T** to open the CLI. 
# 
# If using a Mac, the CLI that you've opened should look similar to the image below.  
# 
# ![CLI on Mac](https://i.imgur.com/jB12VVc.png)
# 
# If you're using a different operating system, the CLI will look slightly different -- but thankfully the commands are identical for all operating systems!
# 
# ### Introduction to file paths
# ---
# 
# Before we use the CLI, we'll need to discuss some preliminaries.
# 
# As you've learned, all of the files on your computer are arranged as a hierarchy of files; most directories are themselves contained inside a directory, which is contained inside another directory, and so on.  However, there is one, special directory that contains all of the files and directories on your computer, and we refer to this directory as the **root directory**.
# 
# Furthermore, we refer to any directory that is contained inside another directory as a **subdirectory** of that directory.  For instance, in the image at the top of this page, `Winter 18` is a **subdirectory** of the `Documents` directory.  
# 
# We can specify the location of a file (or directory) with its (**absolute**) **file path**, which describes how to arrive at the file (or directory) by navigating through the file structure, using the root directory as a starting point.  For instance, on my Mac, the `Documents` directory is located at the following **file path**:
# ```
# /Users/alexiscook/Documents
# ```
# Note that:
# - We always read **file paths** from left to right.
# - The slash (`/`) at the beginning of the file path denotes the **root directory**.
# - `Users` appears first, showing that `Users` is a **subdirectory** of the **root directory**.  Likewise, `alexiscook` is a **subdirectory** of `Users`, and `Documents` is a **subdirectory** of `alexiscook`.
# 
# But we need not always reference the location of a file, using the **root directory** as a starting reference.  In fact, the CLI maintains a second point of reference that is modifiable, and that we can specify as a starting point (in place of the **root directory**) when specifying a **file path**.  We refer to this point of reference as the **working directory**.  Furthermore, any **file path** expressed with the **working directory** as reference is known as a **relative file path**. We'll explore these topics further in the next section.
# 
# ### `pwd` -- prints working directory name
# ---
# 
# To begin, type `pwd` into the CLI, and press **Return** (or **Enter**).
# ![pwd output](https://i.imgur.com/0u1jcYw.png)
# 
# This command lists the current **working directory**.  When opening a new CLI window on your machine, the **working directory** will always be initialized to the same value.  We refer to this initial working directory as the **home directory**. 
# 
# In this case, as can be seen in the picture above, my **home directory**, and the current value of my **working directory**, is `/Users/alexisbcook`.
# 
# As discussed in the previous section, we can use the **working directory** to specify a **relative file path**.  For instance, the **relative file path** for the `Documents` directory is:
# ```
# Documents
# ```
# Note that:
# - As with **absolute file paths**, **relative file paths** are read from left to right.
# - All file paths that don't begin with a slash (`/`) are **relative file paths**, and so the remaining information in the file path indicates how to navigate from the **working directory**.
# - `Documents` appears first, indicating that `Documents` is a subdirectory of the **working directory**.
# 
# Note that **relative file paths** are generally shorter (and, for this reason, often more convenient) than **absolute file paths**.  For this reason, we tend to use **relative file paths** more often!
# 
# ### `ls` -- lists the contents of a directory
# ---
# Next, run `ls` in the CLI.  This command lists the set of all files and directories in the **working directory**.
# 
# ![pwd output](https://i.imgur.com/6wdrV8y.png)
# 
# ### `cd` -- changes the working directory
# ---
# 
# Run `cd /`.  This changes the working directory from the **home directory** to the **root directory**.  (_Remember that the **root directory** is denoted by `/`._)  
# 
# > The `cd` command is always followed by either an **absolute** or **relative file path** containing the location of the directory that we'd like to be the new **working directory.** 
# 
# Next, list the files in the root directory by running `ls`.  
# 
# Remember that the **home directory** on my machine was `/Users/alexiscook`, and so my root directory should contain a `Users` directory -- check this for yourself in the output provided in the image below.
# 
# Next, on your machine, use the `cd` command to change the **working directory** back to your **home directory**.  To do this, you can use the `cd` command to enter the corresponding directories, one subdirectory at a time.  For instance, in my case, I enter the `Users` subdirectory by running `cd Users`, and from there, I need only enter the `alexiscook` subdirectory by running `cd alexiscook`.  
# 
# Once you've finished, list the files in the **home directory** by typing `ls`.  Verify that the contents match the output returned when we listed the contents of the home directory in the previous section.
# 
# ![cd output](https://i.imgur.com/xkwUKXC.png)
# 
# Note that it's possible (and preferred) to move between multiple directories at once; for instance, instead of first navigating to the `Users` directory (with `cd Users`) and then later navigating to the `alexiscook` directory in a later command (`cd alexiscook`), we could have instead executed a single command to accomplish both moves at once: `cd Users/alexiscook`.
# 
# So far, you've learned how to move **down** a hierarchical structure of files: so, you've learned how to enter subdirectories (and subdirectories of subdirectories).  But it's also possible to quickly move **up** the hierarchy -- in other words, we can change the **working directory** from one directory, to the directory containing it.
# 
# For instance, to navigate up ONE directory, we run 
# ```
# cd ..
# ```
# where the two dots (`..`) refer to the directory immediately above the working directory.  Note that this is a **relative file path**.
# 
# Likewise, to navigate up TWO directories, we could run 
# ```
# cd ../..
# ```
# 
# Similarly, we can combine the dots (`..`) with what we already know about **relative file paths** to trace more complex paths.  For instance, say my current **working directory** is set to the **home directory** (`/Users/alexisbcook`).  Then, running `cd ../Shared` will change my working directory to `/Users/Shared`.  
# 
# As a final note, the **home directory** is denoted by `~`, and you can quickly change the **working directory** to the **home directory** by running
# ```
# cd ~
# ```

# ### `mv` -- moves or renames a file
# ---
# 
# To move a file to a new directory, we use a command like the following: 
# ```
# mv path/to/file path/to/new_directory/
# ```
# 
# For instance, consider the image below.
# 
# ![sample file structure](https://i.imgur.com/yYQ4oZ6.png)
# 
# In this picture, the home directory (`~`) has two subdirectories `dir_1` and `dir_2`.  `dir_1` contains one text file (`file_a.txt`), whereas `dir_2` holds two text files (`file_b.txt`, `file_c.txt`).
# 
# Say that the **working directory** in the pictured example is the home directory.  Then, to move `file_b.txt` to the `dir_1` directory, we need only run: 
# ```
# mv dir_2/file_b.txt dir_1
# ```  
# This results in the following file structure:
# 
# ![result of `mv dir_2/file_b.txt dir_1`](https://i.imgur.com/2j0cL1t.png)
# 
# Working from the same example, to change the name of `file_c.txt` to `file_d.txt`, we need only run: 
# ```
# mv dir_2/file_c.txt dir_2/file_d.txt
# ```
# This "move" is effectively just a renaming of the original file (since we have not moved it to a new directory).
# 
# ### `cp` -- copies a file
# ---
# To copy a file, we use:
# ```
# cp path/to/old_file path/to/new_file
# ```
# 
# In the image below, you can see an example usage of this command; the file structure depicted on the left shows how the files were organized before the command was run, and the structure in the box on the right shows the result of running the command.
# ![example for running `cp`](https://i.imgur.com/VkroaHD.png)
# > **IMPORTANT NOTE**: For all examples relating to the hypothetical file structure pictured above, we assume the **working directory** is the **home directory** (`~`). 
# 
# 
# ### `rm` -- removes a file
# ---
# To remove a file, we use:
# ```
# rm path/to/file
# ```
# 
# ![example for running `rm`](https://i.imgur.com/6PJ6dOS.png)
# 
# ### `mkdir` -- makes a new directory
# ---
# To make a new directory, we use:
# ```
# mkdir path/to/new_directory/
# ```
# ![example for running `mkdir`](https://i.imgur.com/mD0Zt0k.png)
# 
# ### `rmdir` -- removes a directory
# ---
# To remove a directory, we use:
# ```
# rmdir path/to/directory/
# ```
# ![example for running `rmdir`](https://i.imgur.com/UDtjpq9.png)

# ### `man` -- displays manual pages
# ---
# In this tutorial, we have only scratched the surface of available Unix commands.  You can peruse a list of many more commands [here](https://en.wikipedia.org/wiki/List_of_Unix_commands).
# 
# Unix also has extensive documentation that can be directly accessed from the CLI.  For instance, to look up the documentation for `ls`, you need only run: `man ls`.  This should return output similar to the picture below.  
# 
# ![`man ls` output](https://i.imgur.com/N5cLhj1.png)
# 
# We refer to the documentation as a [**man page**](https://en.wikipedia.org/wiki/Man_page) (short for **manual page**).  
# 
# The *__NAME__* section is a quick description of what the command does; in this case, it tells us something we already know: `ls` is used to list the contents of a directory.
# 
# It is often the case that all of the documentation won't fit on the screen; to move up and down the page, you can push the UP and DOWN buttons on your keyboard.  
# 
# Scrolling DOWN the **man page** for `ls` reveals that there are several options that can be used to customize the behavior of `ls`.
# 
# ![`ls` options](https://i.imgur.com/uoJaS0l.png)
# 
# (Note that the complete list of available options is listed in the *__SYNOPSIS__* section towards the top of the **man page**.)
# 
# For instance, if you scroll through the options (which are listed in alphabetical order), you'll notice that running `ls` with `-l` should return the list of files in "long format".
# 
# ![`ls -l` option](https://i.imgur.com/EaPkDOv.png)
# 
# To test this new option, we need to first type `q` to quit and return to the command prompt.
# 
# Once back at the command prompt, type `ls -l`.  On my machine, running this command from the home directory yields the following:
# 
# ![`ls -l` output](https://i.imgur.com/KXXCQ1O.png)
# 
# In this case, the output is far more descriptive than when we ran `ls` earlier in the tutorial; among other information, it returns the total size of each directory.
# 
# Note that it's possible to run Unix commands with multiple options at once: for instance, running `ls -la` in the prompt will return the list in long format (option `-l`) AND also include directory entries whose names begin with a dot (option `-a`).
