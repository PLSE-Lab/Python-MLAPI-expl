#!/usr/bin/env python
# coding: utf-8

# # Notes on POSIX find
# 
# * `find` is the POSIX tool for finding files matching criteria of interest. Trey used it to good effect in several different cases at work, so I've decided I need to sit down and spend a bit more time familiarizing myself with it.
# * The basic format of `find` is:
# 
#   ```
#   find [-H] [-L] [-P] path... [expression]
#   ```
#   
#   The arguments configure symlink following behavior and are broadly unimportant.
#   
#   One path is required, many are possible (multidirectory find). Paths are prefixed with switches controlling what attribute is being searched: `-name` for the filename, `-path` for the absolute path.
# * Boolean operators are `-and` (AND), `-or` (OR), and `-not` (NOT).
# * To specify a search that is for a specific type use the `-type` switch. The two options that are used in practice are `directory` and `file`.
# * To specify that a certain directory be excluded append `-prune` as a suffix. E.g. `find . -name '*.js' -path '/js/' -prune`.
# * The default output action is to print the name of the file (this is `-print`). You can do more, e.g. specify `-ls` to print extended file information.
# * You can specify a command to be run on the input filename using `-exec`. The syntax for the command that follows is _really weird_:
# 
#   ```
#   find . -name 'Results.js' -exec head {} \;
#   ```
#   
#   The quoting of the semicolon `;` is required to escape it out of shell.
#   
#   This is extremely hand when used in combination with `grep` and logs:
#   
#   ```
#   find . -name '*.conf' -exec grep host '{}' \;
#   ```
#   
#   This finds all lines in configuration files with `host` in them.
