#!/usr/bin/env python
# coding: utf-8

# # 1 - Introduction
# 
# The ARC competition host, Francois Chollet, suggested in his paper "On the Measure of Intelligence" (https://arxiv.org/abs/1911.01547) that a Domain Specific Language and program synthesis may be a good way to approach this challenge. A number of notebooks were shared with good looking DSLs, capable of encoding the solution to a number of tasks, but it quickly became apparent that the program synthesis part was going to be very hard.
# 
# It would seem ideal to create a DSL where every command does one small thing well. The power of the approach would then be in the program synthesis. However, given the time constraints for the competition and without a background in program synthesis, I thought I'd try something a little different. By making the commands do more, making them more like macros, the composition of commands became easier. This approach would likely limit the overall generality of the solution, but I hoped that as a proof of concept, it would provide a feasible approach to the challenge within the time constraints.
# 
# My initial inspiration came from noticing that many tasks require the input to be split into panels/tiles/objects and then either one panel is selected as the output, or all the panels are combined; for example by a logical operation. The split-filter-combine commands in combination therefore attacked the output smaller than input class of tasks. Although the proof of concept has been extended to have a go at all tasks, the initial split-filter-combine methodology is still very obvious in the program search code.
# 
# The code below is something of a work in progress. There are todos left to do, comments still to be written, etc. But I hope it will provide some insight and inspiration, as I'm sure  further work on ARC will continue.
# 
# ### 1.1 - Results and Points for Next Time
# 
# * Training: 69
# * Evaluation: 61
# * Private Test: 4 (LB 0.96)
# 
# Solving 4 leaderboard tasks shows this approach can work, but falls a long way short of what's need to "solve" ARC completely. However, it does show that a DSL employing a simple search approach cannot do very well on the ARC challenge, which was really the point of this competition.
# 
# Key things I'd do differently next time.
# 
# * The DSL needs to be simpler and more generic. 
# * The DSL should have a reverse operation for each operation, to enable bi-directional search. 
# * The interpreter's datastructure should contain hierarchical array data and keep more metadata. 
# * Much more work is needed on the program synthesis part of the approach. Starting with a proper review of the current SOTA.
# * When searching for programs to solve a task, given there are errors in some training examples, allowing programs that solve all but one of the training tasks would be more robust.
# * The approach should find the 3 most likely programs to solve the task, not just the first.

# In[ ]:


import os
import sys
import time
import operator

import json
import numpy as np
import pandas as pd

from pathlib import Path
from tqdm.notebook import tqdm
from itertools import permutations
from collections import Counter, namedtuple
from copy import deepcopy

import datetime

import matplotlib.pyplot as plt
from matplotlib import colors

from multiprocessing import Pool


# In[ ]:


# From: https://www.kaggle.com/nagiss/abstraction-and-reasoning-view-all-data

cmap = colors.ListedColormap(
    ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',
     '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
norm = colors.Normalize(vmin=0, vmax=9)
# 0:black, 1:blue, 2:red, 3:greed, 4:yellow,
# 5:gray, 6:magenta, 7:orange, 8:sky, 9:brown
plt.figure(figsize=(5, 2), dpi=200)
plt.imshow([list(range(10))], cmap=cmap, norm=norm)
plt.xticks(list(range(10)))
plt.yticks([])
plt.show()


# # 2 - Macro Domain Specific Language
# 
# This DSL uses the syntax of: command sub-command arg1 arg2 ...
# 
# The interpreter keeps a list of numpy arrays as global data. The list is populated with an input array when program execution begins and all commands act on this global array. When a program has been executed it is expected that a single array is left in the list, and that array forms the answer to the input challenge.
# 
# The DSL is specified below in a dict, as a set of commands, with sub-commands, sub-commands with arguments. The arguments have allowed value lists. The intention is that the data structure can be used to help automate the program search step.
# 
# In hindsight, the current proof of concept doesn't make good use of the dsl dict for error handling. Also the DSL definition could have usefully had information about whether a command can increase of decrease the number of objects in the working list, or whether it will leave the length of that list the same, as such information could have helped guide the program synthesis.
# 
# In this proof of concept, error handling is mostly via asserts, so execution of a program should be within a try-catch.

# In[ ]:


# Some constants used throughout. Note that although the input and output data only contains values within range(10), some
# special values are used in this code. See map_special_values later for definitiions.
MAX_VALUE = 14
A_LOT = 20
TOO_MANY = 100
LIKELY_BACKGROUNDS = [0,10,11,12,13]


# In[ ]:


dsl = dict()

CmdDef = namedtuple('CmdDef', 'subcmddict, description')
SubCmd = namedtuple('SubCmd', 'arglist, description')
CmdArg = namedtuple('CmdArg', 'placeholder, allowed_values')

### Command: identity
dsl['identity'] = CmdDef(dict(), "The identity command applies the identity transform to all arrays in the working list. That is it does nothing. Just for testing, and explanation.")


### Command: abstract
# TODO - Turn this into a scale command and add upscaling.
abstract_subcmddict = dict()
abstract_simple_arglist = [CmdArg('output_size', range(1,6))]
abstract_subcmddict['simple'] = SubCmd(abstract_simple_arglist, "Abstract each array in the working list to a square array of the given size, containing just the most common value from the array.")
dsl['abstract'] = CmdDef(abstract_subcmddict, "The abstract command reduces each array in the working list down to some more abstract representation.")


### Command: assemble
assemble_subcmddict = dict()
assemble_original_arglist = [CmdArg('base', ['original', 'zeros', 'majority_value'])]
assemble_subcmddict['original'] = SubCmd(assemble_original_arglist, "Assemble all arrays in the working list by re-writing them back into their original positions, starting with the original input.")
assemble_auto_grid_arglist = []
assemble_subcmddict['auto_grid'] = SubCmd(assemble_auto_grid_arglist, "Assemble all arrays in the working list by writing them into a new grid, of automatically determnied configurationn.")
assemble_histogram_arglist = [CmdArg('flip', ['none', 'lr', 'ud']),
                              CmdArg('rot90', range(4))]
assemble_subcmddict['histogram'] = SubCmd(assemble_histogram_arglist, "For each panel, take the majority value and create a historgram, with option flip and rotate.")
dsl['assemble'] = CmdDef(assemble_subcmddict, "The assemble command builds an output input from the arrays in the working list by various methods.")


### Command: combine
combine_subcmddict = dict()
combine_logical_op_arglist = [CmdArg('pre_invert', [True, False]),
                               CmdArg('logical_op', ['and', 'or', 'xor']),
                               CmdArg('post_invert', [True, False]),
                               CmdArg('final_colour', range(10))]
combine_subcmddict['logical_op'] = SubCmd(combine_logical_op_arglist, "Combine all arrays in the working list into a single array by performing logical operations.")
combine_overwrite_arglist = [CmdArg('transparent', range(11)),
                             CmdArg('permutation', None)]
combine_subcmddict['overwrite'] = SubCmd(combine_overwrite_arglist, "Combine all arrays in the working list by writing arrays in the given permutation into an output array. Later arrays overwrite early arrays. The given transparent value allows the previous output array to be seen through.")
dsl['combine'] = CmdDef(combine_subcmddict, "The combine command combines all arrays in the working list into a single array by various methods.)")


### Command: filter
filter_subcmddict = dict()
filter_by_value_arglist = [CmdArg('action', ['remove', 'keep']),
                           CmdArg('value', list(reversed(range(MAX_VALUE)))),
                           CmdArg('condition', ['most', 'least', 'odd_one_out'])]
filter_subcmddict['by_value'] = SubCmd(filter_by_value_arglist, "Updates the working list to keep/remove the panels with the most/least occurences of the given value.")
filter_by_not_value_arglist = [CmdArg('action', ['remove', 'keep']),
                               CmdArg('value', list(reversed(range(MAX_VALUE)))),
                               CmdArg('condition', ['most', 'least', 'odd_one_out'])]
filter_subcmddict['by_not_value'] = SubCmd(filter_by_not_value_arglist, "Updates the working list to keep/remove the panels with the most/least occurences of any value except the given value.")
filter_by_value_gte_arglist = [CmdArg('action', ['remove', 'keep']),
                               CmdArg('value', list(reversed(range(MAX_VALUE)))),
                               CmdArg('threshold', range(5))]
filter_subcmddict['by_value_gte'] = SubCmd(filter_by_value_gte_arglist, "Updates the working list to keep/remove the panels with greater than or equal to the given threshold of the given value.")
filter_by_majority_value_arglist = [CmdArg('action', ['remove', 'keep']),
                                    CmdArg('condition', ['most', 'least'])]
filter_subcmddict['by_majority_value'] = SubCmd(filter_by_majority_value_arglist, "Updates the working list to keep/remove the panels where majority value is is the most or least across the working list.")
filter_by_size_arglist = [CmdArg('action', ['remove', 'keep']),
                          CmdArg('condition', ['most', 'least', 'odd_one_out'])]
filter_subcmddict['by_size'] = SubCmd(filter_by_size_arglist, "Updates the working list to keep/remove the panels with largest or smallest size.")
filter_unique_values_arglist = [CmdArg('action', ['remove', 'keep']),
                                CmdArg('condition', ['most', 'least', 'odd_one_out'])]
filter_subcmddict['unique_values'] = SubCmd(filter_unique_values_arglist, "Updates the working list to keep/remove the panels with the most/least unique values.")
filter_by_index_arglist = [CmdArg('action', ['remove', 'keep']),
                           CmdArg('index', range(30))]
filter_subcmddict['by_index'] = SubCmd(filter_by_index_arglist, "Updates the working list to keep/remove the panels with the given index.")
filter_by_shape_count_arglist = [CmdArg('action', ['remove', 'keep']),
                                  CmdArg('shape', ['cross', 'x', 'enclosure']),
                                  CmdArg('background', LIKELY_BACKGROUNDS),
                                  CmdArg('condition', ['most', 'least', 'odd_one_out'])]
filter_subcmddict['by_shape_count'] = SubCmd(filter_by_shape_count_arglist, "Updates the working list to keep/remove the panels with the most/least occurences of the given object.")
filter_commonality_arglist = [CmdArg('action', ['remove', 'keep']),
                              CmdArg('most_or_least', ['most', 'least'])]
filter_subcmddict['commonality'] = SubCmd(filter_commonality_arglist, "Updates the working list to keep/remove the most or least common panel.")
filter_has_symmetry_arglist = [CmdArg('action', ['remove', 'keep'])]
filter_subcmddict['has_symmetry'] = SubCmd(filter_has_symmetry_arglist, "Updates the working list to keep/remove the panels with, depending on the argument given.")
filter_rectangular_arglist = [CmdArg('action', ['remove', 'keep']),
                               CmdArg('min_size', range(2,5))]
filter_subcmddict['rectangular'] = SubCmd(filter_rectangular_arglist, "Updates the working list to keep/remove the panels which are rectangular blocks.")
filter_enclosed_arglist = [CmdArg('action', ['remove', 'keep'])]
filter_subcmddict['enclosed'] = SubCmd(filter_enclosed_arglist, "Updates the working list to keep/remove the panels which are enclosed within another panel.")
dsl['filter'] = CmdDef(filter_subcmddict, "The filter command considers the working list and keeps or removes panels according to the given criteria.")


### Command: move
move_subcmddict = dict()
move_by_value_arglist = [CmdArg('value', list(reversed(range(MAX_VALUE)))),
                         CmdArg('direction', ['N', 'E', 'S', 'W']),
                         CmdArg('distance', range(10))]
move_subcmddict['by_value'] = SubCmd(move_by_value_arglist, "Updates the top corners of the working list to move objects by the given amount depending on their main colour.")
move_by_shape_arglist = [CmdArg('dimension', ['H', 'V', 'HV']),
                         CmdArg('direction', ['SE', 'NW'])]
move_subcmddict['by_shape'] = SubCmd(move_by_shape_arglist, "Updates the top corners of the working list to move objects by their width or height depending on the given dimension. Objects are move in a south/east diretion (positive) or north/west direction depending on the arguments given.")
dsl['move'] = CmdDef(move_subcmddict, "Move objects in the working list.")


### Command: replicate
replicate_subcmddict = dict()
replicate_and_merge_arglist = [CmdArg('flip', ['none', 'lr', 'ud', 'all']),
                               CmdArg('rotation', [True, False]),
                               CmdArg('offset', ['none', 'auto'])]
replicate_subcmddict['and_merge'] = SubCmd(replicate_and_merge_arglist, "Replicate each panel, with option flip and rotation operations, them immediately merge back into the panel.")
replicate_flower_flip_arglist = [CmdArg('start_pos', range(4))]
replicate_subcmddict['flower_flip'] = SubCmd(replicate_flower_flip_arglist, "Replicate each panel creating a 2x2 grid of flips. Just the starting position needs to be specified.")
replicate_flower_rotate_arglist = [CmdArg('start_pos', range(4))]
replicate_subcmddict['flower_rotate'] = SubCmd(replicate_flower_rotate_arglist, "Replicate each panel creating a 2x2 grid of rotations. Just the starting position needs to be specified.")
dsl['replicate'] = CmdDef(replicate_subcmddict, "Replicate each panel in the working list, according to sub-command specific rules.")

                                       
### Command: snake
snake_subcmddict = dict()
snake_simple_arglist = [CmdArg('start_value', range(12)),
                       CmdArg('direction', ['away', 'N', 'E', 'S', 'W']),
                       CmdArg('action_on_0', ['overwrite', 'turn_right', 'turn_left', 'around_right', 'around_left', 'around_both', 'stop']),
                       CmdArg('action_on_1', ['overwrite', 'turn_right', 'turn_left', 'around_right', 'around_left', 'around_both', 'stop']),
                       CmdArg('action_on_2', ['overwrite', 'turn_right', 'turn_left', 'around_right', 'around_left', 'around_both', 'stop']),
                       CmdArg('action_on_3', ['overwrite', 'turn_right', 'turn_left', 'around_right', 'around_left', 'around_both', 'stop']),
                       CmdArg('action_on_4', ['overwrite', 'turn_right', 'turn_left', 'around_right', 'around_left', 'around_both', 'stop']),
                       CmdArg('action_on_5', ['overwrite', 'turn_right', 'turn_left', 'around_right', 'around_left', 'around_both', 'stop']),
                       CmdArg('action_on_6', ['overwrite', 'turn_right', 'turn_left', 'around_right', 'around_left', 'around_both', 'stop']),
                       CmdArg('action_on_7', ['overwrite', 'turn_right', 'turn_left', 'around_right', 'around_left', 'around_both', 'stop']),
                       CmdArg('action_on_8', ['overwrite', 'turn_right', 'turn_left', 'around_right', 'around_left', 'around_both', 'stop']),
                       CmdArg('action_on_9', ['overwrite', 'turn_right', 'turn_left', 'around_right', 'around_left', 'around_both', 'stop'])]
snake_subcmddict['simple'] = SubCmd(snake_simple_arglist, "Apply a simple snake rule.")
dsl['snake'] = CmdDef(snake_subcmddict, "Draw a 'snake' starting from a given value, according to given rules.")


### Command: sort
sort_subcmddict = dict()
sort_by_value_arglist = [CmdArg('value', list(reversed(range(MAX_VALUE)))),
                         CmdArg('ordering', ['ascending', 'descending'])]
sort_subcmddict['by_value'] = SubCmd(sort_by_value_arglist, "Sorts the working list by the number of occurences of a given value. Asserts if the ordering is partial.")
sort_unique_values_arglist = [CmdArg('ordering', ['ascending', 'descending'])]
sort_subcmddict['unique_values'] = SubCmd(sort_unique_values_arglist, "Sorts the working list by the number of unique values.")
dsl['sort'] = CmdDef(sort_subcmddict, "The sort reorders the working list by a variety of methods.")


### Command: split
split_subcmddict = dict()
split_by_value_arglist = [CmdArg('background', range(11)),
                          CmdArg('crop', [True, False, 'inclusive'])]
split_subcmddict['by_value'] = SubCmd(split_by_value_arglist, "Splits the input grid into a list of panels by value. The given background colour is considered as not part of any object to be extracted. The objects are extracted into arrays of size equal to the original, unless optionally cropped.")
split_fixed_grid_arglist = [CmdArg('rows', range(30)),
                            CmdArg('columns', range(30)),
                            CmdArg('divider', [True, False])]
split_subcmddict['fixed_grid'] = SubCmd(split_fixed_grid_arglist, "Splits the input grid into a list of panels, assumming that the panels form a grid with the given number of rows and columns. If divider is true, leave an extra row/colum after each panel.")
split_fixed_size_arglist = [CmdArg('rows', range(30)),
                            CmdArg('columns', range(30)),
                            CmdArg('divider', [True, False])]
split_subcmddict['fixed_size'] = SubCmd(split_fixed_size_arglist, "Splits the input grid into a list of panels, assumming that each panels has the given number of rows and columns. If divider is true, leave an extra row/colum after each panel.")
split_connected_region_arglist = [CmdArg('background', range(11)),
                                  CmdArg('single_value', [True, False]),
                                  CmdArg('neighbourhood', [4, 8]),
                                  CmdArg('crop', [True, False])]
split_subcmddict['connected_region'] = SubCmd(split_connected_region_arglist, "Splits the input grid into a list of panels, based on connected regions. Extracted regions can contain a single value or multiple values, depending on the single_value argument. The given background colour is considered as not part of any object to be extracted. The objects are extracted into arrays of size equal to the original, unless optionally cropped.")
split_frame_arglist = [CmdArg('background', range(11)),
                       CmdArg('keep_frame', [True, False])]
split_subcmddict['frame'] = SubCmd(split_frame_arglist, "Splits the input grid into a list of panels, based on finding rectangular frames in the input.")
dsl['split'] = CmdDef(split_subcmddict, "The split command takes an input grid and splits it into many panels.")


### Command: transform
transform_subcmddict = dict()
transform_crop_arglist = [CmdArg('background', LIKELY_BACKGROUNDS)]
transform_subcmddict['crop'] = SubCmd(transform_crop_arglist, "Crop all images in the working list to remove outer rows/colums of the given background colour.")
transform_flip_arglist = [CmdArg('direction', ['lr', 'ud'])]
transform_subcmddict['flip'] = SubCmd(transform_flip_arglist, "Flip all images in the working list, either left-right or up-down.")
transform_invert_arglist = []
transform_subcmddict['invert'] = SubCmd(transform_invert_arglist, "Invert all images in the working list that have 2 colours. Images with more or less than 2 colours are left untouched.")
transform_rot90_arglist = [CmdArg('count', [1, 2, 3])]
transform_subcmddict['rot90'] = SubCmd(transform_rot90_arglist, "Rotate all images in the working list by (count * 90) degrees.")
dsl['transform'] = CmdDef(transform_subcmddict, "The transform command applies simple transforms to all the arrays in the working list.")


### Command: value_map
value_map_subcmddict = dict()
value_map_simple_arg_list = [CmdArg('from', range(MAX_VALUE)),
                             CmdArg('to', range(MAX_VALUE))]
value_map_subcmddict['simple'] = SubCmd(value_map_simple_arg_list, "Simply apply a fixed mapping of initial values to final values for all items in the working list.")
value_map_enclosures_count_arg_list = [CmdArg('background', range(MAX_VALUE)),
                                       CmdArg('count', range(8)),
                                       CmdArg('value', range(10))]
value_map_subcmddict['enclosures_count'] = SubCmd(value_map_enclosures_count_arg_list, "Change the values of objects based on the count of enclosures they have.")
value_map_shape_match_arg_list = [CmdArg('source_value_not', range(10)),
                                  CmdArg('allow_rotations', [True, False])]
value_map_subcmddict['shape_match'] = SubCmd(value_map_shape_match_arg_list, "Match panels by shape, then apply the value from a source panel to all others of the same shape.")
dsl['value_map'] = CmdDef(value_map_subcmddict, "Map from initial values to final values in a variety of ways.")


def dsl_usage(command=None):
    if command is None:
        commands = dsl.keys()
    else:
        commands = {command}
        
    for cmd in commands:
        print(f"Command {cmd}: {dsl[cmd].description}")        
        for subcmd in dsl[cmd].subcmddict.keys():
            args = ""
            for arg in dsl[cmd].subcmddict[subcmd].arglist:
                args += f"<{arg.placeholder}> "
            print(f"\t{cmd} {subcmd} {args}: {dsl[cmd].subcmddict[subcmd].description}")
        print("\n")
        
dsl_usage()


# ### 2.1 - The Interpreter
# 
# This runs programs using the DSL, and maintains the program's state.

# In[ ]:


def split_by_value(input_array, background, crop, debug=False):
    """
    """
    panels = []
    top_corners = []
    values = np.unique(input_array)
    for v in values:
        if v != background:
            new_panel = np.where(input_array == v, v, background)
            if crop == 'True':
                new_panel, top_corner = crop_background(new_panel, background)
            elif crop == 'False':
                top_corner = (0, 0)
            elif crop == 'inclusive':
                temp_panel, top_corner = crop_background(new_panel, background)
                new_panel = input_array[top_corner[0]:top_corner[0]+temp_panel.shape[0], top_corner[1]:top_corner[1]+temp_panel.shape[1]]
            else:
                assert False, f"Bad value for crop {crop} given to split by_value."
            
            panels.append(new_panel)
            top_corners.append(top_corner)
    if debug:        
        print(f"Returning {len(panels)} from split_by_value.")
    return panels, top_corners
    

def split_fixed_internal(input_array, panel_rows, panel_columns, grid_rows, grid_columns, divider, debug=False):
    """
    """
    panels = []
    top_corners = []
    for gr in range(grid_rows):
        for gc in range(grid_columns):
            if divider:
                panel = input_array[(gr*panel_rows)+gr:((gr+1)*panel_rows)+gr,(gc*panel_columns)+gc:((gc+1)*panel_columns)+gc]
                top_corner = ((gr*panel_rows)+gr, (gc*panel_columns)+gc)
            else:
                panel = input_array[(gr*panel_rows):((gr+1)*panel_rows),(gc*panel_columns):(gc+1)*panel_columns]
                top_corner = ((gr*panel_rows), (gc*panel_columns))
          
            panels.append(panel)
            top_corners.append(top_corner)
            if debug:
                print(f"Panel {gr} by {gc}:")
                print(panel)
    return panels, top_corners

def split_fixed_grid(input_array, rows, columns, divider, debug=False):
    """Given a fixed grid we can calculate the size of each panel from the input array shape."""
    input_rows, input_columns = input_array.shape
    # Using floor division here means we don't need to worry about the presence of a divider.
    panel_rows = input_rows//rows
    panel_columns = input_columns//columns    
    return split_fixed_internal(input_array, panel_rows, panel_columns, rows, columns, divider, debug)
    
def split_fixed_size(input_array, rows, columns, divider, debug=False):
    """Given a fixed panel size we can calculate the dimensions of the grid of panels from the input array shape."""
    input_rows, input_columns = input_array.shape
    # Using floor division here means we don't need to worry about the presence of a divider.
    grid_rows = input_rows//rows
    grid_columns = input_columns//columns    
    return split_fixed_internal(input_array, rows, columns, grid_rows, grid_columns, divider, debug)

def find_region(input_array, background, region_value, object_map, visited, row, column, eight_connected=False):
    if row < 0 or row >= input_array.shape[0] or column < 0 or column >= input_array.shape[1]:
        return
    
    assert background is not None or region_value is not None, "Can't pass None for both background and region_value."
    
    if background is None:
        background = [x for x in range(10) if x != region_value]
        
    val = input_array[row,column]
    if val in background or visited[row,column]:
        visited[row,column] = True    
    elif region_value is not None and val != region_value:
        # We're looking for a region of a single value, and this is a different value. Don't mark visited
        # as this must belong to another region.
        pass
    else:
        object_map[row,column] = True
        visited[row,column] = True
        # Recurse
        find_region(input_array, background, region_value, object_map, visited, row-1, column, eight_connected=eight_connected)
        find_region(input_array, background, region_value, object_map, visited, row+1, column, eight_connected=eight_connected)
        find_region(input_array, background, region_value, object_map, visited, row, column-1, eight_connected=eight_connected)
        find_region(input_array, background, region_value, object_map, visited, row, column+1, eight_connected=eight_connected)
        if eight_connected:
            find_region(input_array, background, region_value, object_map, visited, row-1, column-1, eight_connected=True)
            find_region(input_array, background, region_value, object_map, visited, row-1, column+1, eight_connected=True)
            find_region(input_array, background, region_value, object_map, visited, row+1, column-1, eight_connected=True)
            find_region(input_array, background, region_value, object_map, visited, row+1, column+1, eight_connected=True)
            
    
def crop_background(input_array, background, single_value=False, debug=False):
    assert background < 10, "Un-mapped special value in crop_background."

    #print(input_array)
    crop_coords = np.argwhere(input_array != background)
    if len(crop_coords) == 0:
        # Input array must have been all background.
        return input_array, (0, 0)
    r_min, c_min = crop_coords.min(axis=0)
    r_max, c_max = crop_coords.max(axis=0)
    result = input_array[r_min:r_max+1, c_min:c_max+1]
    if single_value:
        result = np.where(result == background, 0, result)
    return result, (r_min, c_min)
            
def split_connected_region(input_array, background, single_value, crop, eight_connected=False, debug=False):
    """
    """
    assert background < 10, "Un-mapped special value in split_connected_region."

    objects = []
    top_corners = []
    visited = np.zeros_like(input_array, dtype=np.bool)
    for row in range(input_array.shape[0]):
        for column in range(input_array.shape[1]):
            val = input_array[row,column]
            if val == background or visited[row,column]:
                visited[row,column] = True
            else:
                object_map = np.zeros_like(input_array, dtype=np.bool)
                region_value = val if single_value else None
                find_region(input_array, [background], region_value, object_map, visited, row, column, eight_connected=eight_connected)
                object_array = np.where(object_map, input_array, background)
                #if debug:
                #    print(object_array)
                if crop:
                    object_array, top_corner = crop_background(object_array, background, single_value)
                objects.append(object_array)
                top_corners.append(top_corner)
    if debug:
        print(f"Returning {len(objects)} from split_connected_region.")
    return objects, top_corners

def is_frame(input_array, top_corner, shape):
    # It's a frame if all values on the boundary of the specified region are the same.
    left_edge = np.unique(input_array[top_corner[0]:top_corner[0]+shape[0], top_corner[1]])
    left   = len(left_edge) == 1
    right  = np.array_equal(left_edge, np.unique(input_array[top_corner[0]:top_corner[0]+shape[0], top_corner[1]+shape[1]-1]))
    top    = np.array_equal(left_edge, np.unique(input_array[top_corner[0], top_corner[1]:top_corner[1]+shape[1]]))
    bottom = np.array_equal(left_edge, np.unique(input_array[top_corner[0]+shape[0]-1, top_corner[1]:top_corner[1]+shape[1]]))
    return left, right, top, bottom

def split_frames(input_array, background, keep_frame, debug=False):
    """
    """
    objects = []
    top_corners = []
    visited = np.zeros_like(input_array, dtype=np.bool)
    
    MIN_FRAME_SIZE = 2
    if input_array.shape[0] < MIN_FRAME_SIZE or input_array.shape[1] < MIN_FRAME_SIZE:
        if debug:
            print("Input array too small for split_frames.")
        return objects, top_corners
        
    for row in range(input_array.shape[0]-(MIN_FRAME_SIZE-1)):
        for column in range(input_array.shape[1]-(MIN_FRAME_SIZE-1)):
            val = input_array[row,column]
            if val == background or visited[row,column]:
                visited[row,column] = True
            else:              
                frame_rows = MIN_FRAME_SIZE
                frame_cols = MIN_FRAME_SIZE
                object_array = None
                viable_frame_rows = None
                viable_frame_cols = None
                left, right, top, bottom = is_frame(input_array, (row, column), (frame_rows, frame_cols))
                while top and left and row + frame_rows <= input_array.shape[0] and column + frame_cols <= input_array.shape[1]:
                    if np.any(visited[row:row+frame_rows,column:column+frame_cols]):
                        # We're running over a previous frame, abort.
                        break
                    left, right, top, bottom = is_frame(input_array, (row, column), (frame_rows, frame_cols))
                    if top and left and right and bottom:
                        # Frame complete!
                        if debug:
                            print(f"Found frame with value: {val} and shape ({frame_rows}, {frame_cols})")
                        viable_frame_rows = frame_rows
                        viable_frame_cols = frame_cols
                        
                        # Don't break here, we may extend the panel further.
                        free_move = None
                        non_free_move = None
                        if (row + frame_rows + 1) <= input_array.shape[0]:
                            left2, right2, top2, bottom2 = is_frame(input_array, (row, column), (frame_rows + 1, frame_cols))
                            if top2 and left2 and right2 and bottom2:
                                free_move = (1, 0)
                            if top2 and left2:
                                non_free_move = (1, 0)
                        if (column + frame_cols + 1) <= input_array.shape[1]:
                            left2, right2, top2, bottom2 = is_frame(input_array, (row, column), (frame_rows, frame_cols + 1))
                            if top2 and left2 and right2 and bottom2:
                                free_move = (1, 1) if free_move is not None else (0, 1)
                            if top2 and left2:
                                non_free_move = (1, 1) if free_move is not None else (0, 1)
                        if free_move is not None:
                            frame_rows += free_move[0]
                            frame_cols += free_move[1]
                        elif non_free_move is not None:
                            frame_rows += non_free_move[0]
                            frame_cols += non_free_move[1]
                        else:
                            break
                    else:
                        if not right:
                            frame_cols += 1
                        if not bottom:
                            frame_rows += 1
                
                visited[row,column] = True
                if viable_frame_rows is not None:
                    frame_rows = viable_frame_rows
                    frame_cols = viable_frame_cols
                    visited[row:row+frame_rows,column:column+frame_cols] = True
                    if keep_frame:
                        object_array = input_array[row:row+frame_rows,column:column+frame_cols]
                        top_corner = (row, column)
                    else:
                        object_array = input_array[row+1:row+frame_rows-1,column+1:column+frame_cols-1]
                        top_corner = (row+1, column+1)
                    # Check for zero dimension.
                    if object_array.shape[0] > 0 and object_array.shape[1] > 0:
                        objects.append(object_array)
                        top_corners.append(top_corner)
                        
                        
        
        visited[row,column] = True

    if debug:
        print(f"Returning {len(objects)} from split_frames.")
    return objects, top_corners

def count_crosses_and_xes(input_array, background, diagonal, debug=False):
    """
    """
    if background >= 10:
        assert False, "Special values should have been mapped already."

    count = 0
    # Iterate over the array, avoiding the boundary. Can't have the centre of a cross on the boundary.
    for row in range(1,input_array.shape[0]-1):
        for column in range(1,input_array.shape[1]-1):
            if input_array[row,column] != background:
                val = input_array[row,column]
                if diagonal:
                    if input_array[row-1,column-1] == val and input_array[row-1,column+1] == val and input_array[row+1,column-1] == val and input_array[row+1,column+1] == val:
                        count += 1
                else:
                    if input_array[row-1,column] == val and input_array[row,column-1] == val and input_array[row+1,column] == val and input_array[row,column+1] == val:
                        count += 1
    return count

def count_enclosures(input_array, background, debug=False):
    """Here we going to look for 4-connected enclosed regions. That is areas of background that are enclosed and do not
       touch the boundary."""
    if background >= 10:
        assert False, "Special values should have been mapped already."

    count = 0
    visited = np.zeros_like(input_array, dtype=np.bool)
    for row in range(input_array.shape[0]):
        for column in range(input_array.shape[1]):
            val = input_array[row,column]
            if val != background or visited[row,column]:
                visited[row,column] = True
            else:
                object_map = np.zeros_like(input_array, dtype=np.bool)
                find_region(input_array, None, background, object_map, visited, row, column, eight_connected=False)
                if np.any(object_map[0,:]) or np.any(object_map[-1,:]) or np.any(object_map[:,0]) or np.any(object_map[:,-1]):
                    # A region that tocuhes the bounday. Not an enclosure.
                    pass
                else:
                    count += 1
    if debug:
        print(f"Returning {count} from count_enclosures.")
    return count
    
def get_most_common_value(input_array, ignore_zero=False):
    values, counts = np.unique(input_array, return_counts=True)
    sort_index = np.flip(np.argsort(counts))
    sorted_values = values[sort_index]
    most_common = sorted_values[0]
    if ignore_zero and most_common == 0 and len(sorted_values) > 1:
        most_common = sorted_values[1]
    return most_common


def split_auto_grid(input_array, debug=False):
    # Check every row and column for a unique value. Look for the most common spacing between them to get the size of panels to be extracted.
    last_row = None
    divider_row_spacing = []
    for row in range(input_array.shape[0]):
        if len(np.unique(input_array[row,:])) == 1:
            if last_row is None:
                spacing = row                
            else:
                spacing = row - last_row - 1
            if spacing != 0:
                divider_row_spacing.append(spacing)
            last_row = row

    last_col = None
    divider_col_spacing = []
    for col in range(input_array.shape[1]):
        if len(np.unique(input_array[:,col])) == 1:
            if last_col is None:
                spacing = col
            else:
                spacing = col - last_col - 1
            if spacing != 0:
                divider_col_spacing.append(spacing)
            last_col = col

    if len(divider_row_spacing) == 0 and len(divider_col_spacing) == 0:
        # No rows/columns with a unique value. Not a grid problem.
        return [], []
    
    if len(divider_row_spacing) == 0:
        panel_rows = input_array.shape[0]
    else:
        panel_rows = Counter(divider_row_spacing).most_common(1)[0][0]
    if len(divider_col_spacing) == 0:
        panel_cols = input_array.shape[1]
    else:      
        panel_cols = Counter(divider_col_spacing).most_common(1)[0][0]
        
    if debug:
        print(f"Extracting panels of size ({panel_rows},{panel_cols})")
        
    objects = []
    top_corners = []
    row = 0
    while row < input_array.shape[0]:        
        if len(np.unique(input_array[row,:])) == 1:
            row += 1
        else:
            col = 0
            while col < input_array.shape[1]:
                if len(np.unique(input_array[:,col])) == 1:
                    col += 1
                else:
                    # Must be the start of a panel to extract.
                    objects.append(input_array[row:row+panel_rows, col:col+panel_cols].copy())
                    top_corners.append((row, col))
                    col += panel_cols
            row += panel_rows
                
    if debug:
        print(f"Returning {len(objects)} from split_auto_grid.")
    return objects, top_corners            


# In[ ]:


class ScoringHelper:
    """
    A helper class to store scores and then choose a winner according to a given condition.
    """
    
    def __init__(self):
        self.indicies = []
        self.scores = []
        
    def add_score(self, index, score):
        self.indicies.append(index)
        self.scores.append(score)
        
    def get_winner(self, condition):
        # TODO - Add majority and minority as conditions.
        if condition == 'most' or condition == 'least':
            best_score = -1 if condition == 'most' else sys.maxsize
            best_index = None
            for i in range(len(self.indicies)):
                if (condition == 'most' and self.scores[i] > best_score) or                    (condition == 'least' and self.scores[i] < best_score):
                        best_score = self.scores[i]
                        best_index = self.indicies[i]
                elif self.scores[i] == best_score:
                    # The result needs to be the most or least, if some other item has the same score,
                    # this could return a winner just based on the order the inputs were seen, which is not stable.
                    best_index = None
            return best_index
        elif condition == "odd_one_out":
            unique_scores = [x for x, n in Counter(self.scores).items() if n == 1]
            if len(unique_scores) == 1:
                return self.indicies[self.scores.index(unique_scores[0])]
            else:
                return None
        else:
            assert False, f"Bad condition {condition}."

    def get_ordering(self, ordering):
        scores_tuple = zip(self.scores, self.indicies)
        reverse = (ordering == "descending")
        scores_tuple = sorted(scores_tuple, key = lambda x: x[0], reverse=reverse)
        result = [i for s, i in scores_tuple]
        return result
        


# In[ ]:


# Most progs start with a split command. Profiling shows that the majority of the runtime goes in split commands. So implement a cache.
# The cache is only to be used if the split command is the first in the prog. I'd like to key this of task_id, train_or_test and example index
# but too many places already call the interpreter without having that info and time is short... So, just store one result, and the reference
# input array, as the searching code tends to go over the same task example over and over.
# TODO - Implement as an object.

split_result_cache_input_array = np.zeros((1,1))
split_result_cache_args = ""
split_result_cache_working_list = []
split_result_cache_metadata = []
split_result_cache_accesses = 0
split_result_cache_hits = 0


# In[ ]:


class Interpreter:
    """
    This class implements the DSL interpreter. The Interpreter object stores the working list (list of numpy arrays) and a metadata list.
    The meta data list is currently just used to store the top corner of the arrays relative to a global frame of reference (typically the
    original input array). The interpreter also stores the original input array.
    
    A typical program would split the input array into objects via the split command, filter some of those objects to keep or remove
    objects meeting certain criteria. Finally, the objects in the working list can be assembled in some way with the assemble command. For
    example, they could be placed on their original locations, on top of the original input array with 'assemble original original'.
    """
    
    def __init__(self):
        self.original_input_array = None
        self.working_list = None
        self.metadata = None
        
    def map_special_values(self, input_array, value, debug=False):
        """
        Given an input_array and a value, map that value to a real value (0-9) according to various special case:
        - Value 10 goes to most common value from the original input array.
        - Value 11 goes to least common value from the original input array.
        - Value 12 goes to most common value from the given input array.
        - Value 13 goes to least common value from the given input array.
        - Value 14 goes to the divider colour populated by split. Will assert if no divider colour available.
        """
        if value < 10:
            return value
        
        if value in [10,11]:
            array_to_use = self.original_input_array
        elif value in [12,13]:
            array_to_use = input_array
        elif value == 14:
            # TODO - Implement this.
            assert False, "Not implemented yet"
            
        if value in [10, 11, 12, 13]:
            values, counts = np.unique(array_to_use, return_counts=True)              
            sort_index = np.flip(np.argsort(counts))
            sorted_values = values[sort_index]
            most_common = sorted_values[0]
            least_common = sorted_values[-1]

            result = value
            if value in [10, 12]:
                result = most_common
            elif value in [11, 13]:
                result = least_common

        if debug:
            print(f"Changed value from {value} to {result}.")
        return result

    def execute(self, input_array, prog, partial=False, debug=False):
        """
        Initialize the interpreter's working list with the given input array and execute the given program.
        """
        
        if self.original_input_array is None:
            self.original_input_array = input_array.copy()
            self.working_list = [input_array.copy()]
            self.metadata = [dict(top_corner = (0,0))]
        
        line_no = 0
        for line in prog:
            line_no += 1
            if debug:
                print(f"{datetime.datetime.now()} Executing: {line}")
            line_split = line.split(' ', 1)
            command = line_split[0]
            rest = line_split[1] if len(line_split) > 1 else None
            if command == 'identity':
                # Take the working_list and apply the identity transform. That is do nothing.
                pass
            elif command == 'abstract':
                self.abstract(rest, debug=debug)
            elif command == 'assemble':
                self.assemble(rest, debug=debug)
            elif command == 'combine':
                self.combine(rest, debug=debug)
            elif command == 'filter':
                self.filter(rest, debug=debug)
            elif command == 'move':
                self.move(rest, debug=debug)
            elif command == 'replicate':
                self.replicate(rest, debug=debug)
            elif command == 'snake':
                self.snake(rest, debug=debug)                
            elif command == 'sort':
                self.sort(rest, debug=debug)
            elif command == 'split':
                global split_result_cache_input_array
                global split_result_cache_args
                global split_result_cache_working_list
                global split_result_cache_metadata
                global split_result_cache_accesses
                global split_result_cache_hits

                if line_no == 1 and len(self.working_list) == 1:
                    # Consider the cache!
                    split_result_cache_accesses += 1
                    if  split_result_cache_args == rest and np.array_equal(self.working_list[0], split_result_cache_input_array):
                        # We've just done this split, re-use the results.
                        split_result_cache_hits += 1
                        self.working_list = deepcopy(split_result_cache_working_list)
                        self.metadata = deepcopy(split_result_cache_metadata)
                    else:
                        # Split and cache the results.
                        split_result_cache_input_array = self.working_list[0].copy()
                        split_result_cache_args = rest
                        self.split(rest, debug=debug)
                        split_result_cache_working_list = deepcopy(self.working_list)
                        split_result_cache_metadata = deepcopy(self.metadata)
                else:
                    self.split(rest, debug=debug)
            elif command == 'transform':
                self.transform(rest, debug=debug)
            elif command == 'value_map':
                self.value_map(rest, debug=debug)                
            else:
                dsl_usage()
                assert False, f"Unknown command: {command}."
            
        if debug:
                print("Execution complete.")
                
        if partial:
            return self.working_list
        else:
            assert len(self.working_list) == 1, "Program should complete with exactly 1 item in the working list."
            return self.working_list[0]

        
    def abstract(self, args, debug=False):
        """
        """
        assert args is not None, dsl_usage('abstract')

        args_split = args.split(' ', 1)
        sub_command = args_split[0]
        rest = args_split[1] if len(args_split) > 1 else None

        new_working_list = []
        if sub_command == 'simple':
            assert len(rest.split()) == 1, dsl_usage('abstract')
            output_size = int(rest.split()[0])
            
            for i in range(len(self.working_list)):
                item = self.working_list[i]
                values, counts = np.unique(item, return_counts=True)
                #print(values, counts)
                sort_index = np.flip(np.argsort(counts))
                sorted_values = values[sort_index]
                most_common = sorted_values[0]
                result = np.full((output_size, output_size), most_common, dtype=int)
                new_working_list.append(result)
        else:
            dsl_usage('abstract')
            assert False, f"Unknown sub-command: {sub_command}."

        self.working_list = new_working_list

        
    def assemble(self, args, debug=False):
        """
        """
        assert args is not None, dsl_usage('assemble')
        assert len(self.working_list), "Cannot assemble with an empty working list!"

        args_split = args.split(' ', 1)
        sub_command = args_split[0]
        rest = args_split[1] if len(args_split) > 1 else None
        
        new_working_list = []
        if sub_command == 'original':
            assert len(rest.split()) == 1, dsl_usage('assemble')
            base = rest.split()[0]
            if base == 'original':
                result = self.original_input_array.copy()
            elif base == 'zeros':
                result = np.zeros(self.original_input_array.shape, dtype=int)
            elif base == 'majority_value':
                result = np.full(self.original_input_array.shape, get_most_common_value(self.original_input_array), dtype=int)
                
            for i in range(len(self.working_list)):
                item = self.working_list[i]                
                # Using the top_corner metadata, write each panel back in it's original place.
                if debug:                  
                    print(self.metadata[i])
                top_corner = self.metadata[i]['top_corner']
                shape = item.shape
                resize = np.zeros(self.original_input_array.shape, dtype=int)
                resize[top_corner[0]:top_corner[0]+shape[0], top_corner[1]:top_corner[1]+shape[1]] = item
                result = np.where(resize == 0, result, resize)
            new_working_list.append(result)
        elif sub_command == 'histogram':
            assert len(rest.split()) == 2, dsl_usage('assemble')
            flip = rest.split()[0]
            rot90 = int(rest.split()[1])
            
            scores = [0] * 10
            for i in range(len(self.working_list)):
                item = self.working_list[i]
                most_common = get_most_common_value(item)
                scores[most_common] += 1
                scores = np.array(scores)
            result_shape = (np.count_nonzero(scores), np.max(scores))
            result = np.zeros(result_shape)
            row = 0
            while np.count_nonzero(scores) > 0:
                val = np.argmax(scores)
                for col in range(scores[val]):
                    result[row, col] = val
                row += 1
                scores[val] = 0
            if flip == 'lr':
                result = np.fliplr(result)
            elif flip == 'ud':
                result = np.flipud(result)
            if rot90 != 0:
                result = np.rot90(result, rot90, axes=(0, 1))
            new_working_list.append(result)
        elif sub_command == 'auto_grid':
            top_corners = []
            for md in self.metadata:
                top_corners.append(md['top_corner'])
            row_parts = sorted([r for r, c in top_corners])
            col_parts = sorted([c for r, c in top_corners])
            
            # Consider an increasing band. If n top-corners sit inside m bands, we have an n by m grid.
            grid_rows = 0
            grid_cols = 0
            band = 0
            
            if len(self.working_list) == 1:
                # Short-cut for the one panel case.
                grid_rows = 1
                grid_cols = 1
                band = 100                
                
            while band < 10:
                for dir in ['row', 'col']:
                    if dir == 'row':
                        parts = row_parts
                    else:
                        parts = col_parts
                    idx = 0
                    groups = []
                    while(idx < len(parts)):
                        part = parts[idx]
                        group_size = 0
                        while(idx < len(parts) and parts[idx] - part <= band):
                            group_size += 1
                            #print(f"part: {part}, idx: {idx}, parts[idx] : {parts[idx]}, group_size: {group_size}")
                            idx +=1
                        groups.append(group_size)
                    # If all groups are the same size, and the number of groups times the size of each group equals the total we have
                    # a solution. Note that all groups being of size 1 is not a solution - that just means that nothing lines up!
                    if len(set(groups)) == 1 and len(groups) * set(groups).pop() == len(parts) and set(groups).pop() != 1:
                        if debug:
                            print(f"band: {band}, dir: {dir}, n: {len(groups)}, m: {set(groups).pop()}")
                        if dir == 'row':
                            grid_rows = len(groups)
                            grid_cols = set(groups).pop()
                        else:
                            grid_cols = len(groups)
                            grid_rows = set(groups).pop()
                        band = 100 # break out of outer loop
                        break
                band += 1    
                
            if debug:
                print(f"grid_rows: {grid_rows}, grid_cols: {grid_cols}")
            first_sort = sorted(top_corners, key=lambda k: [k[0], k[1]])
            sorted_top_corners = []
            for gr in range(grid_rows):
                sorted_top_corners.extend(sorted(first_sort[gr*grid_cols:gr*grid_cols+grid_cols], key=lambda k: [k[1], k[0]]))
            tc_to_item_dict = {tc: i for i, tc in zip(self.working_list, top_corners)}
            
            # Now the top-corners are sorted, simply take items in chunks of grid_cols.
            # Make the result big enough, we'll crop it later.
            result = np.full((1000, 1000), -1, dtype=int)
            offset = (0, 0)
            idx = 0
            for gr in range(grid_rows):
                row_height = 0
                for gc in range(grid_cols):
                    tc = sorted_top_corners[idx]                    
                    item = tc_to_item_dict[tc]
                    shape = item.shape
                    row_height = max(row_height,shape[0])
                    if debug:                  
                        print(f"offset: {offset}")
                    result[offset[0]:offset[0]+shape[0], offset[1]:offset[1]+shape[1]] = item
                    offset = (offset[0], offset[1] + item.shape[1])
                    idx += 1
                offset = (offset[0] + row_height, 0)
                assert offset[0] <= 30 and offset[1] <= 30, "Result beyond 30x30 limit in assemble auto_grid."
                
            result, _ = crop_background(result, -1, debug=debug)
            # There should be any of the non-value -1 left, but if there is, replace with 0.
            result = np.where(result == -1, 0, result)
            new_working_list.append(result)
        else:
            dsl_usage('assemble')
            assert False, f"Unknown sub-command: {sub_command}."

        self.working_list = new_working_list


    def combine(self, args, debug=False):
        """
        """
        assert args is not None, dsl_usage('combine')

        args_split = args.split(' ', 1)
        sub_command = args_split[0]
        rest = args_split[1] if len(args_split) > 1 else None
        
        new_working_list = []
        if sub_command == 'logical_op':
            assert len(rest.split()) == 4, dsl_usage('combine')
            pre_invert = rest.split()[0] == 'True'
            logical_op = rest.split()[1]
            post_invert = rest.split()[2] == 'True'
            colour = int(rest.split()[3])           
            if debug:
                print(f"pre_invert: {pre_invert}, logical_op: {logical_op}, post_invert: {post_invert}, colour: {colour}")
                
            output = None
            for item in self.working_list:
                # Binarize the current panel.
                working_panel = np.where(item == 0, False, True)
                
                if pre_invert:
                    working_panel = np.logical_not(working_panel)
    
                if output is None:
                    output = working_panel.copy()
                elif logical_op == 'and':
                    output = np.logical_and(output, working_panel)
                elif logical_op == 'or':
                    output = np.logical_or(output, working_panel)
                elif logical_op == 'xor':
                    output = np.logical_xor(output, working_panel)
            
            if post_invert:
                output = np.logical_not(output)

            output = np.where(output, colour, 0)
            new_working_list.append(output)
        elif sub_command == 'overwrite':
            assert len(rest.split()) == 2, dsl_usage('combine')
            transparent = int(rest.split()[0])
            permutation = rest.split()[1].strip('[]').split(',')
            if len(permutation) == 1 and permutation[0] == '':
                permutation = range(len(self.working_list))
            if debug:
                print(f"transparent: {transparent}, permutation: {permutation}")

            output = None
            # Go through the working list in the given order.
            for index in permutation:
                item = self.working_list[int(index)]
                if output is None:
                    output = item.copy()
                else:
                    # Overwrite the output with the current item, showing the previous output through where the value
                    # is the given transparent value.
                    output = np.where(item == transparent, output, item)
            
            new_working_list.append(output)
        else:
            dsl_usage('combine')
            assert False, f"Unknown sub-command: {sub_command}."

        self.working_list = new_working_list

        
    def filter(self, args, debug=False):
        """
        """
        assert args is not None, dsl_usage('filter')

        args_split = args.split(' ', 1)
        sub_command = args_split[0]
        rest = args_split[1] if len(args_split) > 1 else None
        
        new_working_list = []
        new_metadata = []
        meets_criteria = []
        if sub_command == 'by_value':
            assert len(rest.split()) == 3, dsl_usage('filter')
            action = rest.split()[0]
            condition = rest.split()[2]            
            helper = ScoringHelper()
            for i in range(len(self.working_list)):
                item = self.working_list[i]
                value = self.map_special_values(item, int(rest.split()[1]), debug=debug)                
                values, counts = np.unique(item, return_counts=True)
                counts_dict = dict(zip(values, counts))
                helper.add_score(i, counts_dict.get(value, 0))
            meets_criteria.append(helper.get_winner(condition))
        elif sub_command == 'by_not_value':
            assert len(rest.split()) == 3, dsl_usage('filter')
            action = rest.split()[0]            
            condition = rest.split()[2]
            helper = ScoringHelper()
            for i in range(len(self.working_list)):
                item = self.working_list[i]
                value = self.map_special_values(item, int(rest.split()[1]), debug=debug)                
                values, counts = np.unique(item, return_counts=True)
                counts_dict = dict(zip(values, counts))
                # Add up counts, except the given value.
                count = 0
                for v in counts_dict.keys():
                    if v != value:
                        count += counts_dict[v]
                helper.add_score(i, count)
            meets_criteria.append(helper.get_winner(condition))
        elif sub_command == 'by_value_gte':
            assert len(rest.split()) == 3, dsl_usage('filter')
            action = rest.split()[0]
            threshold = int(rest.split()[2])
            for i in range(len(self.working_list)):
                item = self.working_list[i]
                value = self.map_special_values(item, int(rest.split()[1]), debug=debug)                
                values, counts = np.unique(item, return_counts=True)
                counts_dict = dict(zip(values, counts))
                if value in counts_dict.keys() and counts_dict[value] >= threshold: 
                    meets_criteria.append(i)
        elif sub_command == 'by_majority_value':
            assert len(rest.split()) == 2, dsl_usage('filter')
            action = rest.split()[0]
            condition = rest.split()[1]
            scores = [0] * 10
            indicies = []    # Don't do [[]] * 10, as that gives the SAME LIST 10 times!
            for i in range(10):
                indicies.append([])
            for i in range(len(self.working_list)):
                item = self.working_list[i]
                most_common = get_most_common_value(item)
                scores[most_common] += 1
                indicies[most_common].append(i)
                
            if condition == 'most':
                val_of_interest = np.argmax(np.array(scores))
            elif condition == 'least':
                val_of_interest = np.argmin(np.array(scores))
            else:
                assert False, f"Bad condition in filter {sub_command}: {condition}"
            if debug:
                print(f"val_of_interest: {val_of_interest}")
            meets_criteria.extend(indicies[val_of_interest])
        elif sub_command == 'by_size':
            assert len(rest.split()) == 2, dsl_usage('filter')
            action = rest.split()[0]
            condition = rest.split()[1]
            helper = ScoringHelper()
            for i in range(len(self.working_list)):
                helper.add_score(i, self.working_list[i].size)
            meets_criteria.append(helper.get_winner(condition))
        elif sub_command == 'unique_values':
            assert len(rest.split()) == 2, dsl_usage('filter')
            action = rest.split()[0]
            condition = rest.split()[1]
            helper = ScoringHelper()
            for i in range(len(self.working_list)):
                helper.add_score(i, len(np.unique(self.working_list[i])))
            meets_criteria.append(helper.get_winner(condition))
        elif sub_command == 'by_index':
            assert len(rest.split()) == 2, dsl_usage('filter')
            action = rest.split()[0]
            index = int(rest.split()[1])
            if index < len(self.working_list):
                meets_criteria.append(index)
        elif sub_command == 'by_shape_count':
            assert len(rest.split()) == 4, dsl_usage('filter')
            action = rest.split()[0]
            shape = rest.split()[1]
            condition = rest.split()[3]            
            helper = ScoringHelper()
            for i in range(len(self.working_list)):
                item = self.working_list[i]
                background = self.map_special_values(item, int(rest.split()[2]), debug=debug)                
                if shape == 'cross':
                    helper.add_score(i, count_crosses_and_xes(item, background, False, debug=debug))
                elif shape == 'x':
                    helper.add_score(i, count_crosses_and_xes(item, background, True, debug=debug))
                elif shape == "enclosure":
                    helper.add_score(i, count_enclosures(item, background, debug=debug))
                    
            meets_criteria.append(helper.get_winner(condition))
        elif sub_command == 'commonality':
            assert len(rest.split()) == 2, dsl_usage('filter')
            action = rest.split()[0]
            most_or_least = rest.split()[1]            
            helper = ScoringHelper()
            already_matched = [False] * len(self.working_list)
            for i in range(len(self.working_list)):
                if not already_matched[i]:
                    commonality_count = 0
                    for j in range(len(self.working_list)):
                        if i != j and not already_matched[j]:
                            if np.array_equal(self.working_list[i], self.working_list[j]):
                                if debug:
                                    print(f"i: {i}, j: {j}, commonality_count: {commonality_count}")
                                commonality_count += 1
                                already_matched[j] = True
                
                    helper.add_score(i, commonality_count)
                    already_matched[i] = True
            meets_criteria.append(helper.get_winner(most_or_least))
        elif sub_command == 'has_symmetry':
            assert len(rest.split()) == 1, dsl_usage('filter')
            action = rest.split()[0]
            chosen_item = None
            for i in range(len(self.working_list)):
                item = self.working_list[i]
                if np.array_equal(item, np.fliplr(item)) or np.array_equal(item, np.flipud(item)):
                    meets_criteria.append(i)                
        elif sub_command == 'rectangular':
            assert len(rest.split()) == 2, dsl_usage('filter')
            action = rest.split()[0]
            min_size  = int(rest.split()[1])
            for i in range(len(self.working_list)):
                item = self.working_list[i]
                if item.shape[0] < min_size or item.shape[1] < min_size:
                    # Too small
                    pass
                else:
                    # If this is rectangular, there will be no background (0) values on the boundary.
                    if np.count_nonzero(item[0,:]) == item.shape[1] and np.count_nonzero(item[-1,:]) == item.shape[1] and np.count_nonzero(item[:,0]) == item.shape[0]  and np.count_nonzero(item[:,-1]) == item.shape[0]:
                        meets_criteria.append(i)
        elif sub_command == 'enclosed':
            assert len(rest.split()) == 1, dsl_usage('filter')
            action = rest.split()[0]
            for i in range(len(self.working_list)):
                item1 = self.working_list[i]
                tc1 = self.metadata[i]['top_corner']
                for j in range(len(self.working_list)):
                    if i != j:
                        item2 = self.working_list[j]
                        tc2 = self.metadata[j]['top_corner']
                        if tc2[0] < tc1[0] and tc2[1] < tc1[1] and tc2[0]+item2.shape[0] > tc1[0]+item1.shape[0] and tc2[1]+item2.shape[1] > tc1[1]+item1.shape[1]:
                            # Completely enclosed.
                            meets_criteria.append(i)
                            break
                        elif tc2[0] <= tc1[0] and tc2[1] < tc1[1] and tc2[0]+item2.shape[0] >= tc1[0]+item1.shape[0] and tc2[1]+item2.shape[1] > tc1[1]+item1.shape[1]:
                            # May overlap up to top or bottom row.
                            meets_criteria.append(i)
                            break
                        elif tc2[0] < tc1[0] and tc2[1] <= tc1[1] and tc2[0]+item2.shape[0] > tc1[0]+item1.shape[0] and tc2[1]+item2.shape[1] >= tc1[1]+item1.shape[1]:
                            # May overlap up to first or last column.
                            meets_criteria.append(i)
                            break
        else:
            dsl_usage('filter')
            assert False, f"Unknown sub-command: {sub_command}."

        # Now apply the action on the filtered items.
        if debug:
            print(f"In filter {sub_command}, meets_criteria = {meets_criteria}")
            
        if action == 'keep':
            new_working_list = [self.working_list[x] for x in meets_criteria]
            new_metadata = [self.metadata[x] for x in meets_criteria]
        elif action == 'remove':
            for i in range(len(self.working_list)):
                if i not in meets_criteria:
                    new_working_list.append(self.working_list[i])
                    new_metadata.append(self.metadata[i])
        else:
            assert False, f"Bad action {action} given in filter by_value."

        self.working_list = new_working_list
        self.metadata = new_metadata        

        
    def move(self, args, debug=False):
        """
        """
        assert args is not None, dsl_usage('move')

        args_split = args.split(' ', 1)
        sub_command = args_split[0]
        rest = args_split[1] if len(args_split) > 1 else None
        
        new_metadata = []
        if sub_command == 'by_value':
            assert len(rest.split()) == 3, dsl_usage('move')            
            direction = rest.split()[1]
            distance = int(rest.split()[2])
            for i in range(len(self.working_list)):
                item = self.working_list[i]
                value = self.map_special_values(item, int(rest.split()[0]), debug=debug)
                
                majority_val = get_most_common_value(item)
                new_top_corner = self.metadata[i]['top_corner']
                if majority_val == value:
                    if direction == 'N':
                        new_top_corner = (max(0, new_top_corner[0] - distance), new_top_corner[1])
                    elif direction == 'S':
                        new_top_corner = (min(new_top_corner[0] + distance, self.original_input_array.shape[0]), new_top_corner[1])
                    elif direction == 'E':
                        new_top_corner = (new_top_corner[0], min(new_top_corner[1] + distance, self.original_input_array.shape[1]))
                    elif direction == 'W':
                        new_top_corner = (new_top_corner[0], max(0, new_top_corner[1] - distance))
                                    
                nmd = dict(top_corner = new_top_corner)
                new_metadata.append(nmd)

        elif sub_command == 'by_shape':
            assert len(rest.split()) == 2, dsl_usage('move')
            dimension  = rest.split()[0]
            direction = rest.split()[1]
            for i in range(len(self.working_list)):
                item = self.working_list[i]
                new_top_corner = self.metadata[i]['top_corner']
                if direction == 'SE':
                    if dimension == 'V' or dimension == 'HV':
                        new_top_corner = (new_top_corner[0] + item.shape[0], new_top_corner[1])
                    elif dimension == 'H' or dimension == 'HV':
                        new_top_corner = (new_top_corner[0], new_top_corner[1] + item.shape[1])
                elif direction == 'NW':
                    if dimension == 'V' or dimension == 'HV':
                        new_top_corner = (new_top_corner[0] - item.shape[0], new_top_corner[1])
                    elif dimension == 'H' or dimension == 'HV':
                        new_top_corner = (new_top_corner[0], new_top_corner[1] - item.shape[1])
                   
                nmd = dict(top_corner = new_top_corner)
                new_metadata.append(nmd)
        else:
            dsl_usage('move')
            assert False, f"Unknown sub-command: {sub_command}."
        
        self.metadata = new_metadata        


    def replicate(self, args, debug=False):
        """
        """
        assert args is not None, dsl_usage('replicate')
        
        args_split = args.split(' ', 1)
        sub_command = args_split[0]
        rest = args_split[1] if len(args_split) > 1 else None

        new_working_list = []
        new_top_corners = []
        for i in range(len(self.working_list)):
            item = self.working_list[i]
            top_corner = self.metadata[i]['top_corner']
            if sub_command == 'and_merge':
                assert len(rest.split()) == 3, dsl_usage('replicate')
                flips = rest.split()[0]
                if rest.split()[1] == 'True':
                    rotations = range(4)
                else:
                    rotations = range(1)
                offset_type = rest.split()[2]
                
                replicants = []
                for rot in rotations:
                    replicant = np.rot90(item, k=rot, axes=(0, 1))
                    if flips == 'none':
                        replicants.append(replicant)
                    elif flips == 'all':
                        replicants.append(np.fliplr(replicant))
                        replicants.append(np.flipud(replicant))
                    elif flips == 'lr':
                        replicants.append(np.fliplr(replicant))
                    elif flips == 'ud':
                        replicants.append(np.flipud(replicant))
                    else:
                        assert False, f"Bad value {flips} passed for flip argument."

                # We don't know that the axis of symmetry was central, so need to try various shifts and pick best.
                MAX_SHIFT = 0
                if offset_type == 'auto':
                    if max(item.shape[0], item.shape[1]) <=5:
                        MAX_SHIFT = 0
                    elif max(item.shape[0], item.shape[1]) <=10:
                        MAX_SHIFT = 1
                    else:
                        MAX_SHIFT = 2

                temp_item = np.zeros((2*MAX_SHIFT+item.shape[0], 2*MAX_SHIFT+item.shape[1]), dtype=int)
                temp_item[MAX_SHIFT:MAX_SHIFT+item.shape[0],MAX_SHIFT:MAX_SHIFT+item.shape[1]] = item
                for rep in replicants:
                    shifts = []
                    helper = ScoringHelper()
                    idx = 0 
                    for rshift in range(-MAX_SHIFT, MAX_SHIFT+1):
                        for cshift in range(-MAX_SHIFT, MAX_SHIFT+1):                                
                            temp = np.zeros((2*MAX_SHIFT+item.shape[0], 2*MAX_SHIFT+item.shape[1]), dtype=int)
                            temp[MAX_SHIFT+rshift:MAX_SHIFT+rshift+item.shape[0],MAX_SHIFT+cshift:MAX_SHIFT+cshift+item.shape[1]] = rep
                            diff = np.where(temp == temp_item, 0, 1)
                            if debug:
                                print(f"rshift: {rshift}, cshift: {cshift}, score: {np.sum(diff)}")
                            helper.add_score(idx, np.sum(diff))
                            shifts.append((rshift, cshift))
                            idx += 1
                    best_shift = (0,0)
                    best_idx = helper.get_winner('least')
                    if best_idx is not None:
                        best_shift = shifts[best_idx]

                    rshift, cshift = best_shift
                    if debug:
                        print(f"best_shift: {best_shift}")
                        print(f"rshift: {rshift}, cshift: {cshift}")
                    temp = np.zeros((2*MAX_SHIFT+item.shape[0], 2*MAX_SHIFT+item.shape[1]), dtype=int)
                    temp[MAX_SHIFT+rshift:MAX_SHIFT+rshift+item.shape[0],MAX_SHIFT+cshift:MAX_SHIFT+cshift+item.shape[1]] = rep
                    temp_item = np.where(temp_item == 0, temp, temp_item)
                    
                new_item, offset = crop_background(temp_item, 0, debug=True)    
                new_working_list.append(new_item)
                new_top_corners.append((top_corner[0] - MAX_SHIFT + offset[0], top_corner[1] - MAX_SHIFT + offset[1]))
                if debug:
                    print(f"top_corner: {top_corner}, offset: {offset}")
                    print(new_top_corners[-1])
            elif sub_command == 'flower_flip':
                assert len(rest.split()) == 1, dsl_usage('replicate')
                starting_pos = int(rest.split()[0])
                # Starting pos: 0 1
                #               2 3
                
                # Add the original item
                new_working_list.append(item)
                new_top_corners.append(top_corner)
                
                # The lr-flip
                new_item = np.fliplr(item)
                if starting_pos in [0, 2]:
                    new_top_corner = (top_corner[0], top_corner[1] + item.shape[1])
                else:
                    new_top_corner = (top_corner[0], top_corner[1] - item.shape[1])
                new_working_list.append(new_item)
                new_top_corners.append(new_top_corner)

                # The ud-flip
                new_item = np.flipud(item)
                if starting_pos in [0, 1]:
                    new_top_corner = (top_corner[0] + item.shape[0], top_corner[1])
                else:
                    new_top_corner = (top_corner[0] - item.shape[0], top_corner[1])
                new_working_list.append(new_item)
                new_top_corners.append(new_top_corner)

                # The both-flip
                new_item = np.fliplr(np.flipud(item))
                if starting_pos == 0:
                    new_top_corner = (top_corner[0] + item.shape[0], top_corner[1] + item.shape[1])
                elif starting_pos == 1:
                    new_top_corner = (top_corner[0] + item.shape[0], top_corner[1] - item.shape[1])
                elif starting_pos == 2:
                    new_top_corner = (top_corner[0] - item.shape[0], top_corner[1] + item.shape[1])
                elif starting_pos == 3:
                    new_top_corner = (top_corner[0] - item.shape[0], top_corner[1] - item.shape[1])
                else:
                    assert False, f"Bad starting_pos {starting_pos} in replicate flower_flip"
                    
                new_working_list.append(new_item)
                new_top_corners.append(new_top_corner)
            elif sub_command == 'flower_rotate':
                assert len(rest.split()) == 1, dsl_usage('replicate')
                starting_pos = int(rest.split()[0])
                # Starting pos: 0 1
                #               2 3
                
                # Add the original item
                new_working_list.append(item)
                new_top_corners.append(top_corner)
                
                # The 1 rotate
                new_item = np.rot90(item, k=1, axes=(0, 1))
                if starting_pos in [0, 2]:
                    new_top_corner = (top_corner[0], top_corner[1] + item.shape[1])
                else:
                    new_top_corner = (top_corner[0], top_corner[1] - item.shape[1])
                new_working_list.append(new_item)
                new_top_corners.append(new_top_corner)

                 # The 2 rotate
                new_item = np.rot90(item, k=2, axes=(0, 1))
                if starting_pos in [0, 1]:
                    new_top_corner = (top_corner[0] + item.shape[0], top_corner[1])
                else:
                    new_top_corner = (top_corner[0] - item.shape[0], top_corner[1])
                new_working_list.append(new_item)
                new_top_corners.append(new_top_corner)

                # The 3 rotate
                new_item = np.rot90(item, k=3, axes=(0, 1))
                if starting_pos == 0:
                    new_top_corner = (top_corner[0] + item.shape[0], top_corner[1] + item.shape[1])
                elif starting_pos == 1:
                    new_top_corner = (top_corner[0] + item.shape[0], top_corner[1] - item.shape[1])
                elif starting_pos == 2:
                    new_top_corner = (top_corner[0] - item.shape[0], top_corner[1] + item.shape[1])
                elif starting_pos == 3:
                    new_top_corner = (top_corner[0] - item.shape[0], top_corner[1] - item.shape[1])
                else:
                    assert False, f"Bad starting_pos {starting_pos} in replicate flower_flip"
                    
                new_working_list.append(new_item)
                new_top_corners.append(new_top_corner)
            else:
                dsl_usage('replicate')
                assert False, f"Unknown sub-command: {sub_command}."
                                        
        self.working_list = new_working_list
        self.metadata = []
        for tc in new_top_corners:
            self.metadata.append(dict(top_corner = tc))
        
            
    def snake(self, args, debug=False):
        """
        """
        assert args is not None, dsl_usage('snake')
        DIRECTIONS = {'N', 'E', 'S', 'W'}
        DIR_AS_OFFSET = dict(N = (-1, 0),
                             E = ( 0, 1),
                             S = ( 1, 0),
                             W = ( 0,-1))
        TURN_RIGHT = dict(N = 'E',
                          E = 'S',
                          S = 'W',
                          W = 'N')
        TURN_LEFT  = dict(N = 'W',
                          E = 'N',
                          S = 'E',
                          W = 'S')

        args_split = args.split(' ', 1)
        sub_command = args_split[0]
        rest = args_split[1] if len(args_split) > 1 else None

        new_working_list = []
        for item in self.working_list:
            if sub_command == 'simple':
                assert len(rest.split()) == 12, dsl_usage('snake')
                start_value = int(rest.split()[0])
                direction = rest.split()[1]
                actions = []
                for i in range(2,12):
                    actions.append(rest.split()[i])
                if debug:
                    print(f"start_value: {start_value}, direction: {direction}, actions: {actions}")
                snake_map = np.zeros(item.shape, dtype=bool)
                safety = 0
                for row in range(item.shape[0]):
                    for col in range(item.shape[1]):
                        val = item[row,col]
                        if val == start_value and not snake_map[row,col]:
                            cr = row
                            cc = col
                            cd = direction
                            if cd == 'away':
                                if row == 0:
                                    cd = 'S'
                                elif col == 0:
                                    cd = 'E'
                                elif row == item.shape[0]-1:
                                    cd = 'N'
                                elif col == item.shape[1]-1:
                                    cd = 'W'
                                else:
                                    assert False, "Cannot start in free space with direction == 'away'."
                            
                            starting_dir = cd
                            going_around = False
                            other_paths = set()
                            while(cr + DIR_AS_OFFSET[cd][0] >= 0 and cr + DIR_AS_OFFSET[cd][0] < item.shape[0] and
                                  cc + DIR_AS_OFFSET[cd][1] >= 0 and cc + DIR_AS_OFFSET[cd][1] < item.shape[1]):
                                safety += 1
                                assert safety < 1000, "Too many iterations in snake, bailing out."
                                
                                made_early_move = False
                                consider_shortcut = False
                                
                                if going_around:
                                    if consider_shortcut:
                                        # Can the snake carry on in the original direction, regardless of the around direction, by cutting across the corner?
                                        pass 
                                        # TODO - Finish this bit.
                                        
                                    lookahead = item[cr + DIR_AS_OFFSET[starting_dir][0], cc + DIR_AS_OFFSET[starting_dir][1]]
                                    if actions[lookahead] == 'overwrite':
                                        # We were going around, but now we can go back to the original direction as the way is clear.
                                        cd = starting_dir
                                        going_around = False

                                lookahead = item[cr + DIR_AS_OFFSET[cd][0], cc + DIR_AS_OFFSET[cd][1]]
                                #if debug:
                                #    print(f"lookahead: {lookahead}")
                                #    print(f"actions[lookahead]: {actions[lookahead]}")
                                if actions[lookahead] == 'overwrite':
                                    item[cr + DIR_AS_OFFSET[cd][0], cc + DIR_AS_OFFSET[cd][1]] = val
                                    snake_map[cr + DIR_AS_OFFSET[cd][0], cc + DIR_AS_OFFSET[cd][1]] = True
                                    cr += DIR_AS_OFFSET[cd][0]
                                    cc += DIR_AS_OFFSET[cd][1]
                                    if not (cr + DIR_AS_OFFSET[cd][0] >= 0 and cr + DIR_AS_OFFSET[cd][0] < item.shape[0] and
                                            cc + DIR_AS_OFFSET[cd][1] >= 0 and cc + DIR_AS_OFFSET[cd][1] < item.shape[1]):
                                        # About to terminate loop, grab extra paths.
                                        if len(other_paths) > 0:
                                            if debug:
                                                print("Taking other paths.")
                                            cr, cc, cd, going_around = other_paths.pop()
                                elif actions[lookahead] == 'turn_right':
                                    cd = TURN_RIGHT[cd]
                                elif actions[lookahead] == 'turn_left':
                                    cd = TURN_LEFT[cd]
                                elif actions[lookahead] == 'around_right':
                                    cd = TURN_RIGHT[cd]
                                    going_around = True
                                elif actions[lookahead] == 'around_left':
                                    cd = TURN_LEFT[cd]
                                    going_around = True
                                elif actions[lookahead] == 'around_right_sc':
                                    cd = TURN_RIGHT[cd]
                                    going_around = True
                                    consider_shortcut = True
                                elif actions[lookahead] == 'around_left_sc':
                                    cd = TURN_LEFT[cd]
                                    going_around = True
                                    consider_shortcut = True
                                elif actions[lookahead] == 'around_both':
                                    going_around = True                                    
                                    other_paths.add((cr, cc, TURN_LEFT[cd], going_around))
                                    cd = TURN_RIGHT[cd]
                                elif actions[lookahead] == 'stop':
                                    if len(other_paths) > 0:
                                        if debug:
                                            print("Taking other paths.")
                                        cr, cc, cd, going_around = other_paths.pop()
                                    else:
                                        break
                                else:
                                    assert False, f"Bad action {actions[lookahead]} in snake simple."
                                                                                        
                new_working_list.append(item)
            else:
                dsl_usage('snake')
                assert False, f"Unknown sub-command: {sub_command}."
                                        
        self.working_list = new_working_list

        
    def split(self, args, debug=False):
        assert args is not None, dsl_usage('split')
        
        args_split = args.split(' ', 1)
        sub_command = args_split[0]
        rest = args_split[1] if len(args_split) > 1 else None
        
        new_working_list = []
        new_metadata = []
        for item in self.working_list:
            if sub_command == 'auto_grid':
                panels, top_corners = split_auto_grid(item, debug=debug)
            elif sub_command == 'by_value':
                assert len(rest.split()) == 2, dsl_usage('split')
                background = self.map_special_values(item, int(rest.split()[0]), debug=debug)
                crop = rest.split()[1]
                panels, top_corners = split_by_value(item, int(background), crop)
            elif sub_command == 'fixed_grid':
                assert len(rest.split()) == 3, dsl_usage('split')
                rows, columns, divider = rest.split()
                panels, top_corners = split_fixed_grid(item, int(rows), int(columns), divider == 'True')
            elif sub_command == 'fixed_size':
                assert len(rest.split()) == 3, dsl_usage('split')
                rows, columns, divider = rest.split()
                panels, top_corners = split_fixed_size(item, int(rows), int(columns), divider == 'True')
            elif sub_command == 'connected_region':
                assert len(rest.split()) == 4, dsl_usage('split')
                background = self.map_special_values(item, int(rest.split()[0]), debug=debug)
                _, single_value, neighbourhood, crop = rest.split()                
                eight_connected = True if int(neighbourhood) == 8 else False
                panels, top_corners = split_connected_region(item, background, single_value == 'True', crop == 'True', eight_connected=eight_connected, debug=debug)
            elif sub_command == 'frame':
                assert len(rest.split()) == 2, dsl_usage('split')
                background = self.map_special_values(item, int(rest.split()[0]), debug=debug)                    
                keep_frame = rest.split()[1]              
                panels, top_corners = split_frames(item, background, keep_frame == 'True', debug=debug)
            else:
                dsl_usage('split')
                assert False, f"Unknown sub-command: {sub_command}."

            new_working_list.extend(panels)
            for tc in top_corners:
                new_metadata.append(dict(top_corner = tc))

        self.working_list = new_working_list
        self.metadata = new_metadata

    def sort(self, args, debug=False):
        """
        """
        assert args is not None, dsl_usage('sort')

        args_split = args.split(' ', 1)
        sub_command = args_split[0]
        rest = args_split[1] if len(args_split) > 1 else None
        
        new_working_list = []
        if sub_command == 'unique_values':
            assert len(rest.split()) == 1, dsl_usage('sort')
            ordering = rest.split()[0]
            helper = ScoringHelper()
            for i in range(len(self.working_list)):
                helper.add_score(i, len(np.unique(self.working_list[i])))
            new_order = helper.get_ordering(ordering)
            for idx in new_order:
                new_working_list.append(self.working_list[idx])
        elif sub_command == 'by_value':
            assert len(rest.split()) == 2, dsl_usage('sort')
            ordering = rest.split()[1]            
            helper = ScoringHelper()
            for i in range(len(self.working_list)):
                item = self.working_list[i]
                value = self.map_special_values(item, int(rest.split()[0]), debug=debug)                
                values, counts = np.unique(item, return_counts=True)
                counts_dict = dict(zip(values, counts))
                helper.add_score(i, counts_dict.get(value, 0))
            new_order = helper.get_ordering(ordering)
            for idx in new_order:
                new_working_list.append(self.working_list[idx])
        else:
            dsl_usage('sort')
            assert False, f"Unknown sub-command: {sub_command}."
                
        self.working_list = new_working_list

    def transform(self, args, debug=False):
        """
        """
        assert args is not None, dsl_usage('transform')
        
        args_split = args.split(' ', 1)
        sub_command = args_split[0]
        rest = args_split[1] if len(args_split) > 1 else None

        new_working_list = []
        for item in self.working_list:
            if sub_command == 'crop':
                assert len(rest.split()) == 1, dsl_usage('transform')
                background = self.map_special_values(item, int(rest.split()[0]), debug=debug)  
                cropped_item, top_corner = crop_background(item, background, debug=debug)
                new_working_list.append(cropped_item)
            elif sub_command == 'flip':
                assert len(rest.split()) == 1, dsl_usage('transform')
                if rest.split()[0] == 'lr':                    
                    new_working_list.append(np.fliplr(item))
                elif rest.split()[0] == 'ud':
                    new_working_list.append(np.flipud(item))
                else:
                    dsl_usage('transform')
                    assert False, f"Bad value ({rest.split()[0]}) given to 'transform flip'."
            elif sub_command == 'invert':
                values = np.unique(item)
                if len(values) == 2:
                    new_working_list.append(np.where(item == values[0], values[1], values[0]))
                else:
                    new_working_list.append(item)
            elif sub_command == 'rot90':
                assert len(rest.split()) == 1, dsl_usage('transform')
                new_working_list.append(np.rot90(item, k=int(rest.split()[0]), axes=(0, 1)))
            else:
                dsl_usage('transform')
                assert False, f"Unknown sub-command: {sub_command}."
                                        
        self.working_list = new_working_list

        
    def value_map(self, args, debug=False):
        """
        """
        assert args is not None, dsl_usage('value_map')
        
        args_split = args.split(' ', 1)
        sub_command = args_split[0]
        rest = args_split[1] if len(args_split) > 1 else None

        new_working_list = []
        for idx in range(len(self.working_list)):
            item = self.working_list[idx]
            new_item = item.copy()
            if sub_command == 'simple':
                assert len(rest.split()) == 2, dsl_usage('value_map')
                from_val = self.map_special_values(item, int(rest.split()[0]), debug=debug)                
                to_val = self.map_special_values(item, int(rest.split()[1]), debug=debug)
                new_item = np.where(new_item == from_val, to_val, new_item)
                new_working_list.append(new_item)
            elif sub_command == 'enclosures_count':
                assert len(rest.split()) == 3, dsl_usage('value_map')
                background = self.map_special_values(item, int(rest.split()[0]), debug=debug)                
                count = int(rest.split()[1])
                value = int(rest.split()[2])
                if count_enclosures(item, background, debug=debug) == count:
                    new_item = np.where(new_item != background, value, background)
                new_working_list.append(new_item)
            elif sub_command == 'shape_match':
                assert len(rest.split()) == 2, dsl_usage('value_map')
                source_value_not = self.map_special_values(item, int(rest.split()[0]), debug=debug)
                allow_rotations = rest.split()[1] == 'True'
                most_common = get_most_common_value(item)
                if most_common == source_value_not:
                    # This is not a source, so needs mapping. Look for a matching shape.
                    bin_item = np.where(item == 0, False, True)
                    for j in range(len(self.working_list)):
                        if j != idx:
                            potential_source = self.working_list[j]
                            source_value = get_most_common_value(potential_source)
                            if source_value != source_value_not:
                                # Do we have a match by shape?
                                bin_source = np.where(potential_source == 0, False, True)
                                rot90s = range(1)
                                if allow_rotations:
                                    rot90s = range(4)
                                
                                for rot90 in rot90s:
                                    if np.array_equal(bin_item, np.rot90(bin_source, k=rot90, axes=(0, 1))):
                                        # Yes, we have a match. Job done.
                                        if debug:
                                            print(f"Matched item {idx} to source {j}, so changed value to {source_value}.")
                                        new_item = np.where(bin_item == True, source_value, 0)
                                        break

                new_working_list.append(new_item)
            else:
                dsl_usage('value_map')
                assert False, f"Unknown sub-command: {sub_command}."
                             
        assert len(self.working_list) == len(new_working_list), "Have lost items in value_map. This is not allowed."
        self.working_list = new_working_list


# # 3 - Examples and Testing
# 
# Setup paths to the competition data. Also borrow from https://www.kaggle.com/ademiquel/data-preprocessing-correcting-tasks-with-errors to correct some known errors in the competition data. The dataset includes a small custom set of tasks developed for testing - see https://www.kaggle.com/andypenrose/visualize-extra-arc-tasks-for-testing for more details.
# 
# When developing the DSL, it was important to keep testing it on previous known good examples, so as to avoid breaking anything. A set of tests are therefore built in below. If any previous good example fails to correctly answer a given task, it asserts to avoid wasting time with running beyond when there is a bug.

# In[ ]:


root_path = Path('/kaggle/input/abstraction-and-reasoning-challenge')
training_path = root_path / 'training'
evaluation_path  = root_path / 'evaluation'
test_path = root_path / 'test'
extra_tasks_path = Path('/kaggle/input/extra-arc-tasks-for-testing/')

train_tasks = { task.stem: json.load(task.open()) for task in training_path.iterdir() } 
valid_tasks = { task.stem: json.load(task.open()) for task in evaluation_path.iterdir() } 
test_tasks = { task.stem: json.load(task.open()) for task in test_path.iterdir() }
extra_tasks = { task.stem: json.load(task.open()) for task in extra_tasks_path.iterdir() }

print(f"Num training tasks: {len(train_tasks)}")
print(f"Num evaluation tasks: {len(valid_tasks)}")
print(f"Num test tasks: {len(test_tasks)}")
print(f"Num extra tasks: {len(extra_tasks)}")


# In[ ]:


# From: https://www.kaggle.com/ademiquel/data-preprocessing-correcting-tasks-with-errors

# A discussion thread asks us to list any errors that we have found in the tasks. Since there are quite a few tasks with errors, I think that it's worth it to have them corrected. So I went through all of the tasks reported on that thread and corrected them manually. This is now the first step I take when I preprocess the data. You can find the code below. I hope you find it useful.

# Correct wrong cases:
# 025d127b
for i in range(9, 12):
    for j in range(3, 8):
        train_tasks['025d127b']['train'][0]['output'][i][j] = 0
for i in range(7, 10):
    for j in range(3, 6):
        train_tasks['025d127b']['train'][0]['output'][i][j] = 2
train_tasks['025d127b']['train'][0]['output'][8][4] = 0
# ef135b50
train_tasks['ef135b50']['test'][0]['output'][6][4] = 9
# bd14c3bf
for i in range(3):
    for j in range(5):
        if valid_tasks['bd14c3bf']['test'][0]['input'][i][j] == 1:
            valid_tasks['bd14c3bf']['test'][0]['input'][i][j] = 2
# a8610ef7
for i in range(6):
    for j in range(6):
        if valid_tasks['a8610ef7']['test'][0]['output'][i][j] == 8:
            valid_tasks['a8610ef7']['test'][0]['output'][i][j] = 5
valid_tasks['a8610ef7']['train'][3]['input'][0][1] = 2
valid_tasks['a8610ef7']['train'][3]['input'][5][1] = 2
# 54db823b
valid_tasks['54db823b']['train'][0]['output'][2][3] = 3
valid_tasks['54db823b']['train'][0]['output'][2][4] = 9
# e5062a87
for j in range(3, 7):
    train_tasks['e5062a87']['train'][1]['output'][1][j] = 2
# 1b60fb0c
train_tasks['1b60fb0c']['train'][1]['output'][8][8] = 0
train_tasks['1b60fb0c']['train'][1]['output'][8][9] = 0
# 82819916
train_tasks['82819916']['train'][0]['output'][4][5] = 4
# fea12743
for i in range(11, 16):
    for j in range(6):
        if valid_tasks['fea12743']['train'][0]['output'][i][j] == 2:
            valid_tasks['fea12743']['train'][0]['output'][i][j] = 8
# 42a50994
train_tasks['42a50994']['train'][0]['output'][1][0] = 8
train_tasks['42a50994']['train'][0]['output'][0][1] = 8
# f8be4b64
for j in range(19):
    if valid_tasks['f8be4b64']['test'][0]['output'][12][j] == 0:
        valid_tasks['f8be4b64']['test'][0]['output'][12][j] = 1
valid_tasks['f8be4b64']['test'][0]['output'][12][8] = 0
# d511f180
train_tasks['d511f180']['train'][1]['output'][2][2] = 9
# 10fcaaa3
train_tasks['10fcaaa3']['train'][1]['output'][4][7] = 8
# cbded52d
train_tasks['cbded52d']['train'][0]['input'][4][6] = 1
# 11852cab
train_tasks['11852cab']['train'][0]['input'][1][2] = 3
# 868de0fa
for j in range(2, 9):
    train_tasks['868de0fa']['train'][2]['input'][9][j] = 0
    train_tasks['868de0fa']['train'][2]['input'][10][j] = 1
    train_tasks['868de0fa']['train'][2]['input'][15][j] = 0
    train_tasks['868de0fa']['train'][2]['input'][16][j] = 1
train_tasks['868de0fa']['train'][2]['input'][15][2] = 1
train_tasks['868de0fa']['train'][2]['input'][15][8] = 1
# 6d58a25d
train_tasks['6d58a25d']['train'][0]['output'][10][0] = 0
train_tasks['6d58a25d']['train'][2]['output'][6][13] = 4
# a9f96cdd
train_tasks['a9f96cdd']['train'][3]['output'][1][3] = 0
# 48131b3c
valid_tasks['48131b3c']['train'][2]['output'][4][4] = 0
# 150deff5
aux = train_tasks['150deff5']['train'][2]['output'].copy()
train_tasks['150deff5']['train'][2]['output'] = train_tasks['150deff5']['train'][2]['input'].copy()
train_tasks['150deff5']['train'][2]['input'] = aux
# 17cae0c1
for i in range(3):
    for j in range(3, 6):
        valid_tasks['17cae0c1']['test'][0]['output'][i][j] = 9
# e48d4e1a
train_tasks['e48d4e1a']['train'][3]['input'][0][9] = 5
train_tasks['e48d4e1a']['train'][3]['output'][0][9] = 0
# 8fbca751
valid_tasks['8fbca751']['train'][1]['output'][1][3] = 2
valid_tasks['8fbca751']['train'][1]['output'][2][3] = 8
# 4938f0c2
for i in range(12):
    for j in range(6,13):
        if train_tasks['4938f0c2']['train'][2]['input'][i][j]==2:
            train_tasks['4938f0c2']['train'][2]['input'][i][j] = 0
for i in range(5,11):
    for j in range(7):
        if train_tasks['4938f0c2']['train'][2]['input'][i][j]==2:
            train_tasks['4938f0c2']['train'][2]['input'][i][j] = 0
# 9aec4887
train_tasks['9aec4887']['train'][0]['output'][1][4] = 8
# b0f4d537
for i in range(9):
    valid_tasks['b0f4d537']['train'][0]['output'][i][3] = 0
    valid_tasks['b0f4d537']['train'][0]['output'][i][4] = 1
valid_tasks['b0f4d537']['train'][0]['output'][2][3] = 3
valid_tasks['b0f4d537']['train'][0]['output'][2][4] = 3
valid_tasks['b0f4d537']['train'][0]['output'][5][3] = 2
# aa300dc3
valid_tasks['aa300dc3']['train'][1]['input'][1][7] = 5
valid_tasks['aa300dc3']['train'][1]['output'][1][7] = 5
valid_tasks['aa300dc3']['train'][1]['input'][8][2] = 5
valid_tasks['aa300dc3']['train'][1]['output'][8][2] = 5
# ad7e01d0
valid_tasks['ad7e01d0']['train'][0]['output'][6][7] = 0
# a8610ef7
valid_tasks['a8610ef7']['train'][3]['input'][0][1] = 0
valid_tasks['a8610ef7']['train'][3]['input'][5][1] = 0
valid_tasks['a8610ef7']['train'][3]['output'][0][1] = 0
valid_tasks['a8610ef7']['train'][3]['output'][5][1] = 0
# 97239e3d
valid_tasks['97239e3d']['test'][0]['input'][14][6] = 0
valid_tasks['97239e3d']['test'][0]['input'][14][10] = 0
# d687bc17
train_tasks['d687bc17']['train'][2]['output'][7][1] = 4


# I used different variable names...
training_tasks = train_tasks
evaluation_tasks = valid_tasks
all_tasks = {**training_tasks, **evaluation_tasks, **extra_tasks}


# In[ ]:


# Used in many notebooks, this version of plot_task is adapted from: https://www.kaggle.com/nanoix9/a-naive-image-manipulation-dsl

def plot_one(ax, input_matrix, i, train_or_test, input_or_output, title_color='black'):
    ax.imshow(input_matrix, cmap=cmap, norm=norm)
    ax.grid(True,which='both',color='lightgrey', linewidth=0.5)    
    ax.set_yticks([x-0.5 for x in range(1+len(input_matrix))])
    ax.set_xticks([x-0.5 for x in range(1+len(input_matrix[0]))])     
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title('{} - {:d} - {}'.format(train_or_test, i, input_or_output), color=title_color)
    
def plot_task(task, prog=None, debug=False):
    """
    Plots the first train and test pairs of a specified task,
    using same color scheme as the ARC app
    """    
    
    extra_line = 1 if prog is not None else 0

    def _plot_helper(train_or_test):
        num_imgs = len(task[train_or_test])
        fig, axs = plt.subplots(2 + extra_line, num_imgs, figsize=(3*num_imgs,3*2))
        for i in range(num_imgs):
            imgs = task[train_or_test][i]
            input_img = np.array(imgs['input'], dtype=np.int)
            output_img = np.array(imgs.get('output', np.zeros_like(input_img)), dtype=np.int)
            if num_imgs > 1:
                axs_col = axs[:, i]
            else:
                axs_col = axs
            plot_one(axs_col[0], input_img, i,train_or_test,'input')
            plot_one(axs_col[1], output_img, i,train_or_test,'output')
            if prog is not None:
                interp = Interpreter()
                pred = interp.execute(input_img, prog, debug=debug)
                color = 'green' if output_img.shape == pred.shape and np.all(output_img == pred) else 'red'
                plot_one(axs_col[2], pred, i, train_or_test, 'prediction', title_color=color)
        plt.tight_layout()
        plt.show()
    
    _plot_helper('train')
    _plot_helper('test')


# In[ ]:


# Testing area...

task_id = 'f9a67cb5'
prog = ['snake simple 2 away overwrite stop stop stop stop stop stop stop around_both stop']


task = all_tasks[task_id]
plot_task(task, prog, debug=False)


# In[ ]:


def check_predictions(task, prog, train_or_test, assert_on_diff=False):
    """
    Check the given prog solves the given task. Can be used to check train or test parts of the task depending on the
    train_or_test argument. If used in testing, setting assert_on_diff can be used to halt the run if a previous good
    example breaks.
    """
       
    for input_array, output_array in zip([np.array(x['input']) for x in task[train_or_test]],
                                         [np.array(x['output']) for x in task[train_or_test]]):
        
        interp = Interpreter()
        try:
            result = interp.execute(input_array, prog)
            if not np.array_equal(result, output_array):
                if assert_on_diff:
                    assert False, f"Unexpected prediction failure. prog={prog}"
                return False
        except:
            if assert_on_diff:
                assert False, f"Unexpected prediction failure. prog={prog}"
            return False            
        
    return True


# In[ ]:


SHOW_EXAMPLES = False

test_examples = [('1c0d0a4b', ['split auto_grid', 'value_map simple 8 2', 'transform invert' ,'assemble original zeros']),
                 ('7953d61e', ['replicate flower_rotate 0' ,'assemble auto_grid']),
                 ('0c786b71', ['replicate flower_flip 3' ,'assemble auto_grid']),
                 ('f9a67cb5', ['snake simple 2 away overwrite stop stop stop stop stop stop stop around_both stop']),
                 ('96a8c0cd', ['snake simple 2 away overwrite around_left overwrite around_right stop stop stop stop stop stop']),
                 ('3345333e', ['split by_value 0 True', 'filter by_size keep most', 'replicate and_merge lr False auto', 'assemble original zeros']),
                 ('e88171ec', ['split frame 10 False', 'value_map simple 0 8', 'assemble original original']),
                 ('5614dbcf', ['split fixed_size 3 3 False', 'abstract simple 1', 'assemble auto_grid']),
                 ('56ff96f3', ['split by_value 10 True', 'value_map simple 12 13', 'assemble original zeros']),
                 ('1a6449f1', ['split frame 10 False', 'filter by_size keep most']),
                 ('f45f5ca7', ['split connected_region 0 True 4 True', 'move by_value 2 E 2', 'move by_value 4 E 3', 'move by_value 3 E 4', 'move by_value 8 E 1', 'assemble original zeros']),
                 ('1f85a75f', ['split connected_region 0 False 4 True', 'filter by_value keep 11 most']),
                 ('2dee498d', ["split fixed_grid 1 3 False", "filter by_index keep 0"]),
                 ('2dc579da', ["split fixed_grid 2 2 True", "filter unique_values keep most"]),
                 ('a87f7484', ["split fixed_size 3 3 False", "filter by_value keep 0 least"]),
                 ('ce4f8723', ["split fixed_grid 2 1 True", "combine logical_op False or False 3"]),
                 ('cf98881b', ["split fixed_grid 1 3 True", "combine overwrite 0 [2,1,0]"]),
                 ('ae4f1146', ["split connected_region 0 False 4 True", "filter by_value keep 1 most"]),
                 ('39a8645d', ["split connected_region 0 True 8 True", "filter commonality keep most"]),
                 ('e74e1818', ["split by_value 0 True", "transform flip ud", "assemble original zeros"]),
                 ('54db823b', ['split connected_region 0 False 8 True', 'filter by_value remove 9 least', 'assemble original zeros']),
                 ('a934301b', ['split connected_region 0 False 8 True', 'filter by_value_gte remove 8 2', 'assemble original zeros']),
                 ('85b81ff1', ['split connected_region 0 False 4 True', 'sort by_value 0 descending', 'assemble original zeros']),
                 ('67636eac', ['split connected_region 0 False 8 True', 'assemble auto_grid']),
                 ('64a7c07e', ['split connected_region 0 True 8 True', 'move by_shape H SE', 'assemble original zeros']),
                 ('2c0b0aff', ['split connected_region 0 False 4 True', 'filter by_shape_count keep cross 12 most']),
                 ('73ccf9c2', ['split connected_region 0 True 8 True', 'filter has_symmetry remove']),
                 ('845d6e51', ['split connected_region 0 True 4 True', 'value_map shape_match 3 True', 'assemble original zeros']),
                 ('810b9b61', ['split connected_region 0 True 4 True', 'value_map enclosures_count 0 1 3', 'assemble original zeros']),
                 ('b2862040', ['split connected_region 10 True 8 True', 'value_map enclosures_count 0 1 8', 'assemble original majority_value']),
                 ('0a1d4ef5', ['split connected_region 0 True 4 True', 'filter rectangular keep 3', 'abstract simple 1', 'assemble auto_grid']),
                 ('48d8fb45', ['split connected_region 0 False 8 True', 'filter by_value keep 5 most', 'value_map simple 5 0', 'transform crop 0']),
                 ('9ddd00f0', ['replicate and_merge none True auto']),
                 ('9f236235', ['split auto_grid', 'abstract simple 1', 'assemble auto_grid', 'transform flip lr']),
                 ('b8825c91', ['split by_value 0 True', 'filter by_value_gte remove 4 1', 'assemble original zeros', 'replicate and_merge none True none']),
                 ('aee291af', ['split frame 10 True', 'filter commonality keep least']),
                 ('c909285e', ['split by_value 0 inclusive', 'filter by_size keep least']),
                 ('9565186b', ['split connected_region 0 True 4 True', 'filter by_value remove 10 most', 'value_map simple 12 5', 'assemble original original']),
                 ('bf699163', ['split by_value 10 True', 'filter enclosed keep']),
                 ('8ee62060', ['transform flip lr', 'split connected_region 0 False 4 True', 'transform flip lr', 'assemble original zeros'])]

for task_id, prog in test_examples:
    task = all_tasks[task_id]
    if SHOW_EXAMPLES:
        print(prog)
        plot_task(task, prog, debug=False)
    try:
        check_predictions(task, prog, 'train', assert_on_diff=True)
        check_predictions(task, prog, 'test', assert_on_diff=True)
    except:
        print(f"Error for task_id: {task_id}")
        raise
        
print(f"Checked {len(test_examples)} examples of using the DSL.")


# # 4 - Search Algorithm
# 
# The search algorithm finds programs to solve tasks. It is not sophiscated enough to deserve being called a program synthesis engine! The logic is mostly built from observing command patterns that are succesful on the training and evaluation sets, then iterating over argument combinations. Ideally this would be totally replaced with something that actually learnt from the training data.
# 
# build_prog_for_task_v2 takes the task, and considers the application of numerous split commands. Applicable split commands, filtering out those that create too many arrays in the working list, are then considered. The rest of the program search is carried out by either complete_prog_for_single_frame or complete_prog_for_multiple_frames depending on how many frames are returned from the split command.
# 
# Buried in the hidden code section is search_for_final_commands_args. This function takes a starting program and considers permutations of a given command to complete the program. The sub-commands to be used are given. Then all argument permutations are computed using the allow ranges for arguments as given in the DSL definition. Arguments hints can be passed to speed up the search when extra information is already known (like which input values are actually present in all the input arrays for a task). search_for_final_commands_args can also take a list of post-ops to be tried after the main command.

# In[ ]:


def does_10_map_to_something_other_than_0(task):
    for train_input in [np.array(x['input']) for x in task['train']]:
        interp = Interpreter()
        _ = interp.execute(train_input, ['identity'])
        if interp.map_special_values(train_input, 10) != 0:
            return True
    return False

def does_11_map_to_something_other_than_0(task):
    for train_input in [np.array(x['input']) for x in task['train']]:
        interp = Interpreter()
        _ = interp.execute(train_input, ['identity'])
        if interp.map_special_values(train_input, 11) != 0:
            return True
    return False

def do_all_task_inputs_contain_0(task):
    for train_input in [np.array(x['input']) for x in task['train']]:
        if np.count_nonzero(train_input == 0) == 0:
            return False
    return True

    
def determine_split_simple(task, pre_commands=None, debug=False):
    """
    """
    output_shapes = set()
    grid_sizes = set()
    divider = set()
    for train_input, train_output in zip([np.array(x['input']) for x in task['train']],[np.array(x['output']) for x in task['train']]):
        if train_input.shape[0] <= train_output.shape[0] and train_input.shape[1] <= train_output.shape[1]:
            # Output not smaller than input, cannot be a split and select/combine task.
            return None
        
        if pre_commands is not None:            
            interp = Interpreter()
            train_input = interp.execute(train_input, pre_commands, debug=debug)
            if debug:
                print(f"Train input:\n{train_input}")

        output_shapes.add(train_output.shape)
        grid_size = ((train_input.shape[0]//train_output.shape[0]),(train_input.shape[1]//train_output.shape[1]))
        grid_sizes.add(grid_size)
        if train_output.shape[0]*grid_size[0] == train_input.shape[0] and            train_output.shape[1]*grid_size[1] == train_input.shape[1]:
            # This input is exactly made up of a grid of tiles the size of the output. No dividers.
            divider.add(False)
        elif train_output.shape[0]*grid_size[0]+grid_size[0]-1 == train_input.shape[0] and              train_output.shape[1]*grid_size[1]+grid_size[1]-1 == train_input.shape[1]:
            # There are extra rows and columns enough to account fo dividers.
            # TODO - Check the divider rows/columns are all a single value to avoid wasting time on cases that are not really
            # made up of multiple panels.
            divider.add(True)
        else:
            # Inconclusive, probably not made up of multiple panels in a regular grid.
            return None

    if debug:
        print(f"output_shapes: {output_shapes}")
        print(f"grid_sizes: {grid_sizes}")
        print(f"divider: {divider}")

    if len(divider) != 1:
        # Search was inconclusive about whether a divider is present. Not supported.
        return None
    
    split_cmd = None
    if len(output_shapes) == 1:
        # If all the train outputs have the same shape, assume a grid of fixed sized panels.
        output_shape = output_shapes.pop()
        split_cmd = f"split fixed_size {output_shape[0]} {output_shape[1]} {divider.pop()}"
    elif len(grid_sizes) == 1:
        # If all the train inputs appear to be have the same grid of panels, assume a fixed grid of different sized panels.
        grid_size = grid_sizes.pop()
        split_cmd = f"split fixed_grid {grid_size[0]} {grid_size[1]} {divider.pop()}"
        
    return split_cmd

def determine_split_connected_region(task, background, single_value, neighbourhood, culling_commands=[], debug=False):
    """
    """
    split_cmd = f"split connected_region {background} {single_value} {neighbourhood} True"
    prog = None
    max_panels = 0
    
    culling_commands_plus_none = [None]
    culling_commands_plus_none.extend(culling_commands)
    
    for culling_command in culling_commands_plus_none:
        prog = None
        max_panels = 0
        for train_input, train_output in zip([np.array(x['input']) for x in task['train']],[np.array(x['output']) for x in task['train']]):
            interp = Interpreter()
            prog = [split_cmd]
            if culling_command is not None:
                prog.append(culling_command)
            panels = interp.execute(train_input, prog, partial=True, debug=debug)
            if debug:
                print(f"Found {len(panels)} panels.")
            if len(panels) > 0:
                max_panels = max(max_panels, len(panels))
                none_match_shape = True
                for p in panels:
                    if p.shape == train_output.shape:
                        none_match_shape = False

                if none_match_shape:
                    if debug:
                        print(f"No panels match output shape {train_output.shape}.")                    
                    if len(panels) > A_LOT:
                        # Too many panels and none have the correct shape, probably not actual objects that have been extracted.
                        # break, maybe another culling command will help.
                        prog = None
                        break
                elif len(panels) > TOO_MANY:
                    # Too many panels, probably not actual objects that have been extracted.
                    # break, maybe another culling command will help.
                    prog = None
                    break
            else:
                return None, None
        
        if prog is not None:
            return prog, max_panels
        
    return None, None
    
def get_panel_info_for_split(task, input_or_output, split_prog):
    singular_override = False
    for line in split_prog:
        if 'fixed_grid' in line.split() and input_or_output == 'output':
            singular_override = True
            
    result_counts = []
    result_shapes = []
    for train_input in [np.array(x[input_or_output]) for x in task['train']]:
        if singular_override:
            result_shapes.append({train_input.shape})
            result_counts.append(1)
        else:
            interp = Interpreter()
            panels = interp.execute(train_input, split_prog, partial=True)
            result_shapes.append({x.shape for x in panels})
            result_counts.append(len(panels))
    return result_counts, result_shapes
    
def get_min_max_panels(task, split_prog):
    """
    """
    counts, _ = get_panel_info_for_split(task, 'input', split_prog)
    return min(counts), max(counts)

def does_prog_give_single_panel_with_same_size_as_output(task, prog):
    """
    """
    for train_input, train_output in zip([np.array(x['input']) for x in task['train']],[np.array(x['output']) for x in task['train']]):
        interp = Interpreter()
        try:
            prediction = interp.execute(train_input, prog)
        except:
            return False
        else:
            if prediction.shape != train_output.shape:
                return False

    # If we go here, the prog runs on all train inputs and gives a single output prediction with the same shape
    # as the corresponding output.
    return True

def panels_same_size_per_input(task, prog):
    """
    """
    for train_input in [np.array(x['input']) for x in task['train']]:
        interp = Interpreter()
        panels = interp.execute(train_input, prog, partial=True)
        if len(panels) is None:
            return False
        
        panel_shape = None
        for p in panels:
            if panel_shape is None:
                panel_shape = p.shape
            elif panel_shape != p.shape:
                return False
    
    return True

def get_output_is_always_an_input_panel(task, pre_commands):
    output_is_an_input_panel= True
    for train_input, train_output in zip([np.array(x['input']) for x in task['train']],[np.array(x['output']) for x in task['train']]):
        interp = Interpreter()
        panels = interp.execute(train_input, pre_commands, partial=True)

        this_output_is_an_input_panel = False
        for p in panels:
            if p.shape == train_output.shape and np.array_equal(p, train_output):
                this_output_is_an_input_panel = True
            
        if not this_output_is_an_input_panel:
            output_is_an_input_panel = False
            break
    
    return output_is_an_input_panel

def get_all_input_or_output_values(task, input_or_output):
    array_values = set()
    for train_array in [np.array(x[input_or_output]) for x in task['train']]:
        array_values = array_values.union(set(np.unique(train_array)))
    return array_values

def get_values_added_in_diff(task):
    values_in_diff = set()
    for train_input, train_output in zip([np.array(x['input']) for x in task['train']],[np.array(x['output']) for x in task['train']]):
        if train_input.shape != train_output.shape:
            return None
        diff = np.where(train_input != train_output, train_output, -1)
        local_values = set(np.unique(diff))
        if -1 in local_values:
            local_values.remove(-1)
        values_in_diff = values_in_diff.union(local_values)
    return values_in_diff


def search_for_final_commands_args(task, prog_prefix, command, sub_commands, arg_hints=None, post_ops=None, final_sequence=None, terminate_early=True, use_scoring=False, filter_opt_info=None, debug=False):
    """
    This function takes a starting program and considers permutations of the given command to complete the program. The sub-commands
    to be used are given. Then all argument permutations are computed using the allow ranges for arguments as given in the DSL definition.
    Arguments hints can be passed to speed up the search when extra information is already known (like which input values are actually
    present in all the input arrays for a task). search_for_final_commands_args can also take a list of post-ops to be tried after
    the main command.
    """
    assert not use_scoring or len(sub_commands) == 1, "use_scoring doesn't currently support multiple post_ops."
    assert not use_scoring or len(sub_commands) == 1, "use_scoring should only be used with a single sub_command."
    
    overall_working_cmds = []
    filter_with_impact = []
    for subcmd in sub_commands:
        # Calculate the number of combinations of allowed args for this sub-command.
        combinations = 1
        args = dsl[command].subcmddict[subcmd].arglist
        if len(args) == 0:
            all_arg_strings = [""]
        else:
            for i in range(len(args)):
                if arg_hints is not None and subcmd in arg_hints.keys() and arg_hints[subcmd][i] is not None:
                    combinations *= len(arg_hints[subcmd][i])
                else:
                    assert args[i].allowed_values is not None, "Must provide arg_hint for arguments without statically known allowed values."
                    combinations *= len(args[i].allowed_values)

            if debug:
                print(f"Arguments combinations for {command} {subcmd} total {combinations}.")
        
            assert combinations < 1000000, "Too many combinations to search in search_for_final_commands_args."
                
            # For the first argument, spread the allowed values evenly over the number of cominbations. That means repeating each allowed value "repeats = combinations//len(allowed_values)" times.
            # For the second argument, cycle through the allowed values, repeating each value "repeats = repeats/len(allowed_values)" times. And so on...
            all_arg_strings = [""] * combinations
            repeats = combinations

            for i in range(len(args)):
                arg = args[i]
                allowed_values = []
                if arg_hints is not None and subcmd in arg_hints.keys() and arg_hints[subcmd][i] is not None:
                    allowed_values = arg_hints[subcmd][i]
                else:
                    allowed_values = arg.allowed_values

                index = 0
                repeats = repeats//len(allowed_values)
                while index < combinations:
                    for v in allowed_values:
                        for i in range(repeats):
                            all_arg_strings[index] += " " + str(v)
                            index += 1
                        
        #if debug:
        #    for argcomb in all_arg_strings:
        #        cmd = f"{command} {subcmd}{argcomb}"
        #        print(cmd)

        num_post_ops = len(post_ops) if post_ops is not None else 0

        working_cmds = None
        working_idxs = None

        scores = [0] * combinations
        all_trial_cmds = [""] * combinations
        train_idx = -1
        for train_input, train_output in zip([np.array(x['input']) for x in task['train']],[np.array(x['output']) for x in task['train']]):
            train_idx += 1
            local_working_cmds = []
            local_working_idxs = []
            for argidx in range(combinations):
                if not use_scoring and working_idxs is not None and len(working_idxs) > 0 and argidx not in working_idxs:
                    # We're not using scoring, we have a list of working commands from earlier train inputs, and this command
                    # isn't in the list.
                    continue
                    
                argcomb = all_arg_strings[argidx]
                for post_op_index in range(-1,num_post_ops):
                    
                    cmd = f"{command} {subcmd}{argcomb}"
                    trial_cmds = [cmd]
                    all_trial_cmds[argidx] = cmd
                    
                    if post_op_index >= 0:
                        if isinstance(post_ops[post_op_index], list):
                            trial_cmds.extend(post_ops[post_op_index])
                        else:
                            trial_cmds.append(post_ops[post_op_index])

                    prog = prog_prefix.copy()                    
                    prog.extend(trial_cmds)

                    if final_sequence is not None:
                        prog.extend(final_sequence)
                        
                    if debug:
                        print(prog)
                        
                    interp = Interpreter()
                    try:
                        objects = interp.execute(train_input, prog, partial=True, debug=debug)
                        if len(objects) == 1:
                            prediction = objects[0]
                            if np.array_equal(prediction, train_output):
                                local_working_cmds.append(trial_cmds)
                                local_working_idxs.append(argidx)
                                # Note that even if we are using scoring, the score increment here is zero as the arrays are equal.
                            elif use_scoring:
                                diff = train_output - prediction
                                scores[argidx] += np.count_nonzero(diff)
                        elif filter_opt_info is not None and post_op_index == -1 and train_idx == 0:
                            if len(objects) == 0:
                                # Filter left no objects left, no point carrying on.
                                break
                            elif len(objects) == filter_opt_info[0][train_idx]:
                                # The filtering didn't change the number of panels from the input. No point trying all post-ops.
                                break
                            else:
                                # Filtering has given a changed number of objects, without removing them all.                                                                
                                filter_with_impact.append(trial_cmds)
                    except Exception as inst:                       
                        if debug:
                            print(f"Error executing: {prog}")
                            print(type(inst))    # the exception instance
                            print(inst.args)     # arguments stored in .args
                            print(inst) 
                        # If this is the first post_op, no point continuing through the others!
                        #if post_op_index == 0:
                        #    break
                        
            if debug:
                print(f"working_cmds: {working_cmds}")
                print(f"local_working_cmds: {local_working_cmds}")
                
            if working_cmds is None:
                working_cmds = local_working_cmds
                working_idxs = local_working_idxs
            else:
                working_cmds = [x for x in working_cmds if x in local_working_cmds]
                working_idxs = [x for x in working_idxs if x in local_working_idxs]
                
            if working_cmds == [] and not use_scoring:
                # No consistent command found based on this sub-command; try next.
                break
                
        if not use_scoring and working_cmds is not None and len(working_cmds) > 0 and terminate_early:
            return working_cmds[0]
        else:
            overall_working_cmds.extend(working_cmds)
    
    if debug:
        print(f"Found {len(overall_working_cmds)} working {command} commands.")
        print(overall_working_cmds)
    
    if not use_scoring and len(overall_working_cmds) > 0:
        return overall_working_cmds[0]

    # We may not have found a command that gives a perfect result, but if using scoring, we'll at least return the command that gives the best score.
    # Even if we have found a perfect result, we must come through here, as otherwise the post-op gets applied incorrectly.
    if use_scoring:
        helper = ScoringHelper()
        for idx in range(len(scores)):
            helper.add_score(idx, scores[idx])
        winner = helper.get_winner('least')
        if winner is not None:
            if debug:
                print(f"Best scoring command: {all_trial_cmds[winner]}")
            return all_trial_cmds[winner]

    if debug and len(filter_with_impact) > 0:
        print(f"Filter commands with impact: {len(filter_with_impact)}")
        
    return None

def search_for_simple_value_maps(task, prog, post_op, debug=False):
    """
    """
    input_values = get_all_input_or_output_values(task, 'input')
    output_values = get_all_input_or_output_values(task, 'output')
    input_values.update([10, 11])
    output_values.update([10, 11])

    pre_prog = prog.copy()
    middle_cmds = []
    for in_val in input_values:
        # This search needs th ability to map x -> x, otherwise it makes bad, 'least worst' matches, that fail.
        arg_hints = dict(simple = [[in_val], list(output_values)])
        if debug:
            print(f"value_map hints: {arg_hints}")
        val_map_cmd = search_for_final_commands_args(task, pre_prog, 'value_map', ['simple'], arg_hints=arg_hints, post_ops=[post_op], use_scoring=True, debug=debug)
        if val_map_cmd is not None:
            pre_prog.append(val_map_cmd)
            middle_cmds.append(val_map_cmd)
    if len(middle_cmds) > 0:
        # No need to add middle commands, already added to pre_prog.
        temp_prog = pre_prog.copy()
        temp_prog.append(post_op)
        if debug:
            print(f"Trying out: {temp_prog}")
        if check_predictions(task, temp_prog, 'train'):
            if debug:
                print("Success!")
            return temp_prog                
                
def try_combine_commands(task, prog, debug=False):
    # Choose a combine mechanism. Limit the search with some heuristics.
    post_ops = ["transform crop 10", "transform crop 0", "transform flip lr", "transform flip ud", "transform rot90 1", "transform rot90 2", "transform rot90 3", "transform invert"]
    combine_cmds = None
    output_colours = set()
    for train_output in [np.array(x['output']) for x in task['train']]:
        for v in np.unique(train_output):
            if v != 0:
                output_colours.add(v)
    if len(output_colours) == 1:
        # There is only 1 output colour that isn't black. Therefore assume the input panels are combined logically and the unique output colour corresponds to True.
        if debug:
            print("Investigating logical combination.")

        arg_hints = dict()
        arg_hints['logical_op'] = [None, None, None, [output_colours.pop()]]
        combine_cmds = search_for_final_commands_args(task, prog,'combine', ['logical_op'], arg_hints=arg_hints, post_ops=post_ops, debug=debug)
    else:
        # Multi-colour output. Assume the input panels have been combined in some combination. Try the "combine overwrite" commands with black as transparent and
        ## brute-force all permutations.
        if debug:
            print("Investigating permutations.")

        min_panels, max_panels = get_min_max_panels(task, prog)
        if min_panels != max_panels:
            # Different number of panels per train input. Just combine in the order they come in.
            arg_hints = dict()
            arg_hints['overwrite'] = [[0], [[]]]
            combine_cmds = search_for_final_commands_args(task, prog,'combine', ['overwrite'], arg_hints=arg_hints, debug=debug)
        else:
            if max_panels > 8:
                if debug:
                    print("Too many panels to brute-force all combinations.")
            else:                    
                # Get all permutations of num_panels, then format as a list of strings with no spaces.
                l = []
                for p in permutations(range(max_panels)):
                    l.append(str(list(p)).replace(' ',''))
                arg_hints = dict()
                arg_hints['overwrite'] = [[0], l]
                combine_cmds = search_for_final_commands_args(task, prog, 'combine', ['overwrite'], arg_hints=arg_hints, debug=debug)

    if combine_cmds is not None:
        prog.extend(combine_cmds)
        return prog
    
    return None


# In[ ]:


def complete_prog_for_single_frame(task, prog, all_output_sizes_equal_or_smaller_than_inputs, all_output_sizes_equal_to_inputs, debug=False):

    if debug:
        print(f"In complete_prog_for_single_frame with prog = {prog}")
        
    # Are we done already?
    if check_predictions(task, prog, 'train'):
        return prog
        
    prog_output_is_right_size = does_prog_give_single_panel_with_same_size_as_output(task, prog)
    
    post_ops = None
    if not prog_output_is_right_size:
        post_ops = ['assemble original original', 'assemble original zeros']
        if does_10_map_to_something_other_than_0(task):
            post_ops.append('assemble original majority_value')
        
    scale_cmds = search_for_final_commands_args(task, prog, 'abstract', ['simple'], post_ops=post_ops, debug=debug)
    if scale_cmds is not None:
        prog.extend(scale_cmds)
        return prog            

    middle_cmds = search_for_final_commands_args(task, prog, 'transform', ['crop', 'flip', 'invert', 'rot90'], post_ops=post_ops, debug=debug)
    if middle_cmds is not None:
        prog.extend(middle_cmds)
        return prog

    if prog_output_is_right_size:
        value_map_prog = search_for_simple_value_maps(task, prog, post_op='identity', debug=debug)
        if value_map_prog is not None:
            return value_map_prog

    if prog_output_is_right_size:
        replicate_sub_cmds = ['and_merge']
    else:
        replicate_sub_cmds = ['flower_flip', 'flower_rotate']
        post_ops.append('assemble auto_grid')        
    replicate_cmds = search_for_final_commands_args(task, prog, 'replicate', replicate_sub_cmds, post_ops=post_ops, debug=debug)
    if replicate_cmds is not None:
        prog.extend(replicate_cmds)
        return prog
    
    return None

def complete_prog_for_multiple_frames(task, split_prog, all_output_sizes_equal_or_smaller_than_inputs, all_output_sizes_equal_to_inputs, debug=False):
    
    input_panel_counts, input_panel_shapes = get_panel_info_for_split(task, 'input', split_prog)
    output_panel_counts, output_panel_shapes = get_panel_info_for_split(task, 'output', split_prog)
    
    same_counts = [x == y for x, y in zip(input_panel_counts, output_panel_counts)]
    same_shapes = [x == y for x, y in zip(input_panel_shapes, output_panel_shapes)]
    need_reduction = [y < x for x, y in zip(input_panel_counts, output_panel_counts)]
    single_output_panel = [x == 1 for x in output_panel_counts]
    if debug:
        print(input_panel_counts)
        print(output_panel_counts)
        print(same_counts)
        print(same_shapes)
        print(need_reduction)
        print(single_output_panel)

    if all_output_sizes_equal_to_inputs and all(same_counts) and all(same_shapes):
        # Same objects in input and output. Either need to move, sort, or re-colour them.
        post_ops = ['assemble original zeros', 'assemble original majority_value', 'assemble original original']

        for post_op in post_ops:
            value_map_prog = search_for_simple_value_maps(task, split_prog, post_op=post_op, debug=debug)
            if value_map_prog is not None:
                return value_map_prog
                
        # Try colour by enclosures.
        output_values = get_all_input_or_output_values(task, 'output')
        middle_cmds = []
        for val in output_values:
            arg_hints = dict(enclosures_count = [[0], None, [val]])
            val_map_cmd = search_for_final_commands_args(task, split_prog, 'value_map', ['enclosures_count'], arg_hints=arg_hints, post_ops=post_ops, use_scoring=True, debug=debug)
            if val_map_cmd is not None:
                middle_cmds.append(val_map_cmd)
        if len(middle_cmds) > 0:
            temp_prog = split_prog.copy()
            temp_prog.extend(middle_cmds)
            temp_prog.append('assemble original original')
            if debug:
                print(f"Trying out: {temp_prog}")
            if check_predictions(task, temp_prog, 'train'):
                return temp_prog

        # Try colour by shape_match.
        middle_cmds = search_for_final_commands_args(task, split_prog, 'value_map', ['shape_match'], post_ops=post_ops, debug=debug)
        if middle_cmds is not None:
            temp_prog = split_prog.copy()
            temp_prog.extend(middle_cmds)
            return temp_prog

        # Try a simple move then assemble.
        middle_cmds = search_for_final_commands_args(task, split_prog, 'move', ['by_shape'], post_ops=post_ops, debug=debug)
        if middle_cmds is not None:
            temp_prog = split_prog.copy()
            temp_prog.extend(middle_cmds)
            return temp_prog

        # Try a sort then assemble.
        middle_cmds = search_for_final_commands_args(task, split_prog, 'sort', ['by_value', 'unique_values'], post_ops=post_ops, debug=debug)
        if middle_cmds is not None:
            temp_prog = split_prog.copy()
            temp_prog.extend(middle_cmds)
            return temp_prog

        # Try move by colour.
        output_values = get_all_input_or_output_values(task, 'output')
        middle_cmds = []
        for val in output_values:
            arg_hints = dict(by_value = [[val], None, None])
            move_by_val_cmd = search_for_final_commands_args(task, split_prog, 'move', ['by_value'], arg_hints=arg_hints, post_ops=['assemble original zeros'], use_scoring=True, debug=debug)
            if move_by_val_cmd is not None:
                middle_cmds.append(move_by_val_cmd)
        if len(middle_cmds) > 0:
            temp_prog = split_prog.copy()
            temp_prog.extend(middle_cmds)
            temp_prog.append('assemble original zeros')
            if debug:
                print(f"Trying out: {temp_prog}")
            if check_predictions(task, temp_prog, 'train'):
                return temp_prog
                
        
    if all(need_reduction) or not all(same_shapes):
        filter_subcmds = ['by_size', 'commonality', 'unique_values', 'has_symmetry', 'rectangular', 'enclosed', 'by_shape_count', 'by_majority_value', 'by_index', 'by_value', 'by_value_gte', 'by_not_value']
        arg_hints = dict()
        for subcmd in filter_subcmds:
            arg_hints[subcmd] = [None, None, None, None, None, None]
        arg_hints['by_index'] = [None, range(min(30, max(input_panel_counts)))]
        if all(single_output_panel):
            # Post-ops based on not needing assemble.
            post_ops = ["transform crop 10", "transform crop 0", "transform flip lr", "transform flip ud", "transform rot90 1", "transform rot90 2", "transform rot90 3", "transform invert", 'abstract simple 1', 'abstract simple 2', 'abstract simple 3']
        else:
            if all_output_sizes_equal_to_inputs:
                post_ops = ['assemble original original', 'assemble original zeros']
                if does_10_map_to_something_other_than_0(task):
                    post_ops.append('assemble original majority_value')
            else:
                post_ops = ['assemble auto_grid', ['abstract simple 1', 'assemble auto_grid'], ['abstract simple 2', 'assemble auto_grid'], ['abstract simple 3', 'assemble auto_grid']]
        
        filter_cmds = search_for_final_commands_args(task, split_prog, 'filter', filter_subcmds, arg_hints=arg_hints, post_ops=post_ops, filter_opt_info=(input_panel_counts, output_panel_counts), debug=debug)
        if filter_cmds is not None:
            prog = split_prog.copy()
            prog.extend(filter_cmds)
            return prog 

    if all(need_reduction) or not all(same_shapes):
        if all_output_sizes_equal_or_smaller_than_inputs and panels_same_size_per_input(task, split_prog):
            combine_prog = try_combine_commands(task, split_prog, debug=debug)
            if combine_prog is not None:
                return combine_prog
            
    # Always try just putting all the objects back together, in combination with some simple transforms!
    transform_cmds = [None, "transform flip lr", "transform flip ud", "transform rot90 1", "transform rot90 2", "transform rot90 3", "transform invert"]
    if all_output_sizes_equal_to_inputs:
        assemble_cmds = ['assemble original original', 'assemble original zeros']
        if does_10_map_to_something_other_than_0(task):
            assemble_cmds.append('assemble original majority_value')
    else:
        transform_cmds.extend(["transform crop 10", "transform crop 0"])
        assemble_cmds = ['assemble auto_grid']    
    
    for transform_cmd in transform_cmds:
        for assemble_cmd in assemble_cmds:
            prog = split_prog.copy()
            if transform_cmd is not None:
                prog.append(transform_cmd)
            prog.append(assemble_cmd)
            prog = complete_prog_for_single_frame(task,
                                                  prog,
                                                  all_output_sizes_equal_or_smaller_than_inputs=all_output_sizes_equal_or_smaller_than_inputs,
                                                  all_output_sizes_equal_to_inputs=all_output_sizes_equal_to_inputs)
            if prog is not None:
                return prog
        
    return None

def build_prog_for_task_v2(task_id, task, debug=False):
    prog = None

    all_output_sizes_equal_or_smaller_than_inputs = True
    all_output_sizes_equal_to_inputs = True
    for train_input, train_output in zip([np.array(x['input']) for x in task['train']],[np.array(x['output']) for x in task['train']]):
        if train_input.shape[0] < train_output.shape[0] and train_input.shape[1] < train_output.shape[1]:
            # inputs are smaller than outputs
            all_output_sizes_equal_or_smaller_than_inputs = False
            all_output_sizes_equal_to_inputs = False
        elif train_input.shape[0] != train_output.shape[0] or train_input.shape[1] != train_output.shape[1]:
            all_output_sizes_equal_to_inputs = False

    # Special case for snake tasks.
    if all_output_sizes_equal_to_inputs:
        values_added_in_diff = get_values_added_in_diff(task)
        all_input_values = get_all_input_or_output_values(task, 'input')
        if len(values_added_in_diff) == 1 and len(all_input_values) <= 5:
            for starting_val in values_added_in_diff:
                arg_hints = dict()
                hints = [[starting_val], None]
                for i in range(10):
                    if i == starting_val:
                        # Allow snakes to cross.
                        hints.append(['overwrite'])
                    elif i in all_input_values:
                        # No hint, iteratre therough all options.
                        hints.append(None)
                    else:
                        # Not seen in any input, treat as stop.
                        hints.append(['stop'])
                        
                arg_hints['simple'] = hints
                snake_cmds = search_for_final_commands_args(task, [], 'snake', ['simple'], arg_hints=arg_hints, debug=debug)
                if snake_cmds is not None:
                    return snake_cmds
                
    split_progs = [['identity']]
    split_cmd = determine_split_simple(task, debug=debug)
    split_progs.append([split_cmd])
    split_cmd = determine_split_simple(task, ['transform crop 10'], debug=debug)
    split_progs.append(['transform crop 10', split_cmd])
    split_progs.append(['split auto_grid'])
    culling_commands = ['filter rectangular keep 2', 'filter rectangular keep 3']
    if do_all_task_inputs_contain_0(task):
        split_progs.extend([["split by_value 0 True"], ["split by_value 0 inclusive"], ['split frame 0 False'], ['split frame 0 True']])
        split_prog, _ = determine_split_connected_region(task, 0, True, 8, culling_commands=culling_commands, debug=debug)
        split_progs.append(split_prog)
        split_prog, _ = determine_split_connected_region(task, 0, False,  8, culling_commands=culling_commands, debug=debug)
        split_progs.append(split_prog)
        split_prog, _ = determine_split_connected_region(task, 0, True, 4, culling_commands=culling_commands, debug=debug)
        split_progs.append(split_prog)
        split_prog, _ = determine_split_connected_region(task, 0, False,  4, culling_commands=culling_commands, debug=debug)
        split_progs.append(split_prog)
    if does_10_map_to_something_other_than_0(task):
        split_progs.extend([["split by_value 10 True"], ["split by_value 10 inclusive"], ['split frame 10 False'], ['split frame 10 True']])
        split_prog, _ = determine_split_connected_region(task, 10, True, 8, culling_commands=culling_commands, debug=debug)
        split_progs.append(split_prog)
        split_prog, _ = determine_split_connected_region(task, 10, False,  8, culling_commands=culling_commands, debug=debug)
        split_progs.append(split_prog)
        split_prog, _ = determine_split_connected_region(task, 10, True, 4, culling_commands=culling_commands, debug=debug)
        split_progs.append(split_prog)
        split_prog, _ = determine_split_connected_region(task, 10, False,  4, culling_commands=culling_commands, debug=debug)
        split_progs.append(split_prog)
    if does_11_map_to_something_other_than_0(task):    
        split_prog, _ = determine_split_connected_region(task, 11, True, 8, culling_commands=culling_commands, debug=debug)
        split_progs.append(split_prog)
        split_prog, _ = determine_split_connected_region(task, 11, False,  8, culling_commands=culling_commands, debug=debug)
        split_progs.append(split_prog)
        split_prog, _ = determine_split_connected_region(task, 11, True, 4, culling_commands=culling_commands, debug=debug)
        split_progs.append(split_prog)
        split_prog, _ = determine_split_connected_region(task, 11, False,  4, culling_commands=culling_commands, debug=debug)
        split_progs.append(split_prog)

    # Some of the above commands can contain None. Filter it.
    new_split_progs = []
    for prog in split_progs:
        if prog is not None and None not in prog:
            new_split_progs.append(prog)
    if debug:
        print(f"{datetime.datetime.now()} len(split_progs): {len(split_progs)}, len(new_split_progs): {len(new_split_progs)}")
        print(f"split_progs: {split_progs}")
        print(f"new_split_progs: {new_split_progs}")
    split_progs = new_split_progs
    
    for split_prog in split_progs:
        min_panels, max_panels = get_min_max_panels(task, split_prog)
        if debug:
            print(f"{datetime.datetime.now()} prog: {split_prog} returns min_panels, max_panels = {min_panels}, {max_panels}")
        if min_panels == 1 and max_panels == 1:
            prog = complete_prog_for_single_frame(task,
                                                  split_prog,
                                                  all_output_sizes_equal_or_smaller_than_inputs=all_output_sizes_equal_or_smaller_than_inputs,
                                                  all_output_sizes_equal_to_inputs=all_output_sizes_equal_to_inputs,
                                                  debug=debug)
            if prog is not None:
                return prog
        elif min_panels > 0:
            if max_panels <= A_LOT:
                # Multiple objects extracted from the split. Can we just put them back together?
                prog = complete_prog_for_multiple_frames(task,
                                                        split_prog,
                                                        all_output_sizes_equal_or_smaller_than_inputs=all_output_sizes_equal_or_smaller_than_inputs,
                                                        all_output_sizes_equal_to_inputs=all_output_sizes_equal_to_inputs,
                                                        debug=debug)
                if prog is not None:
                    return prog
            
    return None


# In[ ]:


task_id = '60b61512'

if task_id in training_tasks:
    print("From training")
elif task_id in evaluation_tasks:
    print("From evaluation")
task = all_tasks[task_id]

prog = None
plot_task(task, prog, debug=False)

before_time = time.perf_counter()
import cProfile
# Useful for keeping runtime under control. 
#cProfile.run('build_prog_for_task_v2("test", task, debug=True)')
prog = build_prog_for_task_v2("test", task, debug=False)
after_time = time.perf_counter()
print(f"Time taken: {after_time - before_time}s")
print(f"Split cache accesses/hits: {split_result_cache_accesses}/{split_result_cache_hits}")

print(prog)
plot_task(task, prog)


# # 5 - Evaluation
# 
# Now evaluate the DSL and program search code against the training or evaluation sets.
# 
# This code also monitors "bad progs" - that is programs that are found to solve the training examples of a task, but do not solve the test componenets of the task.

# In[ ]:


#tasks_to_eval = []
#tasks_to_eval = all_tasks
#tasks_to_eval = training_tasks
#tasks_to_eval = evaluation_tasks
tasks_to_eval = extra_tasks 

SLOW = 200

# From training:
skip_because_slow = []
# From evaluation:
skip_because_slow.extend(['af22c60d', 'de493100', '2f0c5170', '981571dc'])

temp_tasks_to_eval = dict()
for task_id in tasks_to_eval.keys():
    if task_id not in skip_because_slow:
        temp_tasks_to_eval[task_id] = tasks_to_eval[task_id]
tasks_to_eval = temp_tasks_to_eval

print(len(tasks_to_eval))

found_bad_prog = []
successes = []
slow_tasks = []

def solve_task(task_id):
    task = tasks_to_eval[task_id]
    #print(task_id)    
    before_time = time.perf_counter()
    prog = build_prog_for_task_v2(task_id, task)
    after_time = time.perf_counter()
    
    success = False
    if prog is not None and check_predictions(task, prog, 'train') and check_predictions(task, prog, 'test'):
        success = True            
    
    return task_id, prog, success, (after_time - before_time)

# Use multi-processing to speed this up!
pool = Pool()   
results = list(tqdm(pool.imap(solve_task, tasks_to_eval.keys()), total=len(tasks_to_eval.keys())))
pool.close()

for task_id, prog, success, time_taken in results:
    if prog:
        if success:
            successes.append((task_id, prog))
        else:
            found_bad_prog.append((task_id, prog))
    if time_taken > SLOW:
        slow_tasks.append((task_id, time_taken))
            
print(f"Built a program for {len(found_bad_prog) + len(successes)} tasks.")
print(f"Succesfully predicted outputs for {len(successes)} tasks.")

print(f"\nSlow tasks, over {SLOW}s:")
for task_id, time_taken in slow_tasks:
    print(f"{task_id} took {time_taken}s.")


# In[ ]:


for task_id, prog in found_bad_prog:
    task = tasks_to_eval[task_id]
    print(f"task_id: {task_id}, prog: {prog}")
    plot_task(task, prog)


# In[ ]:


for task_id, prog in successes:
    print(f"{task_id}: {prog}")
    #plot_task(all_tasks[task_id], prog)
print('\n\n')


# # 6 - Submission

# In[ ]:


def flattener(pred):
    str_pred = str([row for row in pred])
    str_pred = str_pred.replace(', ', '')
    str_pred = str_pred.replace('[[', '|')
    str_pred = str_pred.replace('][', '|')
    str_pred = str_pred.replace(']]', '|')
    return str_pred


# In[ ]:


sub = pd.read_csv(root_path / 'sample_submission.csv')
sub['output'] = "|0|"

def solve_task_for_test(task_id):
    task = test_tasks[task_id]
  
    result = []
    prog = build_prog_for_task_v2(task_id, task)
    if prog is not None:    
        test_num = -1
        for test in task['test']:
            test_num += 1

            # Create solutions
            test_input = np.array(test['input'])
            interp = Interpreter()
            try:
                solution = interp.execute(test_input, prog)

                task_name = task_id + "_" + str(test_num)
                print(f"SOLUTION for {task_name}")
                print(flattener(solution.tolist()))
                result.append((task_name, flattener(solution.tolist())))
            except:
                pass
            
    return result
    
# If this is a commit run on the public test set, skip it. That makes it quicker to do the real submit on the private test set.
if '00576224' in test_tasks.keys():
    get_ipython().system('cp ../input/abstraction-and-reasoning-challenge/sample_submission.csv submission.csv')
else:
    pool = Pool()   
    results = pool.imap(solve_task_for_test, test_tasks.keys())
    pool.close()

    for result in results:
        for task_name, solution in result:
            sub.loc[sub['output_id'] == task_name, ['output']] = solution
            
    sub.to_csv('submission.csv', index=False)

