#!/usr/bin/env python
# coding: utf-8

# The two main functions are:
# 1. Identification and attribution of objects and shapes
# 2. Using search to establish relations between the attributes of different objects

# In[ ]:


import numpy as np
import pandas as pd
import copy
import time
from tqdm import tqdm


# # Data Loading

# In[ ]:


import json
import os
from os.path import join as path_join


def load_data(path):
    tasks = pd.Series()
    for file_path in os.listdir(path):
        task_file = path_join(path, file_path)

        with open(task_file, 'r') as f:
            task = json.load(f)

        tasks[file_path[:-5]] = task
    return tasks


# In[ ]:


train_tasks = load_data('../input/abstraction-and-reasoning-challenge/training/')
evaluation_tasks = load_data('../input/abstraction-and-reasoning-challenge/evaluation/')
test_tasks = load_data('../input/abstraction-and-reasoning-challenge/test/')

train_tasks.head()


# In[ ]:


class Box():
    
    def __init__(self, coords = (0,0), color = 1, style = "individual", parent_object = None):
        super(Box, self).__init__()      
        self.coords = coords #tuple coordinate
        self.color = color #one of the 10 scalar choices
        self.style = style #boundary, edge, line, individual, crossline, pinpal, body are some styles available.. more can be added
        self.parent_object = parent_object
    
    def get_cardinal_neighbors(self, image_xlen, image_ylen):
        
        x = self.coords[0]
        y = self.coords[1]   
        neibors = []
        
        if y <= image_ylen-2:
            neibors.append((x, y+1))
        else:
            pass
        if x <= image_xlen-2:
            neibors.append((x+1,y))
        else:
            pass
        if y != 0:
            neibors.append((x, y-1))
        else:
            pass
        if x != 0:
            neibors.append((x-1, y))
        else:
            pass
        
        return neibors
                   


# In[ ]:


class ACRObject():
    def __init__(self, boxes=[], object_no = None, comp_of = None, style = "independent"):
        super(ACRObject, self).__init__()      
        self.boxes = set(boxes) #set of boxes
        self.object_no = object_no #scalar id number in corresponding dict
        self.links = set([])
        self.components = set([]) #set of components
        self.component_of = comp_of
        self.style = style
        
    def get_centroid(self):
        center_x = 0
        center_y = 0
        #print("getting centroid..")
        for box in self.boxes:
            center_x += box.coords[0]
            center_y += box.coords[1]
            #print("box x", box.coords[0])
            #print("box y", box.coords[1])
        return (center_x/len(self.boxes), center_y/len(self.boxes))
    
    def get_centroid_with_comps(self):
        center_x = 0
        center_y = 0
        count_boxes = 0
        for box in self.boxes:
            center_x += box.coords[0]
            center_y += box.coords[1]
            count_boxes += 1
        for comp in self.components:
            for box in comp.boxes:
                center_x += box.coords[0]
                center_y += box.coords[1]
                count_boxes += 1
        return (center_x/count_boxes, center_y/count_boxes)
    
    def get_moment(self):
        centroid = self.get_centroid()
        total_distance = 0
        for box in self.boxes:
            distance = ((box.coords[0]-centroid[0])**2 + (box.coords[1]-centroid[1])**2)**0.5
            total_distance += distance
        return (total_distance)/len(self.boxes)
    
    def get_moment_with_comps(self):
        #print("getting moment of object")
        centroid = self.get_centroid_with_comps()
        #print("centroid of object ", centroid)
        total_distance = 0
        total_boxes = 0
        for box in self.boxes:
            distance = ((box.coords[0]-centroid[0])**2 + (box.coords[1]-centroid[1])**2)**0.5
            total_distance += distance
            total_boxes += 1
        for comp in self.components:
            for box in comp.boxes:
                distance = ((box.coords[0]-centroid[0])**2 + (box.coords[1]-centroid[1])**2)**0.5
                total_distance += distance
                total_boxes += 1
        return (total_distance)/total_boxes
    
    def extend_boxes(self,box):
        self.boxes.add(box)
        self.centroid = self.get_centroid() #update the centroid
        
    def get_minmax_x(self):
        min_x = 10000
        max_x = 0
        for box in self.boxes:
            if box.coords[0] < min_x:
                min_x = box.coords[0]
            if box.coords[0] > max_x:
                max_x = box.coords[0]
        return (min_x,max_x)
    
    def get_minmax_y(self):
        min_y = 10000
        max_y = 0
        for box in self.boxes:
            if box.coords[1] < min_y:
                min_y = box.coords[1]
            if box.coords[1] > max_y:
                max_y = box.coords[1]
        return (min_y,max_y)


# In[ ]:


class Grid():
    def __init__(self, image):
        super(Grid, self).__init__()      
        self.image = np.array(image)
        self.width = self.image.shape[0]
        self. height = self.image.shape[1]
        self.box_dict = {}
        self.object_dict = {}
    
    def get_boxes(self):
        for i in range(self.width):
            for j in range(self.height):
                if (i,j) not in self.box_dict.keys():
                    self.box_dict[i,j] = Box((i,j),self.image[i,j])
    
    def conglomerate(self, box):
        ACRO = box.parent_object
        neib_coords = box.get_cardinal_neighbors(self.width, self.height)
        #print("coagulate for box ", box.coords)
        for neib in neib_coords:
            #print("nieb box ", neib)
            try:
                neib_box = self.box_dict[neib]
            except:
                print("box dict ", self.box_dict.keys())
                print(self.width, self.height, neib, box.coords)
            if neib_box.color == box.color and                neib_box.parent_object != None: #if box already was assigned
                ACRO = neib_box.parent_object
                box.parent_object = ACRO
                ACRO.extend_boxes(box)

        for neib in neib_coords:
            #print("nieb box ", neib)
            try:
                neib_box = self.box_dict[neib]
            except:
                print("box dict ", self.box_dict.keys())
                print(self.width, self.height, neib, box.coords)        
            if neib_box.color == box.color and                neib_box.parent_object == None: #if box is not adopted yet
                neib_box.parent_object = ACRO
                ACRO.extend_boxes(neib_box)
                #ACRO = self.conglomerate(neib_box)
            
        return ACRO
    
    def get_objects(self):
        for box_xy, box in self.box_dict.items():
            #print("box ", box_xy)
            #key n=0 ne=1 e=2 se=3 s=4 sw=5 w=6 nw=7
            if box.parent_object == None:  #don't forget the following 4 lines whenever creating new object
                total_objects = len(self.object_dict.keys())
                new_object = ACRObject([box],object_no=total_objects+1)
                self.object_dict[total_objects+1] = new_object
                box.parent_object = new_object
                consolidated_object = self.conglomerate(box)
            
    def link_objects(self, ACRO1, ACRO2): #link will be a searchable action
        ACRO1.link_list.add(ACRO2)
        ACRO2.link_list.add(ACRO1)
        
    def move_object(self, move=(0,0), ACRO=None): #move tuple will be a searchable parameter
        
        check = True
        for box in ACRO.boxes:
            xn = box.coords[0]+move[0]
            yn = box.coords[1]+move[1]
            if xn > self.width-1 or xn < 0:
                check = False
            elif yn > self.height-1 or yn < 0:
                check = False
            if check:
                if self.image[xn,yn] != 0: #if the box to move is occupied
                    check = False    
        if check:
            for box in ACRO.boxes:
                xn = box.coords[0]+move[0]
                yn = box.coords[1]+move[1]
                old_coords = box.coords
                box.coords = (xn,yn)
                old_box = self.box_dict[xn,yn] #the box at destination coordinates
                old_box.coords = old_coords #this is more like a swap than a move

                self.image[xn,yn] = box.color #update the grid image new loc
                self.box_dict[xn,yn] = box #update box dict new location

                self.image[old_coords[0],old_coords[1]] = old_box.color #update image old loc
                self.box_dict[old_coords[0], old_coords[1]] = old_box #update box dict old loc
            return "ok"
        else:
            return False
    
    def copy_object(self, copy=(0,0), ACRO=None): #copy tuple will be a searchable parameter
        
        total_objects = len(self.object_dict.keys())
        ACRO2 = ACRObject()
        ACRO2 = ACRO
        ACRO2.object_no=total_objects+1
        self.object_dict[total_objects+1] = ACRO2
        
        max_x = ACRO2.get_minmax_x()[1]
        min_x = ACRO2.get_minmax_x()[0]
        max_y = ACRO2.get_minmax_y()[1]
        min_y = ACRO2.get_minmax_y()[0]
                
        check = True
        
        for box in ACRO2.boxes:
            
            box.parent_object = ACRO2  #since it was copied from ACRO1, we need to update it
            
            xn = box.coords[0]+copy[0]
            yn = box.coords[1]+copy[1]   
            
            if copy[0] < (max_x - min_x) and copy[1] < (max_y - min_y): # if copy distance is too short
                check = False
            elif xn > self.width-1 or xn < 0:
                check = False
            elif yn > self.height-1 or yn < 0:
                check = False
            if check:
                if self.image[xn,yn] != 0: #if the box to move is occupied
                    check = False
        if check:
            for box in ACRO2.boxes:
                
                xn = box.coords[0]+copy[0]
                yn = box.coords[1]+copy[1] 

                box.coords = (xn,yn) #update box coords
                self.image[xn,yn] = box.color #update the grid image
                self.box_dict[xn,yn] = box #update box dict
            return "ok"
        else:
            return "fail"
                    
    def find_object_components(self):
        print("finding components")
        for obj_grp, obj in object_dict.items():
            neib_objs = set()
            print("obj number", obj_grp)
            for box_no, box in enumerate(obj.boxes):
                neib_coords = box.get_cardinal_neighbors(self.width, self.height)
                for neib in neib_coords:
                    neib_box = self.box_dict[neib]
                    if neib_box.color != 0: #we do not need to add background object
                        neib_objs.add(neib_box.parent_object)
            if len(neib_objs) == 1: #if there is only one neighboring object 
                if obj.component_of != neib_objs[0]: #if it is not a component of the intended component
                    if neib_obj.component_of != None: #if the neibhoring object is not already a component
                        obj.components.add(neib_objs[0])
                        neib_objs.component_of = obj

    def link_objects(self, Grid2):
        print("linking objects")
        for obj_grp1, obj1 in self.object_dict.items():
            print("processing obj no ", obj_grp1)
            print("obj centroid ", obj1.get_centroid())
            obj1_mwc = obj1.get_moment_with_comps()
            print("object moment with comps", obj1_mwc)
            for obj2_grp, obj2 in Grid2.object_dict.items():
                obj2_mwc = obj2.get_moment_with_comps()
                if obj2 != obj1 and obj2_mwc == obj1_mwc and obj2_mwc != 0.0: #we don't want to link all single cells
                    if obj1 not in obj2.links and obj2 not in obj1.links:
                        obj1.links.add(obj2)
                        obj2.links.add(obj1)            


# In[ ]:


class Single_Object_Action():
    def __init__(self, obj = None, params = (), action = "", grid = None):
        self.object = obj
        self.params = params #tuple
        self.action = action
        self.grid = grid
    def apply(self):
        result = eval("self.grid."  + self.action + "(self.params,self.object)")    


# In[ ]:


class PlanningGraph(): 
    def __init__(self, input_grid, output_grid):
        self.ingrid = input_grid
        self.goalgrid = output_grid
        self.list_of_grps_of_act_obj_grid = []
   
    def get_input_objects(self):
        self.input_objects = self.ingrid.get_objects()
    
    def get_output_objects(self):
        self.output_objects = self.goalgrid.get_objects()
        
    def get_following_grids(self, grid):
        # Grid can be also thought of as a node in the planning graph.
        single_object_actions = ["copy_object"] #move_object and many others to be added here
        double_object_actions = ["copy_with_ref_obj", "superimpose_obj"] #many others to be added here
        list_next_grids = []
        for obj_no, obj in grid.object_dict.items():
            for action in single_object_actions:
                for x in range(grid.width):
                    for y in range(grid.height):
                        next_grid = copy.deepcopy(grid)
                        proposed_action = Single_Object_Action(obj,(x,y),action,next_grid)
                        if proposed_action.apply() == "ok":
                            next_grid.former_actions.append(proposed_action)
                            list_next_grids.append(next_grid)
        return list_next_grids
    
    def compare_images(self,grid1,grid2):
        total_boxes = 0
        total_correct = 0
        print("image shapes")
        print("grd1", grid1.image.shape)
        print("grd2", grid2.image.shape)
        print("grd1 wid and hei", grid1.width, grid1.height)
        for i in range(grid1.width):
            for j in range(grid1.height):
                total_boxes += 1
                if grid1.image[i,j] == grid2.image[i,j]:
                    total_correct += 1
        return (total_correct/total_boxes)
    
    def compare_grids(self,grid1,grid2):
        print("comparing grids")
        objs_in_gr1_not_gr2 = []
        objs_in_gr2_not_gr1 = []
        
        for obj_gr1, obj1 in grid1.object_dict.items():
            check = False
            for obj_gr2, obj2 in grid2.object_dict.items():
                if obj2.get_centroid() == obj1.get_centroid() and                    obj2.get_moment() == obj1.get_moment():
                    check = True
            if not check:
                objs_in_gr1_not_gr2.append(obj1.get_centroid())
            else:
                pass
            
        for obj_gr2, obj2 in grid2.object_dict.items():
            check = False
            for obj_gr1, obj1 in grid1.object_dict.items():
                if obj1.get_centroid() == obj2.get_centroid() and                    obj1.get_moment() == obj2.get_moment():
                    check = True
            if not check:
                objs_in_gr2_not_gr1.append(obj2.get_centroid())
            else:
                pass
                
        return  objs_in_gr1_not_gr2, objs_in_gr2_not_gr1
            
    def search_graph(self,grid, score_dict): #start with ingrid and score_dict={ingrid:0}
        for next_grid in self.get_following_grids(grid):
            print(next_grid)
            score = self.compare_images(next_grid, self.goalgrid)
            score_dict[score] = next_grid
        print("score_dict", score_dict)
        best_grid = max(score_dict,key=score_dict.get)
        max_score = score_dict[best_grid]
        if max_score > score_dict[grid]:
            return self.search_graph(best_grid, score_dict)
        else:
            return best_grid


# # Model

# In[ ]:


sample_pic = train_tasks[10]['train'][1]["input"]
sample_out = train_tasks[10]['train'][1]["output"]

sample_grid = Grid(copy.deepcopy(sample_pic))
sample_grid.get_boxes()
print("sample grid width", sample_grid.width)
print("sample grid height", sample_grid.height)

sample_grid.get_objects()
sample_grid.link_objects(sample_grid) #link within itself
obj2copy = sample_grid.object_dict[2]
allowed = sample_grid.copy_object((2,0),obj2copy)
print("is copy allowed?", allowed)

grid_image = np.ones((sample_grid.width,sample_grid.height))

for box_co, box in sample_grid.box_dict.items():
    grid_image[box_co[0],box_co[1]] = box.parent_object.object_no


# In[ ]:


import matplotlib.pyplot as plt
from matplotlib import colors


cmap = colors.ListedColormap(
        ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',
         '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25', '#530C25', '#3429C3', '#4E34CD', '#9A2B14'])
norm = colors.Normalize(vmin=0, vmax=10)
    
def plot_pictures(pictures, labels):
    fig, axs = plt.subplots(1, len(pictures), figsize=(2*len(pictures),32))
    for i, (pict, label) in enumerate(zip(pictures, labels)):
        axs[i].imshow(np.array(pict), cmap=cmap, norm=norm)
        axs[i].set_title(label)
    plt.show()
    

def plot_sample(sample, predict=None):
    if predict is None:
        plot_pictures([sample['input'], sample['output']], ['Input', 'Output'])
    else:
        plot_pictures([sample['input'], sample['output'], predict], ['Input', 'Output', 'Predict'])


# In[ ]:


print("objects and links")
for obj_no in sample_grid.object_dict.keys():
    obj = sample_grid.object_dict[obj_no]
    print("object number", obj_no)
    print("object centroid", obj.get_centroid())
    count = 0
    for linked_obj in obj.links:
        count += 1
        print("linked obj no ", count)
        print("linked obj centroid ", linked_obj.get_centroid())    


# In[ ]:


plot_pictures([sample_pic, grid_image], ['sample_input','grid_image'])


# In[ ]:


def solve_train(train_task):

    input_pic = train_task['train'][1]["input"]
    output_pic = train_task['train'][1]["output"]

    input_grid = Grid(copy.deepcopy(input_pic))
    input_grid.get_boxes()
    input_grid.get_objects()

    output_grid = Grid(copy.deepcopy(output_pic))
    output_grid.get_boxes()
    output_grid.get_objects()

    if input_grid.image.shape != output_grid.image.shape:
        print("input output images must be same size!")
    else:
        graph = PlanningGraph(input_grid, output_grid)
        solution_grid = graph.search_graph(graph.ingrid,{graph.ingrid:0})
        solution_image = solution_grid.image

        plot_pictures([input_pic, output_pic, solution_image], ['sample_input','sample_output','solution_image'])
        


# In[ ]:


def find_changes(train_task):
    
    input_pic = train_task['train'][1]["input"]
    output_pic = train_task['train'][1]["output"]

    input_grid = Grid(copy.deepcopy(input_pic))
    input_grid.get_boxes()
    input_grid.get_objects()
        
    output_grid = Grid(copy.deepcopy(output_pic))
    output_grid.get_boxes()
    output_grid.get_objects()
  
    graph = PlanningGraph(input_grid, output_grid)
    
    objs_in_gr1_not_gr2, objs_in_gr2_not_gr1 = graph.compare_grids(input_grid,output_grid)
   
    for obj_cntr in objs_in_gr1_not_gr2:
        cntr1 = obj_cntr[0]
        cntr2 = obj_cntr[1]
        print(f'Centroid is {cntr1:.2f} {cntr2:.2f}')

    plot_pictures([input_pic, output_pic], ['sample_input','sample_output'])


# In[ ]:


for train_task in train_tasks:
    find_changes(train_task)


# In[ ]:


seconds = time.time()
for train_task in train_tasks:
    solve_train(train_task)
    break
    if time.time() - seconds > 20000:
        break

