#!/usr/bin/env python
# coding: utf-8

# In this work I'm gonna design & implement a functional-style DSL based on simple image manipulation like movement, flipping, connected region extraction, color separation, etc. It's called "naive" because it's intuitive and straightforward.
# 
# <p><font color="green" size=3>If you like this kernel please upvote. It encorages me to produce more quality content!</font></p>

# In[ ]:


import numpy as np

import os
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import colors


data_path = Path('/kaggle/input/abstraction-and-reasoning-challenge/')
training_path = data_path / 'training'
evaluation_path = data_path / 'evaluation'
test_path = data_path / 'test'

training_tasks = sorted(os.listdir(training_path))
evaluation_tasks = sorted(os.listdir(evaluation_path))
test_tasks = sorted(os.listdir(test_path))
print(len(training_tasks), len(evaluation_tasks), len(test_tasks))

def read_data_file(task_filename):
    with open(task_filename, 'r') as f:
        task = json.load(f)
    return task

def get_data(dataset, index):
    if dataset == 'training':
        p = training_path
        t = training_tasks
    elif dataset == 'evaluation':
        p = evaluation_path
        t = evaluation_tasks
    elif dataset == 'test':
        p = test_path
        t = test_tasks
    
    return read_data_file(str(p / t[index]))


# This Naive Image Manipulation (NIM) DSL contains a set of basic functions and two form of operations. Function is pure which means it doesn't modify the input.
# 
# - `f` is a function that maps input to output. The input and/or output can be
#     + `array`: an image
#     + `[array]`: a list of images
#     + `Color`: color in image
#     + `int`: attributes or statistics of a image
#     + `bool`: conditions or testing
#     + functions: high-order functions
# - two operations:
#     + `>>`: composition, or pipeline. `f >> g` is equivalent to `g(f(x))`
#     + `[]`: parametisation. e.g. `move[1]` shifts input image right by 1. `cgt[1]` checks if input value is greater than 1. Seme functions have zero parameter, e.g. `flipx` flips the input image horizontally
#     
# The functions can be roughly put into categories (current function set is far from complete, so this DSL can only solve a subset of tasks. More basic operations need to be added):
# 
# - image transformation: `array -> array`
# - image separation: `array -> [array]`
# - image merging: `[array] -> array`
# - attribute/statistics: `array -> int`
# - testing: `array -> bool`, `int -> bool`, `[bool] -> bool`
# - high order: like map, filter, etc
# 
# In the following, each function only has one input appearing at the first place in the parameter list of a Python's function. The rest are parameterisable ones which should be passed with a constant value in the form of `foo[parameter]`. This is a drawback in cases where functions require two inputs, like `mask` which applys a mask to an image. In such case, the function has to take a list of images (actually it should be a list of two elements in the case of `mask`). I'm not sure this one-input constraint is sufficient for this task. I would appreciate if you come up with an idea to support or improve it.

# In[ ]:


from typing import Any, List, Callable, TypeVar, NewType
Image = np.ndarray
Color = NewType('Color', str)
T = TypeVar('T', Image, List, Color, int, bool)
S = TypeVar('S', Image, List, Color, int, bool)

color_codes = ['k',     'b',    'r',   'g',     'y',      'e',    'p',      'o',      'z',     'n']
color_names = ['black', 'blue', 'red', 'green', 'yellow', 'gray', 'purple', 'orange', 'azure', 'brown']
color_indices = {}
for i, (n, c) in enumerate(zip(color_names, color_codes)):
    color_indices[n] = i
    color_indices[c] = i
color_indices['grey'] = color_indices['gray']

TYPE_NAMES = {
    Image: 'Image',
    List: 'List',
    Color: 'Color',
    int: 'Int',
    bool: 'Bool',
    Any: 'Any',
    T: 'T',
    S: 'S'
}

def _get_type_name(typ):
    if hasattr(typ, '__origin__') and typ.__origin__ is List:
        return '{}[{}]'.format(TYPE_NAMES[typ.__origin__],
            ','.join(_get_type_name(a) for a in typ.__args__))
    return TYPE_NAMES.get(typ, str(typ))

def _is_type_compatible(t, s):
#     print(t, s)
#     print(t.__origin__, s.__origin__)
#     print(t.__origin__ is list, s.__origin__ is list)
    if t == s:
        return True
    if t in (T, S) or s in (T, S):
        return True
    if (t is List or t.__origin__ is List)         and (s is List or s.__origin__ is List):
        return True
    return False

BACKGROUND = 0

_neighbor_offsets = {
    4: [(1, 0), (-1, 0), (0, 1), (0, -1)],
    8: [(1, 0), (-1, 0), (0, 1), (0, -1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
}
def _expand_region_indices(img, i, j, neighbor=4):
    h, w = img.shape
    seed_color = img[i, j]
    idx = np.zeros_like(img, dtype=np.bool)
    region = []
    region.append((i, j))
    while len(region) > 0:
        ii, jj = region.pop()
        if img[ii, jj] != seed_color:
            continue
        idx[ii, jj] = True
        for di, dj in _neighbor_offsets[neighbor]:
            ni, nj = ii + di, jj + dj
            if ni >= 0 and ni < h and nj >= 0 and nj < w                     and not idx[ni, nj]:
                region.append((ni, nj))
    return idx


class _Func(object):
    def __init__(self, func, *args):
        #print(func, args)
        self._func = func
        if not isinstance(func, _Func):
            self._input_type = func.__annotations__[func.__code__.co_varnames[0]]
            self._output_type = func.__annotations__['return']
        else:
            raise ValueError('should not reach here')
        self._args = args
        
    def __call__(self, x):
        #print(self)
        #print(self._args)
        return self._func(x, *self._args)
    
    def __getitem__(self, args):
        # print(self, self._args, args)
        if type(args) == tuple:
            return _Func(self._func, *self._args, *args)
        else:
            return _Func(self._func, *self._args, args)
        
    def __rshift__(self, other):
        if not _is_type_compatible(other._input_type, self._output_type):
            raise ValueError('Cannot compose functions: {}: {}, {}: {}'                              .format(self, self._get_type_anno(), 
                                     other, other._get_type_anno()))
        f = lambda x: other(self(x))
        f.__name__ = str(self) + ' >> ' + str(other)
        f.__annotations__['x'] = self._input_type
        f.__annotations__['return'] = other._output_type
        return _Func(f)
        
    def __str__(self):
        if len(self._args) > 0:
            s = self._func.__name__ + '[' + ', '.join(
                ('<A>' if isinstance(s, np.ndarray) else str(s)) for s in self._args) + ']'
        else:
            s = self._func.__name__
        #s += ': ' + self._get_type_anno()
        return s
    
    def _get_type_anno(self):
        return '{} -> {}'.format(_get_type_name(self._input_type), _get_type_name(self._output_type))

    def to_str_with_anno(self):
        return '{}: {}'.format(self, self._get_type_anno())
    
def Func(func):
    f = _Func(func)
    print('func created:\t' + f.to_str_with_anno().replace(': ', ':\t'))
    return f
    
def _creator(func, debug=False):
    def _func(inp):
        f = func(inp)
        if debug:
            print(f)
        return f(inp)
    return _func

# Func creator is a lazy initialisation of functions with the input image as parameter
FCreator = _creator
FCreatorD = lambda f: _creator(f, True)

def zeros_image(w, h) -> Image:
    return np.zeros(h, w, dtype=np.int)

@Func
def ident(img: T) -> T:
    return img
    
########### size #############

@Func
def zoom_out(img: Image, nx, ny=None, xofst=0, yofst=None) -> Image:
    if ny is None:
        ny = nx
    if yofst is None:
        yofst = xofst
        
    if xofst == 0 and yofst == 0:
        return np.array(np.kron(img, np.ones((ny, nx))), dtype=np.int)
    else:
        h, w = img.shape
        th = h * ny + yofst * (h - 1)
        tw = w * nx + xofst * (w - 1)
        # print(h, w, th, tw)
        ret = np.zeros((th, tw), dtype=img.dtype)
        for iiy, iy in enumerate(range(0, th, ny+yofst)):
            for iix, ix in enumerate(range(0, tw, nx+xofst)):
                #print(ix, iy)
                ret[iy:iy+ny, ix:ix+nx] = img[iiy, iix]
        return ret
    
@Func
def tile(img: Image, nx, ny=None, xofst=0, yofst=None) -> Image:
    if ny is None:
        ny = nx
    if yofst is None:
        yofst = xofst
        
    if xofst == 0 and yofst == 0:
        #return np.array(np.kron(np.ones((n, n)), img), dtype=np.int)
        return np.tile(img, (ny, nx))
    else:
        h, w = img.shape
        th = h * ny + yofst * (ny - 1)
        tw = w * nx + xofst * (nx - 1)
        # print(h, w, th, tw)
        ret = np.zeros((th, tw), dtype=img.dtype)
        for iy in range(0, th, h+yofst):
            for ix in range(0, tw, w+xofst):
                #print(ix, iy)
                ret[iy:iy+h, ix:ix+w] = img
        return ret
    
@Func
def extend(img: Image, x, y) -> Image:
    h, w = img.shape
    ret = np.zeros((h + y, w + x), dtype=img.dtype)
    ret[0:h, 0:w] = img
    return ret

########### copy #############

@Func
def dup(img: T, n) -> List[T]:
    return [img for i in range(n)]

# @Func
# def dup2d(img, n, m):
#     return [[np.array(img) for i in range(m)] for j in range(n)]

########### split #############

@Func
def split_color(img: Image) -> List[Image]:
    ''' Split an image into a list of single-colored images'''
    
    color = np.unique(img)
    return [np.where(img == c, c, 0) for c in color if c > BACKGROUND]

def _split_conn(img, neighbor=4):
    regions = []
    mem = np.zeros_like(img, dtype=np.bool)
    h, w = img.shape
    for j in range(w):
        for i in range(h):
            p = img[i, j]
            if p <= BACKGROUND or mem[i, j]:
                continue
            conn_idx = _expand_region_indices(img, i, j, neighbor)
            mem[conn_idx] = True
            regions.append(np.where(conn_idx, img, BACKGROUND))
    return regions

@Func
def split_conn(img: Image) -> List[Image]:
    ''' Split an image into a list of images each containing a single connected region'''
    
    return _split_conn(img, 4)

@Func
def split_conn8(img: Image) -> List[Image]:
    ''' Split an image into a list of images each containing a single connected region.
      Pixels of 8 neighbors are all considered "connected"
    '''
    
    return _split_conn(img, 8)

split = split_conn  # alias

@Func
def split_even(img: Image, nx: int, ny: int) -> List[Image]:
    '''Split an image evenly'''
    
    h, w = img.shape
    if h % ny != 0 or w % nx != 0:
        raise ValueError('can not split image evenly to {}x{}'.format(nx, ny))
    dh = h // ny
    dw = w // nx
    res = []
    for x in range(nx):
        for y in range(ny):
            #s = np.zeros_like(img)
            #print(img)
            #print(img[(y*dh):((y+1)*dh), (x*dw):((x+1)*dw)])
            #s[(y*dh):((y+1)*dh), (x*dw):((x+1)*dw)] = img[(y*dh):((y+1)*dh), (x*dw):((x+1)*dw)]
            s = img[(y*dh):((y+1)*dh), (x*dw):((x+1)*dw)]
            res.append(s)
    return res

############ position ############

########### transform ##########

# @Func
def move_right(img: Image, ofst) -> Image:
    h, w = img.shape
    ret = np.zeros_like(img)
    ret[:, ofst:w] = img[:, 0:w-ofst]
    return ret

# @Func
def move_left(img: Image, ofst) -> Image:
    h, w = img.shape
    ret = np.zeros_like(img)
    ret[:, 0:w-ofst] = img[:, ofst:w]
    return ret

# @Func
def move_down(img: Image, ofst) -> Image:
    h, w = img.shape
    ret = np.zeros_like(img)
    ret[ofst:h, :] = img[0:h-ofst, :]
    return ret

# @Func
def move_up(img: Image, ofst) -> Image:
    h, w = img.shape
    ret = np.zeros_like(img)
    ret[0:h-ofst, :] = img[ofst:h, :]
    return ret

@Func
def move(img: Image, dx: int, dy: int) -> Image:
    a = img
    if dx > 0:
        a = move_right(img, dx)
    elif dx < 0:
        a = move_left(img, -dx)
    
    if dy > 0:
        a = move_down(a, dy)
    elif dy < 0:
        a = move_up(a, -dy)
    
    return a
    
@Func
def flipx(img: Image) -> Image:
    return np.flip(img, 1)

@Func
def flipy(img: Image) -> Image:
    return np.flip(img, 0)

############# high order ##############

@Func
def fmap(imgs: List[T], func: Callable[[T], S]) -> List[S]:
    ''' Just map. Add "f" at the front to avoid conflict'''
    
    return list(map(func, imgs))
    
@Func
def fzip(list_of_lists: List[List[T]]) -> List[List[T]]:
    ''' Just zip. Add "f" at the front to avoid conflict'''
    
    return [list(row) for row in zip(*list_of_lists)]
    
@Func
def dfzip(list_and_value: List) -> List[List]:
    ''' Similar to `zip` but only contains two elements. The first one is a list and 
      the second one a single value which will be duplicated into the same length
      as the first.
    '''
    
    if len(list_and_value) != 2:
        raise NotImplementedError()
    arr, value = list_and_value
    l1 = [value] * len(arr)
    return [list(row) for row in zip(arr, l1)]
    
@Func
def mfmap(imgs: List, *funcs: Callable) -> List:
    '''Multi function map. This one is a bit trick. I think the correct signature should be
        imgs: Tuple<T1, T2, ..., Tn>
        funcs: Tuple<T1->S1, T2->S2, ..., Tn->Sn>
        returns: Tuple<S1, S2, ..., Sn>
    '''
    
    if not (type(imgs) == list and len(imgs) == len(funcs)):
        raise ValueError('function lists and argument lists must be in same length')
    return [f(i) for f, i in zip(funcs, imgs)]
    
# @Func
# def dmfmap(imgs, *funcs):
#     if not isinstance(imgs, np.ndarray)
#         raise ValueError('argument should be a single image')
#     return [f(imgs) for f in funcs]

@Func
def ffilter(imgs: List[T], func: Callable[[T], bool]) -> List[T]:
    return list(filter(func, imgs))
    
@Func
def access(imgs: List[T], index: int) -> T:
    return imgs[index]

first = access[0]

# work with operation
@Func
def apply_on(img: T, func: Callable[[List[T]], S], param: T) -> S:
    return func([img, param])

@Func
def apply_under(img: T, func: Callable[[List[T]], S], param: T) -> S:
    return func([param, img])

@Func
def lapply_on(imgs: List[T], func: Callable[[List[T]], S], param: T) -> S:
    if type(imgs) != list:
        raise ValueError('input must be a list of arrays')
    return func(imgs + [param])

@Func
def lapply_under(imgs: List[T], func: Callable[[List[T]], S], param: T) -> S:
    if type(imgs) != list:
        raise ValueError('input must be a list of arrays')
    return func([param] + imgs)

################## color ###################

@Func
def set_color(img: Image, c: Color) -> Image:
    ret = np.zeros_like(img)
    ret[img > BACKGROUND] = color_indices.get(c, c)
    return ret

@Func
def has_color(img: Image, c: Color) -> bool:
    return np.any(img == color_indices.get(c, c))

################## shape #################
def _get_bound(img):
    h, w = img.shape
    x0 = w - 1
    x1 = 0
    y0 = h - 1
    y1 = 0
    for x in range(w):
        for y in range(h):
            if img[y, x] == BACKGROUND:
                continue
            x0 = min(x0, x)
            x1 = max(x1, x)
            y0 = min(y0, y)
            y1 = max(y1, y)
    return x0, x1, y0, y1

@Func
def bound(img: Image) -> Image:
    x0, x1, y0, y1 = _get_bound(img)
    bound = np.zeros_like(img)
    bound[y0:(y1+1), x0:(x1+1)] = 1
    return bound

################## operation ################

@Func
def inot(img: Image) -> Image:
    return np.array(np.logical_not(img), dtype=img.dtype)

@Func
def mask(param: List[Image]) -> Image:
    if len(param) != 2:
        raise ValueError('parameter must be a list of two images')
    mask, img = param
    return np.where(mask > BACKGROUND, img, BACKGROUND)

@Func
def mask_color(param: List[Image]) -> Image:
    if len(param) != 2:
        raise ValueError('parameter must be a list of two images')
    mask, img = param
    colors = np.unique(mask[mask > BACKGROUND])
    if len(colors) > 1:
        raise ValueError('mask must be single colored')
    return np.where(img > BACKGROUND, colors[0], BACKGROUND)

@Func
def crop(param: List[Image]) -> Image:
    if len(param) != 2:
        raise ValueError('parameter must be a list of two images')
    mask, img = param
    x0, x1, y0, y1 = _get_bound(mask)
    return img[y0:(y1+1), x0:(x1+1)]
    
@Func
def ior(imgs: List[Image]) -> Image:
    r = imgs[0]
    for i in imgs[1:]:
        r = np.where(r > BACKGROUND, r, i)
    return r

@Func
def iand(imgs: List[Image]) -> Image:
    r = imgs[0]
    for i in imgs[1:]:
        r = np.where(r == BACKGROUND, BACKGROUND, i)
    return r

@Func
def ixor(imgs: List[Image]) -> Image:
    r = np.where(imgs[0] > BACKGROUND, 1, 0)
    for i in imgs[1:]:
        r = r ^ np.where(i > BACKGROUND, 1, 0)
    return r

@Func
def binarise(img: Image) -> Image:
    return np.array(np.where(img > BACKGROUND, 1, BACKGROUND), dtype=img.dtype)

############# numerics (counting function) ###########

@Func
def num_borders(img: Image) -> int:
    idx = [ np.any(img[0, :] > BACKGROUND),
            np.any(img[-1, :] > BACKGROUND),
            np.any(img[:, 0] > BACKGROUND),
            np.any(img[:, -1] > BACKGROUND) ]
    val = np.ones(4, dtype=np.int)
    return sum(val[idx])
    
@Func
def count_nonzero(img: Image) -> int:
    return np.count_nonzero(img)
    
############ booleans (testing function) ##############
@Func
# def cnot(img: T, func: Callable[[T], bool]) -> bool:
def cnot(b: bool) -> bool:
    return not b

@Func
def cgt(a: int, b: int) -> bool:
    return a > b

@Func
def clt(a: int, b: int) -> bool:
    return a < b

@Func
def cge(a: int, b: int) -> bool:
    return a >= b

@Func
def cle(a: int, b: int) -> bool:
    return a <= b

@Func
def ceq(a: int, b: int) -> bool:
    return a == b

@Func
def cneq(a: int, b: int) -> bool:
    return a != b

############## combinations #############

class Lambda(object):
    def __init__(self, create):
        self._create = create
        
    def __getitem__(self, args):
        if type(args) == tuple:
            return self._create(*args)
        else:
            return self._create(args)
    
@Lambda
def divide_by_color(i):
    return split >> ffilter[has_color[i]] >> first >> inot >> split_conn

@Lambda
def get_color(i):
    return split_color >> ffilter[has_color[i]] >> first

@Lambda
def fzmap(f):
    return fzip >> fmap[f]

@Lambda
def dfzmap(f):
    return dfzip >> fmap[f]

@Lambda
def dmfmap(*funcs):
    return dup[len(funcs)] >> mfmap[funcs]

@Lambda
def mfzmap(*funcs):
    return fzip >> mfmap[funcs]

@Lambda
def dmfzmap(*funcs):
    return dfzip >> mfmap[funcs]

@Lambda
def sq_tile(n, ofst=0):
    return tile[n, n, ofst, ofst]

@Lambda
def sq_zoom_out(n, ofst=0):
    return zoom_out[n, n, ofst, ofst]

crop_space = dmfmap[bound, ident] >> crop

@Lambda
def bounded_op(f):
    return dmfmap[bound, f] >> mask

@Lambda
def is_on_border(n):
    return num_borders >> cgt[n]

@Lambda
def get_first(f):
    return ffilter[f] >> first

@Lambda
def movex(dx):
    return move[dx, 0]

@Lambda
def movey(dy):
    return move[0, dy]

dilate4 = dmfmap[ident, movex[-1], movex[1], movey[-1], movey[1]] >> ior
dilate_corner = dmfmap[ident, move[-1, -1], move[-1, 1], move[1, -1], move[1, 1]] >> ior
dilate8 = dmfmap[dilate4, dilate_corner] >> ior

erode = dmfmap[ident, movex[-1], movex[1], movey[-1], movey[1]] >> iand

# print(split_by_color)
# print(split_by_color[2])

def printa(arrays):
    if len(arrays) == 0:
        print(arrays)
        return
    
    if isinstance(arrays[0], np.ndarray):
        for a in arrays:
            print(a)
    elif isinstance(arrays[0][0], np.ndarray):
        for i, arrs in enumerate(arrays):
            arrs_txt = [str(a).split('\n') for a in arrs]
            # print(arrs_txt)
            for j, row in enumerate(zip(*arrs_txt)):
                print('{}\t'.format(i if j == 0 else ' ') + '\t'.join(row))
    else:
        print(arrays)

codes = [
#     split_color,
#     split_conn,
#     split_even[3, 1],
#     move_right[1],
#     move_left[1],
#     move_up[1],
#     move_down[1],
#     zoom_out[3],
#     zoom_out[4, 5, 3, 1],
#     tile[3, 2, 1],
#     extend[3, 3],
]

# codes = [
# #     #move_right[1] >> move_up[1] >> move_left[1],
# #     #move_right[1] >> move_left[1],
# #     #dup[2] >> fmap[move_down[2]],
# #     #dup[4] >> mfmap[move_down[0], move_down[1], move_down[2], move_down[3]] >> fmap[inot],
# #     split >> fmap[binarise] >> fmap[inot]
#     dmfmap[sq_tile[3], zoom_out[3]] >> mask
# #     split >> ffilter[cnot[is_on_border]],
# #     dup[2] >> mfmap[(split >> fmap[bound]), ident] >> fzip >> fmap[crop]
# ]

x = np.array([[1, 1, 2], [1, 3, 2], [0, 2, 2]])
# printa(x)
for code in codes:
    print('------ {} -------'.format(code))
    printa(code(x))


# In[ ]:


cmap = colors.ListedColormap(
        ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',
         '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
norm = colors.Normalize(vmin=0, vmax=9)
    
def plot_one(ax, input_matrix, i, train_or_test, input_or_output, title_color='black'):
    ax.imshow(input_matrix, cmap=cmap, norm=norm)
    ax.grid(True,which='both',color='lightgrey', linewidth=0.5)    
    ax.set_yticks([x-0.5 for x in range(1+len(input_matrix))])
    ax.set_xticks([x-0.5 for x in range(1+len(input_matrix[0]))])     
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title('{} - {:d} - {}'.format(train_or_test, i, input_or_output), color=title_color)
    
def plot_task(task, solve=None):
    """
    Plots the first train and test pairs of a specified task,
    using same color scheme as the ARC app
    """    
    
    extra_line = 1 if solve is not None else 0
    if isinstance(solve, _Func):
        print(solve.to_str_with_anno())
    
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
            if solve is not None:
                pred = solve(input_img)
                color = 'green' if output_img.shape == pred.shape and np.all(output_img == pred) else 'red'
                plot_one(axs_col[2], pred, i, train_or_test, 'predication', title_color=color)
        plt.tight_layout()
        plt.show()
    
    _plot_helper('train')
    _plot_helper('test')

def check_solution(dataset_name, index, solution=None):
    task = get_data(dataset_name, index)
    plot_task(task, solution)
    
def debug_solution(dataset_name, index, solution, train_or_test='train', sample_index=0):
    task = get_data(dataset_name, index)
    matrix = task[train_or_test][sample_index]['input']
    print('------ {} -------'.format(solution.to_str_with_anno()))
    printa(solution(np.array(matrix)))
    


# In[ ]:


# check_solution('test', 99)


# # Solutions 
# 
# The following are hand-crafted solutions for some tasks viable by this DSL as demostrations for its capability. 

# In[ ]:


pgm = dmfmap[zoom_out[3], tile[3]] >> mask
check_solution('training', 0, pgm)


# In[ ]:


@FCreatorD
def pgm_tr1(inp):
    return inot >> split >> ffilter[is_on_border[1] >> cnot] >> ior >> set_color['yellow'] >> apply_on[ior, inp]

# debug_solution('training', 1, pgm)
check_solution('training', 1, pgm_tr1)


# In[ ]:


# solution not found yet
# pgm = extend[0, 3]
# debug_solution('training', 2, pgm)
# check_solution('training', 2, pgm)


# In[ ]:


# pgm = mfmap[(split >> ffilter[has_color[5]] >> first >> inot >> split_conn), ident] \
@FCreator
def pgm_tr5(inp):
    return divide_by_color['gray'] >> fmap[apply_on[crop, inp]]         >> iand >> set_color['red'] 
# debug_solution('training', 5, pgm_tr5)
check_solution('training', 5, pgm_tr5)


# In[ ]:


pgm = dmfmap[
    get_color['gray'],
    dmfmap[divide_by_color['gray'], ident] >> dfzmap[crop] >> ffilter[has_color['azure'] >> cnot] >> first >> sq_zoom_out[3, 1]
] >> ior
# debug_solution('training', 10, pgm)
check_solution('training', 10, pgm)


# In[ ]:


# pgm = mfmap[(split >> ffilter[has_color[5]] >> first >> inot >> split_conn), ident] \
pgm = dmfmap[divide_by_color['blue'], ident]         >> dfzmap[crop]         >> ior         >> inot         >> set_color['azure'] 
# debug_solution('training', 25, pgm)
check_solution('training', 25, pgm)


# In[ ]:


pgm = crop_space
check_solution('training', 30, pgm)


# In[ ]:


def get_pattern(inp):
    return split_conn >> get_first[num_borders >> ceq[4]]             >> dmfmap[
                ident,
                inot >> split_conn >> first] \
            >> mfmap[ident, apply_on[crop, inp]] \
            >> mask_color
    
@FCreatorD
def pgm_tr32(inp):
     return get_pattern(inp) >> sq_tile[3, 1] >> apply_under[ior, inp]
    
# pgm = get_pattern
# debug_solution('training', 32, pgm)
check_solution('training', 32, pgm_tr32)


# In[ ]:


pgm = crop_space >> split_even[2, 2] >> access[0]
# debug_solution('training', 38, pgm)
check_solution('training', 38, pgm)


# In[ ]:


pgm = move[0, 1]
# debug_solution('training', 52, pgm)
check_solution('training', 52, pgm)


# In[ ]:


@FCreatorD
def pgm_tr54(inp):
    return divide_by_color['azure'] >> mfmap[
            set_color['black'],
            set_color['yellow'],
            set_color['black'],
            set_color['red'],
            set_color['purple'],
            set_color['blue'],
            set_color['black'],
            set_color['green'],
            set_color['black'],
        ] \
        >> lapply_on[ior, inp]

# debug_solution('training', 54, pgm_tr54)
check_solution('training', 54, pgm_tr54)


# In[ ]:


pgm = crop_space >> tile[2, 1]
# debug_solution('training', 56, pgm)
check_solution('training', 56, pgm)


# In[ ]:


pgm = split_even[3, 1] >> access[0]
# debug_solution('training', 66, pgm)
check_solution('training', 66, pgm)


# In[ ]:


@FCreatorD
def pgm_tr69(inp):
    return split_color >> get_first[has_color['azure']]         >> bounded_op[inot >> set_color['green']]         >> apply_on[ior, inp]
# debug_solution('training', 69, pgm_tr69)
check_solution('training', 69, pgm_tr69)


# In[ ]:


@FCreatorD
def pgm_tr71(inp):
    return divide_by_color['yellow'] >> fmap[apply_on[crop, inp]] >> ixor >> set_color['green']
# debug_solution('training', 71, pgm_tr71)
check_solution('training', 71, pgm_tr71)


# In[ ]:


fill = split_conn >> fmap[bounded_op[inot >> set_color['blue']]] >> ior
pgm = dmfmap[fill, ident] >> ior
# debug_solution('training', 80, pgm)
check_solution('training', 80, pgm)


# In[ ]:


@FCreatorD
def pgm_tr94(inp):
    return set_color['blue'] >> dilate8 >> apply_under[ior, inp]
# debug_solution('training', 94, pgm_tr94)
check_solution('training', 94, pgm_tr94)


# In[ ]:


pgm = split_conn8 >> ffilter[count_nonzero >> cgt[1]] >> ior
# debug_solution('training', 96, pgm)
check_solution('training', 96, pgm)
# is there something wrong with the first sample? I think the two dots at top left corner 
# should be included in the output


# In[ ]:


@FCreatorD
def pgm_tr97(inp):
    return erode >> inot >> apply_on[mask, inp]

# debug_solution('training', 97, pgm_tr97)
check_solution('training', 97, pgm_tr97)


# In[ ]:


pgm = split_color >> dmfmap[ident, fmap[bound]] >> fzmap[mask_color] >> ior
# debug_solution('training', 131, pgm)
check_solution('training', 131, pgm)


# In[ ]:


@FCreatorD
def pgm_tr146(inp):
    return split_conn >> ffilter[count_nonzero >> cgt[1]] >> ior >> set_color['azure'] >> apply_on[ior, inp]

# debug_solution('training', 146, pgm_tr146)
check_solution('training', 146, pgm_tr146)


# In[ ]:


@FCreatorD
def pgm_tr165(inp):
    return bounded_op[inot] >> set_color['red'] >> apply_under[ior, inp]
# debug_solution('training', 165, pgm_tr165)
check_solution('training', 165, pgm_tr165)

