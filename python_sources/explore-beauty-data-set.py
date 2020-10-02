#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.image as mpimg
from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))
dir=r'/kaggle/input'
dir_list=os.listdir(dir)
print (dir_list) 
source_dir=os.path.join (dir, dir_list[0])
print (source_dir)
data_list=os.listdir(source_dir)
print(data_list)
test_dir=os.path.join(source_dir, 'test')
test_classes=os.listdir(test_dir)
print (test_classes)
average_dir=os.path.join(test_dir, 'average')
beautiful_dir=os.path.join(test_dir, 'beautiful')
print('average directory is ', average_dir, '   beautiful_dir is ', beautiful_dir)


# In[ ]:


def print_in_color(txt_msg,fore_tupple,back_tupple,):
    #prints the text_msg in the foreground color specified by fore_tupple with the background specified by back_tupple 
    #text_msg is the text, fore_tupple is foregroud color tupple (r,g,b), back_tupple is background tupple (r,g,b)
    rf,gf,bf=fore_tupple
    rb,gb,bb=back_tupple
    msg='{0}' + txt_msg
    mat='\33[38;2;' + str(rf) +';' + str(gf) + ';' + str(bf) + ';48;2;' + str(rb) + ';' +str(gb) + ';' + str(bb) +'m' 
    print(msg .format(mat))
    print('\33[39m')
    return

def print_samples( dir,rows, columns, ):
    fig = plt.figure(figsize=(20,100))
    for row in range(rows):
        for column in range(columns):
            i= row * columns + column +1
            #print('row= ',row, '  column= ',column, '  I= ',i)
            filename='00'+ str(i) + '.jpg'
            file_path=os.path.join(dir,filename) 
            img = mpimg.imread(file_path)
            a = fig.add_subplot(rows, columns, i)
            imgplot=plt.imshow(img)
            a.axis("off")
            a.set_title(filename)
    return


# In[ ]:


msg='{0:15s}ROW 1-IMAGES OF BEAUTIFUL WOMEN   ROW 2= IMAGES OF AVERAGE WOMEN'.format(' ')        
print_in_color(msg, (0,255,0),(0,0,0))
dir=beautiful_dir
print_samples(dir, 1,8)
dir=average_dir
print_samples(dir, 1,8)
