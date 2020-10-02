#!/usr/bin/env python
# coding: utf-8

# # Bounding box data for the whale flukes
# <p>This notebook explores the Humpback Whale Identification - Fluke Location dataset.<p>
# <p>This dataset is associated with the Humpback Whale Identification Challenge dataset. It contains the location of points on the edge of the fluke for 1200 pictures randomly selected from the Humpback Whale Identification Challenge training set. Points are selected to capture the leftmost and rightmost points, as well as the highest and lowest points. Additional points are added to help determine the fluke bounding box following an affine transformation on the image.</p>
# <p>The intent of this dataset is to build a model for locating the whale fluke inside the image. In the context of the Humpback Whale Identification Challenge, such a model can then be used to crop images around the region of interest.</p>

# # Read the data

# In[ ]:


with open('../input/humpback-whale-identification-fluke-location/cropping.txt', 'rt') as f: data = f.read().split('\n')[:-1]
len(data) # Number of rows in the dataset


# Show the first 5 lines.

# In[ ]:


for line in data[:5]: print(line)


# Convert data to a list of tuples. Each tuple contains the picture filename and a list of coordinates.

# In[ ]:


data = [line.split(',') for line in data]
data = [(p,[(int(coord[i]),int(coord[i+1])) for i in range(0,len(coord),2)]) for p,*coord in data]
data[0] # First row of the dataset


# We can show the coordinates on the original image

# In[ ]:


from PIL import Image as pil_image
from PIL.ImageDraw import Draw

def read_raw_image(p):
    return pil_image.open('../input/whale-categorization-playground/train/' + p)

def draw_dot(draw, x, y):
    draw.ellipse(((x-5,y-5),(x+5,y+5)), fill='red', outline='red')

def draw_dots(draw, coordinates):
    for x,y in coordinates: draw_dot(draw, x, y)

filename,coordinates = data[0]
img = read_raw_image(filename)
draw = Draw(img)
draw_dots(draw, coordinates)
img


# The list of coordinates can be used to determine a bounding box around the fluke.

# In[ ]:


def bounding_rectangle(list):
    x0, y0 = list[0]
    x1, y1 = x0, y0
    for x,y in list[1:]:
        x0 = min(x0, x)
        y0 = min(y0, y)
        x1 = max(x1, x)
        y1 = max(y1, y)
    return x0,y0,x1,y1

box = bounding_rectangle(coordinates)
box


# In[ ]:


draw.rectangle(box, outline='red')
img


# # Affine transformation
# Using affine transformation is a basic data augmentation technique used when training deep convolution network.<br/>
# The example below shows a 10-degree rotation.

# In[ ]:


# Suppress annoying stderr output when importing keras.
import sys
old_stderr = sys.stderr
sys.stderr = open('/dev/null', 'w')
from keras.preprocessing.image import img_to_array,array_to_img
sys.stderr = old_stderr

import numpy as np
from numpy.linalg import inv
from scipy.ndimage import affine_transform

def transform_img(x, affine):
    matrix   = affine[:2,:2]
    offset   = affine[:2,2]
    x        = np.moveaxis(img_to_array(x), -1, 0) # Change to channel first
    channels = [affine_transform(channel, matrix, offset, order=1, mode='constant', cval=np.average(channel)) for channel in x]
    return array_to_img(np.moveaxis(np.stack(channels, axis=0), 0, -1)) # Back to channel last, and image format

width, height = img.size
rotation = np.deg2rad(10)
# Place the origin at the center of the image
center = np.array([[1, 0, -height/2], [0, 1, -width/2], [0, 0, 1]]) 
# Rotate
rotate = np.array([[np.cos(rotation), np.sin(rotation), 0], [-np.sin(rotation), np.cos(rotation), 0], [0, 0, 1]])
# Restaure the origin
decenter = inv(center)
# Combine the transformations into one
m   = np.dot(decenter, np.dot(rotate, center))
img = transform_img(img, m)
img


# A new bounding box can be computed for the rotated image:<br/>
# Each point on the perimeter is transformed, and a new bounding box is computed on the transformed coordinates.<br/>
# Notice how different points are selected when constructing the bounding box.

# In[ ]:


def coord_transform(coordinates, m):
    result = []
    for x,y in coordinates:
        y,x,_ = m.dot([y,x,1]).astype(np.int)
        result.append((x,y))
    return result

transformed_coordinates = coord_transform(coordinates, inv(m))
transformed_coordinates


# In[ ]:


transformed_box = bounding_rectangle(transformed_coordinates)
transformed_box


# In[ ]:


draw = Draw(img)
draw.rectangle(transformed_box, outline='yellow')
img


# The idea here is to use the computed bounding boxes to train a bounding box model for the whale flukes.<br/>
# When using image affine transformations as a form of data augmentation, the individual points make it possible to compute the bounding box of the transformed image.

# # How valuable is image cropping?

# From my experiments, it appears that building an image cropping model is useful but not critical in achieving good model accuracy.<br/>
# To determine this, I trained essentially identical versions of the identification model with and without image cropping:
# 
# * 0.714 : score without image cropping
# * 0.766 : score with image cropping
# 
# While image cropping provides a substantial improvement, good accuracy is still possible without it.<br/>
# *N.B.: I don't have a "no cropping" equivalent for my final submission, so I am using older submissions for comparison.*

# # Java program to record boundary points
# I used the following small Java programme to record the points in the dataset.
# ```java
# import java.io.*;
# import java.util.*;
# import javafx.application.*;
# import javafx.event.*;
# import javafx.scene.*;
# import javafx.scene.canvas.*;
# import javafx.scene.image.*;
# import javafx.scene.input.*;
# import javafx.scene.layout.*;
# import javafx.stage.*;
# 
# public class WhaleTagger extends Application
# {
#   private List<String>  list   = new ArrayList<>();
#   private int           index  = 0;
#   private StringBuilder buffer = new StringBuilder();
#   private int           tagged;
# 
#   public static void main(String[] args)
#   {
#     Application.launch(args);
#   }
# 
#   @Override
#   public void start(Stage primaryStage) throws Exception
#   {
#     initfiles();
#     Image     image     = getNextImage();
#     int       width     = (int)image.getWidth();
#     int       height    = (int)image.getHeight();
# 
#     Canvas          canvas = new Canvas(width, height);
#     GraphicsContext gc     = canvas.getGraphicsContext2D();
#     gc.drawImage(image, 0, 0, canvas.getWidth(), canvas.getHeight());
# 
#     // Reset the Canvas when the user double-clicks
#     canvas.addEventHandler(MouseEvent.MOUSE_CLICKED, new EventHandler<MouseEvent>()
#     {
#       @Override
#       public void handle(MouseEvent e)
#       {
#         if (e.getClickCount() == 1)
#         {
#           int x = (int)e.getX();
#           int y = (int)e.getY();
#           gc.clearRect(x - 2, y - 2, 5, 5);
#           buffer.append(",");
#           buffer.append(x);
#           buffer.append(",");
#           buffer.append(y);
#           System.out.println(buffer);
#         }
#       }
#     });
# 
# 
#     canvas.setFocusTraversable(true);
#     canvas.requestFocus();
#     canvas.setOnKeyPressed(new EventHandler<KeyEvent>()
#     {
#       @Override
#       public void handle(KeyEvent event)
#       {
#         if (event.getCode() == KeyCode.SPACE || event.getCode() == KeyCode.ENTER)
#         {
#           if (buffer.length() > 0)
#           {
#             ++tagged;
#             list.set(index, list.get(index) + buffer.toString());
#             System.out.println(list.get(index));
#           }
#           Image image = getNextImage();
#           canvas.setWidth(image.getWidth());
#           canvas.setHeight(image.getHeight());
#           canvas.getGraphicsContext2D().drawImage(image, 0, 0, canvas.getWidth(), canvas.getHeight());
#           primaryStage.sizeToScene();
#         }
#         else if (event.getCode() == KeyCode.BACK_SPACE)
#         {
#           System.out.println("Resetting");
#           Image image = getNextImage();
#           canvas.getGraphicsContext2D().drawImage(image, 0, 0, canvas.getWidth(), canvas.getHeight());
#         }
#         else if (event.getCode() == KeyCode.ESCAPE)
#         {
#           done();
#         }
#       }
#     });
# 
#     // Add the Canvas to the Scene, and show the Stage
#     Pane root = new Pane();
#     root.getChildren().add(canvas);
#     Scene scene = new Scene(root);
#     primaryStage.setTitle("Whale tagging");
#     primaryStage.sizeToScene();
#     primaryStage.setScene(scene);
#     primaryStage.show();
#   }
# 
#   private Image getNextImage()
#   {
#     try
#     {
#       for (;;)
#       {
#         if (index == list.size()) done();
#         if (!list.get(index).contains(",")) break;
#         ++index;
#       }
#       File file = new File("humpback-whale-identification\\train\\" + list.get(index));
#       Image image = new Image(file.toURI().toURL().toString());
#       buffer.setLength(0);
#       System.out.println();
#       System.out.println(tagged + 1 + ") " + list.get(index) + " " + (int)image.getWidth() + " x " + (int)image.getHeight());
#       return image;
#     }
#     catch (Exception e)
#     {
#       e.printStackTrace();
#     }
#     return null;
#   }
# 
#   private void done()
#   {
#     try
#     {
#       System.out.println("Writing filelist.txt");
#       PrintStream stream = new PrintStream(new FileOutputStream("filelist.txt"));
#       for (String s : list) stream.println(s);
#       stream.close();
#       System.out.println("Exiting");
#       System.exit(0);
#     }
#     catch (Exception e)
#     {
#       e.printStackTrace();
#     }
#   }
# 
#   private void initfiles() throws Exception
#   {
#     tagged = 0;
#     int untagged = 0;
#     BufferedReader reader = new BufferedReader(new FileReader("filelist.txt"));
#     for (;;)
#     {
#       String line = reader.readLine();
#       if (line == null || line.length() == 0) break;
#       list.add(line);
#       if (line.contains(","))
#         ++tagged;
#       else
#         ++untagged;
#     }
#     reader.close();
#     System.out.println("Read " + tagged + " tagged and " + untagged + " untagged entries");
#   }
# }
# ```
