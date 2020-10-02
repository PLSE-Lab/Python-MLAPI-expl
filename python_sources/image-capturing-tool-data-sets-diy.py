
from tkinter import *
from tkinter.colorchooser import askcolor
from tkinter import messagebox
from tkinter.ttk import Progressbar
import time
import os
import cv2
import pandas as pd
import numpy as np
#import sounddevice as sd #to play a sound at the end of series  sd.play(np.random.rand(3000), 10000)


#download and install pillow:
# http://www.lfd.uci.edu/~gohlke/pythonlibs/#pillow
from PIL import Image, ImageTk

folder=os.path.join('..','input')


camera_port=0 # if you have more than one camera it might have index 1,2,3 etc.
test_image_taken=0 # starting image number (to be added at the end of the filename)
delaty_between_frames=0.01  # depends on the rotation speed. Fater the rotation, shorter the internal can be. 
last_color="#ffffff" # default color added to the filename

# download LEGO inventory from     https://www.kaggle.com/rtatman/lego-database#inventory_parts.csv
df_parts=pd.read_csv(os.path.join(folder,'parts.csv'))

# a cleanup to limit the number of categories
for ss in ["Sticker for","Sticker Sheet","Duplo","DUPLO","Minifig", "MINI", "Torso","Booklet"]:

    df_parts=df_parts[~(df_parts["name"].str.contains(ss))]


df_parts["part_num_name"]=df_parts["part_num"]+" "+df_parts["name"]


# Just a generic list to populate the listbox
lbox_list = list(df_parts["part_num_name"])

# Here, we are creating our class, Window, and inheriting from the Frame
# class. Frame is a class from the tkinter module. (see Lib/tkinter/__init__)
class Window(Frame):

    # Define settings upon initialization. Here you can specify
    def __init__(self, master=None):
        
        # parameters that you want to send through the Frame class. 
        Frame.__init__(self, master)   

        #reference to the master widget, which is the tk window                 
        self.master = master

        #with that, we want to then run init_window, which doesn't yet exist
        self.init_window()

    #Creation of init_window
    def init_window(self):
        global canvas, coloredrectangle, nColorlabel, last_color
        self.master.geometry("1200x600+200+200")
        # changing the title of our master widget      
        self.master.title("Capture series of images and create LEGO parts data set")

        # allowing the widget to take the full space of the root window
        self.pack(fill=BOTH, expand=1)

        # creating a menu instance
        menu = Menu(self.master)
        self.master.config(menu=menu)

        # create the file object)
        file = Menu(menu)

        # adds a command to the menu option, calling it exit, and the
        # command it runs on event is client_exit
        file.add_command(label="Exit", command=self.client_exit)

        #added "file" to our menu
        menu.add_cascade(label="File", menu=file)


        # create the file object)
        edit = Menu(menu)

        # adds a command to the menu option, calling it exit, and the
        # command it runs on event is client_exit
        edit.add_command(label="Capture Img", command=self.captureImg)
        #edit.add_command(label="Show Text", command=self.showText)

        #added "file" to our menu
        menu.add_cascade(label="Edit", menu=edit)
        
        self.upframe = Frame(self.master ) # for image
        self.upframe.pack(side = TOP, fill=X )

        self.uplftframe = Frame(self.upframe) # for image
        self.uplftframe.pack(side = LEFT, fill=X)
        
        self.uprghttframe = Frame(self.upframe, background="gray" ) # for image
        self.uprghttframe.pack(side = RIGHT, fill=X )
        

        
        bottomframe = Frame(self.master) # for buttons
        bottomframe.pack( side = BOTTOM, fill=X )  
        
        paramframe = Frame(self.master) # for buttons
        paramframe.pack( side = BOTTOM, fill=X )


        
        
        bCapture = Button(bottomframe, text = "Test Camera", command = self.captureImg)
        bCapture.pack(side="left", anchor = "w", padx=15, pady=10)



        
        bCaptureSeries = Button(bottomframe, text = "Capture Series", command = self.captureImgSeries)
        bCaptureSeries.pack(side="left", anchor = "w", padx=45, pady=10)
        
        self.progress=Progressbar(bottomframe,orient=HORIZONTAL,length=100,mode='determinate')
        self.progress.pack(side="left", anchor = "w", padx=45, pady=10)
        
        bClose = Button(bottomframe, text = "Close", command = self.client_exit)
        bClose.pack(side="right", anchor = "w", padx=15, pady=10)
        
        ######################### PARAM FRAME
        nImgsEntrylabel = Label(paramframe, text="Number of images in series")
        nImgsEntrylabel.pack(side="left", anchor = "w", padx=15, pady=10)
        
        self.nImgsEntry_var = StringVar()
        self.nImgsEntry_var.set("100")
        self.nImgsEntry = Entry(paramframe, textvariable=self.nImgsEntry_var, width=7)

        self.nImgsEntry.pack(side="left", anchor = "w", padx=5, pady=10)
        
        colorButton= Button(paramframe, text='Select Color', command=self.getColor)
        colorButton.pack(side="left", anchor = "w", padx=15, pady=10)
        
        canvas = Canvas(paramframe, width = 30, height = 30)
        canvas.pack(side="left", anchor = "w", padx=5, pady=10)
        coloredrectangle=canvas.create_rectangle(0, 0, 30, 30, fill=last_color)
        
        nColorlabel = Label(paramframe, text=last_color)
        nColorlabel.pack(side="left", anchor = "w", padx=5, pady=10)
        
        
        self.search_var = StringVar()
        self.search_var.trace("w", lambda name, index, mode: self.update_list())
        
        searchCategory = Label(self.uplftframe, text="Search Category")
        self.searchentry = Entry(self.uplftframe, textvariable=self.search_var, width=13)
        self.lbox = Listbox(self.uplftframe, width=75, height=15)
         
        searchCategory.pack(side="top", anchor = "w", padx=5, pady=10)
        self.searchentry.pack(side="top", anchor = "w", padx=5, pady=10)
        self.lbox.pack(side="top", anchor = "w", padx=5, pady=10)
         
        # Function for updating the list/doing the search.
        # It needs to be called here to populate the listbox.
        self.update_list()
        
 
        

    def captureImg(self):
        global camera_port, test_image_taken, folder
        

                
        camera = cv2.VideoCapture(camera_port)
        time.sleep(1.0)  # Initialization timeout. If you don't wait, the image will be dark or autofocus will not adjust properly
        return_value, image = camera.read()
        filename='test_camera_image.png'
        cv2.imwrite(filename, image)
        load = Image.open('test_camera_image.png')
        render = ImageTk.PhotoImage(load)
        del(camera)  # so that others can use the camera as soon as possible


        if test_image_taken==0:
            # labels can be text or images
            self.img = Label(self.uprghttframe, image=render)
            self.img.image = render
            self.img.pack(side="left", anchor = "w", padx=5, pady=10)
            test_image_taken=1
        else:
            self.img.configure(image=render)
            self.img.image = render
         
            
    def captureImgSeries(self):
        global camera_port,test_image_taken, folder, delaty_between_frames, last_color
        
        
        
        if len(self.lbox.curselection())>0:
            category_name=self.lbox.get(self.lbox.curselection()).replace("/",",").replace("\\",",")
            print("Selected category:", category_name)
            
            # check if folder exists
            
            
            if not os.path.exists(folder+category_name):
                os.makedirs(folder+category_name)
                # create subfolder
            
            camera = cv2.VideoCapture(camera_port)
            time.sleep(1.0)  # Initialization timeout. If you don't wait, the image will be dark or autofocus will not adjust properly
            max_images=int(self.nImgsEntry_var.get())
            
            file_num=1
            for k in range(0,max_images):
                return_value, image = camera.read()
                filename=os.path.join(folder+category_name,'img-'+last_color +'-'+str(file_num)+'.png')
                
                while os.path.isfile(filename): # if file with the sma ename exists - increase file counter
                    file_num=file_num+1
                    filename=os.path.join(folder+category_name,'img-'+last_color +'-'+str(file_num)+'.png')
                    
                cv2.imwrite(filename, image,[cv2.IMWRITE_PNG_COMPRESSION, 9])
                time.sleep(delaty_between_frames)
                
                pct=int(100.*k/max_images)
                if int(pct/50.0)*50==pct: # update once per 2 %
                    #sd.play(np.random.rand(300), 10000)
                    self.progress['value']=pct
                    self.update_idletasks()
                
            del(camera)  # so that others can use the camera as soon as possible
            
            if k>0:
                load = Image.open(filename) # read the last file
                render = ImageTk.PhotoImage(load)
                
                self.progress['value']=100
                self.update_idletasks()
        
        
                if test_image_taken==0:
                    # labels can be text or images
                    self.img = Label(self.uprghttframe, image=render)
                    self.img.image = render
                    self.img.pack(side="left", anchor = "w", padx=5, pady=10)
                    test_image_taken=1
                else:
                    self.img.configure(image=render)
                    self.img.image = render
                    
            self.progress['value']=0
            self.update_idletasks()
            
            #sd.play(np.random.rand(3000), 10000)
         
        else:
            m=messagebox.showerror("ERROR", "Please select one category from the list")
        
    def update_list(self):
        # used for text search
        global lbox_list
        search_term = self.search_var.get()

        self.lbox.delete(0, END)
     
        for item in lbox_list:
            if search_term.lower() in item.lower():
                self.lbox.insert(END, item)
        
    def getColor(self):
        # allows to set a color that will be added to the filename in HEX format
        global canvas, coloredrectangle, nColorlabel, last_color
        color = askcolor() 
        print (color)
        last_color=color[1]
        canvas.itemconfig(coloredrectangle, fill=last_color)
        nColorlabel.config(text=color[1])



    def client_exit(self):
        #exit()
        print("Exiting")
        #quit()
        root.destroy()

root = Tk()


#creation of an instance
app = Window(root)


#mainloop 
root.mainloop()  
root.destroy()
















