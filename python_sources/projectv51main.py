import tkinter
import cv2
from tkinter import messagebox
from tkinter.ttk import *
import PIL.Image, PIL.ImageTk
import os
import numpy as np
# import styler
import time

# import tensorflow as tf
# from tensorflow.python.keras import models 
# from tensorflow.python.keras import losses
# from tensorflow.python.keras import layers
# from tensorflow.python.keras import backend as K
# from tensorflow.python.keras.preprocessing import image as kp_image
# import tensorflow.contrib.eager as tfe
# tf.enable_eager_execution()

global v
global languages
global config
global init_image
global opt
global art
global art_list
 
class App:
	def __init__(self,window,window_title):
		# Configuring the Main Window
		self.windowConfigure(window,window_title)
		
		# Configuring the Grid
		self.setGridConfigure(5,3)
		
		# self.v = tkinter.IntVar()
		# self.v.set(2)
		
		self.languages = ["The_weeping_women","The_starry_night","Monalisa","Wave_of_kanagava","Composition_Vii"]
		self.iterations = ["Iteration 100","Iteration 200","Iteration 300","Iteration 400","Iteration 500","Iteration 600","Iteration 700","Iteration 800","Iteration 900","Iteration 1000"]
		self.iterVal = tkinter.StringVar(self.window)
		self.iterVal.set(self.iterations[9])
		self.v = tkinter.StringVar(self.window)
		self.v.set(self.languages[2])
		
		self.styleTransferUpdate = None

		self.frame1 = tkinter.Frame(self.window)
		self.frame1.grid(row = 0,columnspan=3,sticky = tkinter.W+tkinter.E+tkinter.N+tkinter.S)
		self.header_photo = self.resizedImage('img/header.png',width = self.window.winfo_screenwidth())
		self.header = tkinter.Label(self.frame1,image = self.header_photo,width = self.window.winfo_screenwidth(),height=80)
		self.header.pack(fill=tkinter.X)
		
		self.frame2 = tkinter.Frame(self.window,bg="#00204f")
		self.frame2.grid(row = 1,column=0,sticky = tkinter.W+tkinter.E+tkinter.N+tkinter.S)
		self.content_photo = self.resizedImage('img/camerathumbnail.png',width = 200)		
		self.content = tkinter.Label(self.frame2,image = self.content_photo,width = 200, height=200,bg="#00204f")
		self.content.pack()
		
		self.btn_snapshot_photo1 = PIL.ImageTk.PhotoImage(file="img/capturebtn.gif")
		self.btn_snapshot_photo2 = PIL.ImageTk.PhotoImage(file="img/capturebtnhov.gif")
		self.btn_snapshot=tkinter.Button(self.frame2, image=self.btn_snapshot_photo1, command=self.captureImage, bd=0, bg='#00204f')
		self.btn_snapshot.pack(pady=10)
		self.btn_snapshot.focus_set()
		self.btn_snapshot.bind('<Enter>', lambda e: e.widget.configure(image=self.btn_snapshot_photo2))
		self.btn_snapshot.bind('<Leave>', lambda e: e.widget.configure(image=self.btn_snapshot_photo1))
		
		self.frame3 = tkinter.Frame(self.window,bg="#00204f")
		self.frame3.grid(row = 1,column=1,sticky = tkinter.W+tkinter.E+tkinter.N+tkinter.S)
		# self.style_photo = self.resizedImage('img/{0}.jpg'.format(self.languages[self.v.get()]),width = 200)		
		self.style_photo = self.resizedImage('img/{0}.jpg'.format(self.v.get()),width = 200)		
		self.style = tkinter.Label(self.frame3,image = self.style_photo,width=200,height=200, bg="#00204f")
		self.style.pack()
		self.style_list = tkinter.OptionMenu(self.frame3, self.v, *self.languages)
		self.style_list.pack(pady=10)
		self.v.trace('w', self.ShowStyleImage)
		
		self.frame4 = tkinter.Frame(self.window,pady = 30,bg="#00204f")
		self.frame4.grid(row = 1,column=2,sticky = tkinter.W+tkinter.E+tkinter.N+tkinter.S)
		# self.addStyleRadio(self.frame4)
		
		self.btn_transfer_photo1 = PIL.ImageTk.PhotoImage(file="img/transferbtn.gif")
		self.btn_transfer_photo2 = PIL.ImageTk.PhotoImage(file="img/transferbtnhov.gif")
		self.btn_transfer=tkinter.Button(self.frame4, image = self.btn_transfer_photo1, bd = 0, command=self.styleTransfer,bg="#00204f")
		self.btn_transfer.pack(pady=50)
		self.btn_transfer.bind('<Enter>', lambda e: e.widget.configure(image=self.btn_transfer_photo2))
		self.btn_transfer.bind('<Leave>', lambda e: e.widget.configure(image=self.btn_transfer_photo1))
		
		self.frame5 = tkinter.Frame(self.window,bg="#00204f")
		self.frame5.grid(row = 2, column = 2,sticky = tkinter.W+tkinter.E+tkinter.N+tkinter.S)
		self.iter_list = tkinter.OptionMenu(self.frame5, self.iterVal, *self.iterations)
		self.iter_list.pack(pady=10)
		self.iterVal.trace('w', self.ShowIterImage)
		
		
		self.frame6 = tkinter.Frame(self.window,bg="#00204f")
		self.frame6.grid(row = 2, column = 0,columnspan = 2,sticky = tkinter.W+tkinter.E+tkinter.N+tkinter.S)
		self.progress = Progressbar(self.frame6,mode='determinate',maximum=5,length=500,style='black.Horizontal.TProgressbar')
		self.progress.pack(side=tkinter.LEFT,padx = 70)
		
		self.btn_stop_photo1 = PIL.ImageTk.PhotoImage(file="img/stopbtn.gif")
		self.btn_stop_photo2 = PIL.ImageTk.PhotoImage(file="img/stopbtnhov.gif")
		self.btn_stop=tkinter.Button(self.frame6, image = self.btn_stop_photo1, command=self.stopStyleTransfer,bg="#00204f",bd = 0)
		self.btn_stop.pack(side=tkinter.LEFT,padx = 10)
		self.btn_stop.bind('<Enter>', lambda e: e.widget.configure(image=self.btn_stop_photo2))
		self.btn_stop.bind('<Leave>', lambda e: e.widget.configure(image=self.btn_stop_photo1))

		self.frame7 = tkinter.Frame(self.window,bg="#00204f")
		self.frame7.grid(row = 3,column = 0,columnspan = 2,sticky = tkinter.W+tkinter.E+tkinter.N+tkinter.S)
		self.live_style_title = tkinter.Label(self.frame7,text="Live Style Transfer",bg="#00204f")
		self.live_style_title.pack()
		self.output_photo = self.resizedImage('img/finalartthumbnail.jpg', width = 350)		
		self.output = tkinter.Label(self.frame7,image = self.output_photo,width = 350, height=350,bg="#00204f")
		self.output.pack()
		
		# self.frame8 = tkinter.Frame(self.window,bd=2,relief = tkinter.GROOVE,bg="#00204f")
		# self.frame8.grid(row = 3,column = 1,sticky = tkinter.W+tkinter.E+tkinter.N+tkinter.S)
		# self.addIterRadio(self.frame8)
		self.frame9 = tkinter.Frame(self.window,bg="#00204f")
		self.frame9.grid(row = 3,column = 2,sticky = tkinter.W+tkinter.E+tkinter.N+tkinter.S)
		
		self.iter_output_photo = self.resizedImage('img/finalartthumbnail.jpg',width = 350)		
		self.iter_output = tkinter.Label(self.frame9,image = self.iter_output_photo,width = 350,height=300,bg="#00204f"	)
		self.iter_output.pack()
		
		
		self.window.mainloop()
	
	def windowConfigure(self,window,window_title):
		self.window = window
		self.window.title(window_title)
		self.window.state('zoomed')
		self.window.geometry('{0}x{1}'.format(self.window.winfo_screenwidth(), self.window.winfo_screenheight()))
		
		# self.window.attributes('-fullscreen', True)
		self.window.bind('<Escape>',lambda e: self.window.destroy())
		self.window.grid()
	
	def setGridConfigure(self,row,column):
		for r in range(row):
			self.window.rowconfigure(r)
		for c in range(column):
			self.window.columnconfigure(c, weight=1)
		
		
	def addStyleRadio(self,frame):
		tkinter.Label(frame, text="""Choose Style Images:""", justify = tkinter.LEFT, padx = 20).pack()
		for val, language in enumerate(self.languages):
			tkinter.Radiobutton(frame, 
			text=language,
			indicatoron = 0,
			width = 20,
			padx = 20, 
			variable=self.v, 
			command=self.ShowStyleImage,
			value=val).pack()
	
	def addIterRadio(self,frame):
		tkinter.Label(frame, text="""Iterations:""", justify = tkinter.LEFT, padx = 20).pack()
		iterations = ["Iteration 100","Iteration 200","Iteration 300","Iteration 400","Iteration 500","Iteration 600","Iteration 700","Iteration 800","Iteration 900","Iteration 1000"]
		for val, iteration in enumerate(iterations):
			tkinter.Radiobutton(frame, 
			text=iteration,
			indicatoron = 0,
			width = 20,
			padx = 20, 
			variable=self.iterVal, 
			command=self.ShowIterImage,
			value=val).pack()
			
	def ShowIterImage(self):
		self.iter_output_photo = self.resizedArt(self.iterVal.get(),width = 300)
		self.iter_output.image = self.iter_output_photo
		self.iter_output.configure(image = self.iter_output_photo)
			
	def ShowStyleImage(self,*args):
		# self.style_photo = self.resizedImage('img/{0}.jpg'.format(self.languages[self.v.get()]),width = 200)
		self.style_photo = self.resizedImage('img/{0}.jpg'.format(self.v.get()),width = 200)
		self.style.image = self.style_photo
		self.style.configure(image = self.style_photo)

		
	def captureImage(self):
		os.system("python camera.py")
		if(os.path.exists("img/content.jpg")):
			self.content_photo = self.resizedImage('img/content.jpg',width = 200)
			self.content.image = self.content_photo
			self.content.configure(image = self.content_photo)
		
	def resizedImage(self,image,width=300):
		image = cv2.imread(image, cv2.IMREAD_UNCHANGED)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		factor = width/np.max(image.shape)
		width = int(image.shape[1] * factor)
		height = int(image.shape[0] * factor)
		dim = (width, height)
		# resize image
		thumbnail_img = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
		photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(thumbnail_img))
		return photo
		
	def styleTransfer(self):	
		if((self.v.get() >= 0) and (os.path.exists("img/content.jpg"))):
			# c_weight=1e3
			# s_weight=1e-2
			# c_path = "img/content.jpg"
			# s_path = 'img/{0}.jpg'.format(self.languages[self.v.get()])
			# model = styler.get_model() 
			# for layer in model.layers:
				# layer.trainable = False

			# s_features, c_features = styler.get_features(model, c_path, s_path)
			# gram_style_features = [styler.gram_matrix(style_feature) for style_feature in s_features]

			# self.init_image = styler.load_and_process(c_path)
			# self.init_image = tfe.Variable(self.init_image, dtype=tf.float32)
			# self.opt = tf.train.AdamOptimizer(learning_rate=5, beta1=0.99, epsilon=1e-1)
			# optimized_loss, self.art = float('inf'), None

			# loss_weights = (s_weight, c_weight)
			# self.config = {
				   # 'model': model,'loss_weights': loss_weights,'init_image': self.init_image,
				   # 'gram_style_features': gram_style_features,'c_features': c_features
				  # }

			# norm_means = np.array([103.939, 116.779, 123.68])
			# self.min_vals = -norm_means
			# self.max_vals = 255 - norm_means 

			# ctr = 1

			# start_time = time.time()

			# self.art_list = []
			# self.i = 0
			# self.output_photo = self.resizedImage('img/content.jpg',300)
			# self.output.image = self.output_photo
			# self.output.configure(image = self.output_photo)			
			# self.GenerateArt()
			pass
		else:
			tkinter.messagebox.showerror("Image Style Transfer", "Before Transfer the Style, Please choose content and style image")
			
	def resizedArt(self,image,width=300):
		factor = width/np.max(image.shape)
		width = int(image.shape[1] * factor)
		height = int(image.shape[0] * factor)
		dim = (width, height)
		# resize image
		thumbnail_img = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
		photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(thumbnail_img))
		return photo

		
	def GenerateArt(self):
		if(self.i < 10):
			# grads, all_loss = styler.calc_grads(self.config)
			# loss, style_score, content_score = all_loss
			# self.opt.apply_gradients([(grads, self.init_image)])
			# resized = tf.clip_by_value(self.init_image, self.min_vals, self.max_vals)
			# self.init_image.assign(resized)
			 
			# time.sleep(.1)
			# self.art = self.init_image.numpy()
			# self.art = styler.deprocess_img(self.art)
			# if(self.i % 1 == 0):
				# self.art_list.append(self.art)
				# self.output_photo = self.resizedArt(self.art,300)
				# self.output.image = self.output_photo
				# self.output.configure(image=self.output_photo)
			# self.progress['value'] = self.i
			# print(self.i)
			# self.i += 1
			# self.styleTransferUpdate = self.window.after(1,self.GenerateArt)
			pass

	def stopStyleTransfer(self):
		if self.styleTransferUpdate is not None:
			self.window.after_cancel(self.styleTransferUpdate)
	
	def __del__(self):
		if(os.path.exists("img/content.jpg")):
			os.remove('img/content.jpg')
	

		
App(tkinter.Tk(),"Image Style Transfer")
