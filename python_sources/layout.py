import tkinter
import cv2
import PIL.Image, PIL.ImageTk
import camera

class App:
	def __init__(self,window):
		self.window = window
		self.window.geometry('800x800')
		# self.window.geometry('{}x{}'.format(self.window.winfo_screenwidth, self.window.winfo_screenheight))
		
		
		

		# create a canvas and place in frame
		self.canvas = tkinter.Canvas(self.window, width = 300, height = 300, bd=1, relief=tkinter.SUNKEN)
		self.canvas.grid(row = 0, column = 0)	

		self.btn_snapshot=tkinter.Button(self.window, text="Capture", width=50, command=self.captureImage)
		self.btn_snapshot.grid(row=1,column=0)
		

		# Add a PhotoImage to the Canvas
		image = cv2.imread('img/thumbnail.png', cv2.IMREAD_UNCHANGED)
		dim = (300, 300)
		# resize image
		thumbnail_img = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
		photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(thumbnail_img))
		
		self.thumbnail_img = self.canvas.create_image(0, 0, image=photo, anchor=tkinter.NW)

		
		self.window.mainloop()
		
	def captureImage(self):
		self.content_cap_win = tkinter.Toplevel(master = self.window)
		camera.CaptureCam(self.content_cap_win, "Capture Content Image")
		self.content_cap_win.destroy()
		
		# self.canvas.itemconfig(self.thumbnail_img,image=resizedImage('content.jpg'))
		
def resizedImage(image):
	image = cv2.imread(image, cv2.IMREAD_UNCHANGED)
	dim = (300, 300)
	# resize image
	thumbnail_img = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
	photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(thumbnail_img))
	return photo
	

		
App(tkinter.Tk())