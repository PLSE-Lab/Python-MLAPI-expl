from tkinter import *
from tkinter.ttk import *
import time
     
class ProgressBarApp(Tk):

    def __init__(self, *args, **kwargs):
        Tk.__init__(self,*args, **kwargs)

        root = Tk()

        frame_top = Frame(root)
        frame_middle = Frame(root)
        frame_bottom = Frame(root)

        frame_top.pack(expand=True, fill=BOTH, side=TOP)
        frame_middle.pack(expand=True, fill=BOTH, side=TOP)
        frame_bottom.pack(expand=True, fill=BOTH, side=TOP)


        self.label1 = Label(frame_top, text="Progress Bar 1")
        self.label1.pack()

        self.progress_bar1 = Progressbar(frame_top, orient='horizontal', mode='determinate', maximum=30.25)
        self.progress_bar1.pack(expand=True, fill=BOTH, side=TOP)

        self.start_button = Button(frame_middle, text="start", command=lambda :self.start())
        self.start_button.pack(expand=True, fill=BOTH, side=BOTTOM)

        self.button_bar1 = Button(frame_bottom, text="Progress Bar 1", command=lambda :self.progress_bar_1_selected())
        self.button_bar1.pack(side=LEFT)

    def start(self):
        i = 0
        print("started!")
        while (self.progress_bar1["value"] < self.progress_bar1["maximum"]):
            self.progress_bar1.update()
            self.progress_bar1["value"] = i**2
            i += 0.01
            time.sleep(0.01)

    def next_comparison(self):
        print("Next comparison!")

app = ProgressBarApp()
app.mainloop()