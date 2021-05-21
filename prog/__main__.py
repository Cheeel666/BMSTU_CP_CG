from src.model import NLM
from tkinter import *

class Window:
    def __init__(self):
        self.window = Tk()
        self.window.title("NLM")
        self.window.geometry('500x280')
        self.lbl = Label(self.window, text="Введите название изображения:")
        self.lbl.grid(column=0, row=0)
        self.entry = Entry(self.window, width = 50)
        self.entry.grid(column = 0, row = 1)
        self.lbl1 = Label(self.window, text="Введите радиус участка:")
        self.lbl1.grid(column = 0, row = 2)
        self.entry1 = Entry(self.window, width = 50)
        self.entry1.grid(column = 0, row = 3)
        self.lbl2 = Label(self.window, text="Введите радиус окна:")
        self.lbl2.grid(column = 0, row = 4)
        self.entry2 = Entry(self.window, width = 50)
        self.entry2.grid(column = 0, row = 5)
        self.lbl3 = Label(self.window, text="Введите сигму:")
        self.lbl3.grid(column = 0, row = 6)
        self.entry3 = Entry(self.window, width = 50)
        self.entry3.grid(column = 0, row = 7)
        self.btn = Button(self.window, text="Ввод", width = 50, command = self.submit)
        self.btn.grid(column=0, row=8)
        self.window.mainloop()
    def submit(self):
        t = self.entry.get()
        t1 = self.entry1.get()
        t2 = self.entry2.get()
        t3 = self.entry3.get()
        arr = [t,t1,t2,t3]
        NLM.setup(arr).run()

if __name__ == '__main__':
    win = Window()
