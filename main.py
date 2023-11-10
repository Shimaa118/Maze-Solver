import tkinter as tk
from tkinter import filedialog
from mazesolver import MazeSolver  

if __name__ == '__main__':
    root = tk.Tk()
    app = MazeSolver(root)
    app.maze_canvas.bind("<Button-1>", app.on_canvas_click)
    app.maze_canvas.bind("<Button-3>", app.on_canvas_click)


    root.mainloop()
