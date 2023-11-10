import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
from collections import deque
import heapq
import tkinter.messagebox as messagebox 


class MazeSolver:
    def __init__(self, root):
        self.root = root
        self.root.title("Maze Solver")
        self.maze_data = None
        self.root.configure(bg="#FFF8DC")

        # Create a frame to hold labels, buttons, and the select field on the left
        left_frame = tk.Frame(root, bg="#FFF8DC")
        left_frame.pack(side="left", padx=10, pady=10)

        # Select input field
        self.select_label = tk.Label(left_frame, text="Select an Image:", font=("Helvetica", 11),bg="#FFF8DC")
        self.select_label.pack(padx=10, pady=10)

        self.select_entry = tk.Entry(left_frame, width=30)
        self.select_entry.pack(padx=10, pady=10)
        # Select button
        self.select_button = tk.Button(left_frame, text="Select Image", command=self.select_image, width=15,bg='#1E90FF' , fg='white', font=("Helvetica", 11))
        self.select_button.pack(padx=10, pady=0)

        # Start and goal point selection labels
        self.start_label = tk.Label(left_frame, text="Select Start Point (Left Click):", font=("Helvetica", 11) ,bg="#FFF8DC")
        self.start_label.pack(padx=10, pady=0)
        self.start_selected = False

        self.goal_label = tk.Label(left_frame, text="Select Goal Point (Right Click):", font=("Helvetica", 11),bg="#FFF8DC")
        self.goal_label.pack(padx=10, pady=10)
        self.goal_selected = False

        # Algorithm selection dropdown
        self.algorithm_label = tk.Label(left_frame, text="Select Algorithm:", font=("Helvetica", 11),bg="#FFF8DC")
        self.algorithm_label.pack(padx=10, pady=10)
        self.selected_algorithm = tk.StringVar()
        algorithm_frame = tk.Frame(left_frame ,bg="#FFF8DC")
        algorithm_frame.pack(padx=10, pady=10)
        self.selected_algorithm = tk.StringVar()
        self.selected_algorithm.set("BFS")  # Default algorithm
        algorithms = ["BFS", "DFS", "A*"]
        for algorithm in algorithms:
            algorithm_radio = tk.Radiobutton(algorithm_frame, text=algorithm, variable=self.selected_algorithm, value=algorithm ,bg="#FFF8DC")
            algorithm_radio.pack(anchor="w")

        # Solve button
        self.solve_button = tk.Button(left_frame, text="Solve", command=self.solve_maze, font=("Helvetica", 11))
        self.solve_button.pack(padx=10, pady=10)

        # Reset button
        self.reset_button = tk.Button(left_frame, text="Reset", command=self.reset_selection, font=("Helvetica", 11))
        self.reset_button.pack(padx=10, pady=10)

        right_frame = tk.Frame(root)  
        right_frame.pack(side="right", padx=10, pady=10)

        self.maze_canvas = tk.Canvas(right_frame, bg="#FFF8DC")  
        self.maze_canvas.pack()  

        self.maze_image = None
        self.photo = None

        self.start_point = None
        self.goal_point = None

    def select_image(self):
        initial_dir = "/path/to/initial/directory"  

        file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.gif")],
        initialdir=initial_dir)

        if file_path:
            self.load_maze_image(file_path)
            self.select_entry.delete(0, tk.END)  
            self.select_entry.insert(0, file_path) 

    def load_maze_image(self, maze_filename):
        if maze_filename:
            self.maze_image = cv2.imread(maze_filename)
            self.canvas_width = self.maze_image.shape[1]  # Get image width
            self.canvas_height = self.maze_image.shape[0]  # Get image height
            self.maze_canvas.config(width=self.canvas_width, height=self.canvas_height)  # Update canvas dimensions
            self.display_maze()  
            self.maze_data = cv2.cvtColor(self.maze_image, cv2.COLOR_BGR2GRAY)
            _, self.maze_data = cv2.threshold(self.maze_data, 127, 255, cv2.THRESH_BINARY)


    def display_maze(self):
        maze_rgb = cv2.cvtColor(self.maze_image, cv2.COLOR_BGR2RGB)
        self.photo = ImageTk.PhotoImage(image=Image.fromarray(maze_rgb))
        self.maze_canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

    def solve_maze(self):
        if not self.start_point or not self.goal_point:
            messagebox.showinfo("Warning", "Please select both start and goal points on the image.")
            return

        selected_algorithm = self.selected_algorithm.get()

        if selected_algorithm == "BFS":
            solution = self.bfs()
        elif selected_algorithm == "DFS":
            solution = self.dfs()
        elif selected_algorithm == "A*":
            solution = self.A_star_search()


        if solution:
            self.display_solution(solution)

    def bfs(self):
        queue = deque([(self.start_point, [])])
        visited = set()

        while queue:
            current, path = queue.popleft()
            x, y = current

            if current == self.goal_point:
                return path + [current]

            if current in visited:
                continue

            visited.add(current)

            neighbors = self.get_neighbors(x, y)
            for neighbor in neighbors:
                queue.append((neighbor, path + [current]))

        return None

    def dfs(self):
        stack = [(self.start_point, [])]
        visited = set()

        while stack:
            current, path = stack.pop()
            x, y = current

            if current == self.goal_point:
                return path + [current]

            if current in visited:
                continue

            visited.add(current)

            neighbors = self.get_neighbors(x, y)
            for neighbor in neighbors:
                stack.append((neighbor, path + [current]))

        return None

    def reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path

    def A_star_search(self):
        start = self.start_point
        goal = self.goal_point

        def A_star(node):
            x, y = node
            return ((x - goal[0]) ** 2 + (y - goal[1]) ** 2) ** 0.5

        open_set = [(A_star(start), start)]
        came_from = {}

        g_score = [[float('inf')] * self.maze_image.shape[1] for _ in range(self.maze_image.shape[0])]
        g_score[start[1]][start[0]] = 0

        while open_set:
            _, current = heapq.heappop(open_set)

            if current == goal:
                path = self.reconstruct_path(came_from, current)
                return path

            for neighbor in self.get_neighbors(*current):
                x, y = neighbor
                tentative_g_score = g_score[current[1]][current[0]] + 1

                if tentative_g_score < g_score[y][x]:
                    came_from[neighbor] = current
                    g_score[y][x] = tentative_g_score
                    f_score = tentative_g_score + A_star(neighbor)
                    heapq.heappush(open_set, (f_score, neighbor))

        return None



    def get_neighbors(self, x, y):
        neighbors = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]
        return [(nx, ny) for nx, ny in neighbors if 0 <= nx < self.maze_image.shape[1] and 0 <= ny < self.maze_image.shape[0] and self.maze_data[ny, nx] == 255]

    def reset_selection(self):
        self.select_entry.delete(0, tk.END)
        self.maze_canvas.delete("all")
        self.start_point = None
        self.goal_point = None
        self.start_selected = False  
        self.goal_selected = False
        # Re-bind canvas click events
        self.maze_canvas.bind("<Button-1>", self.on_canvas_click)
        self.maze_canvas.bind("<Button-3>", self.on_canvas_click)

    def display_solution(self, solution):
        maze_solution = self.maze_image.copy()
        for x, y in solution:
            cv2.circle(maze_solution, (x, y), 2, (0, 255, 0), -1)
        cv2.imshow('Solved Maze', maze_solution)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def on_canvas_click(self, event):
        if self.start_selected and self.goal_selected:
            return

        x, y = event.x, event.y

        if not self.start_selected:
            self.start_point = (x, y)
            self.start_selected = True
            self.maze_image[y-2:y+2, x-2:x+2] = [0, 0, 255]
            self.start_label.config(text=f"Start Point: ({x}, {y})")
        elif not self.goal_selected:
            self.goal_point = (x, y)
            self.goal_selected = True
            self.maze_image[y-2:y+2, x-2:x+2] = [0, 255, 0]
            self.goal_label.config(text=f"Goal Point: ({x}, {y})")

        self.display_maze()

