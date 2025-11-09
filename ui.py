import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import subprocess
import sys
import shutil
from pathlib import Path

class FaceMorphUI:
    def __init__(self, root):
        self.root = root
        self.root.title("FaceMorph Live")
        self.root.geometry("600x400")
        
        # Create main frame
        main_frame = ttk.Frame(root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Style
        style = ttk.Style()
        style.configure('Header.TLabel', font=('Helvetica', 16, 'bold'))
        
        # Title
        title = ttk.Label(main_frame, text="FaceMorph Live Control Panel", style='Header.TLabel')
        title.grid(row=0, column=0, columnspan=2, pady=20)
        
        # File paths
        self.source_path = tk.StringVar()
        self.target_path = tk.StringVar()
        self.video_path = tk.StringVar()
        
        # Source image selection
        ttk.Label(main_frame, text="Source Image:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.source_path, width=40).grid(row=1, column=1, padx=5)
        ttk.Button(main_frame, text="Browse", command=lambda: self.browse_file(self.source_path, [("Image files", "*.jpg *.jpeg *.png")])).grid(row=1, column=2)
        
        # Target image selection
        ttk.Label(main_frame, text="Target Image:").grid(row=2, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.target_path, width=40).grid(row=2, column=1, padx=5)
        ttk.Button(main_frame, text="Browse", command=lambda: self.browse_file(self.target_path, [("Image files", "*.jpg *.jpeg *.png")])).grid(row=2, column=2)
        
        # Video selection
        ttk.Label(main_frame, text="Input Video:").grid(row=3, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.video_path, width=40).grid(row=3, column=1, padx=5)
        ttk.Button(main_frame, text="Browse", command=lambda: self.browse_file(self.video_path, [("Video files", "*.mp4 *.avi *.mov")])).grid(row=3, column=2)
        
        # Morphing controls
        controls_frame = ttk.LabelFrame(main_frame, text="Morphing Controls", padding="10")
        controls_frame.grid(row=4, column=0, columnspan=3, pady=20, sticky=(tk.W, tk.E))
        
        # Alpha value slider
        self.alpha = tk.DoubleVar(value=0.5)
        ttk.Label(controls_frame, text="Morphing Intensity (Alpha):").grid(row=0, column=0, sticky=tk.W)
        alpha_slider = ttk.Scale(controls_frame, from_=0.0, to=1.0, orient=tk.HORIZONTAL, 
                               variable=self.alpha, length=200)
        alpha_slider.grid(row=0, column=1, padx=10)
        
        # Buttons frame
        buttons_frame = ttk.Frame(main_frame)
        buttons_frame.grid(row=5, column=0, columnspan=3, pady=20)
        
        # Action buttons
        ttk.Button(buttons_frame, text="Live Webcam Morph", 
                  command=self.run_live_morph).grid(row=0, column=0, padx=10)
        ttk.Button(buttons_frame, text="Static Image Morph", 
                  command=self.run_static_morph).grid(row=0, column=1, padx=10)
        ttk.Button(buttons_frame, text="Video Morph", 
                  command=self.run_video_morph).grid(row=0, column=2, padx=10)

    def browse_file(self, path_var, filetypes):
        filename = filedialog.askopenfilename(
            title="Select a file",
            filetypes=filetypes
        )
        if filename:
            path_var.set(filename)

    def run_live_morph(self):
        try:
            if not self.source_path.get():
                messagebox.showerror("Error", "Please select a source image")
                return
                
            # Create a temporary script to run main with the source path
            with open("temp_live_script.py", "w") as f:
                f.write(f"""
import main
main.main(source_path="{self.source_path.get()}")
""")
            
            # Run the temporary script
            subprocess.Popen([sys.executable, "temp_live_script.py"])
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start live morphing: {str(e)}")

    def run_static_morph(self):
        try:
            if not self.source_path.get() or not self.target_path.get():
                messagebox.showerror("Error", "Please select both source and target images")
                return

            # Create a temporary script to run test with the paths
            with open("temp_static_script.py", "w") as f:
                f.write(f"""
import test
from misc.linux_cam_start import linux
linux()
test.test_static_morph("{self.source_path.get()}", "{self.target_path.get()}")
""")
            
            # Run the temporary script
            subprocess.Popen([sys.executable, "temp_static_script.py"])
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start static morphing: {str(e)}")

    def run_video_morph(self):
        try:
            if not self.source_path.get() or not self.video_path.get():
                messagebox.showerror("Error", "Please select both source image and input video")
                return
            
            # Create a temporary script to run video morphing
            with open("temp_morph_script.py", "w") as f:
                f.write(f"""
import trial
trial.morph_video(
    source_path="{self.source_path.get()}",
    video_path="{self.video_path.get()}",
    output_path="morphed_output.mp4",
    alpha={self.alpha.get()}
)
""")
            
            # Run the temporary script
            subprocess.Popen([sys.executable, "temp_morph_script.py"])
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start video morphing: {str(e)}")

def main():
    root = tk.Tk()
    app = FaceMorphUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
