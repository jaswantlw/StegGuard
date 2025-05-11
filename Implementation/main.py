import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import joblib
import os
import cv2
import numpy as np
import threading

class StegGuardApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("StegGuard - Steganography Detection")
        self.geometry("700x600")
        self.configure(bg="#1a1a1a")
        self.resizable(False, False)
        
        self.header_text = "StegGuard - AI - 6H - 4641,4478,4473"
        
        self.model = None
        self.class_names = ["Clean", "Stego"]
        self.current_image = None
        
        self.load_model()
        
        self.configure_styles()
        self.create_widgets()
        
    def configure_styles(self):
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure('Header.TLabel', 
                           background='#2E3B4E', 
                           foreground='white', 
                           font=('Helvetica', 10, 'bold'),
                           anchor=tk.CENTER)
        self.style.configure('TFrame', background='#1a1a1a')
        self.style.configure('TLabel', background='#1a1a1a', foreground='white', font=('Helvetica', 12))
        self.style.configure('TButton', font=('Helvetica', 12), padding=6)
        self.style.map('TButton',
            background=[('active', '#45a049'), ('!disabled', '#4CAF50')],
            foreground=[('!disabled', 'white')]
        )
        
    def load_model(self):
        try:
            model_path = os.path.join('trained_model', 'StegGuard_Random_Forest_Classifier.pkl')
            model_data = joblib.load(model_path)
            self.model = model_data['model']
            self.class_names = model_data.get('class_names', ["Clean", "Stego"])
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
            self.destroy()
            
    def create_widgets(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        
        header_frame = ttk.Frame(self, style='Header.TFrame')
        header_frame.grid(row=0, column=0, sticky='nsew', padx=10, pady=5)
        ttk.Label(header_frame, text=self.header_text, style='Header.TLabel').pack(pady=5)
        
        main_frame = ttk.Frame(self)
        main_frame.grid(row=1, column=0, sticky='nsew', padx=20, pady=10)
        
        self.image_frame = ttk.Frame(main_frame)
        self.image_frame.pack(pady=10)
        self.image_label = ttk.Label(self.image_frame)
        self.image_label.pack()
        
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(pady=15)
        
        self.upload_btn = ttk.Button(btn_frame, text="Upload Image", command=self.upload_image)
        self.upload_btn.pack(side=tk.LEFT, padx=10)
        
        self.detect_btn = ttk.Button(btn_frame, 
                                   text="Detect Steganography", 
                                   command=self.detect_steg, 
                                   state=tk.DISABLED)
        self.detect_btn.pack(side=tk.LEFT, padx=10)
        
        self.result_label = ttk.Label(main_frame, 
                                     text="", 
                                     font=('Helvetica', 12, 'bold'),
                                     wraplength=400)
        self.result_label.pack(pady=10)
        
        ttk.Button(main_frame, text="Help", command=self.show_help).pack(pady=10)
        
    def upload_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.png *.jpg *.jpeg")]
        )
        if file_path:
            try:
                self.current_image = file_path
                self.show_image_preview(file_path)
                self.detect_btn.config(state=tk.NORMAL)
                self.result_label.config(text="")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")
                
    def show_image_preview(self, path):
        img = Image.open(path)
        img.thumbnail((250, 250))
        photo = ImageTk.PhotoImage(img)
        self.image_label.config(image=photo)
        self.image_label.image = photo
        
    def extract_features(self, image_path):
        try:
            image = cv2.imread(image_path)
            if image is None:
                return None
                
            features = []
            for chan in cv2.split(image):
                hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
                cv2.normalize(hist, hist)
                features.extend(hist.flatten())
                
            return np.array(features).reshape(1, -1)
        except Exception as e:
            messagebox.showerror("Error", f"Feature extraction failed: {str(e)}")
            return None
            
    def detect_steg(self):
        if not self.current_image or not self.model:
            return
            
        self.detect_btn.config(state=tk.DISABLED)
        self.result_label.config(text="Analyzing...", foreground="white")
        
        threading.Thread(target=self._run_detection, daemon=True).start()
        
    def _run_detection(self):
        try:
            features = self.extract_features(self.current_image)
            if features is None:
                raise ValueError("Feature extraction failed")
                
            prediction = self.model.predict(features)[0]
            confidence = self.model.predict_proba(features)[0].max()
            
            self.after(0, self._update_result, prediction, confidence)
            
        except Exception as e:
            self.after(0, self._show_error, str(e))
            
    def _update_result(self, prediction, confidence):
        result_text = f"{self.class_names[prediction]} Detected"
        color = "#ff4444" if prediction == 1 else "#4CAF50"
        self.result_label.config(
            text=f"{result_text}\nConfidence: {confidence:.2%}",
            foreground=color
        )
        self.detect_btn.config(state=tk.NORMAL)
        
    def _show_error(self, message):
        messagebox.showerror("Error", message)
        self.detect_btn.config(state=tk.NORMAL)
        self.result_label.config(text="Error occurred", foreground="red")
                
    def show_help(self):
        help_text = """StegGuard - Steganography Detection Tool
        
1. Click 'Upload Image' to select an image file
2. Click 'Detect Steganography' to analyze
3. Results will show detection status

Supported formats: PNG, JPG, JPEG"""
        messagebox.showinfo("Help", help_text)

if __name__ == "__main__":
    app = StegGuardApp()
    app.mainloop()