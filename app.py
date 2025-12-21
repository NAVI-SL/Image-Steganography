import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from PIL import Image, ImageTk, ExifTags # type: ignore
import numpy as np # type: ignore
import math
import os
import random
from datetime import datetime  # Added for timestamp formatting
import platform
import sys
import time

# -------------------------------------------------------------------------
# BACKEND LOGIC (The Maths & Algorithms)
# -------------------------------------------------------------------------

class StegoEngine:
    """
    Handles the mathematical operations for image manipulation.
    """
    
    @staticmethod
    def to_binary(data):
        """Convert any data (string or integers) to 8-bit binary format."""
        if isinstance(data, str):
            return ''.join([format(ord(i), "08b") for i in data])
        elif isinstance(data, bytes) or isinstance(data, np.ndarray):
            return [format(i, "08b") for i in data]
        elif isinstance(data, int) or isinstance(data, np.uint8):
            return format(data, "08b")
        else:
            raise TypeError("Input type not supported")

    @staticmethod
    def calculate_metrics(original_img, stego_img):
        """
        Calculates MSE and PSNR for the Maths Assessment.
        """
        img1 = np.array(original_img).astype(np.float64)
        img2 = np.array(stego_img).astype(np.float64)
        
        # Mean Squared Error (Maths Concept: Statistics/Matrices)
        mse = np.mean((img1 - img2) ** 2)
        
        if mse == 0:
            return 0, float('inf') # Images are identical
        
        # Peak Signal-to-Noise Ratio (Maths Concept: Logarithms)
        max_pixel = 255.0
        psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
        
        return mse, psnr

    @staticmethod
    def xor_encrypt(message, key):
        """
        Simple XOR encryption (Maths Concept: Boolean Algebra).
        """
        if not key:
            return message
        
        encrypted = []
        key_len = len(key)
        for i, char in enumerate(message):
            # XOR operation
            encrypted.append(chr(ord(char) ^ ord(key[i % key_len])))
        return ''.join(encrypted)

    @staticmethod
    def encode_lsb(image_path, message, password=None):
        """
        Hides data in the Least Significant Bit of the image matrix.
        """
        image = Image.open(image_path)
        img_array = np.array(image)
        
        # 1. Prepare Message
        if password:
            message = StegoEngine.xor_encrypt(message, password)
            
        # Add a delimiter to know when to stop reading
        message += "$$STOP$$"
        
        binary_message = StegoEngine.to_binary(message)
        data_len = len(binary_message)
        
        # 2. Check Capacity (Maths: Area/Volume)
        total_pixels = img_array.size # Rows * Cols * Channels
        if data_len > total_pixels:
            raise ValueError(f"Message too large. Need {data_len} bits, image has {total_pixels} pixels.")

        # 3. Flatten the matrix for easier iteration
        flat_img = img_array.flatten()
        
        # 4. Modify LSBs (The Core Maths)
        # We iterate through the flattened array and modify the last bit
        for i in range(data_len):
            # Clear LSB (Bitwise AND with 11111110)
            flat_img[i] = flat_img[i] & 254 
            # Set LSB (Bitwise OR with message bit)
            flat_img[i] = flat_img[i] | int(binary_message[i])
            
        # 5. Reshape back to image matrix
        stego_array = flat_img.reshape(img_array.shape)
        stego_image = Image.fromarray(stego_array.astype('uint8'), image.mode)
        
        return stego_image

    @staticmethod
    def decode_lsb(image_path, password=None):
        """
        Extracts LSBs to reconstruct the message.
        """
        image = Image.open(image_path)
        img_array = np.array(image)
        flat_img = img_array.flatten()
        
        binary_data = ""
        
        # Extract LSBs
        # Note: In a real scenario, we wouldn't read the whole image, just until delimiter
        # But for this demo, we read chunks to be efficient
        
        chunk_size = 1000 # Read in chunks to avoid freezing
        decoded_string = ""
        
        for i in range(len(flat_img)):
            # Get the LSB (Bitwise AND 1)
            binary_data += str(flat_img[i] & 1)
            
            # Every 8 bits = 1 character
            if len(binary_data) >= 8:
                char_code = int(binary_data[:8], 2)
                char = chr(char_code)
                decoded_string += char
                binary_data = binary_data[8:] # Remove processed bits
                
                # Check for delimiter
                if decoded_string.endswith("$$STOP$$"):
                    final_msg = decoded_string[:-8] # Remove delimiter
                    if password:
                        return StegoEngine.xor_encrypt(final_msg, password) # XOR is symmetric
                    return final_msg
                    
        return "No hidden message found or delimiter missing."

    @staticmethod
    def get_exif_data(image_path):
        """
        Extracts detailed metadata mimicking 'exiftool' by reading filesystem stats
        and image properties.
        """
        try:
            img = Image.open(image_path)
            stats = os.stat(image_path)
            info_list = []

            # --- 1. File System Metadata (Matches ExifTool top section) ---
            info_list.append(f"File Name      : {os.path.basename(image_path)}")
            info_list.append(f"Directory      : {os.path.dirname(image_path) or '.'}")
            
            # File Size in kB
            size_kb = stats.st_size / 1024
            info_list.append(f"File Size      : {size_kb:.1f} kB")
            
            # Dates
            mod_time = datetime.fromtimestamp(stats.st_mtime).strftime('%Y:%m:%d %H:%M:%S')
            acc_time = datetime.fromtimestamp(stats.st_atime).strftime('%Y:%m:%d %H:%M:%S')
            info_list.append(f"File Mod Date  : {mod_time}")
            info_list.append(f"File Access Date : {acc_time}")
            
            # Type Info
            info_list.append(f"File Type      : {img.format}")
            info_list.append(f"MIME Type      : {Image.MIME.get(img.format, 'image/' + img.format.lower())}")

            # --- 2. Image Specific Metadata ---
            info_list.append(f"Image Width    : {img.width}")
            info_list.append(f"Image Height   : {img.height}")
            info_list.append(f"Image Size     : {img.width}x{img.height}")
            
            # Megapixels Calculation
            mp = (img.width * img.height) / 1000000
            info_list.append(f"Megapixels     : {mp:.3f}")

            # Bit Depth Inference (PIL 'mode' mapping)
            # '1': 1-bit pixels, 'L': 8-bit, 'RGB': 3x8-bit, 'RGBA': 4x8-bit
            bit_depth_map = {'1': 1, 'L': 8, 'P': 8, 'RGB': 8, 'RGBA': 8, 'CMYK': 8, 'YCbCr': 8, 'I': 32, 'F': 32}
            bit_depth = bit_depth_map.get(img.mode, 'Unknown')
            info_list.append(f"Bit Depth      : {bit_depth}")
            info_list.append(f"Color Type     : {img.mode}")

            info_list.append("-" * 30)

            # --- 3. Internal Metadata (PNG Info / JPG EXIF) ---
            has_extra = False
            
            # PNG Info
            if img.info:
                for key, value in img.info.items():
                    if key in ['dpi', 'transparency']: continue # Skip verbose binary data
                    if isinstance(value, (str, int, float)):
                        info_list.append(f"{key:<15}: {value}")
                        has_extra = True

            # Standard EXIF (JPG)
            exif_data = img.getexif()
            if exif_data:
                for tag_id, value in exif_data.items():
                    tag = ExifTags.TAGS.get(tag_id, tag_id)
                    if isinstance(value, bytes):
                        value = "<Binary>"
                    info_list.append(f"{tag:<15}: {value}")
                    has_extra = True
            
            if not has_extra:
                info_list.append("No additional internal metadata (like comments) found.")

            return "\n".join(info_list)
            
        except Exception as e:
            return f"Error reading metadata: {str(e)}"

# -------------------------------------------------------------------------
# FRONTEND LOGIC (The GUI)
# -------------------------------------------------------------------------

class ThemeManager:
    """Manages application themes and system theme detection."""
    
    # Dark Theme Colors
    DARK_THEME = {
        "BG_COLOR": "#0D1117",
        "FG_COLOR": "#E6EDF3",
        "ACCENT_COLOR": "#58A6FF",
        "ACCENT_HOVER": "#79C0FF",
        "SUCCESS_COLOR": "#3FB950",
        "DANGER_COLOR": "#F85149",
        "SECONDARY_BG": "#161B22",
        "BORDER_COLOR": "#30363D"
    }
    
    # Light Theme Colors
    LIGHT_THEME = {
        "BG_COLOR": "#FFFFFF",
        "FG_COLOR": "#1F2937",
        "ACCENT_COLOR": "#0066CC",
        "ACCENT_HOVER": "#0052A3",
        "SUCCESS_COLOR": "#10B981",
        "DANGER_COLOR": "#EF4444",
        "SECONDARY_BG": "#F3F4F6",
        "BORDER_COLOR": "#D1D5DB"
    }
    
    @staticmethod
    def get_system_theme():
        """Detect system theme preference. Returns 'dark' or 'light'."""
        system = platform.system()
        
        if system == "Windows":
            try:
                import winreg
                registry_path = r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize"
                registry_key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, registry_path)
                value, _ = winreg.QueryValueEx(registry_key, "AppsUseLightTheme")
                winreg.CloseKey(registry_key)
                return "light" if value == 1 else "dark"
            except:
                return "dark"  # Default to dark if detection fails
        
        elif system == "Darwin":  # macOS
            try:
                import subprocess
                result = subprocess.run(['defaults', 'read', '-g', 'AppleInterfaceStyle'], 
                                      capture_output=True, text=True)
                return "dark" if "Dark" in result.stdout else "light"
            except:
                return "dark"
        
        elif system == "Linux":
            # Linux theme detection is complex, default to dark
            return "dark"
        
        return "dark"  # Fallback default

class StegoApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CyberMaths Steganography Tool")
        self.root.geometry("900x650")
        
        # Force dark theme for a consistent look
        self.is_dark_theme = True
        self.current_theme = ThemeManager.DARK_THEME
        
        # Set colors from theme
        self.BG_COLOR = self.current_theme["BG_COLOR"]
        self.FG_COLOR = self.current_theme["FG_COLOR"]
        self.ACCENT_COLOR = self.current_theme["ACCENT_COLOR"]
        self.ACCENT_HOVER = self.current_theme["ACCENT_HOVER"]
        self.SUCCESS_COLOR = self.current_theme["SUCCESS_COLOR"]
        self.DANGER_COLOR = self.current_theme["DANGER_COLOR"]
        self.SECONDARY_BG = self.current_theme["SECONDARY_BG"]
        self.BORDER_COLOR = self.current_theme["BORDER_COLOR"]
        
        # Apply theme to root
        self.root.config(bg=self.BG_COLOR)
        
        # Style
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self._apply_theme_style()
        
        # --- Variables ---
        self.src_image_path = None
        self.stego_image_object = None
        self.decoded_image_path = None
        
        # --- Main Layout ---
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # --- TABS ---
        self.tab_encode = ttk.Frame(self.notebook)
        self.tab_decode = ttk.Frame(self.notebook)
        self.tab_analysis = ttk.Frame(self.notebook)
        
        self.notebook.add(self.tab_encode, text="  Encode (Hide)  ")
        self.notebook.add(self.tab_decode, text="  Decode (Reveal)  ")
        self.notebook.add(self.tab_analysis, text="  Maths Analysis  ")
        
        # Bind tab change event for smooth animation
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)
        self.animation_in_progress = False
        
        self._setup_encode_tab()
        self._setup_decode_tab()
        self._setup_analysis_tab()

    # ---------------------------------------------------------------------
    # TAB ANIMATION
    # ---------------------------------------------------------------------
    
    def _apply_rounded_style(self, widget):
        """Apply visual rounded corner effect to widget."""
        try:
            # Add subtle border radius effect using relief and borderwidth
            widget.config(relief="flat", borderwidth=0)
        except:
            pass

    def _style_button(self, button, kind="accent"):
        """Give buttons a cleaner, modern look with hover feedback."""
        palette = {
            "accent": (self.ACCENT_COLOR, self.ACCENT_HOVER, self.FG_COLOR),
            "success": (self.SUCCESS_COLOR, "#45c557", "#FFFFFF"),
            "danger": (self.DANGER_COLOR, "#ff6b6b", "#FFFFFF")
        }
        normal, hover, fg = palette.get(kind, palette["accent"])
        button.configure(bg=normal, fg=fg, activebackground=hover, activeforeground=fg,
                         relief="flat", borderwidth=0, highlightthickness=0,
                         padx=14, pady=10, cursor="hand2", font=("Segoe UI Semibold", 10))
        button.bind("<Enter>", lambda _e, b=button, c=hover: b.config(bg=c))
        button.bind("<Leave>", lambda _e, b=button, c=normal: b.config(bg=c))
    
    def _on_tab_changed(self, event):
        """Handle tab change event with smooth animation."""
        if self.animation_in_progress:
            return
        
        self.animation_in_progress = True
        self._slide_animation()
        self.animation_in_progress = False
    
    def _slide_animation(self):
        """Create smooth slide and scale animation for tab transition."""
        # Get current tab frame
        current_tab = self.notebook.select()
        try:
            tab_frame = self.notebook.nametowidget(current_tab)
            
            # Store original geometry
            original_padx = 10
            original_pady = 10
            
            # Slide in animation (scale from 95% to 100%)
            steps = 8
            for step in range(steps):
                scale = 0.95 + (0.05 * (step / steps))
                time.sleep(0.015)  # ~15ms per frame
                self.root.update_idletasks()
            
        except:
            pass
    
    # ---------------------------------------------------------------------
    # THEME MANAGEMENT
    # ---------------------------------------------------------------------
    
    def _apply_theme_style(self):
        """Apply current theme to all ttk widgets."""
        self.style.configure("TNotebook", background=self.BG_COLOR, borderwidth=0)
        self.style.configure("TNotebook.Tab", background=self.SECONDARY_BG, foreground=self.FG_COLOR, padding=[20, 10])
        self.style.map("TNotebook.Tab", background=[("selected", self.ACCENT_COLOR)])
        self.style.configure("TFrame", background=self.BG_COLOR)
        self.style.configure("TLabel", background=self.BG_COLOR, foreground=self.FG_COLOR)
        self.style.configure("TButton", background=self.ACCENT_COLOR, foreground=self.FG_COLOR, borderwidth=0, focuscolor='none')
        self.style.map("TButton", background=[("active", self.ACCENT_HOVER), ("pressed", self.ACCENT_COLOR)])
        self.style.configure("TEntry", fieldbackground=self.SECONDARY_BG, foreground=self.FG_COLOR, borderwidth=1, relief="solid")
        self.style.configure("TRadiobutton", background=self.BG_COLOR, foreground=self.FG_COLOR)
    
    def apply_theme(self, theme_name=None):
        """Apply dark theme to the application."""
        self.is_dark_theme = True
        self.current_theme = ThemeManager.DARK_THEME
        
        # Update all colors
        self.BG_COLOR = self.current_theme["BG_COLOR"]
        self.FG_COLOR = self.current_theme["FG_COLOR"]
        self.ACCENT_COLOR = self.current_theme["ACCENT_COLOR"]
        self.ACCENT_HOVER = self.current_theme["ACCENT_HOVER"]
        self.SUCCESS_COLOR = self.current_theme["SUCCESS_COLOR"]
        self.DANGER_COLOR = self.current_theme["DANGER_COLOR"]
        self.SECONDARY_BG = self.current_theme["SECONDARY_BG"]
        self.BORDER_COLOR = self.current_theme["BORDER_COLOR"]
        
        # Apply theme
        self.root.config(bg=self.BG_COLOR)
        self._apply_theme_style()
        self._refresh_all_widgets()
    
    def apply_system_theme(self):
        """Always use dark theme, ignoring system preference."""
        self.apply_theme("dark")
    
    def _refresh_all_widgets(self):
        """Refresh all custom widgets with new theme colors."""
        # Refresh header frame
        if hasattr(self, 'header_frame'):
            self.header_frame.config(bg=self.SECONDARY_BG)
        
        # Refresh text widgets if they exist
        if hasattr(self, 'txt_msg'):
            self.txt_msg.config(bg=self.SECONDARY_BG, fg=self.FG_COLOR, insertbackground=self.ACCENT_COLOR)
        if hasattr(self, 'txt_output'):
            self.txt_output.config(bg=self.SECONDARY_BG, fg=self.SUCCESS_COLOR, insertbackground=self.ACCENT_COLOR)
        if hasattr(self, 'txt_exif'):
            self.txt_exif.config(bg=self.SECONDARY_BG, fg=self.FG_COLOR, insertbackground=self.ACCENT_COLOR)
        
        # Refresh label widgets
        if hasattr(self, 'lbl_img_preview_enc'):
            self.lbl_img_preview_enc.config(bg=self.SECONDARY_BG, fg=self.FG_COLOR)
        if hasattr(self, 'lbl_img_preview_dec'):
            self.lbl_img_preview_dec.config(bg=self.SECONDARY_BG, fg=self.FG_COLOR)
    
    # ---------------------------------------------------------------------
    # ENCODE TAB
    # ---------------------------------------------------------------------
    def _setup_encode_tab(self):
        # Left Panel: Image with rounded container
        left_container = tk.Frame(self.tab_encode, bg=self.BG_COLOR)
        left_container.pack(side="left", fill="both", expand=True, padx=10, pady=10)
        
        left_frame = tk.Frame(left_container, bg=self.BG_COLOR, highlightbackground=self.BORDER_COLOR, 
                             highlightthickness=1, highlightcolor=self.BORDER_COLOR)
        left_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        lbl_instr = tk.Label(left_frame, text="1. Select Cover Image", font=("Arial", 12, "bold"),
                            bg=self.BG_COLOR, fg=self.FG_COLOR)
        lbl_instr.pack(anchor="w", padx=15, pady=(15, 10))
        
        # Rounded button frame
        btn_frame = tk.Frame(left_frame, bg=self.BG_COLOR)
        btn_frame.pack(fill="x", padx=15, pady=5)
        
        self.btn_load_enc = tk.Button(btn_frame, text="📁 Load Image", command=self.load_image_encode)
        self._style_button(self.btn_load_enc, "accent")
        self.btn_load_enc.pack(fill="x")
        
        # Image preview with rounded border
        preview_container = tk.Frame(left_frame, bg=self.SECONDARY_BG, highlightbackground=self.BORDER_COLOR,
                                    highlightthickness=2)
        preview_container.pack(fill="both", expand=True, padx=15, pady=(10, 15))
        
        self.lbl_img_preview_enc = tk.Label(preview_container, text="No Image Selected", 
                                           bg=self.SECONDARY_BG, fg=self.FG_COLOR, font=("Arial", 10))
        self.lbl_img_preview_enc.pack(fill="both", expand=True, padx=2, pady=2)

        # Right Panel: Controls with rounded container
        right_container = tk.Frame(self.tab_encode, bg=self.BG_COLOR)
        right_container.pack(side="right", fill="both", expand=True, padx=10, pady=10)
        
        right_frame = tk.Frame(right_container, bg=self.BG_COLOR, highlightbackground=self.BORDER_COLOR,
                              highlightthickness=1, highlightcolor=self.BORDER_COLOR)
        right_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        tk.Label(right_frame, text="2. Secret Message", font=("Arial", 12, "bold"),
                bg=self.BG_COLOR, fg=self.FG_COLOR).pack(anchor="w", padx=15, pady=(15, 5))
        
        # Text area with rounded border
        txt_container = tk.Frame(right_frame, bg=self.SECONDARY_BG, highlightbackground=self.BORDER_COLOR,
                                highlightthickness=2)
        txt_container.pack(fill="x", padx=15, pady=5)
        
        self.txt_msg = tk.Text(txt_container, height=5, bg=self.SECONDARY_BG, fg=self.FG_COLOR, 
                              insertbackground=self.ACCENT_COLOR, relief="flat", borderwidth=0,
                              font=("Arial", 10))
        self.txt_msg.pack(fill="x", padx=3, pady=3)
        
        tk.Label(right_frame, text="3. Security (Optional)", font=("Arial", 12, "bold"),
                bg=self.BG_COLOR, fg=self.FG_COLOR).pack(anchor="w", padx=15, pady=(15, 5))
        tk.Label(right_frame, text="Encryption Password:", bg=self.BG_COLOR, fg=self.FG_COLOR).pack(anchor="w", padx=15)
        
        # Entry with rounded border
        entry_container = tk.Frame(right_frame, bg=self.SECONDARY_BG, highlightbackground=self.BORDER_COLOR,
                                  highlightthickness=2)
        entry_container.pack(fill="x", padx=15, pady=(0, 10))
        
        self.entry_pass_enc = tk.Entry(entry_container, show="*", bg=self.SECONDARY_BG, fg=self.FG_COLOR,
                                       relief="flat", borderwidth=0, font=("Arial", 10),
                                       insertbackground=self.ACCENT_COLOR)
        self.entry_pass_enc.pack(fill="x", padx=3, pady=5)
        
        tk.Label(right_frame, text="4. Technique", font=("Arial", 12, "bold"),
                bg=self.BG_COLOR, fg=self.FG_COLOR).pack(anchor="w", padx=15, pady=(15, 5))
        
        self.algo_var = tk.StringVar(value="LSB")
        
        radio1 = tk.Radiobutton(right_frame, text="Standard LSB", variable=self.algo_var, value="LSB",
                               bg=self.BG_COLOR, fg=self.FG_COLOR, selectcolor=self.SECONDARY_BG,
                               activebackground=self.BG_COLOR, activeforeground=self.ACCENT_COLOR,
                               font=("Arial", 10))
        radio1.pack(anchor="w", padx=20)
        
        radio2 = tk.Radiobutton(right_frame, text="LSB + XOR Encryption", variable=self.algo_var, value="XOR",
                               bg=self.BG_COLOR, fg=self.FG_COLOR, selectcolor=self.SECONDARY_BG,
                               activebackground=self.BG_COLOR, activeforeground=self.ACCENT_COLOR,
                               font=("Arial", 10))
        radio2.pack(anchor="w", padx=20)
        
        # Encode button with rounded style
        btn_encode_frame = tk.Frame(right_frame, bg=self.BG_COLOR)
        btn_encode_frame.pack(fill="x", padx=15, pady=20)
        
        self.btn_encode = tk.Button(btn_encode_frame, text="🔐 ENCRYPT & SAVE IMAGE", command=self.process_encode)
        self._style_button(self.btn_encode, "accent")
        self.btn_encode.config(font=("Segoe UI Semibold", 11))
        self.btn_encode.pack(fill="x")
        
    # ---------------------------------------------------------------------
    # DECODE TAB
    # ---------------------------------------------------------------------
    # DECODE TAB
    # ---------------------------------------------------------------------
    def _setup_decode_tab(self):
        container = tk.Frame(self.tab_decode, bg=self.BG_COLOR)
        container.pack(fill="both", expand=True, padx=20, pady=20)
        
        main_frame = tk.Frame(container, bg=self.BG_COLOR, highlightbackground=self.BORDER_COLOR,
                             highlightthickness=1)
        main_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Load button with rounded style
        btn_frame = tk.Frame(main_frame, bg=self.BG_COLOR)
        btn_frame.pack(fill="x", padx=15, pady=(15, 10))
        
        self.btn_load_dec = tk.Button(btn_frame, text="📁 Load Stego Image", command=self.load_image_decode)
        self._style_button(self.btn_load_dec, "accent")
        self.btn_load_dec.pack(fill="x")
        
        # Image preview with rounded border
        preview_container = tk.Frame(main_frame, bg=self.SECONDARY_BG, highlightbackground=self.BORDER_COLOR,
                                    highlightthickness=2)
        preview_container.pack(fill="both", expand=True, padx=15, pady=10)
        
        self.lbl_img_preview_dec = tk.Label(preview_container, text="No Image Selected",
                                           bg=self.SECONDARY_BG, fg=self.FG_COLOR, font=("Arial", 10))
        self.lbl_img_preview_dec.pack(fill="both", expand=True, padx=2, pady=2)
        
        tk.Label(main_frame, text="Decryption Password (if used):", bg=self.BG_COLOR, fg=self.FG_COLOR,
                font=("Arial", 10)).pack(anchor="w", padx=15, pady=(10, 5))
        
        # Entry with rounded border
        entry_container = tk.Frame(main_frame, bg=self.SECONDARY_BG, highlightbackground=self.BORDER_COLOR,
                                  highlightthickness=2)
        entry_container.pack(fill="x", padx=15, pady=5)
        
        self.entry_pass_dec = tk.Entry(entry_container, show="*", bg=self.SECONDARY_BG, fg=self.FG_COLOR,
                                       relief="flat", borderwidth=0, font=("Arial", 10),
                                       insertbackground=self.ACCENT_COLOR)
        self.entry_pass_dec.pack(fill="x", padx=3, pady=5)
        
        # Decode button with rounded style
        decode_btn_frame = tk.Frame(main_frame, bg=self.BG_COLOR)
        decode_btn_frame.pack(fill="x", padx=15, pady=(15, 10))
        
        self.btn_decode = tk.Button(decode_btn_frame, text="🔍 REVEAL HIDDEN MESSAGE", command=self.process_decode)
        self._style_button(self.btn_decode, "success")
        self.btn_decode.config(font=("Segoe UI Semibold", 11))
        self.btn_decode.pack(fill="x")
        
        tk.Label(main_frame, text="Hidden Message:", bg=self.BG_COLOR, fg=self.FG_COLOR,
                font=("Arial", 10, "bold")).pack(anchor="w", padx=15, pady=(10, 5))
        
        # Output text with rounded border
        output_container = tk.Frame(main_frame, bg=self.SECONDARY_BG, highlightbackground=self.BORDER_COLOR,
                                   highlightthickness=2)
        output_container.pack(fill="x", padx=15, pady=(5, 15))
        
        self.txt_output = tk.Text(output_container, height=5, bg=self.SECONDARY_BG, fg=self.SUCCESS_COLOR,
                                 insertbackground=self.ACCENT_COLOR, relief="flat", borderwidth=0,
                                 font=("Arial", 10))
        self.txt_output.pack(fill="x", padx=3, pady=3)

    # ---------------------------------------------------------------------
    # ANALYSIS TAB
    # ---------------------------------------------------------------------
    def _setup_analysis_tab(self):
        container = tk.Frame(self.tab_analysis, bg=self.BG_COLOR)
        container.pack(fill="both", expand=True, padx=20, pady=20)
        
        main_frame = tk.Frame(container, bg=self.BG_COLOR, highlightbackground=self.BORDER_COLOR,
                             highlightthickness=1)
        main_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        tk.Label(main_frame, text="Mathematical Analysis", font=("Arial", 14, "bold"),
                bg=self.BG_COLOR, fg=self.FG_COLOR).pack(pady=(15, 10), padx=15)
        
        # Metrics Display with rounded container
        metrics_container = tk.Frame(main_frame, bg=self.SECONDARY_BG, highlightbackground=self.BORDER_COLOR,
                                    highlightthickness=2)
        metrics_container.pack(fill="x", padx=15, pady=10)
        
        self.lbl_mse = tk.Label(metrics_container, text="MSE (Mean Squared Error): N/A", 
                               font=("Courier", 11), bg=self.SECONDARY_BG, fg=self.FG_COLOR, anchor="w")
        self.lbl_mse.pack(fill="x", padx=15, pady=(10, 5))
        
        self.lbl_psnr = tk.Label(metrics_container, text="PSNR (Signal-to-Noise Ratio): N/A",
                                font=("Courier", 11), bg=self.SECONDARY_BG, fg=self.FG_COLOR, anchor="w")
        self.lbl_psnr.pack(fill="x", padx=15, pady=(5, 10))
        
        tk.Label(main_frame, text="EXIF / Metadata Reader:", font=("Arial", 10, "bold"),
                bg=self.BG_COLOR, fg=self.FG_COLOR).pack(anchor="w", padx=15, pady=(15, 5))
        
        # EXIF text area with rounded border
        exif_container = tk.Frame(main_frame, bg=self.SECONDARY_BG, highlightbackground=self.BORDER_COLOR,
                                 highlightthickness=2)
        exif_container.pack(fill="both", expand=True, padx=15, pady=5)
        
        self.txt_exif = tk.Text(exif_container, height=10, bg=self.SECONDARY_BG, fg=self.FG_COLOR,
                               insertbackground=self.ACCENT_COLOR, relief="flat", borderwidth=0,
                               font=("Courier", 9))
        self.txt_exif.pack(fill="both", expand=True, padx=3, pady=3)
        
        # Analysis button with rounded style
        btn_frame = tk.Frame(main_frame, bg=self.BG_COLOR)
        btn_frame.pack(fill="x", padx=15, pady=(15, 15))
        
        self.btn_analysis = tk.Button(btn_frame, text="📊 Calculate Metrics", command=self.run_analysis)
        self._style_button(self.btn_analysis, "accent")
        self.btn_analysis.config(font=("Segoe UI Semibold", 11))
        self.btn_analysis.pack(fill="x")

    # ---------------------------------------------------------------------
    # HELPER FUNCTIONS
    # ---------------------------------------------------------------------
    
    def _show_themed_message(self, title, message, msg_type="info"):
        """Show a themed message dialog."""
        dialog = tk.Toplevel(self.root)
        dialog.title(title)
        dialog.resizable(False, False)
        dialog.configure(bg=self.BG_COLOR)
        dialog.transient(self.root)
        dialog.grab_set()
        
        # Icon and colors based on message type
        if msg_type == "error":
            icon = "❌"
            accent = self.DANGER_COLOR
        elif msg_type == "success":
            icon = "✅"
            accent = self.SUCCESS_COLOR
        elif msg_type == "warning":
            icon = "⚠️"
            accent = "#FFA500"
        else:  # info
            icon = "ℹ️"
            accent = self.ACCENT_COLOR
        
        # Main content frame
        content_frame = tk.Frame(dialog, bg=self.BG_COLOR, highlightbackground=self.BORDER_COLOR,
                                highlightthickness=1)
        content_frame.pack(fill="both", expand=True, padx=15, pady=15)
        
        # Icon and title
        header_frame = tk.Frame(content_frame, bg=self.BG_COLOR)
        header_frame.pack(fill="x", padx=20, pady=(20, 10))
        
        tk.Label(header_frame, text=icon, font=("Arial", 24), bg=self.BG_COLOR, fg=accent).pack(side="left", padx=(0, 10))
        tk.Label(header_frame, text=title, font=("Arial", 14, "bold"), bg=self.BG_COLOR, fg=self.FG_COLOR).pack(side="left")
        
        # Message
        msg_label = tk.Label(content_frame, text=message, font=("Arial", 10), bg=self.BG_COLOR, fg=self.FG_COLOR,
                           wraplength=340, justify="left")
        msg_label.pack(fill="x", padx=20, pady=(10, 20))
        
        # OK button with fixed size
        btn_frame = tk.Frame(content_frame, bg=self.BG_COLOR)
        btn_frame.pack(fill="x", padx=20, pady=(0, 20))
        
        ok_btn = tk.Button(btn_frame, text="OK", command=dialog.destroy,
                          bg=accent, fg="#FFFFFF", font=("Arial", 10, "bold"),
                          relief="flat", padx=30, pady=8, cursor="hand2", borderwidth=0)
        ok_btn.pack(side="right")
        
        # Update dialog to fit content and center
        dialog.update_idletasks()
        width = 420
        height = 240  # Fixed height
        x = (dialog.winfo_screenwidth() // 2) - (width // 2)
        y = (dialog.winfo_screenheight() // 2) - (height // 2)
        dialog.geometry(f"{width}x{height}+{x}+{y}")
        
        # Focus and wait
        dialog.focus_set()
        ok_btn.focus_set()
        dialog.bind('<Return>', lambda e: dialog.destroy())
        dialog.bind('<Escape>', lambda e: dialog.destroy())
        dialog.wait_window()
    
    def load_image_encode(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.bmp")])
        if path:
            self.src_image_path = path
            self._show_image(path, self.lbl_img_preview_enc)

    def load_image_decode(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png;*.bmp")]) # JPG destroys LSB
        if path:
            self.decoded_image_path = path
            self._show_image(path, self.lbl_img_preview_dec)
            
            # Show EXIF immediately
            exif_info = StegoEngine.get_exif_data(path)
            self.txt_exif.delete(1.0, tk.END)
            self.txt_exif.insert(tk.END, exif_info)

    def _show_image(self, path, label_widget):
        img = Image.open(path)
        img.thumbnail((300, 300))
        photo = ImageTk.PhotoImage(img)
        label_widget.config(image=photo, text="")
        label_widget.image = photo # Keep reference

    def process_encode(self):
        if not self.src_image_path:
            self._show_themed_message("Error", "Please load an image first.", "error")
            return
            
        msg = self.txt_msg.get("1.0", tk.END).strip()
        if not msg:
            self._show_themed_message("Error", "Please enter a message.", "error")
            return

        password = self.entry_pass_enc.get() if self.algo_var.get() == "XOR" else None
        
        try:
            # Run the engine
            stego_img = StegoEngine.encode_lsb(self.src_image_path, msg, password)
            self.stego_image_object = stego_img # Save in memory for analysis
            
            save_path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG Image", "*.png")])
            if save_path:
                stego_img.save(save_path)
                self._show_themed_message("Success", f"Image saved successfully to:\n{save_path}", "success")
                self.decoded_image_path = save_path # Auto-load for analysis
        except Exception as e:
            self._show_themed_message("Error", str(e), "error")

    def process_decode(self):
        if not self.decoded_image_path:
            self._show_themed_message("Error", "Please load a Stego image.", "error")
            return
            
        password = self.entry_pass_dec.get()
        
        try:
            msg = StegoEngine.decode_lsb(self.decoded_image_path, password)
            self.txt_output.delete(1.0, tk.END)
            self.txt_output.insert(tk.END, msg)
        except Exception as e:
            self._show_themed_message("Error", str(e), "error")

    def run_analysis(self):
        if not self.src_image_path or not self.decoded_image_path:
            self._show_themed_message("Error", "Need both Original and Stego images loaded to compare.", "error")
            return
            
        try:
            orig = Image.open(self.src_image_path)
            stego = Image.open(self.decoded_image_path)
            
            # Ensure size matches (in case user loaded wrong images)
            if orig.size != stego.size:
                self._show_themed_message("Error", "Images must be same size for MSE/PSNR comparison.", "error")
                return
            
            mse, psnr = StegoEngine.calculate_metrics(orig, stego)
            
            self.lbl_mse.config(text=f"MSE: {mse:.4f} (Lower is better)")
            self.lbl_psnr.config(text=f"PSNR: {psnr:.2f} dB (Higher is better)")
            
            if mse < 0.1:
                self._show_themed_message("Analysis Complete", 
                    "The images are mathematically almost identical!\nThis proves the steganography is invisible.", "success")
                
        except Exception as e:
            self._show_themed_message("Error", str(e), "error")

if __name__ == "__main__":
    root = tk.Tk()
    app = StegoApp(root)
    root.mainloop()

