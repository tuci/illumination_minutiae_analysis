import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
from PIL import Image, ImageTk
import numpy as np

class ImageAlignmentTool:
    def __init__(self, root):
        self.root = root
        self.root.title("Point-Based Multi-Image Alignment Tool")
        self.root.geometry("1400x850")

        # State Variables
        self.ref_path = None
        self.ref_img_orig = None
        self.target_paths = []
        self.target_images_orig = {}  # path -> original image
        self.active_target_path = None
        
        # Points and Transforms per target: path -> list of (ref_pt, target_pt)
        self.point_pairs = {}
        # Cumulative matrices per target: path -> 3x3 or 2x3 matrix
        self.cumulative_matrices = {}

        self.ref_zoom = 1.0
        self.target_zoom = 1.0

        self.setup_ui()

    def setup_ui(self):
        # --- Toolbar ---
        toolbar = ttk.Frame(self.root, padding=5)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(toolbar, text="Load Reference", command=self.load_reference).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Load Targets (Batch)", command=self.load_targets).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Undo Last Pair", command=self.undo_pair).pack(side=tk.LEFT, padx=5)
        ttk.Button(toolbar, text="Clear Points", command=self.clear_points).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Preview Alignment", command=self.preview_alignment).pack(side=tk.LEFT, padx=10)
        ttk.Button(toolbar, text="Save Active", command=self.save_active).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Save All Aligned", command=self.save_all).pack(side=tk.LEFT, padx=2)

        # --- Target Manager Bar ---
        self.target_bar = ttk.Frame(self.root, padding=5)
        self.target_bar.pack(side=tk.TOP, fill=tk.X)
        self.target_label_var = tk.StringVar(value="Active Target: None")
        ttk.Label(self.target_bar, textvariable=self.target_label_var, font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)

        self.target_combobox = ttk.Combobox(self.target_bar, state="readonly", width=40)
        self.target_combobox.pack(side=tk.LEFT, padx=5)
        self.target_combobox.bind("<<ComboboxSelected>>", self.on_target_selected)

        # --- Canvases Frame (Side-by-Side) ---
        canvas_frame = ttk.Frame(self.root, padding=5)
        canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Left Canvas (Reference)
        ref_frame = ttk.LabelFrame(canvas_frame, text="Reference Image (Click points here first)")
        ref_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        ref_zoom_bar = ttk.Frame(ref_frame)
        ref_zoom_bar.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(
            ref_zoom_bar, text="Zoom +",
            command=lambda: self.change_zoom("ref", 1.2)
        ).pack(side=tk.LEFT, padx=3)

        ttk.Button(
            ref_zoom_bar, text="Zoom −",
            command=lambda: self.change_zoom("ref", 1 / 1.2)
        ).pack(side=tk.LEFT, padx=3)

        ttk.Button(
            ref_zoom_bar, text="Fit",
            command=lambda: self.reset_zoom("ref")
        ).pack(side=tk.LEFT, padx=3)
        self.ref_canvas = tk.Canvas(ref_frame, bg="gray")
        self.ref_canvas.pack(fill=tk.BOTH, expand=True)
        self.ref_canvas.bind("<Button-1>", self.on_ref_click)
        self.ref_canvas.bind("<MouseWheel>", lambda event: self.zoom_image(event, "ref"))
        self.ref_canvas.bind("<ButtonPress-3>", self.start_pan)
        self.ref_canvas.bind("<B3-Motion>", self.pan_image)

        # Right Canvas (Target)
        target_frame = ttk.LabelFrame(canvas_frame, text="Active Target Image (Click matching point here second)")
        target_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        target_zoom_bar = ttk.Frame(target_frame)
        target_zoom_bar.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(
            target_zoom_bar, text="Zoom +",
            command=lambda: self.change_zoom("target", 1.2)
        ).pack(side=tk.LEFT, padx=3)

        ttk.Button(
            target_zoom_bar, text="Zoom −",
            command=lambda: self.change_zoom("target", 1 / 1.2)
        ).pack(side=tk.LEFT, padx=3)

        ttk.Button(
            target_zoom_bar, text="Fit",
            command=lambda: self.reset_zoom("target")
        ).pack(side=tk.LEFT, padx=3)
        self.target_canvas = tk.Canvas(target_frame, bg="gray")
        self.target_canvas.pack(fill=tk.BOTH, expand=True)
        self.target_canvas.bind("<Button-1>", self.on_target_click)
        self.target_canvas.bind("<MouseWheel>", lambda event: self.zoom_image(event, "target"))
        self.target_canvas.bind("<ButtonPress-3>", self.start_pan)
        self.target_canvas.bind("<B3-Motion>", self.pan_image)

        # --- Bottom Panel (Status & Table) ---
        bottom_frame = ttk.Frame(self.root, padding=5)
        bottom_frame.pack(side=tk.BOTTOM, fill=tk.X)

        self.status_var = tk.StringVar(value="Status: Please load a Reference image and Target images to begin.")
        ttk.Label(bottom_frame, textvariable=self.status_var, font=("Arial", 9, "italic")).pack(side=tk.TOP, anchor=tk.W)

        # Pairs Table
        self.table_text = tk.Text(bottom_frame, height=4, state=tk.DISABLED, bg="#f4f4f4")
        self.table_text.pack(side=tk.TOP, fill=tk.X, pady=2)

        # Temporary state for tracking alternating clicks
        self.pending_ref_pt = None

    def load_reference(self):
        path = filedialog.askopenfilename(title="Select Reference Image", filetypes=[("Image Files", "*.jpg *.png *.jpeg *.bmp *.tif")])
        if path:
            self.ref_path = path
            self.ref_img_orig = cv2.imread(path)
            self.redraw_ref()
            self.status_var.set(f"Loaded reference: {os.path.basename(path)}")

    def load_targets(self):
        paths = filedialog.askopenfilenames(title="Select Target Images", filetypes=[("Image Files", "*.jpg *.png *.jpeg *.bmp *.tif")])
        if paths:
            for p in paths:
                if p not in self.target_paths:
                    self.target_paths.append(p)
                    self.target_images_orig[p] = cv2.imread(p)
                    self.point_pairs[p] = []
                    self.cumulative_matrices[p] = np.eye(3, dtype=np.float32) # Identity matrix
            
            self.target_combobox['values'] = [os.path.basename(p) for p in self.target_paths]
            if not self.active_target_path and self.target_paths:
                self.target_combobox.current(0)
                self.active_target_path = self.target_paths[0]
                self.target_label_var.set(f"Active Target: {os.path.basename(self.active_target_path)}")
                self.redraw_target()
            self.status_var.set(f"Loaded {len(paths)} target(s). Total targets: {len(self.target_paths)}")

    def on_target_selected(self, event):
        idx = self.target_combobox.current()
        if idx >= 0:
            self.active_target_path = self.target_paths[idx]
            self.target_label_var.set(f"Active Target: {os.path.basename(self.active_target_path)}")
            self.pending_ref_pt = None
            self.redraw_ref()
            self.redraw_target()
            self.update_table()

    def get_display_scale(self, orig_img, canvas):
        # Determine scaling factor to fit the image inside the canvas frame
        canvas_w = max(canvas.winfo_width(), 400)
        canvas_h = max(canvas.winfo_height(), 400)
        
        orig_h, orig_w = orig_img.shape[:2]
        
        scale_w = canvas_w / orig_w
        scale_h = canvas_h / orig_h
        scale = min(scale_w, scale_h, 1.0) # Don't upscale small images, only downscale large ones
        
        return scale

    def redraw_ref(self):
        if self.ref_img_orig is None:
            return
        
        # Calculate scale based on current canvas size
        self.ref_scale = (self.get_display_scale(self.ref_img_orig, self.ref_canvas) * self.ref_zoom)
        disp_w = int(self.ref_img_orig.shape[1] * self.ref_scale)
        disp_h = int(self.ref_img_orig.shape[0] * self.ref_scale)
        
        img_disp = cv2.resize(self.ref_img_orig, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
        
        # Draw recorded points (scaled down to display size)
        if self.active_target_path and self.active_target_path in self.point_pairs:
            for i, pair in enumerate(self.point_pairs[self.active_target_path]):
                pt_orig = pair[0]
                pt_disp = (int(pt_orig[0] * self.ref_scale), int(pt_orig[1] * self.ref_scale))
                cv2.circle(img_disp, pt_disp, 6, (0, 0, 255), -1)
                cv2.putText(img_disp, str(i+1), (pt_disp[0]+8, pt_disp[1]-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Draw pending reference point
        if self.pending_ref_pt:
            pt_disp = (int(self.pending_ref_pt[0] * self.ref_scale), int(self.pending_ref_pt[1] * self.ref_scale))
            cv2.circle(img_disp, pt_disp, 6, (0, 255, 255), -1)

        self.ref_photo = self.convert_cv_to_photo(img_disp, self.ref_canvas)
        self.ref_canvas.delete("all")
        self.ref_canvas.create_image(0, 0, anchor=tk.NW, image=self.ref_photo)
        self.ref_canvas.configure(scrollregion=(0, 0, disp_w, disp_h))

    def redraw_target(self):
        if not self.active_target_path or self.active_target_path not in self.target_images_orig:
            return
        
        target_orig = self.target_images_orig[self.active_target_path]
        self.target_scale = (self.get_display_scale(target_orig, self.target_canvas) * self.target_zoom)
        disp_w = int(target_orig.shape[1] * self.target_scale)
        disp_h = int(target_orig.shape[0] * self.target_scale)
        
        img_disp = cv2.resize(target_orig, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
        
        # Draw recorded points (scaled down to display size)
        if self.active_target_path in self.point_pairs:
            for i, pair in enumerate(self.point_pairs[self.active_target_path]):
                pt_orig = pair[1]
                pt_disp = (int(pt_orig[0] * self.target_scale), int(pt_orig[1] * self.target_scale))
                cv2.circle(img_disp, pt_disp, 6, (0, 255, 0), -1)
                cv2.putText(img_disp, str(i+1), (pt_disp[0]+8, pt_disp[1]-8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        self.target_photo = self.convert_cv_to_photo(img_disp, self.target_canvas)
        self.target_canvas.delete("all")
        self.target_canvas.create_image(0, 0, anchor=tk.NW, image=self.target_photo)
        self.target_canvas.configure(scrollregion=(0, 0, disp_w, disp_h))

    def start_pan(self, event):
        event.widget.scan_mark(event.x, event.y)

    def pan_image(self, event):
        event.widget.scan_dragto(event.x, event.y, gain=1)

    def on_ref_click(self, event):
        if self.ref_img_orig is None or not self.active_target_path:
            messagebox.showwarning("Warning", "Please load a reference image and select an active target first.")
            return
        
        # Convert click coordinates from display scale back to original full-res scale
        orig_x = int(self.ref_canvas.canvasx(event.x) / self.ref_scale)
        orig_y = int(self.ref_canvas.canvasy(event.y) / self.ref_scale)
        self.pending_ref_pt = (orig_x, orig_y)
        
        self.redraw_ref()
        self.status_var.set(f"Reference point set at {self.pending_ref_pt}. Now click corresponding point on Target Image.")

    def on_target_click(self, event):
        if not self.pending_ref_pt:
            messagebox.showwarning("Warning", "Click a point on the Reference Image first!")
            return
        if not self.active_target_path:
            return
        
        # Convert click coordinates from display scale back to original full-res scale
        orig_x = int(self.target_canvas.canvasx(event.x) / self.target_scale)
        orig_y = int(self.target_canvas.canvasy(event.y) / self.target_scale)
        target_pt = (orig_x, orig_y)
        
        self.point_pairs[self.active_target_path].append((self.pending_ref_pt, target_pt))
        self.pending_ref_pt = None
        
        self.redraw_ref()
        self.redraw_target()
        self.update_table()
        self.status_var.set(f"Pair added successfully! Total pairs for current target: {len(self.point_pairs[self.active_target_path])}")

    def change_zoom(self, which, factor):
        if which == "ref":
            self.ref_zoom = float(np.clip(self.ref_zoom * factor, 0.5, 10.0))
            self.redraw_ref()
        else:
            self.target_zoom = float(np.clip(self.target_zoom * factor, 0.5, 10.0))
            self.redraw_target()

    def reset_zoom(self, which):
        if which == "ref":
            self.ref_zoom = 1.0
            self.ref_canvas.xview_moveto(0)
            self.ref_canvas.yview_moveto(0)
            self.redraw_ref()
        else:
            self.target_zoom = 1.0
            self.target_canvas.xview_moveto(0)
            self.target_canvas.yview_moveto(0)
            self.redraw_target()

    def convert_cv_to_photo(self, cv_img, canvas):
            # Resize/fit logic can be added here, keeping original scale for simplicity
            cv_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(cv_rgb)
            return ImageTk.PhotoImage(pil_img)
    def undo_pair(self):
        if self.active_target_path and self.active_target_path in self.point_pairs and self.point_pairs[self.active_target_path]:
            self.point_pairs[self.active_target_path].pop()
            self.pending_ref_pt = None
            self.redraw_ref()
            self.redraw_target()
            self.update_table()
            self.status_var.set("Undid last point pair.")

    def clear_points(self):
        if self.active_target_path and self.active_target_path in self.point_pairs:
            self.point_pairs[self.active_target_path] = []
            self.pending_ref_pt = None
            self.redraw_ref()
            self.redraw_target()
            self.update_table()
            self.status_var.set("Cleared all point pairs for current target.")

    def update_table(self):
        self.table_text.config(state=tk.NORMAL)
        self.table_text.delete("1.0", tk.END)
        if self.active_target_path and self.active_target_path in self.point_pairs:
            pairs = self.point_pairs[self.active_target_path]
            table_str = " | ".join([f"{i+1}: Ref{p[0]} -> Tar{p[1]}" for i, p in enumerate(pairs)])
            self.table_text.insert(tk.END, table_str if table_str else "No points set yet for this target.")
        self.table_text.config(state=tk.DISABLED)

    def preview_alignment(self):
        if not self.active_target_path or len(self.point_pairs[self.active_target_path]) < 3:
            messagebox.showwarning("Not Enough Points", "You need at least 3 point pairs to compute an affine transform!")
            return

        pairs = self.point_pairs[self.active_target_path]
        ref_pts = np.float32([p[0] for p in pairs])
        target_pts = np.float32([p[1] for p in pairs])

        # Compute affine matrix mapping target points -> reference points
        M, _ = cv2.estimateAffine2D(target_pts, ref_pts)
        if M is None:
            messagebox.showerror("Error", "Could not estimate transformation matrix from points.")
            return

        # Warp target original image
        ref_h, ref_w = self.ref_img_orig.shape[:2]
        target_orig = self.target_images_orig[self.active_target_path]
        warped = cv2.warpAffine(target_orig, M, (ref_w, ref_h))

        # Open Live Preview Window with Alpha Bar
        self.open_preview_window(self.ref_img_orig, warped)

    def open_preview_window(self, ref_img, warped_img):
        preview_win = tk.Toplevel(self.root)
        preview_win.title("Live Alignment Preview (Alpha Blending)")
        preview_win.geometry("1000x850")

        # Container frame for canvas with scrollbars or auto-scaling
        container = ttk.Frame(preview_win)
        container.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        canvas = tk.Canvas(container, bg="gray")
        canvas.pack(fill=tk.BOTH, expand=True)

        # Alpha Slider frame
        slider_frame = ttk.Frame(preview_win, padding=5)
        slider_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(slider_frame, text="Ref Only (0%)").pack(side=tk.LEFT)
        alpha_slider = ttk.Scale(slider_frame, from_=0, to=100, orient=tk.HORIZONTAL)
        alpha_slider.set(50)  # Default 50% blend
        alpha_slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
        ttk.Label(slider_frame, text="Target Only (100%)").pack(side=tk.RIGHT)

        def update_blend(val):
            alpha = float(val) / 100.0
            blended = cv2.addWeighted(ref_img, 1.0 - alpha, warped_img, alpha, 0)
            
            # Determine scale to fit preview window nicely (e.g., max 900x700 or fit to window size)
            win_w = max(preview_win.winfo_width() - 40, 600)
            win_h = max(preview_win.winfo_height() - 100, 500)
            
            orig_h, orig_w = blended.shape[:2]
            scale = min(win_w / orig_w, win_h / orig_h, 1.0)
            
            if scale < 1.0:
                disp_w = int(orig_w * scale)
                disp_h = int(orig_h * scale)
                blended_disp = cv2.resize(blended, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
            else:
                blended_disp = blended

            # Convert for Tkinter
            cv_rgb = cv2.cvtColor(blended_disp, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(cv_rgb)
            img_tk = ImageTk.PhotoImage(pil_img)
            
            canvas.image = img_tk  # Prevent garbage collection
            canvas.delete("all")
            canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)

        # Bind slider and window resize event to keep it dynamically responsive
        alpha_slider.config(command=update_blend)
        preview_win.bind("<Configure>", lambda e: update_blend(alpha_slider.get()))
        update_blend(50)  # Initial render

    def save_active(self):
        if not self.active_target_path or len(self.point_pairs[self.active_target_path]) < 3:
            messagebox.showwarning("Warning", "Not enough points or no active target selected.")
            return
        
        pairs = self.point_pairs[self.active_target_path]
        ref_pts = np.float32([p[0] for p in pairs])
        target_pts = np.float32([p[1] for p in pairs])
        M, _ = cv2.estimateAffine2D(target_pts, ref_pts)

        ref_h, ref_w = self.ref_img_orig.shape[:2]
        target_orig = self.target_images_orig[self.active_target_path]
        warped = cv2.warpAffine(target_orig, M, (ref_w, ref_h))

        # Check if it's a mask to decide default extension and file types
        is_mask = "mask" in self.active_target_path.lower()
        default_ext = ".png" if is_mask else ".jpg"
        file_types = [("PNG Image", "*.png")] if is_mask else [("JPEG Image", "*.jpg"), ("PNG Image", "*.png")]

        out_path = filedialog.asksaveasfilename(
            defaultextension=default_ext, 
            filetypes=file_types,
            initialfile="aligned_" + os.path.basename(self.active_target_path)
        )
        
        if out_path:
            cv2.imwrite(out_path, warped)
            messagebox.showinfo("Success", f"Saved aligned target to:\n{out_path}")

    def save_all(self):
        if not self.active_target_path or len(self.point_pairs.get(self.active_target_path, [])) < 3:
            messagebox.showwarning("Warning", "Please select an active target with at least 3 point pairs to define the transformation.")
            return

        if not self.target_paths:
            messagebox.showwarning("Warning", "No target images loaded to save.")
            return

        # Compute transformation matrix from the active target
        pairs = self.point_pairs[self.active_target_path]
        ref_pts = np.float32([p[0] for p in pairs])
        target_pts = np.float32([p[1] for p in pairs])
        M, _ = cv2.estimateAffinePartial2D(target_pts, ref_pts)

        if M is None:
            messagebox.showerror("Error", "Could not estimate transformation matrix from the active target's points.")
            return

        # Use the directory of the first target image as the base location
        base_dir = os.path.dirname(self.target_paths[0])
        out_dir = os.path.join(base_dir, "Aligned")

        # Create the 'Aligned' folder if it doesn't already exist
        os.makedirs(out_dir, exist_ok=True)

        count = 0
        ref_h, ref_w = self.ref_img_orig.shape[:2]

        # Apply transformation M to ALL loaded target images
        for path in self.target_paths:
            target_img = self.target_images_orig.get(path)
            if target_img is not None:
                warped = cv2.warpAffine(target_img, M, (ref_w, ref_h))
                
                orig_filename = os.path.basename(path)
                name_without_ext, _ = os.path.splitext(orig_filename)
                
                if "mask" in orig_filename.lower():
                    final_filename = name_without_ext + ".png"
                else:
                    final_filename = name_without_ext + ".jpg"

                out_path = os.path.join(out_dir, final_filename)
                cv2.imwrite(out_path, warped)
                count += 1

        messagebox.showinfo(
            "Batch Complete", 
            f"Successfully aligned and saved all {count} images using the active target's transformation!\nSaved to folder:\n{out_dir}"
        )

        # Clear the completed batch
        self.target_paths.clear()
        self.target_images_orig.clear()
        self.point_pairs.clear()
        self.cumulative_matrices.clear()

        self.active_target_path = None
        self.pending_ref_pt = None

        self.target_combobox.set("")
        self.target_combobox["values"] = []
        self.target_label_var.set("Active Target: None")

        self.redraw_ref()
        self.target_canvas.delete("all")
        self.update_table()
        
if __name__ == "__main__":
    root = tk.Tk()
    app = ImageAlignmentTool(root)
    root.mainloop()