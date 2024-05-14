import tkinter as tk
from tkinter import filedialog
from ultralytics import YOLO

class YOLOExportGUI:
    def __init__(self, master):
        self.master = master
        master.title("YOLO Model Export GUI")

        self.export_format_options = ['torchscript', 'onnx', 'openvino', 'engine', 'coreml', 'pb', 'tflite']
        self.selected_export_format = tk.StringVar(value='torchscript')
        self.imgsz = tk.StringVar(value="640")
        self.keras = tk.BooleanVar(value=False)
        self.optimize = tk.BooleanVar(value=False)
        self.half = tk.BooleanVar(value=False)
        self.int8 = tk.BooleanVar(value=False)
        self.dynamic = tk.BooleanVar(value=False)
        self.simplify = tk.BooleanVar(value=False)
        self.nms = tk.BooleanVar(value=False)
        self.model_path = tk.StringVar()

        # Create input fields and dropdowns
        self.create_export_dropdown("Select Export Format:", self.selected_export_format, self.export_format_options)
        self.create_entry("Image Size (imgsz):", self.imgsz, 10)
        self.create_checkbox("Optimize:", self.optimize)
        self.create_checkbox("Half Precision (FP16):", self.half)
        self.create_checkbox("INT8 Quantization:", self.int8)
        self.create_checkbox("Dynamic:", self.dynamic)
        self.create_checkbox("Simplify:", self.simplify)
        self.create_checkbox("NMS:", self.nms)
        self.create_checkbox("Keras:", self.keras)

        self.create_entry("Model Path (.pt file):", self.model_path, 30)
        self.create_button("Browse", self.browse_model_path)
        self.create_button("Export Model", self.export_model, "blue")

    def create_export_dropdown(self, label_text, variable, options):
        label = tk.Label(self.master, text=label_text)
        label.pack()

        dropdown = tk.OptionMenu(self.master, variable, *options)
        dropdown.pack()

    def create_entry(self, label_text, variable, width, font=None, bg=None):
        label = tk.Label(self.master, text=label_text)
        label.pack()

        entry = tk.Entry(self.master, textvariable=variable, width=width, font=font, bg=bg)
        entry.pack()

    def create_checkbox(self, label_text, variable):
        check = tk.Checkbutton(self.master, text=label_text, variable=variable)
        check.pack()

    def create_button(self, button_text, command, bg_color=None):
        button = tk.Button(self.master, text=button_text, command=command, bg=bg_color)
        button.pack()

    def browse_model_path(self):
        file_path = filedialog.askopenfilename(filetypes=[("PT files", "*.pt")])
        if file_path:
            self.model_path.set(file_path)

    def export_model(self):
        model_path = self.model_path.get()
        if model_path:
            model = YOLO(model_path)
            kwargs = {
                "format": self.selected_export_format.get(),
                "imgsz": eval(self.imgsz.get()),
                "optimize": self.optimize.get(),
                "half": self.half.get(),
                "int8": self.int8.get(),
                "dynamic": self.dynamic.get(),
                "simplify": self.simplify.get(),
                "nms": self.nms.get(),
                "keras": self.keras.get()
            }
            model.export(**kwargs)
            export_path = model_path.replace('.pt', f'.{kwargs["format"]}')
            print(f"Model exported in {kwargs['format']} format to {export_path}")
        else:
            print("Please select a model file.")

if __name__ == "__main__":
    root = tk.Tk()
    gui = YOLOExportGUI(root)
    root.mainloop()
