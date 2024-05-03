import tkinter as tk
from tkinter import filedialog
from tkinter.font import Font
from ultralytics import YOLO

class YOLOTrainingGUI:
    def __init__(self, master):
        self.master = master
        master.title("YOLO Training GUI")

        self.model_options = ['yolov9c', 'yolov9e', 'yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x','yolov8n-seg', 'yolov8s-seg', 'yolov8m-seg', 'yolov8l-seg', 'yolov8x-seg', 'yolov6-n', 'yolov6-s', 'yolov6-m', 'yolov6-l', 'yolov6-l6', 'yolov5nu', 'yolov5su', 'yolov5mu', 'yolov5lu', 'yolov5xu']
        self.selected_model = tk.StringVar(value=self.model_options[0])
        self.imgsz_options = [320, 384, 448, 512, 576, 640]
        self.selected_imgsz = tk.IntVar(value=self.imgsz_options[5])
        self.boolean_options = ["True", "False"]
        self.selected_cache = tk.StringVar(value=self.boolean_options[0])
        self.selected_amp = tk.StringVar(value=self.boolean_options[1])
        self.selected_val = tk.StringVar(value=self.boolean_options[0])
        self.batch_size_options = [-1, 2, 4, 8, 16, 32, 64]
        self.selected_batch_size = tk.IntVar(value=self.batch_size_options[0])
        self.save_period_options = [-1, 2, 4, 8, 16, 32, 64]
        self.selected_save_period = tk.IntVar(value=self.save_period_options[0])
        self.epoch_options = [15, 25, 50, 75, 100, 200, 300]
        self.selected_epochs = tk.IntVar(value=self.epoch_options[1])
        # Nuovo: Opzioni
        self.worker_options = [1, 2, 3, 4, 5, 6, 7, 8]
        self.selected_workers = tk.IntVar(value=7)
        #self.boolean_options = ["True", "False"]
        self.selected_pretrained = tk.StringVar(value="True")
        self.selected_single_cls = tk.StringVar(value="False")
        self.patience_options = [20, 40, 60, 100]
        self.selected_patience = tk.IntVar(value=40)


        self.small_font = Font(family="Helvetica", size=9)

        self.create_model_dropdown("Select YOLO Model:", "model_selection", self.model_options)
        self.create_entry("Select YAML File:", "yaml_path", 30, font=self.small_font, bg="#f0f0f0")
        self.create_button("Browse", self.browse_yaml)
        
        self.create_dropdown("Image Size (imgsz):", "imgsz_selection", self.imgsz_options, self.selected_imgsz)
        self.create_dropdown("Cache:", "cache_selection", self.boolean_options, self.selected_cache)
        self.create_dropdown("AMP:", "amp_selection", self.boolean_options, self.selected_amp)
        self.create_dropdown("Validation:", "val_selection", self.boolean_options, self.selected_val)
        self.create_dropdown("Batch Size (batch):", "batch_selection", self.batch_size_options, self.selected_batch_size)
        self.create_dropdown("Save Period:", "save_period_selection", self.save_period_options, self.selected_save_period)
        self.create_dropdown("Epochs:", "epoch_selection", self.epoch_options, self.selected_epochs)
        # Nuovi dropdown per Workers ,Pretrained,single_cls
        self.create_dropdown("Workers:", "worker_selection", self.worker_options, self.selected_workers)
        self.create_dropdown("Pretrained:", "pretrained_selection", self.boolean_options, self.selected_pretrained)
        self.create_dropdown("Single Class Mode:", "single_cls_selection", self.boolean_options, self.selected_single_cls)
        self.create_dropdown("Patience:", "patience_selection", self.patience_options, self.selected_patience)


        self.create_button("Train Model", self.train_model, "green")

        self.create_entry("Resume last.pt:", "resume_lastpt_path", 30, font=self.small_font, bg="#f0f0f0")
        self.create_button("Browse for last.pt", self.browse_resume_lastpt)

        self.create_button("Resume Training", self.resume_training, "green")

    def create_model_dropdown(self, label_text, var_name, options):
        self.create_dropdown(label_text, var_name, options, self.selected_model)

    def create_dropdown(self, label_text, var_name, options, variable):
        label = tk.Label(self.master, text=label_text)
        label.pack()

        dropdown = tk.OptionMenu(self.master, variable, *options)
        dropdown.pack()

    def create_entry(self, label_text, var_name, width, font=None, bg=None):
        label = tk.Label(self.master, text=label_text)
        label.pack()

        variable = tk.StringVar()
        entry = tk.Entry(self.master, textvariable=variable, width=width, font=font, bg=bg)
        entry.pack()

        setattr(self, var_name.replace(" ", "_"), variable)

    def create_button(self, button_text, command, bg_color=None):
        button = tk.Button(self.master, text=button_text, command=command, bg=bg_color)
        button.pack()

    def browse_yaml(self):
        file_path = filedialog.askopenfilename(filetypes=[("YAML files", "*.yaml")])
        if file_path:
            self.yaml_path.set(file_path)

    def train_model(self):
        yaml_path = self.yaml_path.get()
        if yaml_path:
            selected_model = self.selected_model.get()
            imgsz = self.selected_imgsz.get()
            batch = self.selected_batch_size.get()
            epochs = self.selected_epochs.get()
            cache = self.selected_cache.get() == 'True'
            amp = self.selected_amp.get() == 'True'
            val = self.selected_val.get() == 'True'
            save_period = self.selected_save_period.get()
            workers = self.selected_workers.get()
            pretrained = self.selected_pretrained.get() == 'True'
            single_cls = self.selected_single_cls.get() == 'True'
            patience = self.selected_patience.get()

            model_path = f"{selected_model}.pt"
            self.model = YOLO(model_path)

            results = self.model.train(data=yaml_path, epochs=epochs, batch=batch, imgsz=imgsz,
                                       cache=cache, amp=amp, val=val, save_period=save_period,
                                       workers=workers, pretrained=pretrained, single_cls=single_cls,patience=patience)
            print("Training results:", results)
        else:
            print("Please select a YAML file.")


    def browse_resume_lastpt(self):
        file_path = filedialog.askopenfilename(filetypes=[("LAST.PT", "*.pt")])
        if file_path:
            self.resume_lastpt_path.set(file_path)

    def resume_training(self):
        resume_lastpt_path = self.resume_lastpt_path.get()
        if resume_lastpt_path:
            selected_model = self.selected_model.get()

            model_path = f"{selected_model}.pt"
            self.model = YOLO(model_path)

            results = self.model.train(resume=True, data=resume_lastpt_path)
            print("Resume Training results:", results)

if __name__ == "__main__":
    root = tk.Tk()
    gui = YOLOTrainingGUI(root)
    root.mainloop()
