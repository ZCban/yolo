import tkinter as tk
from tkinter import filedialog, messagebox
import os
from multiprocessing import Process
from ai_dataset import download_video_from_link, video_to_frame, elimino0_50, remove_corrupted_images, annotate_images, clean_folders

# Impostazione delle directory
FRAMES_PATH = "frames_extracted"
videos_folder = "download"
train_path = os.path.join(os.getcwd(), "train")
images_path = os.path.join(train_path, "images")
labels_path = os.path.join(train_path, "labels")

# Creazione delle directory necessarie
os.makedirs(videos_folder, exist_ok=True)
os.makedirs(FRAMES_PATH, exist_ok=True)
os.makedirs(images_path, exist_ok=True)
os.makedirs(labels_path, exist_ok=True)


class VideoProcessingGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("ONNX Video Processing Tool")
        self.root.geometry("500x550")

        # Titolo
        tk.Label(root, text="ONNX Video Processing", font=("Arial", 14, "bold")).pack(pady=10)

        # URL del Video
        tk.Label(root, text="Inserisci URL del Video:").pack()
        self.entry_url = tk.Entry(root, width=50)
        self.entry_url.pack(pady=5)

        # Impostazione target size
        tk.Label(root, text="Dimensione Target per Frames:").pack()
        self.entry_target_size = tk.Entry(root, width=10)
        self.entry_target_size.insert(0, "512")
        self.entry_target_size.pack(pady=5)

        # Impostazione frames da skippare
        tk.Label(root, text="Frames da Skippare:").pack()
        self.entry_skip_frames = tk.Entry(root, width=10)
        self.entry_skip_frames.insert(0, "2")
        self.entry_skip_frames.pack(pady=5)

        # Checkbox per selezionare attività
        self.chk_download_var = tk.IntVar()
        self.chk_extract_var = tk.IntVar()
        self.chk_filter_var = tk.IntVar()
        self.chk_annotate_var = tk.IntVar()
        self.chk_clean_var = tk.IntVar()

        tk.Checkbutton(root, text="Scarica Video", variable=self.chk_download_var).pack(pady=2)
        tk.Checkbutton(root, text="Estrai Frames", variable=self.chk_extract_var).pack(pady=2)
        tk.Checkbutton(root, text="Filtra immagini (Conf. < 0.50)", variable=self.chk_filter_var).pack(pady=2)
        tk.Checkbutton(root, text="Annota Immagini", variable=self.chk_annotate_var).pack(pady=2)
        tk.Checkbutton(root, text="Pulisci Cartelle", variable=self.chk_clean_var).pack(pady=2)

        # Pulsante per eseguire operazioni selezionate
        tk.Button(root, text="Esegui Selezionati", command=self.run_selected_tasks, bg="green", fg="white").pack(pady=10)

        # Pulsante per uscire
        tk.Button(root, text="Esci", command=root.quit, bg="red", fg="white").pack(pady=5)

    def run_selected_tasks(self):

        if self.chk_download_var.get():
            video_url = self.entry_url.get()
            if video_url:
                download_video_from_link(video_url)
            else:
                messagebox.showerror("Errore", "Inserisci un URL valido.")
                return

        if self.chk_extract_var.get():
            target_size = self.entry_target_size.get()
            frames_to_skip = self.entry_skip_frames.get()

            if not target_size.isdigit() or not frames_to_skip.isdigit():
                messagebox.showerror("Errore", "Valori numerici non validi per target size o frame da skippare.")
                return

            video_to_frame(int(target_size), int(frames_to_skip), videos_folder, FRAMES_PATH)
            remove_corrupted_images(FRAMES_PATH)

        # Operazioni indipendenti eseguite in parallelo
        parallel_tasks = []

        if self.chk_filter_var.get():
            parallel_tasks.append(Process(target=elimino0_50, args=(FRAMES_PATH,)))

        if self.chk_annotate_var.get():
            parallel_tasks.append(Process(target=annotate_images, args=(FRAMES_PATH,)))

        for task in parallel_tasks:
            task.start()
        for task in parallel_tasks:
            task.join()

        # Pulizia (ultima operazione)
        if self.chk_clean_var.get():
            clean_folders(FRAMES_PATH, images_path, labels_path)

        messagebox.showinfo("Successo", "Operazioni selezionate completate con successo!")

        # Reset delle checkbox al termine
        self.chk_download_var.set(0)
        self.chk_extract_var.set(0)
        self.chk_filter_var.set(0)
        self.chk_annotate_var.set(0)
        self.chk_clean_var.set(0)


if __name__ == "__main__":
    root = tk.Tk()
    app = VideoProcessingGUI(root)
    root.mainloop()
