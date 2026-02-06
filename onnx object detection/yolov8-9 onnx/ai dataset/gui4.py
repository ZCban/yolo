import tkinter as tk
from tkinter import messagebox
import os
from multiprocessing import Process
from ai_dataset import (
    download_video_from_link,
    extract_frames,
    remove_corrupted_images,
    elimino0_50,
    annotate_images,
    clean_folders
)


# Percorsi principali
VIDEOS_FOLDER = "download"
FRAMES_FOLDER = os.path.join(VIDEOS_FOLDER, "frames")
TRAIN_PATH = os.path.join(os.getcwd(), "train")
IMAGES_PATH = os.path.join(TRAIN_PATH, "images")
LABELS_PATH = os.path.join(TRAIN_PATH, "labels")

# Creazione delle directory se non esistono
os.makedirs(VIDEOS_FOLDER, exist_ok=True)
os.makedirs(FRAMES_FOLDER, exist_ok=True)
os.makedirs(IMAGES_PATH, exist_ok=True)
os.makedirs(LABELS_PATH, exist_ok=True)


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

        # Impostazione frames da skippare
        tk.Label(root, text="Estrai 1 frame ogni N frames:").pack()
        self.entry_skip_frames = tk.Entry(root, width=10)
        self.entry_skip_frames.insert(0, "10")  # default 1 frame ogni 10
        self.entry_skip_frames.pack(pady=5)

        # Checkbox per selezionare attività
        self.chk_download_var = tk.IntVar()
        self.chk_extract_var = tk.IntVar()
        self.chk_filter_var = tk.IntVar()
        self.chk_annotate_var = tk.IntVar()
        self.chk_clean_var = tk.IntVar()

        tk.Checkbutton(root, text="Scarica Video (H.264 1080p, no audio)", variable=self.chk_download_var).pack(pady=2)
        tk.Checkbutton(root, text="Estrai Frames", variable=self.chk_extract_var).pack(pady=2)
        tk.Checkbutton(root, text="Filtra immagini (Conf. < 0.50)", variable=self.chk_filter_var).pack(pady=2)
        tk.Checkbutton(root, text="Annota Immagini", variable=self.chk_annotate_var).pack(pady=2)
        tk.Checkbutton(root, text="Pulisci Cartelle", variable=self.chk_clean_var).pack(pady=2)

        # Pulsante per eseguire operazioni selezionate
        tk.Button(root, text="Esegui Selezionati", command=self.run_selected_tasks, bg="green", fg="white").pack(pady=10)

        # Pulsante per uscire
        tk.Button(root, text="Esci", command=root.quit, bg="red", fg="white").pack(pady=5)

    def run_selected_tasks(self):
        # 1️⃣ Scarica video
        if self.chk_download_var.get():
            video_url = self.entry_url.get()
            if video_url:
                download_video_from_link(video_url)
            else:
                messagebox.showerror("Errore", "Inserisci un URL valido.")
                return

        # 2️⃣ Estrai frame
        if self.chk_extract_var.get():
            frames_to_skip = self.entry_skip_frames.get()
            if not frames_to_skip.isdigit() or int(frames_to_skip) < 1:
                messagebox.showerror("Errore", "Valore non valido per frames da skippare.")
                return

            extract_frames(
                every_n_frames=int(frames_to_skip),
                download_folder=VIDEOS_FOLDER,
                frames_folder=FRAMES_FOLDER
            )

            remove_corrupted_images(FRAMES_FOLDER)

        # 3️⃣ Operazioni indipendenti in parallelo
        parallel_tasks = []

        if self.chk_filter_var.get():
            parallel_tasks.append(Process(target=elimino0_50, args=(FRAMES_FOLDER,)))

        if self.chk_annotate_var.get():
            parallel_tasks.append(Process(target=annotate_images, args=(FRAMES_FOLDER,)))

        for task in parallel_tasks:
            task.start()
        for task in parallel_tasks:
            task.join()

        # 4️⃣ Pulizia cartelle
        if self.chk_clean_var.get():
            clean_folders(FRAMES_FOLDER, IMAGES_PATH, LABELS_PATH)

        messagebox.showinfo("Successo", "Operazioni selezionate completate con successo!")

        # Reset checkbox
        self.chk_download_var.set(0)
        self.chk_extract_var.set(0)
        self.chk_filter_var.set(0)
        self.chk_annotate_var.set(0)
        self.chk_clean_var.set(0)


if __name__ == "__main__":
    root = tk.Tk()
    app = VideoProcessingGUI(root)
    root.mainloop()
