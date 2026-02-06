import tkinter as tk
from tkinter import simpledialog, messagebox
from PIL import Image, ImageTk
import cv2
import os
import shutil

# Percorsi aggiornati
current_dir = os.path.dirname(os.path.abspath(__file__))
images_folder = os.path.join(current_dir, 'train', 'images')
labels_folder = os.path.join(current_dir, 'train', 'labels')
cartella_verificato = os.path.join(current_dir, 'train', 'verificato')

# Variabili globali
images_list = []
current_index = 0
img_width, img_height = 0, 0

# Funzione per caricare tutte le immagini dalla cartella
def carica_immagini(cartella=images_folder):
    global images_list
    images_list = [f for f in os.listdir(cartella) if f.endswith('.jpg')]

def aggiorna_contatore():
    contatore_label.config(text=f"Immagine {current_index + 1} di {len(images_list)}")

def visualizza_immagine_e_label1(index):
    global img_width, img_height

    if index < 0 or index >= len(images_list):
        return

    file_img = os.path.join(images_folder, images_list[index])
    file_label = os.path.join(labels_folder, images_list[index].replace('.jpg', '.txt'))

    img = Image.open(file_img)
    img_width, img_height = img.size
    img_tk = ImageTk.PhotoImage(img)
    label_img.config(image=img_tk)
    label_img.image = img_tk

    if os.path.exists(file_label):
        with open(file_label, 'r') as f:
            boxes = f.readlines()

        img_cv = cv2.imread(file_img)

        for i, box in enumerate(boxes):
            class_id, x_center, y_center, width, height = map(float, box.split())
            x_center, y_center = int(x_center * img_width), int(y_center * img_height)
            width, height = int(width * img_width), int(height * img_height)
            x1, y1 = int(x_center - width / 2), int(y_center - height / 2)
            x2, y2 = int(x_center + width / 2), int(y_center + height / 2)

            cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_cv, f'{i + 1} (Class {int(class_id)})', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        img_with_boxes = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGRA2BGR))
        img_tk_with_boxes = ImageTk.PhotoImage(img_with_boxes)
        label_img.config(image=img_tk_with_boxes)
        label_img.image = img_tk_with_boxes
    
    aggiorna_contatore()


import cv2
import os
import numpy as np
from tkinter import Tk, Label

# Dimensione massima per adattarsi allo schermo
MAX_WIDTH = 1200
MAX_HEIGHT = 800

def visualizza_immagine_e_label(index):
    global img_width, img_height

    if index < 0 or index >= len(images_list):
        return

    file_img = os.path.join(images_folder, images_list[index])
    file_label = os.path.join(labels_folder, images_list[index].rsplit('.', 1)[0] + '.txt')

    if not os.path.exists(file_img):
        print(f"Image not found: {file_img}")
        return

    # Leggi immagine con OpenCV
    img_cv = cv2.imread(file_img)
    if img_cv is None:
        print(f"Failed to load image: {file_img}")
        return

    original_height, original_width = img_cv.shape[:2]

    # Ridimensiona se l'immagine è troppo grande
    scale = min(MAX_WIDTH / original_width, MAX_HEIGHT / original_height, 1.0)
    img_width, img_height = int(original_width * scale), int(original_height * scale)
    img_resized = cv2.resize(img_cv, (img_width, img_height))

    # Disegna bounding boxes se esistono
    if os.path.exists(file_label):
        with open(file_label, 'r') as f:
            boxes = f.readlines()

        for i, box in enumerate(boxes):
            class_id, x_center, y_center, width, height = map(float, box.split())

            # Adatta le coordinate normalizzate alla dimensione ridimensionata
            x_center, y_center = int(x_center * img_width), int(y_center * img_height)
            width, height = int(width * img_width), int(height * img_height)
            x1, y1 = int(x_center - width / 2), int(y_center - height / 2)
            x2, y2 = int(x_center + width / 2), int(y_center + height / 2)

            cv2.rectangle(img_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_resized, f'{i+1} (Class {int(class_id)})', 
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Converti BGR → RGB per Tkinter
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_tk = ImageTk.PhotoImage(image=Image.fromarray(img_rgb))
    label_img.config(image=img_tk)
    label_img.image = img_tk

    aggiorna_contatore()


def elimina_box():
    global current_index
    if current_index < 0 or current_index >= len(images_list):
        return

    file_label = os.path.join(labels_folder, images_list[current_index].replace('.jpg', '.txt'))
    if not os.path.exists(file_label):
        messagebox.showinfo("Errore", "Nessuna bounding box trovata per questa immagine.")
        return

    with open(file_label, 'r') as f:
        boxes = f.readlines()

    if not boxes:
        messagebox.showinfo("Errore", "Nessuna bounding box presente.")
        return

    box_nums = ""
    while True:
        box_nums = simpledialog.askstring("Elimina box", "Inserisci i numeri delle box da eliminare:", initialvalue=box_nums)
        if box_nums is None:
            return
        box_nums = "".join([c + ", " if c.isdigit() and (i == len(box_nums) - 1 or not box_nums[i + 1].isdigit()) else c for i, c in enumerate(box_nums.replace(" ", ""))])
        box_nums = box_nums.rstrip(", ")
        if all(num.isdigit() and 0 < int(num) <= len(boxes) for num in box_nums.split(", ")):
            break
        messagebox.showerror("Errore", "Inserisci numeri validi separati da virgola.")
    
    try:
        box_indices = sorted(set(int(num.strip()) - 1 for num in box_nums.split(", ")), reverse=True)
        for idx in box_indices:
            if 0 <= idx < len(boxes):
                del boxes[idx]
    except ValueError:
        messagebox.showerror("Errore", "Inserisci numeri validi separati da virgola.")
        return

    with open(file_label, 'w') as f:
        f.writelines(boxes)

    visualizza_immagine_e_label(current_index)

def avanti(event=None):
    global current_index
    if current_index < len(images_list) - 1:
        current_index += 1
        visualizza_immagine_e_label(current_index)

def indietro(event=None):
    global current_index
    if current_index > 0:
        current_index -= 1
        visualizza_immagine_e_label(current_index)

def verifica_immagine(event=None):
    global current_index
    if current_index < 0 or current_index >= len(images_list):
        return
    
    file_img = os.path.join(images_folder, images_list[current_index])
    file_label = os.path.join(labels_folder, images_list[current_index].replace('.jpg', '.txt'))

    cartella_images_verificato = os.path.join(cartella_verificato, "images")
    cartella_labels_verificato = os.path.join(cartella_verificato, "labels")
    os.makedirs(cartella_images_verificato, exist_ok=True)
    os.makedirs(cartella_labels_verificato, exist_ok=True)
    
    shutil.move(file_img, os.path.join(cartella_images_verificato, os.path.basename(file_img)))
    if os.path.exists(file_label):
        shutil.move(file_label, os.path.join(cartella_labels_verificato, os.path.basename(file_label)))

    images_list.pop(current_index)
    if current_index >= len(images_list):
        current_index = len(images_list) - 1
    visualizza_immagine_e_label(current_index)

root = tk.Tk()
root.title("Controllo delle bounding box")

label_img = tk.Label(root)
label_img.pack()

contatore_label = tk.Label(root, text="")
contatore_label.pack()

bottone_indietro = tk.Button(root, text="Indietro", command=indietro)
bottone_indietro.pack(side=tk.LEFT)

bottone_avanti = tk.Button(root, text="Avanti", command=avanti)
bottone_avanti.pack(side=tk.RIGHT)

bottone_verifica = tk.Button(root, text="Verifica e sposta", command=verifica_immagine)
bottone_verifica.pack(side=tk.BOTTOM)

bottone_elimina_box = tk.Button(root, text="Elimina box", command=elimina_box)
bottone_elimina_box.pack(side=tk.BOTTOM)

carica_immagini()
if images_list:
    visualizza_immagine_e_label(current_index)

root.bind("d", avanti)
root.bind("a", indietro)
root.bind("v", verifica_immagine)
root.bind("e", lambda event: elimina_box())

root.mainloop()
