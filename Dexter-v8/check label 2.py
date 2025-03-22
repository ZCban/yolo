import tkinter as tk
from PIL import Image, ImageTk
import cv2
import os
import shutil

# Percorsi aggiornati
current_dir = os.path.dirname(os.path.abspath(__file__))
images_folder = os.path.join(current_dir, 'data', 'detected_target', 'images')
labels_folder = os.path.join(current_dir, 'data', 'detected_target', 'labels')
cartella_verificato = os.path.join(current_dir, 'data', 'detected_target', 'verificato')


# Variabili globali
images_list = []  # Lista di tutte le immagini nella cartella
current_index = 0  # Indice dell'immagine attualmente visualizzata
img_width, img_height = 0, 0


# Funzione per caricare tutte le immagini dalla cartella
def carica_immagini(cartella=images_folder):
    global images_list
    # Lista di tutte le immagini nella cartella
    images_list = [f for f in os.listdir(cartella) if f.endswith('.jpg')]

# Funzione per aggiornare il contatore delle immagini nell'interfaccia
def aggiorna_contatore():
    contatore_label.config(text=f"Immagine {current_index + 1} di {len(images_list)}")

# Funzione per caricare un'immagine e la sua label
def visualizza_immagine_e_label(index):
    global img_width, img_height

    # Controllo dei limiti per evitare errori di indice
    if index < 0 or index >= len(images_list):
        return

    # Aggiornamenti per percorsi nel resto del codice
    file_img = os.path.join(images_folder, images_list[index])
    file_label = os.path.join(labels_folder, images_list[index].replace('.jpg', '.txt'))

    # Carica l'immagine
    img = Image.open(file_img)
    img_width, img_height = img.size

    # Carica l'immagine in formato Tkinter
    img_tk = ImageTk.PhotoImage(img)

    # Mostra l'immagine nell'interfaccia
    label_img.config(image=img_tk)
    label_img.image = img_tk

    # Se esiste il file label, lo carichiamo e disegniamo le bounding box
    if os.path.exists(file_label):
        with open(file_label, 'r') as f:
            boxes = f.readlines()

        # Carica l'immagine con OpenCV per disegnare le box
        img_cv = cv2.imread(file_img)

        # Disegna le bounding box e le classi
        for box in boxes:
            class_id, x_center, y_center, width, height = map(float, box.split())
            x_center, y_center = int(x_center * img_width), int(y_center * img_height)
            width, height = int(width * img_width), int(height * img_height)

            # Calcola le coordinate dei vertici del rettangolo
            x1 = int(x_center - width / 2)
            y1 = int(y_center - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center + height / 2)

            # Disegna il rettangolo (bounding box)
            cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Scrivi la classe sopra la bounding box
            cv2.putText(img_cv, f'Class: {int(class_id)}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (0, 255, 0), 2)

        # Converti l'immagine OpenCV in formato PIL per Tkinter
        img_with_boxes = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
        img_tk_with_boxes = ImageTk.PhotoImage(img_with_boxes)

        # Mostra l'immagine con le bounding box
        label_img.config(image=img_tk_with_boxes)
        label_img.image = img_tk_with_boxes
    else:
        print("File label non trovato per", file_img)

    # Aggiorna il contatore delle immagini
    aggiorna_contatore()

# Funzione per andare all'immagine successiva
def avanti(event=None):
    global current_index
    if current_index < len(images_list) - 1:
        current_index += 1
        visualizza_immagine_e_label(current_index)

# Funzione per tornare all'immagine precedente
def indietro(event=None):
    global current_index
    if current_index > 0:
        current_index -= 1
        visualizza_immagine_e_label(current_index)

# Funzione per spostare l'immagine e la label nella cartella "verificato"
def verifica_immagine(event=None):
    global current_index
    if current_index < 0 or current_index >= len(images_list):
        return
    
    file_img = os.path.join(images_folder, images_list[current_index])
    file_label = os.path.join(labels_folder, images_list[current_index].replace('.jpg', '.txt'))

    # Creazione delle sottocartelle "images" e "labels" all'interno di "verificato"
    cartella_images_verificato = os.path.join(cartella_verificato, "images")
    cartella_labels_verificato = os.path.join(cartella_verificato, "labels")
    os.makedirs(cartella_images_verificato, exist_ok=True)
    os.makedirs(cartella_labels_verificato, exist_ok=True)
    
    # Sposta l'immagine nella sottocartella "images" di "verificato"
    shutil.move(file_img, os.path.join(cartella_images_verificato, os.path.basename(file_img)))
    # Sposta il file label nella sottocartella "labels" di "verificato", se esiste
    if os.path.exists(file_label):
        shutil.move(file_label, os.path.join(cartella_labels_verificato, os.path.basename(file_label)))

    # Rimuove l'immagine dalla lista e aggiorna la visualizzazione
    images_list.pop(current_index)
    if current_index >= len(images_list):
        current_index = len(images_list) - 1
    visualizza_immagine_e_label(current_index)

# Creiamo la finestra principale
root = tk.Tk()
root.title("Controllo delle bounding box")

# Etichetta per visualizzare l'immagine
label_img = tk.Label(root)
label_img.pack()

# Label per visualizzare il contatore delle immagini
contatore_label = tk.Label(root, text="")
contatore_label.pack()

# Bottone per immagine precedente
bottone_indietro = tk.Button(root, text="Indietro", command=indietro)
bottone_indietro.pack(side=tk.LEFT)

# Bottone per immagine successiva
bottone_avanti = tk.Button(root, text="Avanti", command=avanti)
bottone_avanti.pack(side=tk.RIGHT)

# Bottone per verificare l'immagine e spostarla nella cartella "verificato"
bottone_verifica = tk.Button(root, text="Verifica e sposta", command=verifica_immagine)
bottone_verifica.pack(side=tk.BOTTOM)

# Carichiamo tutte le immagini all'avvio
carica_immagini()

# Visualizziamo la prima immagine e aggiorniamo il contatore
if images_list:
    visualizza_immagine_e_label(current_index)

# Bind dei tasti
root.bind("d", avanti)       # Tasto 'd' per andare avanti
root.bind("a", indietro)     # Tasto 'a' per andare indietro
root.bind("v", verifica_immagine)  # Tasto 'v' per verificare e spostare

# Avvia il loop dell'interfaccia grafica
root.mainloop()
