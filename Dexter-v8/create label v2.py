import time
import numpy as np
import torch
from utils.utils_trt import TRTModule, Utility
from PIL import Image, UnidentifiedImageError
import os
import shutil  # Importa shutil per spostare i file
from tqdm import tqdm

# Percorsi delle cartelle
current_dir = os.path.dirname(os.path.abspath(__file__))
input_folder = os.path.join(current_dir, 'data')
model_file = 'misto.trt'
min_conf = 0.42

# Directory del modello
models_path = os.path.join(current_dir, 'models')
model_path = os.path.join(models_path, model_file)

# Load model
model = TRTModule(model_path, device=0)

# Percorsi output
data_folder = os.path.join(current_dir, 'data')
detected_target_folder = os.path.join(data_folder, 'detected_target')
no_target_folder = os.path.join(data_folder, 'no_target')
detected_images_folder = os.path.join(detected_target_folder, 'images')
detected_labels_folder = os.path.join(detected_target_folder, 'labels')

# Creazione delle directory necessarie
os.makedirs(detected_images_folder, exist_ok=True)
os.makedirs(detected_labels_folder, exist_ok=True)
os.makedirs(no_target_folder, exist_ok=True)

# Lista dei file da processare
files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# Processa ogni immagine nella cartella di input
for filename in tqdm(files, desc="Processing Images"):
    image_path = os.path.join(input_folder, filename)
    detected_image_path = os.path.join(detected_images_folder, filename)
    detected_label_path = os.path.join(detected_labels_folder, os.path.splitext(filename)[0] + ".txt")
    no_target_image_path = os.path.join(no_target_folder, filename)

    try:
        # Carica l'immagine
        img = Image.open(image_path)
        img_array = np.array(img)

        # Preprocessa l'immagine
        tensor = torch.as_tensor(np.ascontiguousarray(img_array.transpose(2, 0, 1)[np.newaxis, ...], dtype=np.float32) / 255.0, device='cuda')

        # Esegui l'inferenza
        data = model(tensor)

        # Inline implementation of det_postprocess6
        num_dets, bboxes, scores, labels = (i[0] for i in data)

        # Applica soglia di confidenza e filtra per class ID
        selected = (scores >= min_conf) & (labels == 0)
        bboxes_selected = bboxes[selected].cpu().numpy()  # Converti in NumPy

        if len(bboxes_selected) > 0:
            # Sposta immagine nella cartella detected_target
            shutil.move(image_path, detected_image_path)

            # Salva annotazioni
            with open(detected_label_path, "w") as f:
                for bbox in bboxes_selected:
                    # Converti bbox in formato YOLO
                    x1, y1, x2, y2 = bbox
                    x_center = (x1 + x2) / 2 / img_array.shape[1]
                    y_center = (y1 + y2) / 2 / img_array.shape[0]
                    width = (x2 - x1) / img_array.shape[1]
                    height = (y2 - y1) / img_array.shape[0]

                    # Scrivi annotazione nel file
                    f.write(f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        else:
            # Sposta immagine nella cartella no_target
            shutil.move(image_path, no_target_image_path)

    except (UnidentifiedImageError, OSError) as e:
        print(f"Errore durante l'elaborazione di {filename}: {e}. Elimino l'immagine corrotta.")
        try:
            os.remove(image_path)
        except Exception as delete_error:
            print(f"Impossibile eliminare {filename}: {delete_error}")

print("Elaborazione completata. Le immagini sono state spostate nelle rispettive cartelle.")
