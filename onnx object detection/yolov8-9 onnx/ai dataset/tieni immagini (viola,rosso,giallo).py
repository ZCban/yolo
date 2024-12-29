import cv2
import numpy as np
import os

def process_images(image_folder, color_choice):
    # Definisci gli intervalli di colore in HSV
    colors = {
        "violet": (np.array([140, 100, 135]), np.array([155, 255, 255])),
        "red": [(np.array([0, 120, 70]), np.array([10, 255, 255])), (np.array([170, 120, 70]), np.array([180, 255, 255]))],
        "yellow": (np.array([20, 100, 100]), np.array([30, 255, 255]))
    }

    # Verifica se il colore scelto è valido
    if color_choice not in colors:
        print(f"Colore '{color_choice}' non valido. Scegli tra: {', '.join(colors.keys())}")
        return

    # Definisci il kernel per la dilatazione
    kernel = np.ones((5, 5), np.uint8)

    # Verifica se la cartella esiste
    if not os.path.exists(image_folder):
        print(f"La cartella '{image_folder}' non esiste.")
        return

    # Ottieni la lista di tutte le immagini nella cartella
    images = [f for f in os.listdir(image_folder) if os.path.isfile(os.path.join(image_folder, f))]

    # Itera attraverso tutte le immagini
    for image_name in images:
        # Costruisci il percorso completo dell'immagine
        image_path = os.path.join(image_folder, image_name)

        # Carica l'immagine
        img = cv2.imread(image_path)

        # Se l'immagine non riesce a caricarsi, saltala
        if img is None:
            print(f"Impossibile caricare l'immagine: {image_name}")
            continue

        # Converte l'immagine in HSV
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Crea una maschera per il colore specificato
        if color_choice == "red":
            mask = cv2.inRange(hsv, colors["red"][0][0], colors["red"][0][1]) | cv2.inRange(hsv, colors["red"][1][0], colors["red"][1][1])
        else:
            lower, upper = colors[color_choice]
            mask = cv2.inRange(hsv, lower, upper)

        # Applica la dilatazione alla maschera
        dilated = cv2.dilate(mask, kernel, iterations=1)

        # Trova i contorni
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # Se non ci sono contorni, elimina l'immagine
        if len(contours) == 0:
            print(f"Il colore '{color_choice}' non è presente nell'immagine: {image_name}. Eliminando...")
            os.remove(image_path)
        else:
            print(f"Il colore '{color_choice}' è presente nell'immagine: {image_name}.")

    print("Operazione completata.")

# Esempio di utilizzo
image_folder = r"C:\Users\Admin\Desktop\ghg\old backup\Pyt\yolo-main\onnx object detection\yolov8-9 onnx\data"  # Modifica questo percorso con il percorso corretto
color_choice = "violet"  # Scegli tra: "violet", "red", "yellow"
process_images(image_folder, color_choice)
