import os
import shutil
import os
from tqdm import tqdm

def process_files(directory, txt_folder, img_folder):
    # Assicurati che le cartelle di destinazione esistano
    if not os.path.exists(txt_folder):
        os.makedirs(txt_folder)
    if not os.path.exists(img_folder):
        os.makedirs(img_folder)

    # Elenca tutti i file nella directory
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        # Assicurati che sia un file e non una cartella
        if os.path.isfile(file_path):
            # Elimina i file .txt di dimensione 0
            if filename.endswith('.txt') and os.path.getsize(file_path) == 0:
                os.remove(file_path)
            
            # Sposta i file .txt rimanenti
            elif filename.endswith('.txt'):
                shutil.move(file_path, os.path.join(txt_folder, filename))
            
            # Sposta i file immagine
            elif filename.endswith(('.png', '.jpg', '.jpeg')):
                shutil.move(file_path, os.path.join(img_folder, filename))






def check_images_without_labels(folder_labels, folder_images):
    # Get the list of files in both folders
    files_labels = set(os.listdir(folder_labels))
    files_images = set(os.listdir(folder_images))

    # Extract the filenames without extensions from the label files
    label_basenames = set(os.path.splitext(f)[0] for f in files_labels)

    # Iterate over the files in the images folder
    for file in tqdm(files_images, desc='Checking for missing labels', unit='file'):
        # Get the filename without extension
        filename, _ = os.path.splitext(file)
        # Check if the filename is not present in the label folder
        if filename not in label_basenames:
            # Create the full path of the file in the images folder
            file_path = os.path.join(folder_images, file)
            # Remove the file
            os.remove(file_path)

def check_labels_without_images(folder_labels, folder_images):
    # Get the list of files in both folders
    files_labels = set(os.listdir(folder_labels))
    files_images = set(os.listdir(folder_images))

    # Extract the filenames without extensions from the image files
    image_basenames = set(os.path.splitext(f)[0] for f in files_images)

    # Iterate over the files in the labels folder
    for file in tqdm(files_labels, desc='Checking for missing images', unit='file'):
        # Get the filename without extension
        filename, _ = os.path.splitext(file)
        # Check if the filename is not present in the image folder
        if filename not in image_basenames:
            # Create the full path of the file in the labels folder
            file_path = os.path.join(folder_labels, file)
            # Remove the file
            os.remove(file_path)

if __name__ == "__main__":

    # Percorso della directory da processare
    directory = r'C:\Users\Admin\Desktop\ghg\old backup\Pyt\yolo-main\onnx object detection\yolov8-9 onnx\data'
    # Percorsi delle cartelle di destinazione per i file .txt e immagini
    txt_folder = r'C:\Users\Admin\Desktop\ghg\old backup\Pyt\yolo-main\onnx object detection\yolov8-9 onnx\data\labels'
    img_folder = r'C:\Users\Admin\Desktop\ghg\old backup\Pyt\yolo-main\onnx object detection\yolov8-9 onnx\data\images'
    process_files(directory, txt_folder, img_folder)

    # Check train and valid directories
    print("Checking train set for images without labels...")
    check_images_without_labels(txt_folder, img_folder)
    
    print("Checking train set for labels without images...")
    check_labels_without_images(txt_folder, img_folder)




