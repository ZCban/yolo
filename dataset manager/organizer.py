# organizer.py
import os
import shutil
from PIL import Image
from PyQt5.QtCore import QThread, pyqtSignal

EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


class OrganizerThread1(QThread):
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal(dict)

    def __init__(
        self,
        root_dir,
        split_resolution,
        delete_orphan_images,
        delete_orphan_labels,
        resize_dims=None,
        do_resize=False
    ):
        super().__init__()
        self.root_dir = root_dir
        self.split_resolution = split_resolution
        self.delete_orphan_images = delete_orphan_images
        self.delete_orphan_labels = delete_orphan_labels
        self.resize_dims = resize_dims
        self.do_resize = do_resize

    def run(self):
        splits = ["train", "valid"]
        results = {}

        for split_idx, split in enumerate(splits):
            images_dir = os.path.join(self.root_dir, split, "images")
            labels_dir = os.path.join(self.root_dir, split, "labels")

            if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
                results[split] = {}
                continue

            img_files = [f for f in os.listdir(images_dir)
                         if f.lower().endswith(EXTENSIONS)]
            total_files = len(img_files)
            resolution_counts = {}

            for idx, file in enumerate(img_files, 1):
                img_path = os.path.join(images_dir, file)
                name, _ = os.path.splitext(file)
                label_file = name + ".txt"
                label_path = os.path.join(labels_dir, label_file)

                try:
                    folder_name = None
                    if self.split_resolution and not self.do_resize:
                        with Image.open(img_path) as img:
                            w, h = img.size
                        folder_name = f"{w}x{h}"

                    img_dst_dir = os.path.join(images_dir, folder_name) if folder_name else images_dir
                    lbl_dst_dir = os.path.join(labels_dir, folder_name) if folder_name else labels_dir
                    os.makedirs(img_dst_dir, exist_ok=True)
                    os.makedirs(lbl_dst_dir, exist_ok=True)

                    shutil.move(img_path, os.path.join(img_dst_dir, file))

                    key = folder_name or "original"
                    resolution_counts.setdefault(key, {"images": 0, "labels": 0})
                    resolution_counts[key]["images"] += 1

                    if os.path.exists(label_path):
                        shutil.move(label_path, os.path.join(lbl_dst_dir, label_file))
                        resolution_counts[key]["labels"] += 1
                    elif self.delete_orphan_images:
                        os.remove(os.path.join(img_dst_dir, file))
                        resolution_counts[key]["images"] -= 1

                    # resize + normalizzazione
                    if self.do_resize and self.resize_dims:
                        target_w, target_h = self.resize_dims
                        with Image.open(os.path.join(img_dst_dir, file)) as img:
                            orig_w, orig_h = img.size
                            img = img.resize((target_w, target_h))
                            img.save(os.path.join(img_dst_dir, file))

                        lbl_path = os.path.join(lbl_dst_dir, label_file)
                        if os.path.exists(lbl_path):
                            new_lines = []
                            with open(lbl_path) as f:
                                for line in f:
                                    cls, xc, yc, w, h = line.split()
                                    xc = float(xc) * orig_w / target_w
                                    yc = float(yc) * orig_h / target_h
                                    w = float(w) * orig_w / target_w
                                    h = float(h) * orig_h / target_h
                                    new_lines.append(
                                        f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"
                                    )
                            with open(lbl_path, "w") as f:
                                f.write("\n".join(new_lines))

                except Exception as e:
                    print("Errore:", e)

                progress = int(((split_idx + idx / total_files) / len(splits)) * 100)
                self.progress_signal.emit(progress)

            # rimuove label orfane
            if self.delete_orphan_labels:
                for root, _, files in os.walk(labels_dir):
                    for f in files:
                        if not f.endswith(".txt"):
                            continue
                        name = os.path.splitext(f)[0]
                        img_found = any(
                            os.path.exists(os.path.join(images_dir, name + ext))
                            for ext in EXTENSIONS
                        )
                        if not img_found:
                            os.remove(os.path.join(root, f))

            results[split] = resolution_counts

        self.progress_signal.emit(100)
        self.finished_signal.emit(results)



# organizer.py
import os
import shutil
from PIL import Image
from PyQt5.QtCore import QThread, pyqtSignal

EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


class OrganizerThread(QThread):
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal(dict)

    def __init__(
        self,
        root_dir,
        split_resolution,
        delete_orphan_images,
        delete_orphan_labels,
        resize_dims=None,
        do_resize=False,
        move_small_labels=False,
        threshold=0.1,  # consider labels < 10% della media
    ):
        super().__init__()
        self.root_dir = root_dir
        self.split_resolution = split_resolution
        self.delete_orphan_images = delete_orphan_images
        self.delete_orphan_labels = delete_orphan_labels
        self.resize_dims = resize_dims
        self.do_resize = do_resize
        self.move_small_labels = move_small_labels
        self.threshold = threshold

    def run(self):
        splits = ["train", "valid"]
        results = {}

        for split_idx, split in enumerate(splits):
            images_dir = os.path.join(self.root_dir, split, "images")
            labels_dir = os.path.join(self.root_dir, split, "labels")

            if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
                results[split] = {}
                continue

            img_files = [f for f in os.listdir(images_dir)
                         if f.lower().endswith(EXTENSIONS)]
            total_files = len(img_files)
            resolution_counts = {}

            # --- calcola media area label per spostamento small_labels
            mean_area = 0
            if self.move_small_labels:
                label_areas = []
                for file in img_files:
                    label_path = os.path.join(labels_dir, os.path.splitext(file)[0] + ".txt")
                    if not os.path.exists(label_path):
                        continue
                    with open(label_path) as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) != 5:
                                continue
                            _, _, _, w, h = map(float, parts)
                            label_areas.append(w * h)
                mean_area = sum(label_areas) / len(label_areas) if label_areas else 0

            for idx, file in enumerate(img_files, 1):
                img_path = os.path.join(images_dir, file)
                name, _ = os.path.splitext(file)
                label_file = name + ".txt"
                label_path = os.path.join(labels_dir, label_file)

                try:
                    folder_name = None
                    if self.split_resolution and not self.do_resize:
                        with Image.open(img_path) as img:
                            w, h = img.size
                        folder_name = f"{w}x{h}"

                    img_dst_dir = os.path.join(images_dir, folder_name) if folder_name else images_dir
                    lbl_dst_dir = os.path.join(labels_dir, folder_name) if folder_name else labels_dir
                    os.makedirs(img_dst_dir, exist_ok=True)
                    os.makedirs(lbl_dst_dir, exist_ok=True)

                    shutil.move(img_path, os.path.join(img_dst_dir, file))
                    key = folder_name or "original"
                    resolution_counts.setdefault(key, {"images": 0, "labels": 0})
                    resolution_counts[key]["images"] += 1

                    if os.path.exists(label_path):
                        shutil.move(label_path, os.path.join(lbl_dst_dir, label_file))
                        resolution_counts[key]["labels"] += 1
                    elif self.delete_orphan_images:
                        os.remove(os.path.join(img_dst_dir, file))
                        resolution_counts[key]["images"] -= 1

                    # --- ridimensionamento + normalizzazione
                    if self.do_resize and self.resize_dims:
                        target_w, target_h = self.resize_dims
                        with Image.open(os.path.join(img_dst_dir, file)) as img:
                            orig_w, orig_h = img.size
                            img = img.resize((target_w, target_h))
                            img.save(os.path.join(img_dst_dir, file))

                        lbl_path = os.path.join(lbl_dst_dir, label_file)
                        if os.path.exists(lbl_path):
                            new_lines = []
                            with open(lbl_path) as f:
                                for line in f:
                                    cls, xc, yc, w, h = line.split()
                                    xc = float(xc) * orig_w / target_w
                                    yc = float(yc) * orig_h / target_h
                                    w = float(w) * orig_w / target_w
                                    h = float(h) * orig_h / target_h
                                    new_lines.append(
                                        f"{cls} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"
                                    )
                            with open(lbl_path, "w") as f:
                                f.write("\n".join(new_lines))

                    # --- sposta small_labels
                    if self.move_small_labels and os.path.exists(os.path.join(lbl_dst_dir, label_file)):
                        small = False
                        with open(os.path.join(lbl_dst_dir, label_file)) as f:
                            for line in f:
                                parts = line.strip().split()
                                if len(parts) != 5:
                                    continue
                                _, _, _, w, h = map(float, parts)
                                if mean_area > 0 and (w * h) < self.threshold * mean_area:
                                    small = True
                                    break

                        if small:
                            small_dir_img = os.path.join(self.root_dir, split, "small_labels", "images")
                            small_dir_lbl = os.path.join(self.root_dir, split, "small_labels", "labels")
                            os.makedirs(small_dir_img, exist_ok=True)
                            os.makedirs(small_dir_lbl, exist_ok=True)

                            shutil.move(os.path.join(img_dst_dir, file), os.path.join(small_dir_img, file))
                            shutil.move(os.path.join(lbl_dst_dir, label_file), os.path.join(small_dir_lbl, label_file))

                            # decrementa contatori
                            resolution_counts[key]["images"] -= 1
                            resolution_counts[key]["labels"] -= 1

                except Exception as e:
                    print("Errore:", e)

                # barra progresso
                progress = int(((split_idx + idx / total_files) / len(splits)) * 100)
                self.progress_signal.emit(progress)

            # --- rimuove label orfane
            if self.delete_orphan_labels:
                for root, _, files in os.walk(labels_dir):
                    for f in files:
                        if not f.endswith(".txt"):
                            continue
                        name = os.path.splitext(f)[0]
                        img_found = any(
                            os.path.exists(os.path.join(images_dir, name + ext))
                            for ext in EXTENSIONS
                        )
                        if not img_found:
                            os.remove(os.path.join(root, f))

            results[split] = resolution_counts

        self.progress_signal.emit(100)
        self.finished_signal.emit(results)

