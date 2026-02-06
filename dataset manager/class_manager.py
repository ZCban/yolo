# class_manager.py
import os
import yaml
from PyQt5.QtCore import QThread, pyqtSignal

EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def find_yaml_in_root(root_dir):
    for f in os.listdir(root_dir):
        if f.lower().endswith((".yaml", ".yml")):
            return os.path.join(root_dir, f)
    return None


def load_yaml_classes(yaml_path):
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    names = data["names"]
    if isinstance(names, list):
        classes = {i: n for i, n in enumerate(names)}
    else:
        classes = {int(k): v for k, v in names.items()}
    return classes, data


def save_yaml_classes(yaml_path, data, classes):
    data["nc"] = len(classes)
    data["names"] = [classes[i] for i in sorted(classes)]
    with open(yaml_path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


class DeleteClassesThread(QThread):
    progress_signal = pyqtSignal(int)
    finished_signal = pyqtSignal()

    def __init__(self, root_dir, yaml_path, delete_ids):
        super().__init__()
        self.root_dir = root_dir
        self.yaml_path = yaml_path
        self.delete_ids = set(delete_ids)

    def run(self):
        classes, data = load_yaml_classes(self.yaml_path)

        id_map = {}
        new_classes = {}
        new_id = 0

        for old_id, name in classes.items():
            if old_id not in self.delete_ids:
                id_map[old_id] = new_id
                new_classes[new_id] = name
                new_id += 1

        label_files = []
        for split in ["train", "valid"]:
            lbl = os.path.join(self.root_dir, split, "labels")
            for r, _, files in os.walk(lbl):
                for f in files:
                    if f.endswith(".txt"):
                        label_files.append(os.path.join(r, f))

        total = len(label_files)

        for i, path in enumerate(label_files, 1):
            with open(path) as f:
                lines = f.readlines()

            out = []
            for line in lines:
                parts = line.split()
                cid = int(parts[0])
                if cid in self.delete_ids:
                    continue
                parts[0] = str(id_map[cid])
                out.append(" ".join(parts))

            if out:
                with open(path, "w") as f:
                    f.write("\n".join(out))
            else:
                os.remove(path)

            self.progress_signal.emit(int(i / total * 100))

        save_yaml_classes(self.yaml_path, data, new_classes)
        self.finished_signal.emit()
