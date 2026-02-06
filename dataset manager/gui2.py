# gui.py
import sys
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QFileDialog,
    QVBoxLayout, QCheckBox, QProgressBar, QSpinBox, QHBoxLayout,
    QMessageBox, QListWidget, QListWidgetItem, QDialog
)
from PyQt5.QtCore import Qt

from organizer import OrganizerThread
from class_manager import (
    find_yaml_in_root,
    load_yaml_classes,
    DeleteClassesThread
)


# ================= CLASS MANAGER DIALOG =================
class ClassManagerDialog(QDialog):
    def __init__(self, root_dir):
        super().__init__()
        self.root_dir = root_dir
        self.yaml_path = find_yaml_in_root(root_dir)

        classes, _ = load_yaml_classes(self.yaml_path)

        self.setWindowTitle("Gestione Classi")
        self.resize(400, 500)

        layout = QVBoxLayout(self)

        self.list = QListWidget()
        self.list.setSelectionMode(QListWidget.MultiSelection)

        for cid, name in classes.items():
            item = QListWidgetItem(f"{cid}: {name}")
            item.setData(Qt.UserRole, cid)
            self.list.addItem(item)

        layout.addWidget(self.list)

        self.progress = QProgressBar()
        self.progress.setValue(0)
        layout.addWidget(self.progress)

        btn = QPushButton("Elimina classi selezionate")
        btn.clicked.connect(self.delete_classes)
        layout.addWidget(btn)

    def delete_classes(self):
        ids = [i.data(Qt.UserRole) for i in self.list.selectedItems()]
        if not ids:
            QMessageBox.warning(self, "Errore", "Seleziona almeno una classe")
            return

        self.thread = DeleteClassesThread(self.root_dir, self.yaml_path, ids)
        self.thread.progress_signal.connect(self.progress.setValue)
        self.thread.finished_signal.connect(self.on_done)
        self.thread.start()

    def on_done(self):
        QMessageBox.information(self, "Completato", "Classi eliminate e YAML aggiornato")
        self.accept()


# ================= MAIN GUI =================
class DatasetOrganizer(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("YOLO Dataset Organizer")
        self.setGeometry(100, 100, 520, 580)
        self.root_dir = ""

        layout = QVBoxLayout(self)

        # Root
        self.label_root = QLabel("Cartella root dataset: Non selezionata")
        layout.addWidget(self.label_root)

        btn_root = QPushButton("Seleziona cartella root dataset")
        btn_root.clicked.connect(self.select_root)
        layout.addWidget(btn_root)

        # Checkboxes
        self.cb_split_resolution = QCheckBox("Dividi immagini e label per risoluzione")
        layout.addWidget(self.cb_split_resolution)

        self.cb_delete_orphan_images = QCheckBox("Elimina immagini senza label")
        layout.addWidget(self.cb_delete_orphan_images)

        self.cb_delete_orphan_labels = QCheckBox("Elimina label senza immagine")
        layout.addWidget(self.cb_delete_orphan_labels)

        self.cb_resize = QCheckBox(
            "Ridimensiona + normalizza label (disabilita dividi per risoluzione)"
        )
        layout.addWidget(self.cb_resize)

        # Resize controls
        size_layout = QHBoxLayout()
        self.spin_w = QSpinBox()
        self.spin_w.setRange(1, 10000)
        self.spin_w.setValue(512)

        self.spin_h = QSpinBox()
        self.spin_h.setRange(1, 10000)
        self.spin_h.setValue(512)

        size_layout.addWidget(QLabel("Width:"))
        size_layout.addWidget(self.spin_w)
        size_layout.addWidget(QLabel("Height:"))
        size_layout.addWidget(self.spin_h)
        layout.addLayout(size_layout)

        # Small labels option
        self.cb_small_labels = QCheckBox("Sposta immagini con label troppo piccole")
        layout.addWidget(self.cb_small_labels)

        small_layout = QHBoxLayout()
        self.spin_threshold = QSpinBox()
        self.spin_threshold.setRange(1, 100)
        self.spin_threshold.setValue(10)
        small_layout.addWidget(QLabel("Threshold (% della media):"))
        small_layout.addWidget(self.spin_threshold)
        layout.addLayout(small_layout)

        # Progress bar
        self.progress = QProgressBar()
        self.progress.setAlignment(Qt.AlignCenter)
        self.progress.setValue(0)
        layout.addWidget(self.progress)

        # Buttons
        btn_run = QPushButton("Organizza dataset")
        btn_run.clicked.connect(self.run_organizer)
        layout.addWidget(btn_run)

        btn_classes = QPushButton("Gestisci classi")
        btn_classes.clicked.connect(self.manage_classes)
        layout.addWidget(btn_classes)

    # ================= ACTIONS =================
    def select_root(self):
        d = QFileDialog.getExistingDirectory(self, "Seleziona root dataset")
        if d:
            self.root_dir = d
            self.label_root.setText(f"Cartella root dataset: {d}")

    def manage_classes(self):
        if not self.root_dir:
            QMessageBox.warning(self, "Errore", "Seleziona la cartella root")
            return
        if not find_yaml_in_root(self.root_dir):
            QMessageBox.critical(self, "Errore", "Nessun file YAML trovato nella root")
            return

        ClassManagerDialog(self.root_dir).exec_()

    def run_organizer(self):
        if not self.root_dir:
            QMessageBox.warning(self, "Errore", "Seleziona prima la cartella root dataset")
            return

        do_resize = self.cb_resize.isChecked()
        split_resolution = self.cb_split_resolution.isChecked()

        if do_resize and split_resolution:
            reply = QMessageBox.question(
                self,
                "Attenzione",
                "Hai selezionato il resize.\n"
                "La divisione per risoluzione verrà ignorata.\n\n"
                "Vuoi continuare?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply == QMessageBox.No:
                return
            split_resolution = False

        resize_dims = (self.spin_w.value(), self.spin_h.value()) if do_resize else None

        move_small_labels = self.cb_small_labels.isChecked()
        threshold_fraction = self.spin_threshold.value() / 100  # converti percentuale in frazione

        self.progress.setValue(0)

        self.thread = OrganizerThread(
            root_dir=self.root_dir,
            split_resolution=split_resolution,
            delete_orphan_images=self.cb_delete_orphan_images.isChecked(),
            delete_orphan_labels=self.cb_delete_orphan_labels.isChecked(),
            resize_dims=resize_dims,
            do_resize=do_resize,
            move_small_labels=move_small_labels,
            threshold=threshold_fraction
        )

        self.thread.progress_signal.connect(self.progress.setValue)
        self.thread.finished_signal.connect(self.on_finished)
        self.thread.start()

    def on_finished(self, results):
        summary = ""
        for split, res in results.items():
            summary += f"\n=== {split.upper()} ===\n"
            for k, v in res.items():
                summary += f"{k}: immagini={v['images']}, label={v['labels']}\n"

        QMessageBox.information(self, "Completato", summary)


# ================= RUN =================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = DatasetOrganizer()
    w.show()
    sys.exit(app.exec_())
