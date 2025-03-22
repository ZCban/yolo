import os
import time
import threading
import usb.core
import usb.util

class KeyMouseSimulation():
    def __init__(self):
        # Trova il dispositivo USB con gli ID specificati
        self.dev = usb.core.find(idVendor=0x99ba, idProduct=0x7250)

        # Assicurati che il dispositivo sia stato trovato
        if self.dev is None:
            raise ValueError("Device not found")

        # Configura il dispositivo
        self.dev.set_configuration()


if __name__ == "__main__":
    key_mouse_simulation = KeyMouseSimulation()





