import serial
import win32api
import win32con
import time
from serial.tools import list_ports
from colorama import Fore, Style

class MouseEmulator:
    def __init__(self):
        self.arduino_port = self.find_arduino_port()
        if self.arduino_port:
            self.arduino = serial.Serial(self.arduino_port, 115200)
            print(f"{Fore.CYAN}Arduino trovato sulla porta {Fore.YELLOW}{self.arduino_port}{Fore.CYAN}{Style.RESET_ALL}")
        else:
            print(f"{Fore.RED}Collega l'Arduino al PC{Style.RESET_ALL}")

    def find_arduino_port(self):
        arduino_port = None
        available_ports = list(list_ports.comports())
        for port in available_ports:
            #print(f"{Fore.CYAN}Port: {Fore.MAGENTA}{port.device}{Fore.CYAN}, Description: {Fore.MAGENTA}{port.description}{Style.RESET_ALL}")

            if "USB Serial Device" in port.description or "Arduino" in port.description:
                arduino_port = port.device
                break

        return arduino_port

    def posR(self):
        screen_width, screen_height = self.get_screen_size()
        x1, y1 = win32api.GetCursorPos()[0], win32api.GetCursorPos()[1]
        x = -(screen_width / 2 - x1) if x1 < screen_width / 2 else x1 - screen_width / 2
        y = -(screen_height / 2 - y1) if y1 < screen_height / 2 else y1 - screen_height / 2
        return int(x), int(y)

    def posA(self):
        x1 = win32api.GetCursorPos()[0]
        y1 = win32api.GetCursorPos()[1]
        return int(x1), int(y1)

    def mouse_move(self, x, y):
        x = int(x/4.5) + 127
        y = int(y/4.5) + 127

        if x > 254:
            x = 254
        if x < 0:
            x = 0
        if y > 254:
            y = 254
        if y < 0:
            y = 0

        packet = bytearray([0x7E, x, y, 0x7E])
        self.arduino.write(packet)


    def mouse_click(self):
        self.arduino.write(b'C')




# Creare un'istanza della classe MouseEmulator
#A = MouseEmulator()

#A.mouse_move(110, 100)
#print(A.posA())#absolute
#print(A.posR())#relative
#time.sleep(1000)
