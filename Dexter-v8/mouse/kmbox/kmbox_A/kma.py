import ctypes
from struct import *
import os
import time
import win32api
import win32con
import threading

#kmboxA = ctypes.cdll.LoadLibrary(r"C:\Users\ghjgu\Desktop\old backup\Python\Dexter-v8\mouse\kmbox\kmbox_A\kmbox_dll_64bit.dll")
script_directory = os.path.dirname(os.path.abspath(__file__))
dll_path = os.path.join(script_directory, "kmbox_dll_64bit.dll")

class KeyMouseSimulation():
    kmboxA = ctypes.cdll.LoadLibrary(dll_path)
    kmboxA.KM_init.argtypes = [ctypes.c_ushort, ctypes.c_ushort]
    kmboxA.KM_init.restype = ctypes.c_ushort
    kmboxA.KM_move.argtypes = [ctypes.c_short, ctypes.c_short]
    kmboxA.KM_move.restype = ctypes.c_int
    ts=kmboxA.KM_init(ctypes.c_ushort(0X99BA), ctypes.c_ushort(0xABCD))
    print("Kmbox collegato:{}".format(ts))

 
    def perss(self,vk_key:int):
        KeyMouseSimulation.kmboxA.KM_press(ctypes.c_char(vk_key))

    
    def down(self,vk_key:int):
        KeyMouseSimulation.kmboxA.KM_down(ctypes.c_char(vk_key))

    
    def up(self,vk_key:int):
        KeyMouseSimulation.kmboxA.KM_up(ctypes.c_char(vk_key))

    
    def left(self,vk_key:int):
        KeyMouseSimulation.kmboxA.KM_left(ctypes.c_char(vk_key))

    
    def right(self,vk_key:int):
        KeyMouseSimulation.kmboxA.KM_right(ctypes.c_char(vk_key))

    
    def middle(self,vk_key:int):
        KeyMouseSimulation.kmboxA.KM_middle(ctypes.c_char(vk_key))

    
    def move(self,short_x:int,short_y:int):
        KeyMouseSimulation.kmboxA.KM_move(short_x,short_y)


    def click(self):
        self.left(1)
        self.left(0)


    def get_screen_size(self):
        width = win32api.GetSystemMetrics(win32con.SM_CXSCREEN)
        height = win32api.GetSystemMetrics(win32con.SM_CYSCREEN)
        return width, height

    def posA(self):
        x1=win32api.GetCursorPos()[0]
        y1=win32api.GetCursorPos()[1]      
        return int(x1), int(y1)
    
    def posR(self):
        screen_width, screen_height = self.get_screen_size()
        x1,y1 = win32api.GetCursorPos()[0],win32api.GetCursorPos()[1]
        x = -(screen_width / 2 - x1) if x1 < screen_width / 2 else x1 - screen_width / 2
        y = -(screen_height / 2 - y1) if y1 < screen_height / 2 else y1 - screen_height / 2
        return int(x), int(y)

    def clickTo(self, short_x, short_y):
        short_x1 = short_x // int(1.6)
        short_y1 = short_y // int(1.6)
        self.move(short_x1, short_y1)
        if -2 <= short_x1 <= 2 and -1 <= short_y1 <= 0:
            self.left(1)
            start_time = time.time()
            while time.time() - start_time < 0.1:
                self.left(0)


    def clickTo2(self,short_x:int,short_y:int):
        short_x1 = short_x // int(1.6)
        short_y1 = short_y // int(1.6)
        self.move(short_x1, short_y1)
        if -2 <= short_x1 <= 2 and -1 <= short_y1 <= 0:
            start_time = time.time()
            self.left(1)
            while time.time() - start_time < 0.005:
                self.left(0)

                

    def clickTo3(self,short_x:int,short_y:int):
        if win32api.GetKeyState(0x05)<0 or win32api.GetKeyState(0x14):
            t = threading.Thread(target=self.clickTo(short_x,short_y))
            t.start()




            






# 本页面直接运行简单例子：
# a=KeyMouseSimulation() # chiamata in questo file
# a.move(100, 100)
    
# 其他页面调用运行简单例子：
# import kma
 #a=kma.KeyMouseSimulation() # quando lo richiami in altri file
# a.move(100, 100)

#self.down(0xE1)
#self.up(0xE1)
