import mss
import win32gui
import win32ui
import win32con
import win32api
import numpy as np
import time
import cv2
#import dxcam
import bettercam
import ctypes
from threading import Thread, Lock

#################################################cpu engine##################################################

class ScreenCaptureMSS:
    def __init__(self,screenshotdim):
        # Get screen resolution dynamically
        self.screen_width = int((win32api.GetSystemMetrics(0) / 2)-(screenshotdim/2)) # 0 corresponds to SM_CXSCREEN
        self.screen_height =int((win32api.GetSystemMetrics(1) / 2)-(screenshotdim/2)) # 1 corresponds to SM_CYSCREEN
        #print(self.screen_width,self.screen_height)

        self.width = screenshotdim
        self.height = screenshotdim

        # Initialize mss instance and monitor settings
        self.sct = mss.mss()
        self.monitor = {"top": self.screen_height , "left": self.screen_width , "width": self.width, "height": self.height}

    def capture(self):
        img = self.sct.grab(self.monitor)
        #img = img[:, :, :3]
        return img

    def display_image(self, img):
        cv2.imshow("Captured Image", img)
        cv2.waitKey(1)



class ScreenCaptureMSSthreading:
    def __init__(self, screenshotdim):
        # Calcola la posizione per catturare il centro dello schermo
        self.screen_width = int((win32api.GetSystemMetrics(0) / 2) - (screenshotdim / 2))
        self.screen_height = int((win32api.GetSystemMetrics(1) / 2) - (screenshotdim / 2))

        self.width = screenshotdim
        self.height = screenshotdim

        # Inizializza la cattura con mss
        self.sct = mss.mss()
        self.monitor = {
            "top": self.screen_height,
            "left": self.screen_width,
            "width": self.width,
            "height": self.height
        }

        # Variabili per il threading
        self.frame_lock = Lock()
        self.latest_frame = None
        self.running = True

        # Avvia il thread di cattura
        self.capture_thread = Thread(target=self.capture_screen, daemon=True)
        self.capture_thread.start()

    def capture_screen(self):
        self.sct = mss.mss()
        """Thread che cattura continuamente lo schermo."""
        while self.running:
            try:
                screen_image = np.array(self.sct.grab(self.monitor))
                screen_image = screen_image[:, :, :3]  # Rimuove il canale alfa

                with self.frame_lock:
                    self.latest_frame = screen_image  # Salva il frame catturato
            except Exception as e:
                print(f"Errore nella cattura dello schermo: {e}")

    def get_latest_frame(self):
        """Restituisce l'ultimo frame catturato in modo sicuro."""
        with self.frame_lock:
            return None if self.latest_frame is None else self.latest_frame.copy()

    def display_image(self):
        """Mostra il frame catturato in una finestra OpenCV in tempo reale."""
        while True:
            frame = self.get_latest_frame()
            if frame is not None:
                cv2.imshow("Captured Image", frame)

            # Chiudi la finestra premendo 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.stop()
                break

        cv2.destroyAllWindows()

    def stop(self):
        """Ferma il thread di cattura."""
        self.running = False
        self.capture_thread.join()



class ScreenCaptureGDI:
    def __init__(self, width, height):
        # Get screen resolution dynamically
        self.screen_width = int((win32api.GetSystemMetrics(0) / 2)-(screenshotdim/2)) # 0 corresponds to SM_CXSCREEN
        self.screen_height =int((win32api.GetSystemMetrics(1) / 2)-(screenshotdim/2)) # 1 corresponds to SM_CYSCREEN
        

        self.width = width
        self.height = height
        self.left = self.screen_width 
        self.top = self.screen_height 
        print(self.left,self.top)
        self.cWidth = self.width / 2
        self.cHeight = self.height / 2


        # Create device contexts and bitmap outside of the capture method
        self.hdesktop = win32gui.GetDesktopWindow()
        self.desktop_dc = win32gui.GetWindowDC(self.hdesktop)
        self.img_dc = win32ui.CreateDCFromHandle(self.desktop_dc)
        self.mem_dc = self.img_dc.CreateCompatibleDC()
        self.bitmap = win32ui.CreateBitmap()
        self.bitmap.CreateCompatibleBitmap(self.img_dc, self.width, self.height)
        self.mem_dc.SelectObject(self.bitmap)


    def capture(self):

        # Capture the specified region
        self.mem_dc.BitBlt((0, 0), (self.width, self.height), self.img_dc, (self.left, self.top), win32con.SRCCOPY)

        # Convert the captured bitmap data to a NumPy array
        img = np.frombuffer(self.bitmap.GetBitmapBits(True), dtype=np.uint8)

        # Reshape the NumPy array to a 3-dimensional image
        img.shape = (self.height, self.width, 4)
        return img

    def display_image(self, img):
        cv2.imshow("Captured Image", img)
        cv2.waitKey(1)

#####################################################gpu engine####################################################

class ScreenCaptureDXCam:
    def __init__(self, screenshotdim):
        # Get screen resolution dynamically
        self.screen_width = int((win32api.GetSystemMetrics(0) / 2)-(screenshotdim/2)) # 0 corresponds to SM_CXSCREEN
        self.screen_height = int((win32api.GetSystemMetrics(1) / 2)-(screenshotdim/2)) # 1 corresponds to SM_CYSCREEN
        self.width = screenshotdim
        self.height = screenshotdim

        self.left = self.screen_width
        self.top = self.screen_height
        self.right = self.left + screenshotdim
        self.bottom = self.top + screenshotdim
        self.region = (self.left, self.top, self.right, self.bottom)

        self.region = (self.left, self.top, self.right, self.bottom)
        self.cam=dxcam.create(output_idx=0, output_color="BGR")
        self.cam.start(region=self.region,video_mode=True,target_fps=0)


    def capture(self):
        # Capture frame using dxcam
        img = self.cam.get_latest_frame()
        return img

    def display_image(self, img):
        cv2.imshow("Captured Image", img)
        cv2.waitKey(1)



class ScreenCaptureBETTERCAM:
    def __init__(self, screenshotdim):
        # Get screen resolution dynamically
        self.screen_width = int((win32api.GetSystemMetrics(0) / 2)-(screenshotdim/2)) # 0 corresponds to SM_CXSCREEN
        self.screen_height = int((win32api.GetSystemMetrics(1) / 2)-(screenshotdim/2)) # 1 corresponds to SM_CYSCREEN
        self.width = screenshotdim
        self.height = screenshotdim

        self.left = self.screen_width
        self.top = self.screen_height
        self.right = self.left + screenshotdim
        self.bottom = self.top + screenshotdim
        self.region = (self.left, self.top, self.right, self.bottom)

        self.region = (self.left, self.top, self.right, self.bottom)
        self.cam = bettercam.create(output_idx=0, output_color="BGR")
        #self.cam.start(region=self.region,video_mode=True,target_fps=0)


    def capture(self):
        # Capture frame using dxcam
        #img = self.cam.get_latest_frame()
        img = self.cam.grab(region=self.region)
        return img

    def display_image(self, img):
        cv2.imshow("Captured Image", img)
        cv2.waitKey(1)


class ScreenCaptureDXGI:
    def __init__(self, screenshot_dim):
        # Load the custom DirectX screen capture DLL
        self.dxgx_dll = ctypes.CDLL(r'C:\Users\Admin\Desktop\ghg\old backup\Pyt\dxgi_shot-main\x64\Release\dxgx.dll')

        # Define the argument and return types of the DLL functions
        self.dxgx_dll.create.restype = ctypes.c_void_p
        self.dxgx_dll.init.argtypes = [ctypes.c_void_p]
        self.dxgx_dll.init.restype = ctypes.c_bool
        self.dxgx_dll.shot.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.POINTER(ctypes.c_ubyte)]
        self.dxgx_dll.shot.restype = ctypes.POINTER(ctypes.c_ubyte)
        self.dxgx_dll.destroy.argtypes = [ctypes.c_void_p]

        # Create an instance of DXGIDuplicator
        self.duplicator = self.dxgx_dll.create()

        # Initialize the duplicator
        if not self.dxgx_dll.init(self.duplicator):
            raise RuntimeError("Failed to initialize DXGIDuplicator")

        # Get the center region of the screen
        user32 = ctypes.windll.user32
        screen_width = user32.GetSystemMetrics(0)
        screen_height = user32.GetSystemMetrics(1)
        self.width = screenshot_dim
        self.height = screenshot_dim
        self.x = (screen_width - self.width) // 2
        self.y = (screen_height - self.height) // 2

        self.buffer_size = self.width * self.height * 4  # Assuming 4 bytes per pixel (RGBA)
        self.buffer = (ctypes.c_ubyte * self.buffer_size)()

    def capture(self):
        # Capture the specified region using the DLL
        self.dxgx_dll.shot(self.duplicator, self.x, self.y, self.width, self.height, self.buffer)
        image_data = np.ctypeslib.as_array(self.buffer)
        image_data = image_data.reshape((self.height, self.width, 4))
        return image_data[:, :, :3]  # Convert from RGBA to RGB

    def display_image(self, img):
        # Display the image using OpenCV
        cv2.imshow("Captured Image", img)
        cv2.waitKey(1)

    def __del__(self):
        # Clean up the duplicator resources
        if self.duplicator:
            self.dxgx_dll.destroy(self.duplicator)
            print("DXGI Duplicator resources released.")




##exeple normal capture
# Create ScreenCaptureMSS instance with desired width and height
screenshotdim=320
#screen_capture = ScreenCaptureMSS(screenshotdim)
# or for dxacm 
#screen_capture = ScreenCaptureDXCam(screenshotdim)
#or bettercam
#screen_capture = ScreenCaptureBETTERCAM(screenshotdim)
#or custom DXGI
#screen_capture = ScreenCaptureDXGI (screenshotdim)
#or for GDI
#screen_capture = ScreenCaptureGDI(screenshotdim, screenshotdim)

#while True:
#    img = screen_capture.capture()
#    screen_capture.display_image(img)


#exemple multitrhed capture mss
#screen_capture = ScreenCaptureMSSthreading(screenshotdim)
#screen_capture.display_image()
#frame = screen_capture.get_latest_frame()
#while True:
#    frame = screen_capture.get_latest_frame()
#    if frame is not None:
#        cv2.imshow("Captured Image", frame)

        # Chiudi la finestra premendo 'q'
#        if cv2.waitKey(1):
#            self.stop()
#            break
#screen_capture.stop()

