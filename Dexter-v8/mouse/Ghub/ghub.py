import ctypes
from time import sleep
import win32file
import win32api
import ctypes.wintypes as wintypes
from ctypes import windll

# Define constants for mouse button actions
MOUSE_LEFT_BUTTON = 1
MOUSE_RIGHT_BUTTON = 2
MOUSE_MIDDLE_BUTTON = 4
MOUSE_BUTTON_4 = 8
MOUSE_BUTTON_5 = 16

handle = 0
found = False

def _DeviceIoControl(devhandle, ioctl, inbuf, inbufsiz, outbuf, outbufsiz):
    """See: DeviceIoControl function
    http://msdn.microsoft.com/en-us/library/aa363216(v=vs.85).aspx
    """
    DeviceIoControl_Fn = windll.kernel32.DeviceIoControl
    DeviceIoControl_Fn.argtypes = [wintypes.HANDLE, wintypes.DWORD, wintypes.LPVOID,wintypes.DWORD,wintypes.LPVOID,wintypes.DWORD,ctypes.POINTER(wintypes.DWORD),wintypes.LPVOID]                   
    DeviceIoControl_Fn.restype = wintypes.BOOL

    dwBytesReturned = wintypes.DWORD(0)
    lpBytesReturned = ctypes.byref(dwBytesReturned)

    status = DeviceIoControl_Fn(int(devhandle),ioctl,inbuf,inbufsiz,outbuf,outbufsiz,lpBytesReturned,None)

    return status, dwBytesReturned


class MOUSE_IO(ctypes.Structure):
    _fields_ = [
        ("button", ctypes.c_char),
        ("x", ctypes.c_int),  # Change c_char to c_int
        ("y", ctypes.c_int),  # Change c_char to c_int
        ("wheel", ctypes.c_char),
        ("unk1", ctypes.c_char),
    ]

def device_initialize(device_name: str) -> bool:
    global handle
    try:
        handle = win32file.CreateFileW(device_name, win32file.GENERIC_WRITE, 0, None, win32file.OPEN_ALWAYS, win32file.FILE_ATTRIBUTE_NORMAL, 0)
    except:
        pass
    return bool(handle)

def mouse_open() -> bool:
    global found
    global handle
    if handle: 
        return found
    if device_initialize('\\??\\ROOT#SYSTEM#0002#{1abc05c0-c378-41b9-9cef-df1aba82b015}'): #Would be best to put this in a for loop and only change the 0002
            found = True
    else:
        if device_initialize('\\??\\ROOT#SYSTEM#0001#{1abc05c0-c378-41b9-9cef-df1aba82b015}'): #Logitech CVE thank you ekknod
            found = True

    return found

def call_mouse(buffer) -> bool:
    global handle
    return _DeviceIoControl(handle, 0x2a2010, ctypes.c_void_p(ctypes.addressof(buffer)), ctypes.sizeof(buffer),  0, 0)[0] == 0


def mouse_close() -> None:
    global handle
    win32file.CloseHandle(int(handle))
    handle = 0


def is_mouse_connected(device_name: str) -> bool:
    try:
        handle = win32file.CreateFileW(device_name, win32file.GENERIC_WRITE, 0, None, win32file.OPEN_ALWAYS, win32file.FILE_ATTRIBUTE_NORMAL, 0)
        win32file.CloseHandle(handle)
        return True
    except:
        return False

def mouse_click(button, release=False):
    global handle

    io = MOUSE_IO()
    io.x = 0
    io.y = 0
    io.unk1 = 0
    io.button = button
    io.wheel = 0

    # Simulate the mouse button press
    if not release:
        if not call_mouse(io):
            mouse_close()
            mouse_open()
    else:
        io.button = 0  # Simulate the mouse button release
        if not call_mouse(io):
            mouse_close()
            mouse_open()

def move_mouse_relative(dx, dy):
    global handle

    io = MOUSE_IO()
    io.x = dx
    io.y = dy
    io.unk1 = 0
    io.button = 0
    io.wheel = 0

    if not call_mouse(io):
        mouse_close()
        mouse_open()



# Check if the mouse is connected
if is_mouse_connected('\\??\\ROOT#SYSTEM#0002#{1abc05c0-c378-41b9-9cef-df1aba82b015}') or is_mouse_connected('\\??\\ROOT#SYSTEM#0001#{1abc05c0-c378-41b9-9cef-df1aba82b015}'):
    print("G Hub Mouse is connected")
else:
    print("G Hub Mouse is not connected")

if not mouse_open():
    print("Ghub is not open or something else is wrong")
 



