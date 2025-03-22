import os
import re
import time
import requests
import zipfile
import shutil
import win32com.client
from colorama import Fore, init

init(autoreset=True)

ARDUINO_CLI_ZIP_URL = "https://downloads.arduino.cc/arduino-cli/arduino-cli_latest_Windows_64bit.zip"
ARDUINO_CLI_FILENAME = "arduino-cli.exe"
SKETCH_FILE = r"C:\Users\ghjgu\Desktop\old backup\Python\Dexter-v8\mouse\arduino\arduino_hostshield\usb_hid.ino"
BOARDS_TXT_LOCATION = os.path.expandvars("%LOCALAPPDATA%/Arduino15/packages/arduino/hardware/avr/1.8.6/boards.txt")


def download_and_extract_file(url, filename):
    print(Fore.GREEN + f"Downloading {filename}...")
    response = requests.get(url, stream=True)
    with open(filename, "wb") as fd:
        for chunk in response.iter_content(chunk_size=128):
            fd.write(chunk)
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall("./")
    print(Fore.GREEN + f"{filename} downloaded successfully.")


def update_boards_txt(vid, pid):
    pattern_to_replace = {
        "leonardo.build.vid=": vid,
        "leonardo.build.pid=": pid,
        ".vid=": vid,
        ".pid=": pid
    }

    with open(BOARDS_TXT_LOCATION, 'r') as file:
        data = file.readlines()

    for idx, line in enumerate(data):
        for pattern, replacement in pattern_to_replace.items():
            if pattern in line:
                prefix = line.split(pattern)[0]
                data[idx] = f"{prefix}{pattern}{replacement}\n"

    with open(BOARDS_TXT_LOCATION, 'w') as file:
        file.writelines(data)


def cleanup_files(files):
    print(Fore.YELLOW + "Cleaning up files...")

    for file in files:
        try:
            os.remove(file)
            print(Fore.GREEN + f"Deleted {file}")
        except Exception as e:
            print(Fore.RED + f"Failed to delete {file}. Error: {str(e)}")



def list_mice_devices():
    wmi = win32com.client.GetObject("winmgmts:")
    devices = wmi.InstancesOf("Win32_PointingDevice")
    return [(device.Name, *re.search(r'VID_(\w+)&PID_(\w+)', device.PNPDeviceID).groups()) for device in devices]


def user_select_mouse(mice):
    print(Fore.CYAN + "\nDetecting mice devices...")

    if not mice:
        print(Fore.RED + "No mice detected.")
        return None

    # Automatically select the first mouse
    selected_mouse = mice[0][1:]
    print(f"{Fore.CYAN}Selected mouse automatically - VID: {selected_mouse[0] or 'Not found'}, PID: {selected_mouse[1] or 'Not found'}")
    return selected_mouse


def execute_cli_command(command):
    os.system(f"{ARDUINO_CLI_FILENAME} {command} >NUL 2>&1")

def find_arduino_port():
    arduino_port = None
    available_ports = list(list_ports.comports())
    for port in available_ports:
        
        if "USB Serial Device" in port.description or "Arduino" in port.description:
            arduino_port = port.device
            break

    return arduino_port

def main():
    download_and_extract_file(ARDUINO_CLI_ZIP_URL, "arduino-cli.zip")
    execute_cli_command("core install arduino:avr@1.8.6")
    execute_cli_command("lib install Mouse")
    vid, pid = user_select_mouse(list_mice_devices())
    update_boards_txt("0x" + vid, "0x" + pid)
    execute_cli_command(f"compile --fqbn arduino:avr:leonardo --input-file SKETCH_FILE")
    com_port = find_arduino_port
    execute_cli_command(f"upload--p {com_port} --fqbn arduino:avr:leonardo --input-file SKETCH_FILE")
    #cleanup_files(["arduino-cli.exe", "arduino-cli.zip"])


if __name__ == '__main__':
    try:
        main()
        print(f"spoof succes")
    except Exception as e:
        print(f"An error occurred: {e}")
