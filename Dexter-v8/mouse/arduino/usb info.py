import pywinusb.hid as hid
import time
import struct

# Funzione per convertire un valore short in una sequenza di byte
def short_to_bytes(val):
    return struct.pack('<h', val)

# Trova il dispositivo USB basato su VID e PID
vid = 0x99BA  # Sostituisci con il VID del tuo Arduino
pid = 0x7250  # Sostituisci con il PID del tuo Arduino
report_id = 0
# Cerca il dispositivo
all_devices = hid.HidDeviceFilter(vendor_id=vid, product_id=pid).get_devices()
if not all_devices:
    print("Dispositivo non trovato.")
else:
    device = all_devices[0]
    device.open()
    print("Dispositivo trovato.", device)

    # Aggiungi un ritardo iniziale per consentire ad Arduino di inizializzarsi
    time.sleep(2)
    
    # Invia il comando KM_move(short_x, short_y)
    short_x = 100
    short_y = 50
    # Ensure that the report size matches the expected size
    expected_report_size = 65
    comando = b'KM_move' + short_to_bytes(short_x) + short_to_bytes(short_y)
    comando += b'\0' * (expected_report_size - len(comando))  # Pad with null bytes if needed
    # Invia il comando al dispositivo USB
    report = device.find_output_reports()[0]
    report.set_raw_data(list(comando))
    report.send()


    # Chiudi la connessione
    device.close()

