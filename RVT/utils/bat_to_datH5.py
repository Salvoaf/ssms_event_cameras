import rosbag
import tqdm
import numpy as np
from tqdm import tqdm
import datetime
import os
import sys
import h5py

def bag_to_txt(bag_file, txt_file, topic='/dvs/events'):
    # Apri il file bag
    bag = rosbag.Bag(bag_file, 'r')
    
    # Lista per contenere gli eventi
    events = []
    min_time = 0
    max_time = 0  # Variabile per memorizzare l'ultimo timestamp
    
    # Conta il numero totale di eventi
    total_events = bag.get_message_count(topic_filters=topic)

    # Leggi gli eventi dal topic
    with tqdm(total=total_events) as pbar:
        for topic, msg, t in bag.read_messages(topics=[topic]):
            for event in msg.events:
                if len(events) == 0:
                    min_time = event.ts.to_nsec()
                current_time = event.ts.to_nsec()
                events.append([int((current_time - min_time) / 1e3), event.x, event.y, 0 if event.polarity == False else 1])
                max_time = current_time  # Aggiorna l'ultimo timestamp
                pbar.update(1)  # Aggiorna la barra di avanzamento
    
    # Scrivi gli eventi nel file di testo
    with open(txt_file, 'w') as txt:
        # Scrivi l'intestazione
        txt.write("t\tx\ty\tp\n")
        
        # Scrivi i dati
        for event in events:
            timestamp, x, y, polarity = event
            txt.write(f"{timestamp}\t{x}\t{y}\t{polarity}\n")
    
    # Stampa il primo e l'ultimo timestamp
   
    print(f"Ultimo timestamp: {int((max_time-min_time)/ 1e3)} ns")
    print(f"Converted {len(events)} events from {bag_file} to {txt_file}")
    return int((max_time-min_time)/ 1e3)
EV_TYPE = [('t', 'u4'), ('_', 'i4')]  # Event2D

def write_header(f, height, width, ev_type=0):
    header = (
        f"% Data file containing Event2D events.\n"
        f"% Version 2\n"
        f"% Date 2024-06-12 00:00:00\n"
        f"% Height {height}\n"
        f"% Width {width}\n"
    )
    f.write(header.encode('latin-1'))
    ev_size = np.array([ev_type, 8], dtype=np.uint8)
    ev_size.tofile(f)

def write_event_buffer(f, buffers):
    dtype = EV_TYPE
    data_to_write = np.empty(len(buffers['t']), dtype=dtype)

    x = buffers['x'].astype('i4')
    y = np.left_shift(buffers['y'].astype('i4'), 14)
    p = np.left_shift(buffers['p'].astype('i4'), 28)

    data_to_write['_'] = x + y + p
    data_to_write['t'] = buffers['t']
    
    data_to_write.tofile(f)
    f.flush()

def convert_txt_to_dat(txt_file, dat_file, height, width):
    # Count the total number of lines in the text file
    total_lines = sum(1 for line in open(txt_file)) - 1  # Subtract 1 for the header

    # Load data with progress bar
    data = np.loadtxt(txt_file, dtype=int, delimiter='\t', skiprows=1)
    
    with open(dat_file, 'wb') as f:
        write_header(f, height, width)
        
        buffers = np.zeros(len(data), dtype=[('t', 'u4'), ('x', 'i2'), ('y', 'i2'), ('p', 'i2')])
        buffers['t'] = data[:, 0]
        buffers['x'] = data[:, 1]
        buffers['y'] = data[:, 2]
        buffers['p'] = data[:, 3]
        
        for i in tqdm(range(0, len(buffers), 100)):
            chunk = buffers[i:i+100]
            write_event_buffer(f, chunk)


EV_TYPE = [("t", "u4"), ("_", "i4")]  # Event2D
EV_STRING = "Event2D"

def load_td_data(filename, ev_count=-1, ev_start=0):
    with open(filename, "rb") as f:
        _, ev_type, ev_size, _ = parse_header(f)
        if ev_start > 0:
            f.seek(ev_start * ev_size, 1)
        dtype = EV_TYPE
        dat = np.fromfile(f, dtype=dtype, count=ev_count)
        xyp = None
        if ("_", "i4") in dtype:
            x = np.bitwise_and(dat["_"], 16383)
            y = np.right_shift(np.bitwise_and(dat["_"], 268419072), 14)
            p = np.right_shift(np.bitwise_and(dat["_"], 268435456), 28)
            xyp = (x, y, p)
        return _dat_transfer(dat, dtype, xyp=xyp)

def _dat_transfer(dat, dtype, xyp=None):
    variables = []
    xyp_index = -1
    for i, (name, _) in enumerate(dtype):
        if name == "_":
            xyp_index = i
            continue
        variables.append((name, dat[name]))
    if xyp and xyp_index == -1:
        print("Error dat didn't contain a '_' field !")
        return None
    if xyp_index >= 0:
        dtype = (
            dtype[:xyp_index]
            + [("x", "i2"), ("y", "i2"), ("p", "i2")]
            + dtype[xyp_index + 1 :]
        )
    new_dat = np.empty(dat.shape[0], dtype=dtype)
    if xyp:
        new_dat["x"] = xyp[0].astype(np.uint16)
        new_dat["y"] = xyp[1].astype(np.uint16)
        new_dat["p"] = xyp[2].astype(np.uint16)
    for name, arr in variables:
        new_dat[name] = arr
    return new_dat

def parse_header(f):
    f.seek(0, os.SEEK_SET)
    bod = None
    end_of_header = False
    header = []
    num_comment_line = 0
    size = [None, None]
    while not end_of_header:
        bod = f.tell()
        line = f.readline()
        if sys.version_info > (3, 0):
            first_item = line.decode("latin-1")[:2]
        else:
            first_item = line[:2]
        if first_item != "% ":
            end_of_header = True
        else:
            words = line.split()
            if len(words) > 1:
                if words[1] == "Date":
                    header += ["Date", words[2] + " " + words[3]]
                if words[1] == "Height" or words[1] == b"Height":
                    size[0] = int(words[2])
                    header += ["Height", words[2]]
                if words[1] == "Width" or words[1] == b"Width":
                    size[1] = int(words[2])
                    header += ["Width", words[2]]
            else:
                header += words[1:3]
            num_comment_line += 1
    f.seek(bod, os.SEEK_SET)
    if num_comment_line > 0:
        ev_type = np.frombuffer(f.read(1), dtype=np.uint8)[0]
        ev_size = np.frombuffer(f.read(1), dtype=np.uint8)[0]
    else:
        ev_type = 0
        ev_size = sum([int(n[-1]) for _, n in EV_TYPE])
    bod = f.tell()
    return bod, ev_type, ev_size, size

# Funzione per convertire il file .dat in .dat.h5
def convert_dat_to_h5(dat_file_path, h5_file_path, height, width):
    try:
        # Carica tutti i dati dal file .dat
        events = load_td_data(dat_file_path)
        if events is None:
            print("Errore durante il trasferimento dei dati")
            return

        # Verifica che i dati contengano tutti i campi necessari
        required_fields = ['t', 'x', 'y', 'p']
        for field in required_fields:
            if field not in events.dtype.names:
                raise ValueError(f"Il campo '{field}' non è presente nei dati caricati")

        # Crea un nuovo file HDF5
        with h5py.File(h5_file_path, 'w') as h5_file:
            # Crea dataset per ogni campo
            events_group = h5_file.create_group('events')
            events_group.create_dataset('t', data=events['t'])
            events_group.create_dataset('x', data=events['x'])
            events_group.create_dataset('y', data=events['y'])
            events_group.create_dataset('p', data=events['p'])
            events_group.attrs['height'] = height
            events_group.attrs['width'] = width
        
        if os.path.exists(h5_file_path):
            print(f"Conversione completata con successo: {h5_file_path}")
        else:
            print(f"Errore: il file {h5_file_path} non è stato creato correttamente.")

    except Exception as e:
        print(f"Errore durante la conversione: {e}")

def create_npy(npy_file, max_ts, jump,height, width):
    initial_timestamp = 13599999
    num_rows = int((max_ts - initial_timestamp)/jump)

    #num_rows = 480
    # Genera i timestamp
    new_timestamps = np.array([initial_timestamp + i * jump for i in range(num_rows)], dtype=np.uint64)

    # Genera dati randomici per gli altri campi
    x_values = np.random.randint(0, width, size=num_rows, dtype=np.int32).astype(np.float32)
    y_values = np.random.randint(0, height, size=num_rows, dtype=np.int32).astype(np.float32)
    w_values = np.random.randint(2, 61, size=num_rows, dtype=np.int32).astype(np.float32)
    h_values = np.random.randint(3, 71, size=num_rows, dtype=np.int32).astype(np.float32)
    class_id_values = np.random.randint(0, 2, size=num_rows, dtype=np.uint8)
    confidence_values = np.ones(num_rows, dtype=np.float32)
    track_id_values = np.random.randint(0, 2, size=num_rows, dtype=np.uint32)

    # Crea il nuovo array strutturato
    new_data = np.zeros(num_rows, dtype=[('ts', '<u8'), ('x', '<f4'), ('y', '<f4'), ('w', '<f4'), ('h', '<f4'), ('class_id', 'u1'), ('confidence', '<f4'), ('track_id', '<u4')])
    new_data['ts'] = new_timestamps
    new_data['x'] = x_values
    new_data['y'] = y_values
    new_data['w'] = w_values
    new_data['h'] = h_values
    new_data['class_id'] = class_id_values
    new_data['confidence'] = confidence_values
    new_data['track_id'] = track_id_values

    # Salva il nuovo file
    np.save(npy_file, new_data)
# Esempio di utilizzo
bag_file = '/home/salvatore/uslam_ws/src/rpg_ultimate_slam_open/data/dynamic_translation.bag'
txt_file = '/home/salvatore/ssms_event_cameras/dataset/dynamic_translation.txt'
dat_file = '/home/salvatore/ssms_event_cameras/dataset/ynamic_translation.dat'
# Percorsi dei file .dat e .dat.h5
dat_file = '/home/salvatore/ssms_event_cameras/dataset/dynamic_translation.dat'
h5_file = '/home/salvatore/ssms_event_cameras/dataset/val/dynamic_translationtd.dat.h5'
npy_file = '/home/salvatore/ssms_event_cameras/dataset/val/dynamic_translation.npy'
height, width = 180, 240


max_ts= bag_to_txt(bag_file, txt_file)
convert_txt_to_dat(txt_file, dat_file, height, width)
os.remove(txt_file)
convert_dat_to_h5(dat_file, h5_file, height, width)
os.remove(dat_file)
create_npy(npy_file, max_ts, 5000,height, width)

