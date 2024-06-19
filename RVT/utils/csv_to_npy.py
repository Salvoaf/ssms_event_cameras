import pandas as pd
import numpy as np

# Percorso del file CSV
csv_file_path = '/home/salvatore/ssms_event_cameras/poses/predictions.csv'

# Carica i dati dal file CSV
csv_data = pd.read_csv(csv_file_path)

# Rinomina le colonne per farle corrispondere alla struttura dell'array numpy
csv_data = csv_data.rename(columns={
    't': 'ts',
    'class_confidence': 'confidence'
})

# Definisce il formato desiderato (uguale al primo formato)
desired_dtype = np.dtype([
    ('ts', '<u8'), 
    ('x', '<f4'), 
    ('y', '<f4'), 
    ('w', '<f4'), 
    ('h', '<f4'), 
    ('class_id', 'u1'), 
    ('confidence', '<f4'), 
    ('track_id', '<u4')
])

# Converti il DataFrame in un array strutturato numpy con il formato desiderato
structured_array = np.zeros(csv_data.shape[0], dtype=desired_dtype)

# Copia i dati dal DataFrame all'array strutturato con conversione dei tipi
structured_array['ts'] = csv_data['ts'].astype('<u8')
structured_array['x'] = csv_data['x'].astype('<f4')
structured_array['y'] = csv_data['y'].astype('<f4')
structured_array['w'] = csv_data['w'].astype('<f4')
structured_array['h'] = csv_data['h'].astype('<f4')
structured_array['class_id'] = csv_data['class_id'].astype('u1')
structured_array['confidence'] = csv_data['confidence'].astype('<f4')
structured_array['track_id'] = csv_data['track_id'].astype('<u4')

# Percorso del file di output
output_file_path = '/home/salvatore/ssms_event_cameras/dataset/val/dynamic_translation.npy'

# Salva l'array strutturato come file .npy
np.save(output_file_path, structured_array)

print(f"File salvato con successo in {output_file_path}")
