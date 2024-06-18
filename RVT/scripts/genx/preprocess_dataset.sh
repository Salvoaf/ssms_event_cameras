NUM_PROCESSES=20  # set to the number of parallel processes to use
DATA_DIR=/media/salvatore/gen11/mydt
DEST_DIR=/home/salvatore/ssms_event_cameras/RVT/gen1_frequenciesgen1_200hz/
FREQUENCY=/home/salvatore/ssms_event_cameras/RVT/scripts/genx/conf_preprocess/extraction/frequencies/const_duration_200hz.yaml

python3 preprocess_dataset.py ${DATA_DIR} ${DEST_DIR} conf_preprocess/representation/stacked_hist.yaml ${FREQUENCY} \
conf_preprocess/filter_gen1.yaml -ds gen1 -np ${NUM_PROCESSES}


