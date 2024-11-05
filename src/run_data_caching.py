import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from collections import deque
import shutil

from data import DataGenerator


def cache_data() -> None:
    dg = DataGenerator(memory_cache=False, disk_cache=True)
    
    # Remove old cache to force recaching
    print('Removing cache...')
    shutil.rmtree(dg.cache_directory, ignore_errors=True)
    
    # Remove randomise and train/validation split so that it is easier to track
    # progress with verbose=1
    dg.train_list = dg.get_patient_list(dg.train_directory)
    print('Saving new cache...')
    
    deque(dg.train_generator(augment=False, verbose=1), maxlen=0)
    deque(dg.validation_generator(verbose=1), maxlen=0)
    deque(dg.test_generator(verbose=1), maxlen=0)


if __name__ == '__main__':
    cache_data()
    