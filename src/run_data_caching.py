import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

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
    
    for i in dg.train_generator(augment=False, verbose=1):
        continue
    for i in dg.test_generator(verbose=1):
        continue


if __name__ == '__main__':
    cache_data()
    