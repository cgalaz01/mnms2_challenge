import shutil

from data import DataGenerator


def cache_data() -> None:
    dg = DataGenerator()
    
    # Remove old cache to force recaching
    print('Removing cache...')
    shutil.rmtree(dg.cache_directory, ignore_errors=True)
    
    # Remove randomise and train/validation split so that it is easier to track
    # progress with verbose=1
    dg.train_list = dg.get_patient_list(dg.train_directory)
    
    print('Saving new cache...')
    for i in dg.train_generator(verbose=1):
        continue
    

if __name__ == '__main__':
    cache_data()
    