import shutil

from data import DataGenerator


def cache_data() -> None:
    dg = DataGenerator()
    
    # Remove old cache to force recaching
    print('Removing cache...')
    shutil.rmtree(dg.cache_directory, ignore_errors=True)
    
    print('Saving new cache...')
    # TODO: make verbose
    for i in dg.train_generator():
        continue
    for j in dg.validation_generator():
        continue
    

if __name__ == '__main__':
    cache_data()
    