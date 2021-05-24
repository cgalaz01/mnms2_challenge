import shutil

from data import DataGenerator


def cache_data() -> None:
    dg = DataGenerator()
    
    # Remove old cache to force recaching
    print('Removing cache...')
    shutil.rmtree(dg.cache_directory)
    
    print('Saving new cache...')
    for i in dg.train_generator():
        if i % 2 == 0:
            print(i // 2)
    i = i // 2
    for j in dg.validation_generator():
        if j % 2 == 0:
            print(j // 2 + i)
    

if __name__ == '__main__':
    cache_data()
    