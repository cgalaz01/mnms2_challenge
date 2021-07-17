import shutil

from data import DataGenerator


def cache_data() -> None:
    dg = DataGenerator(memory_cache=False)
    
    # Remove old cache to force recaching
    print('Removing cache...')
    #shutil.rmtree(dg.cache_directory, ignore_errors=True)
    
    # Remove randomise and train/validation split so that it is easier to track
    # progress with verbose=1
    dg.train_list = dg.get_patient_list(dg.train_directory)
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    print('Saving new cache...')
    for i in dg.train_affine_generator(augment=False, verbose=1):
        plt.imshow(i[0]['input_sa'][..., 3])
        width = i[0]['input_la'].shape[0] // 2 - 21 // 2
        height = i[0]['input_la'].shape[1] // 2 - 21 // 2
        plt.gca().add_patch(patches.Rectangle((width, height),21,21,linewidth=1,edgecolor='r',facecolor='none'))
        plt.show()
        plt.close()
        plt.imshow(i[0]['input_la'][..., 0])
        width = i[0]['input_la'].shape[0] // 2 - 21 // 2
        height = i[0]['input_la'].shape[1] // 2 - 21 // 2
        plt.gca().add_patch(patches.Rectangle((width, height),21,21,linewidth=1,edgecolor='r',facecolor='none'))
        print(i[0]['input_sa'].shape)
        plt.show()
        plt.close()
        continue
        

if __name__ == '__main__':
    cache_data()
    