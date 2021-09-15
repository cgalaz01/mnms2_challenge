from typing import Dict, Tuple, List

from collections import Counter

import numpy as np

import SimpleITK as sitk

from data import DataGenerator


def get_spacing(patient_data: Dict[str, sitk.Image],
                spacing_list: List[List[float]]) -> List[List[float]]:
    sa_spacing = patient_data['SA_ED'].GetSpacing()
    la_spacing = patient_data['LA_ED'].GetSpacing()

    spacing_list[0].append(sa_spacing[0])
    spacing_list[1].append(sa_spacing[2])
    spacing_list[2].append(la_spacing[0])
    spacing_list[3].append(la_spacing[2])
    
    return spacing_list


def print_spacing(spacing_list: List[List[float]]) -> None:
    print('Short-Axis in-plane pixel spacing:')
    print(Counter(spacing_list[0]))
    
    print('Short-Axis through-plane pixel spacing:')
    print(Counter(spacing_list[1]))
    
    print('Long-Axis in-plane pixel spacing:')
    print(Counter(spacing_list[2]))
    
    d=dict(Counter(spacing_list[2]))
    d_view = [ (v,k) for k,v in d.items() ]
    d_view.sort(reverse=True) # natively sort tuples by first element
    for v,k in d_view:
        print("%s: %d" % (k,v))
    
    print('Long-Axis through-plane pixel spacing:')
    print(Counter(spacing_list[3]))
    
    
def get_size(patient_data: Dict[str, sitk.Image],
             size_list: List[List[float]]) -> List[List[float]]:
    
    sa_size = patient_data['SA_ED'].GetSize()
    la_size = patient_data['LA_ED'].GetSize()
    
    size_list[0].append(sa_size[0])
    size_list[1].append(sa_size[1])
    size_list[2].append(sa_size[2])
    size_list[3].append(la_size[0])
    size_list[4].append(la_size[1])
    size_list[5].append(la_size[2])
    
    return size_list


def print_size(size_list: List[List[float]]) -> None:
    print('Short-Axis x-size:')
    print(Counter(size_list[0]))
    
    print('Short-Axis y-size:')
    print(Counter(size_list[1]))
    
    print('Short-Axis z-size:')
    print(Counter(size_list[2]))
    
    print('Long-Axis x-size:')
    print(Counter(size_list[3]))
    
    print('Long-Axis y-size:')
    print(Counter(size_list[4]))


def get_value_range(image: sitk.Image, segmentation: sitk.Image, index) -> Tuple[float]:
    image_n = sitk.GetArrayFromImage(image)
    segmentation_n = sitk.GetArrayFromImage(segmentation)
    
    segmentation_n[segmentation_n > 0] = 1
    image_n[segmentation_n == 0] = np.nan    
    
    #np.histogram(image_n, bins=500) 
    from matplotlib import pyplot as plt 
    plt.hist(image_n.flatten(), bins='auto', range=[0, 3500])
    plt.title(str(index))
    plt.title('Short-Axis Heart Value Distribution')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.ylim(0, 1000)
    plt.show()
    plt.close()
    
    min_value = np.nanmin(image_n)
    max_value = np.nanmax(image_n)
    image_n = sitk.GetArrayFromImage(image)
    fig, ax = plt.subplots()
    im = ax.imshow(image_n[4,:, :], cmap='hot', vmin=0, vmax=3500)
    plt.title('Short-Axis Slice')
    plt.axis('off')
    fig.colorbar(im)
    plt.show()
    plt.close()    
    return min_value, max_value


def print_min_max(global_max_sa, global_min_sa, global_max_la, global_min_la) -> None:
    
    print('Short-Axis min: ', global_min_sa)
    print('Short-Axis max: ', global_max_sa)
    
    print('Long-Axis min: ', global_min_la)
    print('Long-Axis max: ', global_max_la)
    
    
def data_analysis() -> None:
    dg = DataGenerator()
    dg.train_list = dg.get_patient_list(dg.train_directory)
    
    spacing_list = [[], [], [], []]
    size_list = [[], [], [], [], [], []]
    
    global_min_sa = 1000
    global_max_sa = -1000
    global_min_la = 1000
    global_max_la = -1000
    
    i = 1
    for patient_directory in dg.train_list:
        #print('Processing: ', patient_directory)
        patient_data = dg.load_patient_data(patient_directory)
        
        spacing_list = get_spacing(patient_data, spacing_list)
        #patient_data = dg.preprocess_patient_data(patient_data, dg.target_spacing, dg.target_size)
        size_list = get_size(patient_data, size_list)
        
        
        min_sa, max_sa = get_value_range(patient_data['SA_ED'], patient_data['SA_ED_gt'], i)
        #print(min_sa)
        if global_min_sa > min_sa:
            global_min_sa = min_sa
        if global_max_sa < max_sa:
            global_max_sa = max_sa
        
        min_sa, max_sa = get_value_range(patient_data['SA_ES'], patient_data['SA_ES_gt'], i)
        if global_min_sa > min_sa:
            global_min_sa = min_sa
        if global_max_sa < max_sa:
            global_max_sa = max_sa
            
        min_la, max_la = get_value_range(patient_data['LA_ED'], patient_data['LA_ED_gt'], i)
        if global_min_la > min_la:
            global_min_la = min_la
        if global_max_la < max_la:
            global_max_la = max_la
        min_la, max_la = get_value_range(patient_data['LA_ES'], patient_data['LA_ES_gt'], i)
        if global_min_la > min_la:
            global_min_la = min_la
        if global_max_la < max_la:
            global_max_la = max_la
        
        i += 1

    print_spacing(spacing_list)
    print_size(size_list)
    print_min_max(global_max_sa, global_min_sa, global_max_la, global_min_la)


if __name__ == '__main__':
    data_analysis()    
    