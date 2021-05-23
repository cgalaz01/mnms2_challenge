from typing import Dict, List

from collections import Counter

import SimpleITK as sitk

from data import DataGenerator


def get_spacing(patient_data: Dict[str, sitk.Image],
                spacing_list: List[List[float]]) -> List[List[float]]:
    sa_spacing = patient_data['sa_ed'].GetSpacing()
    la_spacing = patient_data['la_ed'].GetSpacing()

    spacing_list[0].append(sa_spacing[0])
    spacing_list[1].append(sa_spacing[2])
    spacing_list[2].append(la_spacing[0])
    
    return spacing_list


def print_spacing(spacing_list: List[List[float]]) -> None:
    print('Short-Axis in-plane pixel spacing:')
    print(Counter(spacing_list[0]))
    
    print('Short-Axis through-plane pixel spacing:')
    print(Counter(spacing_list[1]))
    
    print('Long-Axis in-plane pixel spacing:')
    print(Counter(spacing_list[2]))
    
    
def get_size(patient_data: Dict[str, sitk.Image],
             size_list: List[List[float]]) -> List[List[float]]:
    
    sa_size = patient_data['sa_ed'].GetSize()
    la_size = patient_data['la_ed'].GetSize()
    
    size_list[0].append(sa_size[0])
    size_list[1].append(sa_size[1])
    size_list[2].append(sa_size[2])
    size_list[3].append(la_size[0])
    size_list[4].append(la_size[1])
    
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
    

def data_analysis() -> None:
    dg = DataGenerator()
    spacing_list = [[], [], []]
    size_list = [[], [], [], [], []]
    for patient_directory in dg.train_list:
        patient_data = dg.load_patient_data(patient_directory)
        
        spacing_list = get_spacing(patient_data, spacing_list)
        #patient_data = dg.preprocess_patient_data(patient_data, dg.target_spacing, dg.target_size)
        size_list = get_size(patient_data, size_list)

    print_spacing(spacing_list)
    print_size(size_list)
    

if __name__ == '__main__':
    data_analysis()    
    