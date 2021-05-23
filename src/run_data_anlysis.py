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
    

def data_analysis() -> None:
    dg = DataGenerator()
    spacing_list = [[], [], []]
    for patient_directory in dg.train_list:
        patient_data = dg.load_patient_data(patient_directory)
        
        spacing_list = get_spacing(patient_data, spacing_list)

    print_spacing(spacing_list)
    

if __name__ == '__main__':
    data_analysis()    
    