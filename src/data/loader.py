import os

from enum import Enum

from typing import Any, Dict, Tuple, List, Union
from pathlib import Path
from glob import glob

import numpy as np

import SimpleITK as sitk


class FileType(Enum):
    sa_ed = 'SA_ED'
    sa_ed_gt = 'SA_ED_gt'
    sa_es = 'SA_ES'
    sa_es_gt = 'SA_ES_gt'
    la_ed = 'LA_ED'
    la_ed_gt = 'LA_ED_gt'
    la_es = 'LA_ES'
    la_es_gt = 'LA_ES_gt'
    

class DataGenerator():

    
    def __init__(self) -> None:
        file_path = Path(__file__).parent.absolute()
        expected_data_directory = os.path.join('..', '..', 'data')
        
        self.data_directory = Path(os.path.join(file_path, expected_data_directory))
        self.train_directory = Path(os.path.join(self.data_directory, 'training'))
        # For the purposes of model development, the 'validation' set is treated
        # as the test set
        # (It does not have ground truth - validated on submission only)
        self.testing_directory = Path(os.path.join(self.data_directory, 'validation'))
        
        self.train_list = self.get_patient_list(self.train_directory)
        self.train_list = self.randomise_list(self.train_list, seed=4516, inplace=True)
        self.train_list, self.validation_list = self.split_list(self.train_list, split_fraction=0.8)
        self.test_list = self.get_patient_list(self.testing_directory)


    @staticmethod
    def get_patient_list(root_directory: Union[str, Path]) -> List[Path]:
        files = glob(os.path.join(root_directory, "**"))
        files = [Path(i) for i in files]
        
        return files
    
    
    @staticmethod
    def randomise_list(item_list: List[Any], seed: Union[None, int]=None,
                       inplace: bool=True) -> List[Any]:
        if not inplace:
            item_list = item_list.copy()
            
        random_generator = np.random.RandomState(seed)
        random_generator.shuffle(item_list)
        
        return item_list
    
    
    @staticmethod
    def split_list(item_list: List[Any], split_fraction: float) -> Tuple[List[Any]]:
        assert 0 < split_fraction < 1
        
        split_index = int(len(item_list) * split_fraction)
        
        split_1 = item_list[:split_index]
        split_2 = item_list[split_index:]
                
        return split_1, split_2

        
    @staticmethod
    def load_image(patient_directory: Union[str, Path], file_type: FileType) -> sitk.Image:
        
        file_suffix = '*' + file_type.value + '.nii.gz'
        
        file_path = os.path.join(patient_directory, file_suffix)
        file_path = glob(file_path)
        assert len(file_path) == 1
        file_path = file_path[0]
        
        sitk_image = sitk.ReadImage(file_path)
        
        return sitk_image
    
    
    @staticmethod
    def load_patient_data(patient_directory: Union[str, Path]) -> Dict[str, sitk.Image]:
        patient_data = {}
        
        patient_data['sa_ed'] = DataGenerator.load_image(patient_directory, FileType.sa_ed)
        patient_data['sa_ed_gt'] = DataGenerator.load_image(patient_directory, FileType.sa_ed_gt)
        patient_data['sa_es'] = DataGenerator.load_image(patient_directory, FileType.sa_es)
        patient_data['sa_es_gt'] = DataGenerator.load_image(patient_directory, FileType.sa_es_gt)
        patient_data['la_ed'] = DataGenerator.load_image(patient_directory, FileType.la_ed)
        patient_data['la_ed_gt'] = DataGenerator.load_image(patient_directory, FileType.la_ed_gt)
        patient_data['la_es'] = DataGenerator.load_image(patient_directory, FileType.la_es)
        patient_data['la_es_gt'] = DataGenerator.load_image(patient_directory, FileType.la_es_gt)
        
        return patient_data
    
    
    @staticmethod
    def generator(patient_directory) -> Tuple[Dict[str, np.ndarray]]:
        patient_data = DataGenerator.load_patient_data(patient_directory)
        
        for key, image in patient_data.items():
            patient_data[key] = sitk.GetArrayFromImage(image).astype(np.float32)
    
    
        output_data = []
        output_data.append(({'input_sa': patient_data['sa_ed'],
                             'input_la': patient_data['la_ed']},
                            {'output_sa': patient_data['sa_ed_gt'],
                             'output_la': patient_data['la_ed_gt']}))
        output_data.append(({'input_sa': patient_data['sa_es'],
                             'input_la': patient_data['la_es']},
                            {'output_sa': patient_data['sa_es_gt'],
                             'output_la': patient_data['la_es_gt']}))
        return output_data

    
    def train_generator(self) -> Tuple[Dict[str, np.ndarray]]:
        for patient_directory in self.train_list:
            patient_data = self.generator(patient_directory)
            
            yield patient_data[0]
            yield patient_data[1]
        
    
    def validation_generator(self) -> Tuple[Dict[str, np.ndarray]]:
        for patient_directory in self.validation_list:
            patient_data = self.generator(patient_directory)
            
            yield patient_data[0]
            yield patient_data[1]
            
    
    def test_generator(self) -> Tuple[Dict[str, np.ndarray]]:
        for patient_directory in self.test_list:
            patient_data = self.generator(patient_directory)
            
            yield patient_data[0]
            yield patient_data[1]
            
    
    