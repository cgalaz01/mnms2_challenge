import os

from enum import Enum

from typing import Any, Dict, Tuple, List, Union
from pathlib import Path
from glob import glob

import numpy as np

import SimpleITK as sitk

from .preprocess import Preprocess


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
        self.cache_directory = os.path.join('..', '..', 'data_cache')
        self.cache_directory = Path(os.path.join(file_path, self.cache_directory))
        
        self.train_directory = Path(os.path.join(self.data_directory, 'training'))
        # For the purposes of model development, the 'validation' set is treated
        # as the test set
        # (It does not have ground truth - validated on submission only)
        self.testing_directory = Path(os.path.join(self.data_directory, 'validation'))
        
        self.train_list = self.get_patient_list(self.train_directory)
        self.train_list = self.randomise_list(self.train_list, seed=4516, inplace=True)
        self.train_list, self.validation_list = self.split_list(self.train_list, split_fraction=0.8)
        self.test_list = self.get_patient_list(self.testing_directory)
        
        self.target_spacing = (1.25, 1.25, 10)
        self.target_size = (256, 256, 17)
        
        self.n_classes = 4  # Including background


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
        
        patient_data[FileType.sa_ed.value] = DataGenerator.load_image(patient_directory, FileType.sa_ed)
        patient_data[FileType.sa_ed_gt.value] = DataGenerator.load_image(patient_directory, FileType.sa_ed_gt)
        patient_data[FileType.sa_es.value] = DataGenerator.load_image(patient_directory, FileType.sa_es)
        patient_data[FileType.sa_es_gt.value] = DataGenerator.load_image(patient_directory, FileType.sa_es_gt)
        patient_data[FileType.la_ed.value] = DataGenerator.load_image(patient_directory, FileType.la_ed)
        patient_data[FileType.la_ed_gt.value] = DataGenerator.load_image(patient_directory, FileType.la_ed_gt)
        patient_data[FileType.la_es.value] = DataGenerator.load_image(patient_directory, FileType.la_es)
        patient_data[FileType.la_es_gt.value] = DataGenerator.load_image(patient_directory, FileType.la_es_gt)
        
        return patient_data
    
    
    @staticmethod
    def preprocess_patient_data(patient_data: Dict[str, sitk.Image], spacing: Tuple[float],
                                size: Tuple[int]) -> Dict[str, sitk.Image]:
        # Resample images to standardised spacing and size
        patient_data[FileType.sa_ed.value] = Preprocess.resample_image(patient_data[FileType.sa_ed.value],
                                                                       spacing, size, is_label=False)
        patient_data[FileType.sa_es.value] = Preprocess.resample_image(patient_data[FileType.sa_es.value],
                                                                       spacing, size, is_label=False)
        patient_data[FileType.sa_ed_gt.value] = Preprocess.resample_image(patient_data[FileType.sa_ed_gt.value],
                                                                          spacing, size, is_label=True)
        patient_data[FileType.sa_es_gt.value] = Preprocess.resample_image(patient_data[FileType.sa_es_gt.value],
                                                                          spacing, size, is_label=True)
        # Normalise intensities so there are (roughly) [0-1]
        patient_data[FileType.sa_ed.value] = Preprocess.normalise_intensities(patient_data[FileType.sa_ed.value])
        patient_data[FileType.sa_es.value] = Preprocess.normalise_intensities(patient_data[FileType.sa_es.value])
        
        la_spacing = list(spacing)
        la_spacing[2] = patient_data[FileType.la_ed.value].GetSpacing()[2]
        la_size = list(size)
        la_size[2] = 1
        patient_data[FileType.la_ed.value] = Preprocess.resample_image(patient_data[FileType.la_ed.value],
                                                                       la_spacing, la_size, is_label=False)
        patient_data[FileType.la_es.value] = Preprocess.resample_image(patient_data[FileType.la_es.value],
                                                                       la_spacing, la_size, is_label=False)
        patient_data[FileType.la_ed_gt.value] = Preprocess.resample_image(patient_data[FileType.la_ed_gt.value],
                                                                          la_spacing, la_size, is_label=True)
        patient_data[FileType.la_es_gt.value] = Preprocess.resample_image(patient_data[FileType.la_es_gt.value],
                                                                          la_spacing, la_size, is_label=True)
        
        patient_data[FileType.la_ed.value] = Preprocess.normalise_intensities(patient_data[FileType.la_ed.value])
        patient_data[FileType.la_es.value] = Preprocess.normalise_intensities(patient_data[FileType.la_es.value])
        
        return patient_data
        

    def get_cache_directory(self, patient_directory: Union[str, Path]) -> Path:
        path = os.path.normpath(patient_directory)
        split_path = path.split(os.sep)
        # .. / data / training or vlaidation / patient ID
        # only last two are of interest
        cache_directory = Path(os.path.join(self.cache_directory,
                                            split_path[-2],
                                            split_path[-1]))
        
        return cache_directory

    
    def is_cached(self, patient_directory: Union[str, Path]) -> bool:
        patient_cache_directory = self.get_cache_directory(patient_directory)
        
        # Check if folder exists
        if os.path.isdir(patient_cache_directory):
            # and every individual file exist
            for expected_file_name in FileType:
                expected_file_path = os.path.join(patient_cache_directory,
                                                  expected_file_name.value + '.nii.gz')
                if not os.path.exists(expected_file_path):
                    return False
            return True
        
        return False

        
    def save_cache(self, patient_directory: Union[str, Path],
                    patient_data: Dict[str, sitk.Image]) -> None:
        patient_cache_directory = self.get_cache_directory(patient_directory)
        os.makedirs(patient_cache_directory, exist_ok=True)
        
        for key, image in patient_data.items():
            file_path = os.path.join(patient_cache_directory, key + '.nii.gz')
            sitk.WriteImage(image, file_path)
        
    
    def load_cache(self, patient_directory: Union[str, Path]) -> Dict[str, sitk.Image]:
        patient_cache_directory = self.get_cache_directory(patient_directory)
        patient_data = self.load_patient_data(patient_cache_directory)
        
        return patient_data
    
    
    def to_numpy(self, patient_data: Dict[str, sitk.Image]) -> Dict[str, np.ndarray]:
        for key, image in patient_data.items():
            if 'gt' in key:
                numpy_image = sitk.GetArrayFromImage(image).astype(np.uint8)
            else:
                numpy_image = sitk.GetArrayFromImage(image).astype(np.float32)
                
            # Swap axes so ordering is x, y, z rather than z, y, x as stored
            # in sitk
            numpy_image = np.swapaxes(numpy_image, 0, -1)
            
            # Add 'channel' axis for 3D images
            #if 'sa' in key:
            #    numpy_image = np.expand_dims(numpy_image, axis=-1)
            
            # Generate one-hot encoding of the labels (done after swapping to simplify
            # the swap)
            if 'gt' in key:
                if 'LA' in key: # use the 'depth; axis as the channel for the label
                    numpy_image = np.squeeze(numpy_image, axis=-1)
                n_values = self.n_classes
                numpy_image = np.eye(n_values)[numpy_image]
                
            patient_data[key] = numpy_image
        
        return patient_data


    def generator(self, patient_directory: Union[str, Path]) -> Tuple[Dict[str, np.ndarray]]:
        if self.is_cached(patient_directory):
            patient_data = self.load_cache(patient_directory)
        else:
            patient_data = DataGenerator.load_patient_data(patient_directory)
            patient_data = DataGenerator.preprocess_patient_data(patient_data,
                                                                 self.target_spacing,
                                                                 self.target_size)
            self.save_cache(patient_directory, patient_data)

        
        patient_data = self.to_numpy(patient_data)
    
        output_data = []
        output_data.append(({'input_sa': patient_data[FileType.sa_ed.value],
                             'input_la': patient_data[FileType.la_ed.value]},
                            {'output_sa': patient_data[FileType.sa_ed_gt.value],
                             'output_la': patient_data[FileType.la_ed_gt.value]}))
        output_data.append(({'input_sa': patient_data[FileType.sa_es.value],
                             'input_la': patient_data[FileType.la_es.value]},
                            {'output_sa': patient_data[FileType.sa_es_gt.value],
                             'output_la': patient_data[FileType.la_es_gt.value]}))
        return output_data

    
    def train_generator(self) -> Tuple[Dict[str, np.ndarray]]:
        for patient_directory in self.train_list:
            patient_data = self.generator(patient_directory)
            
            yield patient_data[0]   # End diastolic
            yield patient_data[1]   # End systolic
        
    
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
            
    
    