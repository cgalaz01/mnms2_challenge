import os

from enum import Enum

from typing import Any, Dict, Tuple, List, Union
from pathlib import Path
from glob import glob

import numpy as np

import SimpleITK as sitk

from .augment import DataAugmentation
from .preprocess import Preprocess, Registration


class FileType(Enum):
    sa_ed = 'SA_ED'
    sa_ed_gt = 'SA_ED_gt'
    sa_es = 'SA_ES'
    sa_es_gt = 'SA_ES_gt'
    la_ed = 'LA_ED'
    la_ed_gt = 'LA_ED_gt'
    la_es = 'LA_ES'
    la_es_gt = 'LA_ES_gt'
    
    
class ExtraType(Enum):
    reg_affine = 'SA_to_LA_registration_affine'
    
    
class Affine(Enum):
    sa_affine = 'sa_affine'
    la_affine = 'la_affine'
    

class OutputAffine(Enum):
    sa_affine = 'SA_Affine'
    la_affine = 'LA_Affine'    
    

class DataGenerator():

    
    def __init__(self, floating_precision: str = '32') -> None:
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
        self.train_list = self.randomise_list(self.train_list, seed=4594, inplace=True)
        self.train_list, self.validation_list = self.split_list(self.train_list, split_fraction=15/16)
        self.test_list = self.get_patient_list(self.testing_directory)
        
        self.target_spacing = (1.25, 1.25, 10)
        self.target_size = (192, 192, 17)
        
        self.n_classes = 1  # Right ventricle only

        self.floating_precision = floating_precision
        
        # Compute the shape for the inputs and outputs
        self.sa_target_shape = list(self.target_size)
        self.sa_shape = self.sa_target_shape.copy()
        self.sa_target_shape.append(self.n_classes)
        
        self.la_target_shape = list(self.target_size)
        self.la_shape = self.la_target_shape.copy()
        self.la_shape[-1] = 1
        self.la_target_shape[-1] = self.n_classes
        
        self.affine_shape = (4, 4)
        
        self.augmentation = DataAugmentation(seed=1235)


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
    def load_transformation(patient_directory: Union[str, Path], file_type: ExtraType) -> sitk.Transform:
        file_suffix = '*' + file_type.value + '.tfm'
        
        file_path = os.path.join(patient_directory, file_suffix)
        file_path = glob(file_path)
        assert len(file_path) == 1
        file_path = file_path[0]
        
        sitk_transform = sitk.ReadTransform(file_path)
        
        return sitk_transform
    
    
    @staticmethod
    def load_patient_data(patient_directory: Union[str, Path], has_gt: bool = True) -> Dict[str, sitk.Image]:
        patient_data = {}
        
        patient_data[FileType.sa_ed.value] = DataGenerator.load_image(patient_directory, FileType.sa_ed)        
        patient_data[FileType.sa_es.value] = DataGenerator.load_image(patient_directory, FileType.sa_es)        
        patient_data[FileType.la_ed.value] = DataGenerator.load_image(patient_directory, FileType.la_ed)
        patient_data[FileType.la_es.value] = DataGenerator.load_image(patient_directory, FileType.la_es)
        
        if has_gt:
            patient_data[FileType.sa_ed_gt.value] = DataGenerator.load_image(patient_directory, FileType.sa_ed_gt)
            patient_data[FileType.sa_es_gt.value] = DataGenerator.load_image(patient_directory, FileType.sa_es_gt)
            patient_data[FileType.la_ed_gt.value] = DataGenerator.load_image(patient_directory, FileType.la_ed_gt)
            patient_data[FileType.la_es_gt.value] = DataGenerator.load_image(patient_directory, FileType.la_es_gt)
            
        
        return patient_data
    
    
    @staticmethod
    def load_extra_patient_data(patient_directory: Union[str, Path],
                                patient_data: Dict[str, sitk.Image]) -> Dict[str, sitk.Image]:
        
        patient_data[ExtraType.reg_affine.value] = DataGenerator.load_transformation(patient_directory,
                                                                                     ExtraType.reg_affine)
        
        return patient_data

    
    @staticmethod
    def preprocess_patient_data(patient_data: Dict[str, sitk.Image], spacing: Tuple[float],
                                size: Tuple[int], has_gt: bool = True, register: bool = True) -> Dict[str, sitk.Image]:
        # Resample images to standardised spacing and size
        # Short-axis
        sa_spacing = list(spacing)
        sa_spacing[2] = patient_data[FileType.sa_ed.value].GetSpacing()[2]
        patient_data[FileType.sa_ed.value] = Preprocess.resample_image(patient_data[FileType.sa_ed.value],
                                                                       sa_spacing, size, is_label=False)
        patient_data[FileType.sa_es.value] = Preprocess.resample_image(patient_data[FileType.sa_es.value],
                                                                       sa_spacing, size, is_label=False)
        if has_gt:
            patient_data[FileType.sa_ed_gt.value] = Preprocess.resample_image(patient_data[FileType.sa_ed_gt.value],
                                                                              sa_spacing, size, is_label=True)
            patient_data[FileType.sa_es_gt.value] = Preprocess.resample_image(patient_data[FileType.sa_es_gt.value],
                                                                              sa_spacing, size, is_label=True)

        # Long-axis
        la_spacing = list(spacing)
        la_spacing[2] = patient_data[FileType.la_ed.value].GetSpacing()[2]
        la_size = list(size)
        la_size[2] = 1
        patient_data[FileType.la_ed.value] = Preprocess.resample_image(patient_data[FileType.la_ed.value],
                                                                       la_spacing, la_size, is_label=False)
        patient_data[FileType.la_es.value] = Preprocess.resample_image(patient_data[FileType.la_es.value],
                                                                       la_spacing, la_size, is_label=False)
        if has_gt:
            patient_data[FileType.la_ed_gt.value] = Preprocess.resample_image(patient_data[FileType.la_ed_gt.value],
                                                                              la_spacing, la_size, is_label=True)
            patient_data[FileType.la_es_gt.value] = Preprocess.resample_image(patient_data[FileType.la_es_gt.value],
                                                                              la_spacing, la_size, is_label=True)
        
        # Register short-axis to long axis (only for end diastolic for faster execution time)
        if register:
            affine_transform, _ = Registration.register(patient_data[FileType.sa_ed.value],
                                                        patient_data[FileType.la_ed.value])
            patient_data[ExtraType.reg_affine.value] = affine_transform
        
        # Normalise intensities so there are (roughly) [0-1]
        patient_data[FileType.sa_ed.value] = Preprocess.normalise_intensities(patient_data[FileType.sa_ed.value])
        patient_data[FileType.sa_es.value] = Preprocess.normalise_intensities(patient_data[FileType.sa_es.value])
        
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

    
    def is_cached(self, patient_directory: Union[str, Path], has_gt: bool = True) -> bool:
        patient_cache_directory = self.get_cache_directory(patient_directory)
        
        # Check if folder exists
        if os.path.isdir(patient_cache_directory):
            # and every individual file exist
            for expected_file_name in FileType:
                if not has_gt and expected_file_name.value.endswith('_gt'):
                    continue
                expected_file_path = os.path.join(patient_cache_directory,
                                                  expected_file_name.value + '.nii.gz')
                if not os.path.exists(expected_file_path):
                    return False
                
            for expected_file_name in ExtraType:
                expected_file_path = os.path.join(patient_cache_directory,
                                                  expected_file_name.value + '.tfm')
                if not os.path.exists(expected_file_path):
                    return False
            return True
        
        return False

        
    def save_cache(self, patient_directory: Union[str, Path],
                    patient_data: Dict[str, sitk.Image]) -> None:
        patient_cache_directory = self.get_cache_directory(patient_directory)
        os.makedirs(patient_cache_directory, exist_ok=True)
        
        for key, data in patient_data.items():
            if key in (k.value for k in FileType):
                file_path = os.path.join(patient_cache_directory, key + '.nii.gz')
                sitk.WriteImage(data, file_path)
            elif key in (k.value for k in ExtraType):
                file_path = os.path.join(patient_cache_directory, key + '.tfm')
                sitk.WriteTransform(data, file_path)
        
    
    def load_cache(self, patient_directory: Union[str, Path], has_gt: bool = True) -> Dict[str, sitk.Image]:
        patient_cache_directory = self.get_cache_directory(patient_directory)
        patient_data = self.load_patient_data(patient_cache_directory, has_gt)
        patient_data = self.load_extra_patient_data(patient_cache_directory, patient_data)
        
        return patient_data
    
    
    def augment_data(self, patient_data: Dict[str, sitk.Image]) -> Dict[str, sitk.Image]:        

        (patient_data[FileType.sa_ed.value], patient_data[FileType.sa_ed_gt.value],
         sa_affine) = self.augmentation.random_augmentation(patient_data[FileType.sa_ed.value],
                                                            patient_data[FileType.sa_ed_gt.value],
                                                            use_cache=False)
        (patient_data[FileType.sa_es.value], patient_data[FileType.sa_es_gt.value],
         sa_affine) = self.augmentation.random_augmentation(patient_data[FileType.sa_es.value],
                                                            patient_data[FileType.sa_es_gt.value],
                                                            use_cache=True)
                                                            
        (patient_data[FileType.la_ed.value], patient_data[FileType.la_ed_gt.value],
         la_affine) = self.augmentation.random_augmentation(patient_data[FileType.la_ed.value],
                                                            patient_data[FileType.la_ed_gt.value],
                                                            use_cache=False)
        (patient_data[FileType.la_es.value], patient_data[FileType.la_es_gt.value],
         la_affine) = self.augmentation.random_augmentation(patient_data[FileType.la_es.value],
                                                            patient_data[FileType.la_es_gt.value],
                                                            use_cache=True)
        
        patient_data[Affine.sa_affine.value] = sa_affine
        patient_data[Affine.la_affine.value] = la_affine
        
        
        return patient_data
    
    
    def to_numpy(self, patient_data: Dict[str, sitk.Image], has_affine_matrix: bool) -> Dict[str, np.ndarray]:
        
        # Handle 'ExtraType' data first
        if has_affine_matrix:
            if (Affine.sa_affine.value in patient_data and 
                Affine.la_affine.value in patient_data):
                affine_matrix = sitk.CompositeTransform([patient_data[Affine.sa_affine.value].GetInverse(),
                                                         patient_data[ExtraType.reg_affine.value],
                                                         patient_data[Affine.la_affine.value]])
            else:
                affine_matrix = patient_data[ExtraType.reg_affine.value]
            sa_affine = Registration.get_affine_registration_matrix(patient_data[FileType.sa_ed.value],
                                                                    affine_matrix)
            sa_affine = sa_affine.astype(np.float32)
            la_affine = Registration.get_affine_matrix(patient_data[FileType.la_ed.value])
            la_affine = la_affine.astype(np.float32)
        
        # Free from memory (and indexing)
        if ExtraType.reg_affine.value in patient_data:
            del patient_data[ExtraType.reg_affine.value]
        if Affine.sa_affine.value in patient_data:
            del patient_data[Affine.sa_affine.value]
        if Affine.la_affine.value in patient_data:
            del patient_data[Affine.la_affine.value]
        
        # Handle original file data (images and segmentations)
        for key, image in patient_data.items():
            numpy_image = sitk.GetArrayFromImage(image)
            # Swap axes so ordering is x, y, z rather than z, y, x as stored
            # in sitk
            numpy_image = np.swapaxes(numpy_image, 0, -1)
            
            # Generate one-hot encoding of the labels
            #if 'gt' in key:
            #    numpy_image = numpy_image.astype(np.uint8)
            #    if 'LA' in key: # use the 'depth' axis as the channel for the label
            #        numpy_image = np.squeeze(numpy_image, axis=-1)
            #    n_values = self.n_classes
            #    numpy_image = np.eye(n_values)[numpy_image]
            
            # Select right-ventricle labels only
            if 'gt' in key:
                numpy_image = numpy_image.astype(np.uint8)
                if 'SA' in key:
                    numpy_image = np.expand_dims(numpy_image, axis=-1)
                numpy_image[numpy_image != 3] = 0
                numpy_image[numpy_image == 3] = 1
            
            if self.floating_precision == '16':
                numpy_image = numpy_image.astype(np.float16)
            else:
                numpy_image = numpy_image.astype(np.float32)
                
            # Add 'channel' axis for 3D images
            #if 'sa' in key:
            #    numpy_image = np.expand_dims(numpy_image, axis=-1)
                
            patient_data[key] = numpy_image
        
        if has_affine_matrix:
            patient_data[OutputAffine.sa_affine.value] = sa_affine
            patient_data[OutputAffine.la_affine.value] = la_affine
        
        return patient_data
    
    @staticmethod
    def to_structure(patient_data: Dict[str, sitk.Image], has_affine_matrix: bool,
                     has_gt: bool = True):
        output_data = []
        if has_gt:
            output_data.append(({'input_sa': patient_data[FileType.sa_ed.value],
                                 'input_la': patient_data[FileType.la_ed.value]},
                                {'output_sa': patient_data[FileType.sa_ed_gt.value],
                                 'output_la': patient_data[FileType.la_ed_gt.value]}))
            
            output_data.append(({'input_sa': patient_data[FileType.sa_es.value],
                                 'input_la': patient_data[FileType.la_es.value]},
                                {'output_sa': patient_data[FileType.sa_es_gt.value],
                                 'output_la': patient_data[FileType.la_es_gt.value]}))
        else:
            output_data.append(({'input_sa': patient_data[FileType.sa_ed.value],
                                 'input_la': patient_data[FileType.la_ed.value]},))
            
            output_data.append(({'input_sa': patient_data[FileType.sa_es.value],
                                 'input_la': patient_data[FileType.la_es.value]},))
            
        if has_affine_matrix:
            for data in output_data:
                data[0]['input_sa_affine'] = patient_data[OutputAffine.sa_affine.value]
                data[0]['input_la_affine'] = patient_data[OutputAffine.la_affine.value]
                
        return output_data
        

    def generator(self, patient_directory: Union[str, Path], affine_matrix: bool,
                  has_gt: bool = True, augment: bool = False) -> Tuple[Dict[str, np.ndarray]]:
        if self.is_cached(patient_directory, has_gt):
            patient_data = self.load_cache(patient_directory, has_gt)
        else:
            patient_data = DataGenerator.load_patient_data(patient_directory, has_gt)
            patient_data = DataGenerator.preprocess_patient_data(patient_data,
                                                                 self.target_spacing,
                                                                 self.target_size,
                                                                 has_gt,
                                                                 affine_matrix)
            self.save_cache(patient_directory, patient_data)

        if augment:
            patient_data = self.augment_data(patient_data)
        patient_data = self.to_numpy(patient_data, affine_matrix)
    
        output_data = self.to_structure(patient_data, affine_matrix, has_gt)
        return output_data

    
    def sitk_generator(self, patient_directory: Union[str, Path], has_gt: bool = True) -> Tuple[Dict[str, np.ndarray]]:
        """
        Returns pre- and post-processed data in sitk
        """
        if self.is_cached(patient_directory, has_gt):
            pre_patient_data = DataGenerator.load_patient_data(patient_directory, has_gt)
            post_patient_data = self.load_cache(patient_directory, has_gt)
        else:
            pre_patient_data = DataGenerator.load_patient_data(patient_directory, has_gt)
            post_patient_data = DataGenerator.load_patient_data(patient_directory, has_gt)
            post_patient_data = DataGenerator.preprocess_patient_data(post_patient_data,
                                                                      self.target_spacing,
                                                                      self.target_size,
                                                                      has_gt,
                                                                      False)
            self.save_cache(patient_directory, pre_patient_data)
            
        
        pre_output_data = self.to_structure(pre_patient_data, False, has_gt)
        post_output_data = self.to_structure(post_patient_data, False, has_gt)
        
        return pre_output_data, post_output_data
        
        
    def train_generator(self, augment: bool = True, verbose: int = 0) -> Tuple[Dict[str, np.ndarray]]:
        for patient_directory in self.train_list:
            if verbose > 0:
                print('Generating patient: ', patient_directory)
            patient_data = self.generator(patient_directory, affine_matrix=False, augment=augment)
            
            yield patient_data[0]   # End diastolic
            yield patient_data[1]   # End systolic
        
    
    def validation_generator(self, verbose: int = 0) -> Tuple[Dict[str, np.ndarray]]:
        for patient_directory in self.validation_list:
            if verbose > 0:
                print('Generating patient: ', patient_directory)
            patient_data = self.generator(patient_directory, affine_matrix=False)
            
            yield patient_data[0]
            yield patient_data[1]
            
    
    def test_generator(self, verbose: int = 0) -> Tuple[Dict[str, np.ndarray]]:
        for patient_directory in self.test_list:
            if verbose > 0:
                print('Generating patient: ', patient_directory)
            patient_data = self.generator(patient_directory, affine_matrix=False)
            
            yield patient_data[0]
            yield patient_data[1]
            
    
    def test_generator_inference(self, verbose: int = 0) -> Tuple[Dict[str, np.ndarray]]:
        for patient_directory in self.test_list:
            if verbose > 0:
                print('Generating patient: ', patient_directory)
            patient_data = self.generator(patient_directory, affine_matrix=False, has_gt=False)
            pre_patient_data, post_patient_data = self.sitk_generator(patient_directory, has_gt=False)
            
            yield patient_data[0], pre_patient_data[0], post_patient_data[0], patient_directory, 'ed'
            yield patient_data[1], pre_patient_data[1], post_patient_data[1], patient_directory, 'es'
        
        
    def train_affine_generator(self, augment: bool = True, verbose: int = 0) -> Tuple[Dict[str, np.ndarray]]:
        for patient_directory in self.train_list:
            if verbose > 0:
                print('Generating patient: ', patient_directory)
            patient_data = self.generator(patient_directory, affine_matrix=True, augment=augment)
            
            yield patient_data[0]   # End diastolic
            yield patient_data[1]   # End systolic
        
    
    def validation_affine_generator(self, verbose: int = 0) -> Tuple[Dict[str, np.ndarray]]:
        for patient_directory in self.validation_list:
            if verbose > 0:
                print('Generating patient: ', patient_directory)
            patient_data = self.generator(patient_directory, affine_matrix=True)
            
            yield patient_data[0]
            yield patient_data[1]
            
    
    def test_affine_generator(self, verbose: int = 0) -> Tuple[Dict[str, np.ndarray]]:
        for patient_directory in self.test_list:
            if verbose > 0:
                print('Generating patient: ', patient_directory)
            patient_data = self.generator(patient_directory, affine_matrix=True)
            
            yield patient_data[0]
            yield patient_data[1]


    def test_affine_generator_inference(self, verbose: int = 0) -> Tuple[Dict[str, np.ndarray]]:
        for patient_directory in self.test_list:
            if verbose > 0:
                print('Generating patient: ', patient_directory)
            patient_data = self.generator(patient_directory, affine_matrix=True, has_gt=False)
            pre_patient_data, post_patient_data = self.sitk_generator(patient_directory, has_gt=False)
            
            yield patient_data[0], pre_patient_data[0], post_patient_data[0], patient_directory, 'ed'
            yield patient_data[1], pre_patient_data[1], post_patient_data[1], patient_directory, 'es'