import os

from typing import Union
from pathlib import Path

import numpy as np

from skimage.morphology import label

import tensorflow as tf

import SimpleITK as sitk


from data import TensorFlowDataGenerator, DataGenerator
from tf.models import multi_stage_model


class model:
    
    def __init__(self):
        '''
        IMPORTANT: Initializes the model wrapper WITHOUT PARAMETERS, it will be called as model()
        '''
        self.model_weights_path = os.path.join('model_weights', 'multi_stage_model')


    def load_model(self, data_gen: DataGenerator) -> tf.keras.Model:
        activation = 'selu'
        kernel_initializer = 'lecun_normal'
        dropout_rate = 0.
        
        model = multi_stage_model.get_model(data_gen.sa_shape, data_gen.la_shape, data_gen.n_classes,
                                            activation, kernel_initializer, dropout_rate)
        
        # Load weights
        file_path = Path(__file__).parent.absolute()
        model_path = os.path.join(file_path, self.model_weights_path)
        model.load_weights(model_path)
        
        return model
        
    
    def select_largest_region(self, label_image: np.ndarray) -> np.ndarray:
        multi_label_image, label_num = label(label_image, return_num=True,
                                             background=0, connectivity=2)
    
        if label_num > 1:
            # Select and keep only the largest label
            largest_label_size = 0
            largest_index_label = 0
            for i in range(1, label_num):
                label_size = (multi_label_image == i).sum()
                if label_size > largest_label_size:
                    largest_label_size = label_size
                    largest_index_label = i
            
            # Remove any set labels that do not correspond to the largest segmentation
            label_image[multi_label_image != largest_index_label] = 0
            
        return label_image


    def save_predictions(self, sa_prediction: sitk.Image, la_prediction: sitk.Image,
                         patient_id: str, cardiac_phase: str,
                         base_output: Union[str, Path]) -> None:
        os.makedirs(base_output, exist_ok=True)
        
        patient_output = os.path.join(base_output, patient_id)
        os.makedirs(patient_output, exist_ok=True)
        
        cardiac_phase = cardiac_phase.upper()
        sa_file_output = os.path.join(patient_output, patient_id + '_SA_' +  cardiac_phase + '_pred.nii.gz')
        sitk.WriteImage(sa_prediction, sa_file_output)
        
        la_file_output = os.path.join(patient_output, patient_id + '_LA_' +  cardiac_phase + '_pred.nii.gz')
        sitk.WriteImage(la_prediction, la_file_output)
        
        
    def test_prediction(self, input_directory: Union[str, Path],
                        output_directory: Union[str, Path]) -> None:    
        (train_gen, validation_gen,
         test_gen, data_gen) = TensorFlowDataGenerator.get_affine_generators(batch_size=1,
                                                                             max_buffer_size=None,
                                                                             floating_precision='16',
                                                                             memory_cache=False,
                                                                             disk_cache=False,
                                                                             test_directory=input_directory)
        
        model = self.load_model(data_gen)
                                                                             
        threshold = 0
        
        for data, pre_sitk, post_sitk, patient_directory, phase in data_gen.test_affine_generator_inference():
            # Add batch dimension to each of the input values
            for key, value in data[0].items():
                data[0][key] = np.expand_dims(value, 0)
            
            prediction_list = model.predict(data[0])
            prediction_dict = {name: pred for name, pred in zip(model.output_names, prediction_list)}
            
            output_sa = (prediction_dict['output_sa'][0] >= threshold).astype(np.uint8)[..., 0]
            output_sa = self.select_largest_region(output_sa)
            output_sa *= 3  # For right ventricle label
            output_sa = np.swapaxes(output_sa, 0, -1)
            output_sa = sitk.GetImageFromArray(output_sa)
            
            output_sa.CopyInformation(post_sitk[0]['input_sa'])
            output_sa = sitk.Resample(output_sa, pre_sitk[0]['input_sa'],
                                      interpolator=sitk.sitkNearestNeighbor)
            
            output_la = (prediction_dict['output_la'][0] >= threshold).astype(np.uint8)[..., 0]
            output_la = self.select_largest_region(output_la)
            output_la *= 3  # For right ventricle label
            output_la = np.expand_dims(output_la, -1)   # Add 3rd dimension back
            output_la = np.swapaxes(output_la, 0, -1)
            output_la = sitk.GetImageFromArray(output_la)
            
            output_la.CopyInformation(post_sitk[0]['input_la'])
            output_la = sitk.Resample(output_la, pre_sitk[0]['input_la'],
                                      interpolator=sitk.sitkNearestNeighbor)
            
            patient_id = os.path.basename(os.path.normpath(patient_directory))
            self.save_predictions(output_sa, output_la, patient_id, phase, output_directory)
            

    def predict(self, input_folder, output_folder):

        '''
        IMPORTANT: Mandatory. This function makes predictions for an entire test folder. 
        '''

        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        self.test_prediction(input_folder, output_folder)

    
    