import os
import argparse

from typing import Union
from pathlib import Path

import numpy as np

from scipy import ndimage

import tensorflow as tf

import SimpleITK as sitk

from data import TensorFlowDataGenerator
from tf.models import multi_stage_model
from tf.losses.loss import combined_loss
from tf.metrics.metrics import soft_dice


class model:
    
    def __init__(self):
        '''
        IMPORTANT: Initializes the model wrapper WITHOUT PARAMETERS, it will be called as model()
        '''
        self.model_weights_path = os.path.join('model_weights', 'multi_stage_model') + '/'


    def load_model(self) -> tf.keras.Model:
        tf.keras.backend.clear_session()
        activation = 'selu'
        kernel_initializer = 'lecun_normal'
        dropout_rate = 0.
        
        model = multi_stage_model.get_model((192, 192, 17), (192, 192, 1), 1,
                                            activation, kernel_initializer, dropout_rate)
        
        learning_rate = 0.0005
        decay_after_epoch = 30
        steps = 316 # Total data per epoch
        decay_steps = steps / 1 * decay_after_epoch
        learning_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=learning_rate,
            decay_steps=decay_steps,
            decay_rate=0.9,
            staircase=True)
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_schedule)
        
        model.compile(
            optimizer=optimizer,
            loss=combined_loss,
            metrics=[soft_dice],
            loss_weights={'output_sa': 75,
                          'output_la': 1})
        
        # Load weights
        file_path = Path(__file__).parent.absolute()
        model_path = os.path.join(file_path, self.model_weights_path)
        model.load_weights(model_path)
        
        return model

    def select_largest_region(segmentation: sitk.Image) -> sitk.Image:
        connected_components = sitk.ConnectedComponent(segmentation)
        label_shape_stats = sitk.LabelShapeStatisticsImageFilter()
        label_shape_stats.Execute(connected_components)
        
        if label_shape_stats.GetNumberOfLabels() > 0:
            largest_component = max(label_shape_stats.GetLabels(), key=lambda l: label_shape_stats.GetPhysicalSize(l))
            segmentation = sitk.Equal(connected_components, largest_component)
                
        return segmentation

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
         test_gen, data_gen) = TensorFlowDataGenerator.get_generators(batch_size=1,
                                                                      max_buffer_size=None,
                                                                      floating_precision='16',
                                                                      memory_cache=False,
                                                                      disk_cache=False,
                                                                      test_directory=input_directory)
        
        model = self.load_model()
        
        threshold = -1
        
        for data, pre_sitk, post_sitk, patient_directory, phase in data_gen.test_generator_inference():
            # Add batch dimension to each of the input values
            for key, value in data[0].items():
                data[0][key] = np.expand_dims(value, 0)
            
            prediction_list = model.predict(data[0])
            prediction_dict = {name: pred for name, pred in zip(model.output_names, prediction_list)}
            
            output_sa = (prediction_dict['output_sa'][0] >= threshold).astype(np.uint8)[..., 0]
            output_sa = np.swapaxes(output_sa, 0, -1)
            output_sa = sitk.GetImageFromArray(output_sa)
            output_sa = self.select_largest_region(output_sa)
            output_sa *= 3  # For right ventricle label
            
            output_sa.CopyInformation(post_sitk[0]['input_sa'])
            output_sa = sitk.Resample(output_sa, pre_sitk[0]['input_sa'],
                                      interpolator=sitk.sitkNearestNeighbor)
            output_sa_tmp = sitk.GetArrayFromImage(output_sa)
            output_sa_tmp = ndimage.median_filter(output_sa_tmp, size=(1, 5, 5))
            output_sa_tmp = sitk.GetImageFromArray(output_sa_tmp)
            output_sa_tmp.CopyInformation(pre_sitk[0]['input_sa'])
            output_sa = output_sa_tmp
            
            output_la = (prediction_dict['output_la'][0] >= threshold).astype(np.uint8)[..., 0]
            output_la = np.expand_dims(output_la, -1)   # Add 3rd dimension back
            output_la = np.swapaxes(output_la, 0, -1)
            output_la = sitk.GetImageFromArray(output_la)
            output_la = self.select_largest_region(output_la)
            output_la *= 3  # For right ventricle label
            
            output_la.CopyInformation(post_sitk[0]['input_la'])
            output_la = sitk.Resample(output_la, pre_sitk[0]['input_la'],
                                      interpolator=sitk.sitkNearestNeighbor)
            output_la_tmp = sitk.GetArrayFromImage(output_la)
            output_la_tmp = ndimage.median_filter(output_la_tmp, size=(1, 5, 5))
            output_la_tmp = sitk.GetImageFromArray(output_la_tmp)
            output_la_tmp.CopyInformation(pre_sitk[0]['input_la'])
            output_la = output_la_tmp
            
            patient_id = os.path.basename(os.path.normpath(patient_directory))
            self.save_predictions(output_sa, output_la, patient_id, phase, output_directory)


    def predict(self, input_folder, output_folder):

        '''
        IMPORTANT: Mandatory. This function makes predictions for an entire test folder. 
        '''

        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        self.test_prediction(input_folder, output_folder)


def parse_arguments():
    parser = argparse.ArgumentParser(description='Tempera Model Inference')
    parser.add_argument('--input_path', type=str)
    parser.add_argument('--output_path', type=str)

    return parser.parse_args()
    

if __name__ == '__main__':
    m = model()
    parsed_args = parse_arguments()
    input_folder = Path(parsed_args.input_path)
    output_folder = Path(parsed_args.output_path)
    m.predict(input_folder, output_folder)
    