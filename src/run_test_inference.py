import os

import numpy as np
import tensorflow as tf

import SimpleITK as sitk

from data import TensorFlowDataGenerator
from tf.losses.loss import FocalLoss
from tf.layers.transformer import TargetAffineLayer, TargetShapePad, TargetShapeCrop
from tf.metrics.metrics import dice


def load_model(modelpath) -> tf.keras.Model:
    tf.keras.backend.clear_session()
    model = tf.keras.models.load_model(modelpath, custom_objects={'dice': dice,
                                                                  'FocalLoss': FocalLoss,
                                                                  'TargetAffineLayer': TargetAffineLayer,
                                                                  'TargetShapePad': TargetShapePad,
                                                                  'TargetShapeCrop': TargetShapeCrop})
    
    return model


def save_predictions(sa_prediction: sitk.Image, la_prediction: sitk.Image,
                     patient_id: str, cardiac_phase: str) -> None:
    base_output = os.path.join('..', 'Submission')
    os.makedirs(base_output, exist_ok=True)
    
    patient_output = os.path.join(base_output, patient_id)
    os.makedirs(patient_output, exist_ok=True)
    
    cardiac_phase = cardiac_phase.upper()
    sa_file_output = os.path.join(patient_output, patient_id + '_SA_' +  cardiac_phase + '_pred.nii.gz')
    sitk.WriteImage(sa_prediction, sa_file_output)
    
    la_file_output = os.path.join(patient_output, patient_id + '_LA_' +  cardiac_phase + '_pred.nii.gz')
    sitk.WriteImage(la_prediction, la_file_output)
    
    
def test_prediction(model: tf.keras.Model) -> None:    
    (train_gen, validation_gen,
     test_gen, data_gen) = TensorFlowDataGenerator.get_affine_generators(batch_size=1,
                                                                         max_buffer_size=None,
                                                                         floating_precision='16')
                                            
    for data, pre_sitk, post_sitk, patient_directory, phase in data_gen.test_affine_generator_inference():
        # Add batch dimension to each of the input values
        for key, value in data[0].items():
            data[0][key] = np.expand_dims(value, 0)
        
        prediction_list = model.predict(data[0])
        prediction_dict = {name: pred for name, pred in zip(model.output_names, prediction_list)}
        
        output_sa = np.argmax(prediction_dict['output_sa'][0], axis=-1).astype(np.uint8)
        output_sa = np.swapaxes(output_sa, 0, -1)
        output_sa = sitk.GetImageFromArray(output_sa)
        
        output_sa.CopyInformation(post_sitk[0]['input_sa'])
        output_sa = sitk.Resample(output_sa, pre_sitk[0]['input_sa'])
        
        output_la = np.argmax(prediction_dict['output_la'][0], axis=-1).astype(np.uint8)
        output_la = np.expand_dims(output_la, -1)   # Add '3D' dimension back
        output_la = np.swapaxes(output_la, 0, -1)
        output_la = sitk.GetImageFromArray(output_la)
        
        output_la.CopyInformation(post_sitk[0]['input_la'])
        output_la = sitk.Resample(output_la, pre_sitk[0]['input_la'])
        
        patient_id = os.path.basename(os.path.normpath(patient_directory))
        save_predictions(output_sa, output_la, patient_id, phase)
      
    
if __name__ == '__main__':
    model_path = 'path/to/model'
    run_on_cpu = True
    device = 'cpu:0' if run_on_cpu else 'gpu:0'
    with tf.device(device):
        model = load_model(model_path)
        test_prediction(model)