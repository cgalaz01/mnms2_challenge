import os

from typing import Union
from pathlib import Path

import datetime
import random

import numpy as np

import tensorflow as tf
from tensorflow import keras

from tensorboard.plugins.hparams import api as hp

import matplotlib.pyplot as plt

from configuration import HyperParameters
from data import TensorFlowDataGenerator, DataGenerator
from tf.models import multi_stage_model
from tf.losses.loss import FocalLoss, TverskyLoss
from tf.metrics.metrics import dice


__SEED = 1456
os.environ['PYTHONHASHSEED'] = str(__SEED)
random.seed(__SEED)
tf.random.set_seed(__SEED)
np.random.seed(__SEED)


def get_callbacks(prefix: str, checkpoint_directory: str, hparams):
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_directory,
        save_weights_only=True,
        monitor='val_loss',
        mode='min',
        save_best_only=True)
    
    log_dir = os.path.join('logs', 'fit', prefix + datetime.datetime.now().strftime('_%Y%m%d-%H%M%S')) + '/'
    #file_writer = tf.summary.create_file_writer(log_dir + '\\metrics')
    #file_writer.set_as_default()
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    hparams_callback = hp.KerasCallback(log_dir, hparams)
    
    return [model_checkpoint_callback,
            tensorboard_callback,
            hparams_callback]


def visual_summary(model: keras.Model, generator: DataGenerator,
                   output_path: Union[str, Path]) -> None:
    os.makedirs(output_path, exist_ok=True)
    i = 0
    for data in generator.validation_affine_generator():
        # Add batch dimension to each of the input values
        for key, value in data[0].items():
            data[0][key] = np.expand_dims(value, 0)
        
        prediction_list = model.predict(data[0])
        prediction_dict = {name: pred for name, pred in zip(model.output_names, prediction_list)}

        predict_sa = np.argmax(prediction_dict['output_sa'][0], axis=-1).astype(np.uint8)
        gt_sa = np.argmax(data[1]['output_sa'])
        
        predict_la = np.argmax(prediction_dict['output_la'][0], axis=-1).astype(np.uint8)
        gt_la = np.argmax(data[1]['output_la'])
        
        output_file = os.path.join(output_path, str(i) + '.png')
        i += 1
        
        plt.subplot(221)
        plt.title('SA GT')
        plt.imshow(gt_sa[..., gt_sa.shape[-1] // 2])
        plt.subplot(222)
        plt.title('SA Prediction')
        plt.imshow(predict_sa[..., gt_sa.shape[-1] // 2])
        plt.subplot(223)
        plt.title('LA GT')
        plt.imshow(gt_la)
        plt.subplot(224)
        plt.title('LA Prediction')
        plt.imshow(predict_la)
        
        plt.savefig(output_file)
        plt.close()


if __name__ == '__main__':
    hyper_parameters = HyperParameters('grid')
    
    for hparams in hyper_parameters:
        keras.backend.clear_session()
        
        fp = hparams[hyper_parameters.HP_FLOATING_POINT]
        if fp == '16':
            policy = keras.mixed_precision.experimental.Policy('mixed_float16')
            keras.mixed_precision.experimental.set_policy(policy)
    
        use_xla = hparams[hyper_parameters.HP_XLA]
        if use_xla:
            tf.config.optimizer.set_jit('autoclustering')
    
        batch_size = hparams[hyper_parameters.HP_BATCH_SIZE]
        (train_gen, validation_gen,
         test_gen, data_gen) = TensorFlowDataGenerator.get_affine_generators(batch_size,
                                                                             max_buffer_size=None,
                                                                             floating_precision=fp)
        
                                                                      
        model = multi_stage_model.get_model(data_gen.sa_shape, data_gen.la_shape, data_gen.n_classes)
        
        
        learning_rate = hparams[hyper_parameters.HP_LEANRING_RATE]
        if hparams[hyper_parameters.HP_OPTIMISER] == 'adam':
            optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
            
        if hparams[hyper_parameters.HP_LOSS] == 'focal':
            # α should be decreased slightly as γ is increased
            loss = FocalLoss(0.25, 2.0)
        elif hparams[hyper_parameters.HP_LOSS] == 'tversky':
            loss = TverskyLoss(0.5, 0.5)
        elif hparams[hyper_parameters.HP_LOSS] == 'crossentropy':
            loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
            
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=[dice])
        
        epochs = hparams[hyper_parameters.HP_EPOCHS]
        prefix = 'multi_stage_model'
        checkpoint_path = os.path.join('tmp', 'checkpoint', prefix + datetime.datetime.now().strftime('_%Y%m%d-%H%M%S')) + '/'
        model.fit(x=train_gen,
                  validation_data=validation_gen,
                  epochs=epochs,
                  callbacks=get_callbacks(prefix, checkpoint_path, hparams),
                  verbose=1)
        
        model.load_weights(checkpoint_path)
        visual_summary(model, data_gen, 'tmp/output_results')

