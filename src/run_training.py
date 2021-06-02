import os

import datetime
import random

import numpy as np

import tensorflow as tf
from tensorflow import keras

from tensorboard.plugins.hparams import api as hp

from configuration import HyperParameters
from data import TensorFlowDataGenerator
from tf.models import test_model
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
        save_weights_only=False,
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


if __name__ == '__main__':
    hyper_parameters = HyperParameters('grid')
    
    for hparams in hyper_parameters:
        keras.backend.clear_session()
        
        fp = hparams[hyper_parameters.HP_FLOATING_POINT]
        if fp == '16':
            policy = keras.mixed_precision.experimental.Policy('mixed_float16')
            keras.mixed_precision.experimental.set_policy(policy)
    
        batch_size = hparams[hyper_parameters.HP_BATCH_SIZE]
        (train_gen, validation_gen,
         test_gen, data_gen) = TensorFlowDataGenerator.get_affine_generators(batch_size,
                                                                             max_buffer_size=None,
                                                                             floating_precision=fp)
        
                                                                      
        model = test_model.get_affine_model(data_gen.sa_shape, data_gen.la_shape, data_gen.n_classes)
        
        
        learning_rate = hparams[hyper_parameters.HP_LEANRING_RATE]
        if hparams[hyper_parameters.HP_OPTIMISER] == 'adam':
            optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
            
        if hparams[hyper_parameters.HP_LOSS] == 'focal':
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
        prefix = 'test_model'
        checkpoint_path = os.path.join('tmp', 'checkpoint', prefix + datetime.datetime.now().strftime('_%Y%m%d-%H%M%S')) + '/'
        model.fit(x=train_gen,
                  validation_data=validation_gen,
                  epochs=epochs,
                  callbacks=get_callbacks(prefix, checkpoint_path, hparams),
                  verbose=1)
        
