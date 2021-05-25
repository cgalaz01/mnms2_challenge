import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers


def shared_2d_branch(input_shape, kernel_initializer):
    model = keras.Sequential()
    
    model.add(layers.DepthwiseConv2D((9, 9), padding='same', activation='relu',
                            input_shape=input_shape, kernel_initializer=kernel_initializer))
    model.add(layers.DepthwiseConv2D((7, 7), padding='same', activation='relu',
                            kernel_initializer=kernel_initializer))
    model.add(layers.DepthwiseConv2D((5, 5), padding='same', activation='relu',
                            kernel_initializer=kernel_initializer))
    model.add(layers.DepthwiseConv2D((3, 3), padding='same', activation='relu',
                            kernel_initializer=kernel_initializer))
    model.add(layers.DepthwiseConv2D((3, 3), padding='same', activation='relu',
                            kernel_initializer=kernel_initializer))
    model.add(layers.DepthwiseConv2D((3, 3), padding='same', activation='relu',
                            kernel_initializer=kernel_initializer))
    
    return model
    
    

def get_model(sa_input_shape, la_input_shape, num_classes) -> keras.Model:
    # A basic model to test pipeline
    
    kernel_initializer = 'glorot_uniform'
    
    input_sa = keras.Input(shape=sa_input_shape, name='input_sa')
    input_la = keras.Input(shape=la_input_shape, name='input_la')
    
    x_sa = input_sa
    x_la = input_la
    
    # Transform the long-axis image to have the same number of channels as the
    # short-axis (so they can be passed to the shared branch)
    x_la = layers.Conv2D(sa_input_shape[-1], (3, 3), padding='same',
                         kernel_initializer=kernel_initializer)(x_la)
    x_la = layers.Activation('relu')(x_la)
    
    shared_layers = shared_2d_branch(sa_input_shape, kernel_initializer)
    x_la = shared_layers(x_la)
    x_sa = shared_layers(x_sa)
    
    # Now predict each one independantly
    # Short-Axis branch
    # Reshape the image so that it is treated as a 3D image (W, H, D) to (W, H, D, C)
    x_sa = tf.expand_dims(x_sa, axis=-1)
    
    x_sa = layers.Conv3D(16, (5, 5, 3), padding='same', kernel_initializer=kernel_initializer)(x_sa)
    x_sa = layers.Activation('relu')(x_sa)
    
    x_sa = layers.Conv3D(32, (3, 3, 3), padding='same', kernel_initializer=kernel_initializer)(x_sa)
    x_sa = layers.Activation('relu')(x_sa)
    
    x_sa = layers.Conv3D(64, (3, 3, 3), padding='same', kernel_initializer=kernel_initializer)(x_sa)
    x_sa = layers.Activation('relu')(x_sa)
    
    output_sa = layers.Conv3D(num_classes, (1, 1, 1), padding='same',
                              kernel_initializer=kernel_initializer, name='output_sa')(x_sa)
    
    # Long-Axis branch
    x_la = layers.Conv2D(32, (5, 5), padding='same', kernel_initializer=kernel_initializer)(x_la)
    x_la = layers.Activation('relu')(x_la)
    
    x_la = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer=kernel_initializer)(x_la)
    x_la = layers.Activation('relu')(x_la)
    
    x_la = layers.Conv2D(128, (3, 3), padding='same', kernel_initializer=kernel_initializer)(x_la)
    x_la = layers.Activation('relu')(x_la)
    
    output_la = layers.Conv2D(num_classes, (1, 1), padding='same',
                              kernel_initializer=kernel_initializer, name='output_la')(x_la)
    
    
    model = keras.Model([input_sa, input_la], [output_sa, output_la])
    
    return model
    