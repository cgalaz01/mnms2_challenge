import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

from tf.layers.transformer import spatial_target_transformer

    
def shared_2d_branch(input_shape, kernel_initializer):
    shared_input = keras.layers.Input(shape=input_shape)
    
    x = layers.Conv2D(32, (9, 9), padding='same', activation='relu',
                      kernel_initializer=kernel_initializer)(shared_input)
    x = layers.Conv2D(64, (7, 7), padding='same', activation='relu',
                      kernel_initializer=kernel_initializer)(x)
    x = layers.Conv2D(128, (5, 5), padding='same', activation='relu',
                      kernel_initializer=kernel_initializer)(x)
    x = layers.Conv2D(17, (3, 3), padding='same', activation='relu',
                      kernel_initializer=kernel_initializer)(x)
    x = layers.DepthwiseConv2D((7, 7), padding='same', activation='relu',
                               kernel_initializer=kernel_initializer)(x)
    
    shared_model = keras.models.Model(shared_input, x)
    return shared_model


def shared_single_2d_branch(input_shape, kernel_initializer) -> keras.Model:
    shared_input = keras.layers.Input(shape=input_shape)
    
    x = layers.Conv2D(32, (9, 9), padding='same', activation='relu',
                      kernel_initializer=kernel_initializer)(shared_input)
    x = layers.Conv2D(64, (7, 7), padding='same', activation='relu',
                      kernel_initializer=kernel_initializer)(x)
    x = layers.Conv2D(64, (5, 5), padding='same', activation='relu',
                      kernel_initializer=kernel_initializer)(x)
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu',
                      kernel_initializer=kernel_initializer)(x)
    x = layers.Conv2D(input_shape[-1], (1, 1), padding='same', activation='relu',
                      kernel_initializer=kernel_initializer)(x)
    
    shared_model = keras.models.Model(shared_input, x)
    return shared_model
    

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
    
    x_la = shared_layers(x_la)
    
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
    

def get_affine_model(sa_input_shape, la_input_shape, num_classes) -> keras.Model:
    # A basic model to test pipeline and including affine matrices/spatial transformer
    
    kernel_initializer = 'glorot_uniform'
    
    input_sa = keras.Input(shape=sa_input_shape, name='input_sa')
    input_la = keras.Input(shape=la_input_shape, name='input_la')
    
    input_sa_affine = keras.Input(shape=(4, 4), name='input_sa_affine', dtype=tf.float32)
    input_la_affine = keras.Input(shape=(4, 4), name='input_la_affine', dtype=tf.float32)
    
    x_sa = input_sa
    x_la = input_la
    
    
    shared_layers = shared_single_2d_branch(la_input_shape, kernel_initializer)
    
        
    # Create 'channel' axis that will be carried over when unstacking
    x_sa = tf.expand_dims(x_sa, axis=-1)
    # Break the 3D image into single 2D slice input
    x_sa_list = tf.unstack(x_sa, axis=-2)
    # Pass each slice to the shared layer    
    for i in range(len(x_sa_list)):
        x_sa_list[i] = shared_layers(x_sa_list[i])
    
    # Stack back into a 3D image (W, H, D, C)
    x_sa = tf.stack(x_sa_list, axis=-2)
    
    # Short-Axis branch
    x_sa = layers.Conv3D(16, (5, 5, 3), padding='same', kernel_initializer=kernel_initializer)(x_sa)
    x_sa = layers.Activation('relu')(x_sa)
    
    x_sa = layers.Conv3D(32, (3, 3, 3), padding='same', kernel_initializer=kernel_initializer)(x_sa)
    x_sa = layers.Activation('relu')(x_sa)
    
    x_sa = layers.Conv3D(64, (3, 3, 3), padding='same', kernel_initializer=kernel_initializer)(x_sa)
    x_sa = layers.Activation('relu')(x_sa)
    
    output_sa = layers.Conv3D(num_classes, (1, 1, 1), padding='same',
                              kernel_initializer=kernel_initializer, name='output_sa')(x_sa)
    
    # Pass the long-axis slice through the shared layers
    x_la = shared_layers(x_la)
    
    # Long-Axis branch
    x_la = layers.Conv2D(32, (5, 5), padding='same', kernel_initializer=kernel_initializer)(x_la)
    x_la = layers.Activation('relu')(x_la)
    
    x_la = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer=kernel_initializer)(x_la)
    x_la = layers.Activation('relu')(x_la)
    
    x_la = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer=kernel_initializer)(x_la)
    x_la = layers.Activation('relu')(x_la)
      
    x_la = layers.Conv2D(num_classes, (1, 1), padding='same', kernel_initializer=kernel_initializer)(x_la)
    
    # output_sa or x_sa as input to spatial transformer
    x_la_t = spatial_target_transformer(output_sa, input_sa_affine, input_la_affine,
                                        sa_input_shape, la_input_shape)
    
    # Reshape from 3d to 2d (depth size is expected to be 1 after the spatial transformer)
    x_la_t = layers.Reshape((la_input_shape[0], la_input_shape[1], -1))(x_la_t)
    
    x_la = layers.Concatenate()([x_la, x_la_t])
    
    output_la = layers.Conv2D(num_classes, (1, 1), padding='same',
                              kernel_initializer=kernel_initializer, name='output_la')(x_la)
    
    model = keras.Model([input_sa, input_la, input_sa_affine, input_la_affine],
                        [output_sa, output_la])
    
    return model


if __name__ == '__main__':
    x = tf.reshape(tf.range(256*256*17), (256, 256, 17))
    x = tf.expand_dims(x, axis=-1)
    a = tf.unstack(x, axis=-2)
    print(a[0].shape.as_list())
    x = tf.stack(a, axis=-2)
    print(x.shape.as_list())
    
    