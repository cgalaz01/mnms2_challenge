import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

from tf.layers.transformer import spatial_target_transformer



def _inception_block_a(x, num_filters, kernel_initializer, suffix, index):
    # Branch 1
    x1 = layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same',
                             name=suffix + '_inception_a_max_pooling_1_1_' + index)(x)
    x1 = layers.Conv2D(num_filters, (1, 1), (1, 1), padding='same',
                       kernel_initializer=kernel_initializer,
                       name=suffix + '_inception_a_conv2d_1_2_' + index)(x1)
    x1 = layers.Activation('relu', name=suffix + '_inception_a_activation_1_3_' + index)(x1)
    
    # Branch 2
    x2 = layers.Conv2D(num_filters, (1, 1), (1, 1), padding='same',
                       kernel_initializer=kernel_initializer,
                       name=suffix + '_inception_a_conv2d_2_1_' + index)(x)
    x2 = layers.Activation('relu', name=suffix + '_inception_a_activation_2_2_' + index)(x2)
    x2 = layers.Conv2D(num_filters, (3, 3), (1, 1), padding='same',
                       kernel_initializer=kernel_initializer,
                       name=suffix + '_inception_a_conv2d_2_3_' + index)(x2)
    x2 = layers.Activation('relu', name=suffix + '_inception_a_activation_2_4_' + index)(x2)
    
    # Branch 3
    x3 = layers.Conv2D(num_filters, (1, 1), (1, 1), padding='same',
                       kernel_initializer=kernel_initializer,
                       name=suffix + '_inception_a_conv2d_3_1_' + index)(x)
    x3 = layers.Activation('relu', name=suffix + '_inception_a_activation_3_2_' + index)(x3)
    x3 = layers.Conv2D(num_filters, (3, 3), (1, 1), padding='same',
                       kernel_initializer=kernel_initializer,
                       name=suffix + '_inception_a_conv2d_3_3_' + index)(x3)
    x3 = layers.Activation('relu', name=suffix + '_inception_a_activation_3_4_' + index)(x3)
    x3 = layers.Conv2D(num_filters, (3, 3), (1, 1), padding='same',
                       kernel_initializer=kernel_initializer,
                       name=suffix + '_inception_a_conv2d_3_5_' + index)(x3)
    x3 = layers.Activation('relu', name=suffix + '_inception_a_activation_3_6_' + index)(x3)
    
    # Branch 4
    x4 = layers.Conv2D(num_filters, (1, 1), (1, 1), padding='same',
                       kernel_initializer=kernel_initializer,
                       name=suffix + '_inception_a_conv2d_4_1_' + index)(x)
    x4 = layers.Activation('relu', name=suffix + '_inception_a_activation_4_2_' + index)(x4)
    
    # Concatenate branches
    x = layers.Concatenate(axis=-1, name=suffix + '_inception_a_concatenate_' + index)([x, x1, x2, x3, x4])
    # Reduce filter size
    x = layers.Conv2D(num_filters, (1, 1), (1, 1), padding='same',
                      kernel_initializer=kernel_initializer,
                      name=suffix + '_inception_a_conv2d_merge_' + index)(x)
    
    return x


def _inception_block_b(x, num_filters, kernel_initializer, suffix, index):
    # Branch 1
    x1 = layers.Conv2D(num_filters, (1, 1), (1, 1), padding='same',
                       kernel_initializer=kernel_initializer,
                       name=suffix + '_inception_b_conv2d_1_1_' + index)(x)
    x1 = layers.Activation('relu', name=suffix + '_inception_b_activation_1_2_' + index)(x1)
    
    # Branch 2
    x2 = layers.Conv2D(num_filters, (1, 1), (1, 1), padding='same',
                       kernel_initializer=kernel_initializer,
                       name=suffix + '_inception_b_conv2d_2_1_' + index)(x)
    x2 = layers.Activation('relu', name=suffix + '_inception_b_activation_2_2_' + index)(x2)
    x2 = layers.SeparableConv2D(num_filters, (5, 5), (1, 1), padding='same',
                                depthwise_initializer=kernel_initializer,
                                pointwise_initializer=kernel_initializer,
                                name=suffix + '_inception_b_seperable_conv2d_2_3_' + index)(x2)
    x2 = layers.Activation('relu', name=suffix + '_inception_b_activation_2_4_' + index)(x2)
    
    # Branch 3
    x3 = layers.Conv2D(num_filters, (1, 1), (1, 1), padding='same',
                       kernel_initializer=kernel_initializer,
                       name=suffix + '_inception_b_conv2d_3_1_' + index)(x)
    x3 = layers.Activation('relu', name=suffix + '_inception_b_activation_3_2_' + index)(x3)
    x3 = layers.Conv2D(num_filters, (1, 1), (1, 1), padding='same',
                       kernel_initializer=kernel_initializer,
                       name=suffix + '_inception_b_conv2d_3_3_' + index)(x3)
    x3 = layers.Activation('relu', name=suffix + '_inception_b_activation_3_4_' + index)(x3)
    x3 = layers.SeparableConv2D(num_filters, (7, 7), (1, 1), padding='same',
                                depthwise_initializer=kernel_initializer,
                                pointwise_initializer=kernel_initializer,
                                name=suffix + '_inception_b_seperable_conv2d_3_5_' + index)(x3)
    x3 = layers.Activation('relu', name=suffix + '_inception_b_activation_3_6_' + index)(x3)
    
    # Branch 3
    x4 = layers.Conv2D(num_filters, (1, 1), (1, 1), padding='same',
                       kernel_initializer=kernel_initializer,
                       name=suffix + '_inception_b_conv2d_4_1_' + index)(x)
    x4 = layers.Activation('relu', name=suffix + '_inception_b_activation_4_2_' + index)(x4)
    x4 = layers.Conv2D(num_filters, (1, 1), (1, 1), padding='same',
                       kernel_initializer=kernel_initializer,
                       name=suffix + '_inception_b_conv2d_4_3_' + index)(x4)
    x4 = layers.Activation('relu', name=suffix + '_inception_b_activation_4_4_' + index)(x4)
    x4 = layers.SeparableConv2D(num_filters, (9, 9), (1, 1), padding='same',
                                depthwise_initializer=kernel_initializer,
                                pointwise_initializer=kernel_initializer,
                                name=suffix + '_inception_b_seperable_conv2d_4_5_' + index)(x4)
    x4 = layers.Activation('relu', name=suffix + '_inception_b_activation_4_6_' + index)(x4)
    
    # Concatenate branches
    x = layers.Concatenate(axis=-1, name=suffix + '_inception_b_concatenate_' + index)([x, x1, x2, x3, x4])
    # Reduce filter size
    x = layers.Conv2D(num_filters, (1, 1), (1, 1), padding='same',
                      kernel_initializer=kernel_initializer,
                      name=suffix + '_inception_b_conv2d_merge_' + index)(x)
    
    return x
    
    
    
def _shared_2d_branch(input_shape, kernel_initializer) -> keras.Model:
    shared_input = keras.layers.Input(shape=input_shape)
    suffix = 'shared_branch'
    
    x = shared_input
    x = _inception_block_a(x, num_filters=64, kernel_initializer=kernel_initializer,
                           suffix=suffix, index='1')
    x = _inception_block_a(x, num_filters=64, kernel_initializer=kernel_initializer,
                           suffix=suffix, index='2')
    x = _inception_block_b(x, num_filters=64, kernel_initializer=kernel_initializer,
                           suffix=suffix, index='3')
    
    
    shared_model = keras.models.Model(shared_input, x)
    return shared_model


def get_model(sa_input_shape, la_input_shape, num_classes) -> keras.Model:
    kernel_initializer = 'glorot_uniform'
    
    # The short-axis image is expected to have its 3rd dimension as channels: (B, W, H, C)    
    input_sa = keras.Input(shape=sa_input_shape, name='input_sa')
    input_la = keras.Input(shape=la_input_shape, name='input_la')
    
    input_sa_affine = keras.Input(shape=(4, 4), name='input_sa_affine', dtype=tf.float32)
    input_la_affine = keras.Input(shape=(4, 4), name='input_la_affine', dtype=tf.float32)
    
    x_sa = input_sa
    x_la = input_la
    
    
    shared_layers = _shared_2d_branch(la_input_shape, kernel_initializer)
        
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

