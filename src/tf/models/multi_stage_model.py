import math

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

from tf.layers.transformer import spatial_target_transformer

    

def _shared_feature_pyramid_layers(num_pyramid_layers, input_shape, num_filters,
                                   kernel_initializer, activation, dropout_rate,
                                   suffix, index):
    shared_down_level = []
    for i in range(num_pyramid_layers):
        i_s = str(i + 1)
        shared_layers = []
        shared_layers.append(layers.Conv2D(num_filters, (3, 3), (1, 1), padding='same',
                                           kernel_initializer=kernel_initializer,
                                           name=suffix + '_pyramid_down_conv2d_' + i_s + '_1_' + index))
        shared_layers.append(layers.Activation(activation, name=suffix + '_pyramid_down_activation_' + i_s + '_2_' + index))
        shared_layers.append(layers.Conv2D(num_filters, (3, 3), (1, 1), padding='same',
                                           kernel_initializer=kernel_initializer,
                                           name=suffix + '_pyramid_down_conv2d_' + i_s + '_3_' + index))
        shared_layers.append(layers.Activation(activation, name=suffix + '_pyramid_down_activation_' + i_s + '_4_' + index))
        shared_layers.append(layers.MaxPooling2D((2, 2), padding='same',
                             name=suffix + '_pyramid_down_max_pooling_' + i_s + '_5_' + index))
        x_pad_size = input_shape[0] // 4
        y_pad_size = input_shape[1] // 4
        shared_layers.append(layers.ZeroPadding2D((x_pad_size, y_pad_size),
                             name=suffix + '_pyramid_down_padding_' + i_s + '_6_' + index))
        
        shared_down_level.append(shared_layers)
    
    
    shared_up_level = []
    for i in range(num_pyramid_layers):
        i_s = str(i + 1)
        shared_layers = []
        shared_layers.append(layers.Conv2D(num_filters, (3, 3), (1, 1), padding='same',
                                           kernel_initializer=kernel_initializer,
                                           name=suffix + '_pyramid_up_conv2d_' + i_s + '_1_' + index))
        shared_layers.append(layers.Activation(activation, name=suffix + '_pyramid_up_activation_' + i_s + '_2_' + index))
        shared_layers.append(layers.Conv2D(num_filters, (3, 3), (1, 1), padding='same',
                                           kernel_initializer=kernel_initializer,
                                           name=suffix + '_pyramid_up_conv2d_' + i_s + '_3_' + index))
        shared_layers.append(layers.Activation(activation, name=suffix + '_pyramid_up_activation_' + i_s + '_4_' + index))
        
        x_crop_size = input_shape[0] // 4
        y_crop_size = input_shape[1] // 4
        shared_layers.append(layers.Cropping2D((x_crop_size, y_crop_size),
                             name=suffix + '_pyramid_up_cropping_' + i_s + '_5_' + index))
        shared_layers.append(layers.UpSampling2D((2, 2), interpolation='bilinear',
                                                 name=suffix + '_pyramid_upsampling_' + i_s + '_6_' + index))
        
        shared_up_level.append(shared_layers)
    
    
    shared_skip = []
    for i in range(num_pyramid_layers):
        i_s = str(i + 1)
        shared_layers = []
        shared_layers.append(layers.Conv2D(num_filters, (1, 1), (1, 1), padding='same',
                                           kernel_initializer=kernel_initializer,
                                           name=suffix + '_pyramid_skip_conv2d_' + i_s + '_1_' + index))
        shared_layers.append(layers.Activation(activation, name=suffix + '_pyramid_skip_activation_' + i_s + '_2_' + index))
        shared_layers.append(layers.Add(name=suffix + '_pyramid_skip_add_' + i_s + '_3_' + index))
        
        shared_skip.append(shared_layers)
    
    
    ratio = 8
    shared_squeeze_excitation = []
    for i in range(num_pyramid_layers):
        i_s = str(i + 1)
        shared_layers = []
        # Squeeze and Excitation block
        shared_layers.append(layers.GlobalAveragePooling2D(name=suffix + '_pyramid_se_globalaveragepooling2d_' + i_s + '_1_' + index))
        shared_layers.append(layers.Dense(num_filters // ratio, use_bias=False,
                                          kernel_initializer=kernel_initializer,
                                          name=suffix + '_pyramid_se_dense_' + i_s + '_2_' + index))
        shared_layers.append(layers.Activation(activation, name=suffix + '_pyramid_se_activation_' + i_s + '_3_' + index))
        shared_layers.append(layers.Dense(num_filters, use_bias=False,
                                          kernel_initializer=kernel_initializer,
                                          name=suffix + '_pyramid_se_dense_' + i_s + '_4_' + index))
        shared_layers.append(layers.Activation('sigmoid', name=suffix + '_pyramid_se_activation_' + i_s + '_5_' + index))
        shared_layers.append(layers.Multiply(name=suffix + '_pyramid_se_multiply_' + i_s + '_6_' + index))
        
        shared_squeeze_excitation.append(shared_layers)
        
        
    return shared_down_level, shared_up_level, shared_skip, shared_squeeze_excitation

    
def feature_pyramid_layer(x, pyramid_layers, input_shape, num_filters, kernel_initializer,
                          activation, dropout_rate, suffix, index):
    
    x_input = layers.Conv2D(num_filters, (1, 1), (1, 1), padding='same',
                            kernel_initializer=kernel_initializer,
                            name=suffix + '_pyramid_input_conv2d_1_' + index)(x)
    x_input = layers.Activation(activation, name=suffix + '_pyramid_input_activation_2_' + index)(x_input)
    

    # Initialise shared layers for the pyramid
    (shared_down_level, shared_up_level,
     shared_skip, shared_squeeze_excitation) = _shared_feature_pyramid_layers(pyramid_layers,
                                                                              input_shape,
                                                                              num_filters,
                                                                              kernel_initializer,
                                                                              activation,
                                                                              dropout_rate,
                                                                              suffix,
                                                                              index)
    pyramid_output = []
    pyramid_index = 0
    while True:
        x_skip = []
        x = x_input
        # Downsampling
        for i in range(pyramid_index, pyramid_layers):
            x_skip.append(x)
            shared_layers = shared_down_level[i]
            for j in range(len(shared_layers)):
                x = shared_layers[j](x)        

        x_skip.reverse()
        
        # Upsampling
        for i in range(pyramid_index, pyramid_layers):
            shared_layers = shared_up_level[i]
            for j in range(len(shared_layers)):
                x = shared_layers[j](x)
            
            # Skip connection
            shared_skip_layers = shared_skip[i]
            x_s = x_skip[i - pyramid_index]
            for s in range(len(shared_skip_layers) - 1):
                x_s = shared_skip_layers[s](x_s)
            x = shared_skip_layers[-1]([x_s, x])
            
            # Skip squeeze and excitation connection
            shared_se_layers = shared_squeeze_excitation[i]
            x_se = x_skip[i - pyramid_index]
            for se in range(len(shared_se_layers) - 1):
                x_se = shared_se_layers[se](x_se)
            x = shared_se_layers[-1]([x, x_se])
        
                
        pyramid_output.append(x)

        pyramid_index += 1
        if pyramid_index - 1 >= pyramid_layers:
            break
    
        
    x = layers.Concatenate(axis=-1,
                           name=suffix + '_pyramid_output_concatenate_1_' + index)(pyramid_output)
    x = layers.Conv2D(num_filters, (1, 1), (1, 1), padding='same',
                      kernel_initializer=kernel_initializer,
                      name=suffix + '_pyramid_output_conv2d_2_' + index)(x)
    x = layers.Activation(activation, name=suffix + '_pyramid_output_activation_3_' + index)(x)
        
    return x
        
    
def _shared_2d_branch(input_shape, kernel_initializer, activation, dropout_rate) -> keras.Model:
    suffix = 'shared_branch'
    
    shared_input = keras.layers.Input(shape=input_shape, name='input_' + suffix)
    
    x = shared_input

    # Pass input through multi-level feature pyramid pipeline
    x = feature_pyramid_layer(x, pyramid_layers=3, input_shape=input_shape,
                              num_filters=128, kernel_initializer=kernel_initializer,
                              activation=activation, dropout_rate=dropout_rate,
                              suffix=suffix, index='1')
    
    shared_model = keras.models.Model(shared_input, x)
    
    return shared_model


def get_model(sa_input_shape, la_input_shape, num_classes, activation,
              kernel_initializer, dropout_rate) -> keras.Model:
    
    # The short-axis image is expected to have its 3rd dimension as channels: (B, W, H, C)    
    input_sa = keras.Input(shape=sa_input_shape, name='input_sa')
    input_la = keras.Input(shape=la_input_shape, name='input_la')
    
    input_sa_affine = keras.Input(shape=(4, 4), name='input_sa_affine', dtype=tf.float32)
    input_la_affine = keras.Input(shape=(4, 4), name='input_la_affine', dtype=tf.float32)
    
    x_sa = input_sa
    x_la = input_la
    
    # Compute the biases for the output layers
    p_sa = 0.01
    p_la = 0.01
    bias_sa = math.log10((1 - p_sa) / p_sa)
    bias_la = math.log10((1 - p_la) / p_la)
    
    shared_layers = _shared_2d_branch(la_input_shape, kernel_initializer, activation,
                                      dropout_rate)
        
    # Create 'channel' axis that will be carried over when unstacking
    x_sa = layers.Lambda(lambda x: tf.expand_dims(x, axis=-1))(x_sa)
    # Break the 3D image into single 2D slice input
    x_sa_list = layers.Lambda(lambda x: tf.unstack(x, axis=-2))(x_sa)
    # Pass each slice to the shared layer    
    for i in range(len(x_sa_list)):
        x_sa_list[i] = shared_layers(x_sa_list[i])
    
    # Stack back into a 3D image (W, H, D, C)
    x_sa = layers.Lambda(lambda x: tf.stack(x, axis=-2))(x_sa_list)
    x_sa_skip = x_sa
    
    # Short-Axis branch
    x_sa = layers.Conv3D(64, (3, 3, 3), padding='same', kernel_initializer=kernel_initializer,
                          name='sa_conv3d_1_1')(x_sa)
    x_sa = layers.Activation('relu', name='sa_activation_1_2')(x_sa)
    
    x_sa = layers.Conv3D(64, (3, 3, 3), padding='same', kernel_initializer=kernel_initializer,
                          name='sa_conv3d_2_1')(x_sa)
    x_sa = layers.Activation('relu', name='sa_activation_2_2')(x_sa)
    
    x_sa = layers.Conv3D(128, (3, 3, 3), padding='same', kernel_initializer=kernel_initializer,
                          name='sa_conv3d_3_1')(x_sa)
    x_sa = layers.Activation('relu', name='sa_activation_3_2')(x_sa)

    x_sa = layers.Add(name='sa_add_4_1')([x_sa, x_sa_skip])
    
    output_sa = layers.Conv3D(num_classes, (1, 1, 1), padding='same',
                              kernel_initializer=kernel_initializer,
                              bias_initializer=keras.initializers.Constant(bias_sa),
                              name='output_sa')(x_sa)
    
    # Pass the long-axis slice through the shared layers
    x_la = shared_layers(x_la)
    x_la_skip = x_la
    
    # Long-Axis branch
    x_la = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer=kernel_initializer,
                          name='la_conv2d_1_1')(x_la)
    x_la = layers.Activation(activation, name='la_activation_1_2')(x_la)
    
    x_la = layers.Conv2D(64, (3, 3), padding='same', kernel_initializer=kernel_initializer,
                          name='la_conv2d_2_1')(x_la)
    x_la = layers.Activation(activation, name='la_activation_2_2')(x_la)
    
    x_la = layers.Conv2D(128, (3, 3), padding='same', kernel_initializer=kernel_initializer,
                          name='la_conv2d_3_1')(x_la)
    x_la = layers.Activation(activation, name='la_activation_3_2')(x_la)
      
    x_la = layers.Add(name='la_add_4_1')([x_la, x_la_skip])
    
    x_la = layers.Conv2D(num_classes, (1, 1), padding='same', kernel_initializer=kernel_initializer,
                          name='la_conv2d_5_1')(x_la)
    
    # output_sa or x_sa as input to spatial transformer
    x_la_t = spatial_target_transformer(output_sa, input_sa_affine, input_la_affine,
                                        sa_input_shape, la_input_shape)
    
    # # Reshape from 3d to 2d (depth size is expected to be 1 after the spatial transformer)
    x_la_t = layers.Reshape((la_input_shape[0], la_input_shape[1], -1))(x_la_t)

    x_la = layers.Concatenate(name='la_concatenate')([x_la, x_la_t])
    
    x_la = layers.Conv2D(32, (3, 3), padding='same', kernel_initializer=kernel_initializer,
                          name='la_conv2d_6_1')(x_la)
    x_la = layers.Activation(activation, name='la_activation_6_2')(x_la)
    
    output_la = layers.Conv2D(num_classes, (1, 1), padding='same',
                              kernel_initializer=kernel_initializer,
                              bias_initializer=keras.initializers.Constant(bias_la),
                              name='output_la')(x_la)
    
    model = keras.Model(inputs={'input_sa': input_sa, 'input_la': input_la,
                                'input_sa_affine': input_sa_affine, 'input_la_affine': input_la_affine},
                        outputs={'output_sa': output_sa, 'output_la': output_la})
    
    return model

if __name__ == '__main__':
    m = get_model((192, 192, 17), (192, 192, 17), 1, activation='relu',
                  kernel_initializer='lecun_normal', dropout_rate=0.0)
    m.summary()