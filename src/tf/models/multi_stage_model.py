import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

from tf.layers.transformer import spatial_target_transformer



def _inception_block_a(x, num_filters, kernel_initializer, suffix, index):
    # Branch 1
    x1 = layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same',
                             name=suffix + '_inception_a_max_pooling_1_1_' + index)(x)
    x1 = layers.Conv2D(num_filters // 2, (1, 1), (1, 1), padding='same',
                       kernel_initializer=kernel_initializer,
                       name=suffix + '_inception_a_conv2d_1_2_' + index)(x1)
    x1 = layers.Activation('relu', name=suffix + '_inception_a_activation_1_3_' + index)(x1)
    
    # Branch 2
    x2 = layers.Conv2D(num_filters // 2, (1, 1), (1, 1), padding='same',
                       kernel_initializer=kernel_initializer,
                       name=suffix + '_inception_a_conv2d_2_1_' + index)(x)
    x2 = layers.Activation('relu', name=suffix + '_inception_a_activation_2_2_' + index)(x2)
    x2 = layers.Conv2D(num_filters // 2, (3, 3), (1, 1), padding='same',
                       kernel_initializer=kernel_initializer,
                       name=suffix + '_inception_a_conv2d_2_3_' + index)(x2)
    x2 = layers.Activation('relu', name=suffix + '_inception_a_activation_2_4_' + index)(x2)
    
    # Branch 3
    x3 = layers.Conv2D(num_filters // 2, (1, 1), (1, 1), padding='same',
                       kernel_initializer=kernel_initializer,
                       name=suffix + '_inception_a_conv2d_3_1_' + index)(x)
    x3 = layers.Activation('relu', name=suffix + '_inception_a_activation_3_2_' + index)(x3)
    x3 = layers.Conv2D(num_filters // 2, (3, 3), (1, 1), padding='same',
                       kernel_initializer=kernel_initializer,
                       name=suffix + '_inception_a_conv2d_3_3_' + index)(x3)
    x3 = layers.Activation('relu', name=suffix + '_inception_a_activation_3_4_' + index)(x3)
    x3 = layers.Conv2D(num_filters // 2, (3, 3), (1, 1), padding='same',
                       kernel_initializer=kernel_initializer,
                       name=suffix + '_inception_a_conv2d_3_5_' + index)(x3)
    x3 = layers.Activation('relu', name=suffix + '_inception_a_activation_3_6_' + index)(x3)
    
    # Branch 4
    x4 = layers.Conv2D(num_filters // 2, (1, 1), (1, 1), padding='same',
                       kernel_initializer=kernel_initializer,
                       name=suffix + '_inception_a_conv2d_4_1_' + index)(x)
    x4 = layers.Activation('relu', name=suffix + '_inception_a_activation_4_2_' + index)(x4)
    
    # Concatenate branches
    x = layers.Concatenate(axis=-1, name=suffix + '_inception_a_concatenate_' + index)([x, x1, x2, x3, x4])
    # Reduce filter size
    x = layers.Conv2D(num_filters // 2, (1, 1), (1, 1), padding='same',
                      kernel_initializer=kernel_initializer,
                      name=suffix + '_inception_a_conv2d_merge_' + index)(x)
    
    return x


def _inception_block_b(x, num_filters, kernel_initializer, suffix, index):
    # Branch 1
    x1 = layers.Conv2D(num_filters // 2, (1, 1), (1, 1), padding='same',
                       kernel_initializer=kernel_initializer,
                       name=suffix + '_inception_b_conv2d_1_1_' + index)(x)
    x1 = layers.Activation('relu', name=suffix + '_inception_b_activation_1_2_' + index)(x1)
    
    # Branch 2
    x2 = layers.Conv2D(num_filters // 2, (1, 1), (1, 1), padding='same',
                       kernel_initializer=kernel_initializer,
                       name=suffix + '_inception_b_conv2d_2_1_' + index)(x)
    x2 = layers.Activation('relu', name=suffix + '_inception_b_activation_2_2_' + index)(x2)
    x2 = layers.SeparableConv2D(num_filters // 2, (5, 5), (1, 1), padding='same',
                                depthwise_initializer=kernel_initializer,
                                pointwise_initializer=kernel_initializer,
                                name=suffix + '_inception_b_seperable_conv2d_2_3_' + index)(x2)
    x2 = layers.Activation('relu', name=suffix + '_inception_b_activation_2_4_' + index)(x2)
    
    # Branch 3
    x3 = layers.Conv2D(num_filters // 2, (1, 1), (1, 1), padding='same',
                       kernel_initializer=kernel_initializer,
                       name=suffix + '_inception_b_conv2d_3_1_' + index)(x)
    x3 = layers.Activation('relu', name=suffix + '_inception_b_activation_3_2_' + index)(x3)
    x3 = layers.Conv2D(num_filters // 2, (1, 1), (1, 1), padding='same',
                       kernel_initializer=kernel_initializer,
                       name=suffix + '_inception_b_conv2d_3_3_' + index)(x3)
    x3 = layers.Activation('relu', name=suffix + '_inception_b_activation_3_4_' + index)(x3)
    x3 = layers.SeparableConv2D(num_filters // 2, (1, 7), (1, 1), padding='same',
                                depthwise_initializer=kernel_initializer,
                                pointwise_initializer=kernel_initializer,
                                name=suffix + '_inception_b_seperable_conv2d_3_5_' + index)(x3)
    x3 = layers.Activation('relu', name=suffix + '_inception_b_activation_3_6_' + index)(x3)
    
    # Branch 3
    x4 = layers.Conv2D(num_filters // 2, (1, 1), (1, 1), padding='same',
                       kernel_initializer=kernel_initializer,
                       name=suffix + '_inception_b_conv2d_4_1_' + index)(x)
    x4 = layers.Activation('relu', name=suffix + '_inception_b_activation_4_2_' + index)(x4)
    x4 = layers.Conv2D(num_filters // 2, (1, 1), (1, 1), padding='same',
                       kernel_initializer=kernel_initializer,
                       name=suffix + '_inception_b_conv2d_4_3_' + index)(x4)
    x4 = layers.Activation('relu', name=suffix + '_inception_b_activation_4_4_' + index)(x4)
    x4 = layers.SeparableConv2D(num_filters // 2, (9, 9), (1, 1), padding='same',
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
    
    
def _shared_feature_pyramid_layers(num_pyramid_layers, input_shape, num_filters,
                                   kernel_initializer, suffix, index):
    shared_down_level = []
    for i in range(num_pyramid_layers):
        i_s = str(i + 1)
        shared_layers = []
        shared_layers.append(layers.Conv2D(num_filters, (3, 3), (1, 1), padding='same',
                                           kernel_initializer=kernel_initializer,
                                           name=suffix + '_pyramid_down_conv2d_' + i_s + '_1_' + index))
        shared_layers.append(layers.Activation('relu', name=suffix + '_pyramid_down_activation_' + i_s + '_2_' + index))
        shared_layers.append(layers.MaxPooling2D((2, 2), padding='same',
                             name=suffix + '_pyramid_down_max_pooling_' + i_s + '_3_' + index))
        x_pad_size = input_shape[0] // 4
        y_pad_size = input_shape[1] // 4
        shared_layers.append(layers.ZeroPadding2D((x_pad_size, y_pad_size),
                             name=suffix + '_pyramid_down_padding_' + i_s + '_4_' + index))
        
        shared_down_level.append(shared_layers)
    
    
    shared_up_level = []
    for i in range(num_pyramid_layers):
        i_s = str(i + 1)
        shared_layers = []
        shared_layers.append(layers.Conv2D(num_filters, (3, 3), (1, 1), padding='same',
                                           kernel_initializer=kernel_initializer,
                                           name=suffix + '_pyramid_up_conv2d_' + i_s + '_1_' + index))
        shared_layers.append(layers.Activation('relu', name=suffix + '_pyramid_up_activation_' + i_s + '_2_' + index))
        shared_layers.append(layers.UpSampling2D((2, 2), interpolation='bilinear',
                                                 name=suffix + '_pyramid_upsampling_' + i_s + '_3_' + index))
        x_crop_size = input_shape[0] // 2
        y_crop_size = input_shape[1] // 2
        shared_layers.append(layers.Cropping2D((x_crop_size, y_crop_size),
                             name=suffix + '_pyramid_up_cropping_' + i_s + '_4_' + index))
        
        shared_up_level.append(shared_layers)
    
    
    shared_skip = []
    for i in range(num_pyramid_layers - 1):
        i_s = str(i + 1)
        shared_layers = []
        shared_layers.append(layers.Conv2D(num_filters, (1, 1), (1, 1), padding='same',
                                           kernel_initializer=kernel_initializer,
                                           name=suffix + '_pyramid_skip_conv2d_' + i_s + '_1_' + index))
        shared_layers.append(layers.Activation('relu', name=suffix + '_pyramid_skip_activation_' + i_s + '_2_' + index))
        shared_layers.append(layers.Add(name=suffix + '_pyramid_skip_add_' + i_s + '_3_' + index))
        
        shared_skip.append(shared_layers)
    
    
    return shared_down_level, shared_up_level, shared_skip

    
def feature_pyramid_layer(x, pyramid_layers, input_shape, num_filters, kernel_initializer,
                          suffix, index):
    
    x_input = layers.Conv2D(num_filters, (1, 1), (1, 1), padding='same',
                            kernel_initializer=kernel_initializer,
                            name=suffix + '_pyramid_input_conv2d_1_' + index)(x)
    x_input = layers.Activation('relu', name=suffix + '_pyramid_input_activation_2_' + index)(x_input)
    

    # Initialise shared layers for the pyramid
    shared_down_level, shared_up_level, shared_skip = _shared_feature_pyramid_layers(pyramid_layers,
                                                                                     input_shape,
                                                                                     num_filters,
                                                                                     kernel_initializer,
                                                                                     suffix,
                                                                                     index)
    pyramid_output = []
    
    while True:
        x_skip = []
        x = x_input
        # Downsampling
        for i in range(pyramid_layers):
            shared_layers = shared_down_level[i]
            for j in range(len(shared_layers)):
                x = shared_layers[j](x)
            x_skip.append(x)        

        # Remove last element, as last layer does not have a skip connection
        del x_skip[-1]
        x_skip.reverse()
        
        # Upsampling
        
        for i in range(pyramid_layers):
            # Pass skip data and add with main data flow
            if i > 0:
                shared_skip_layers = shared_skip[i - 1]
                x_s = x_skip[i - 1]
                for s in range(len(shared_skip_layers) - 1):
                    x_s = shared_skip_layers[s](x_s)
                x = shared_skip_layers[-1]([x_s, x])
            
            shared_layers = shared_up_level[i]
            for j in range(len(shared_layers)):
                x = shared_layers[j](x)
        
                
        pyramid_output.append(x)
        
        pyramid_layers -= 1
        if pyramid_layers <= 0:
            break
    
        
    x = layers.Concatenate(axis=-1,
                           name=suffix + '_pyramid_output_concatenate_1_' + index)(pyramid_output)
    x = layers.Conv2D(num_filters, (1, 1), (1, 1), padding='same',
                      kernel_initializer=kernel_initializer,
                      name=suffix + '_pyramid_output_conv2d_2_' + index)(x)
    x = layers.Activation('relu', name=suffix + '_pyramid_output_activation_3_' + index)(x)
        
    return x
        
    
def _shared_2d_branch(input_shape, kernel_initializer, downsample=False) -> keras.Model:
    suffix = 'shared_branch'
    
    shared_input = keras.layers.Input(shape=input_shape, name='input_' + suffix)
    
    x = shared_input
    
    if downsample:
        # Downsample image to reduce total memory requirement
        original_dtype = x.dtype
        downsmapled_input_shape = (input_shape[0] // 2, input_shape[1] //2)
        x = tf.image.resize(x, downsmapled_input_shape, method=tf.image.ResizeMethod.BILINEAR,
                            antialias=True, name=suffix + 'image_resize_down')
        # Cast image back to original as 'resize' returns a Tensor of float32
        x = tf.cast(x, original_dtype, name=suffix + 'image_casting_down')
    
    # Pass input through inception pipeline
    x_inc = _inception_block_a(x, num_filters=32, kernel_initializer=kernel_initializer,
                               suffix=suffix, index='1')
    x_inc = _inception_block_a(x_inc, num_filters=64, kernel_initializer=kernel_initializer,
                               suffix=suffix, index='2')
    x_inc = _inception_block_a(x_inc, num_filters=64, kernel_initializer=kernel_initializer,
                               suffix=suffix, index='3')
    x_inc = _inception_block_b(x_inc, num_filters=128, kernel_initializer=kernel_initializer,
                               suffix=suffix, index='4')
    
    # Pass input through multi-level feature pyramid pipeline
    x_pyr = feature_pyramid_layer(x, pyramid_layers=3, input_shape=downsmapled_input_shape,
                                  num_filters=64, kernel_initializer=kernel_initializer,
                                  suffix=suffix, index='1')
    
    x = layers.Add(name=suffix + '_add_1')([x_inc, x_pyr])
    
    if downsample:
        # Upsample image back to original resolution
        upsample_input_shape = (input_shape[0], input_shape[1])
        x = tf.image.resize(x, upsample_input_shape, method=tf.image.ResizeMethod.BILINEAR,
                            name=suffix + 'image_resize_up')
        x = tf.cast(x, original_dtype, name=suffix + 'image_casting_up')
        
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
    
    
    shared_layers = _shared_2d_branch(la_input_shape, kernel_initializer, downsample=False)
        
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
    x_sa = layers.Conv3D(32, (3, 3, 3), padding='same', kernel_initializer=kernel_initializer)(x_sa)
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
    x_la = layers.Conv2D(32, (3, 3), padding='same', kernel_initializer=kernel_initializer)(x_la)
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
    
    x_la = layers.Conv2D(32, (3, 3), padding='same', kernel_initializer=kernel_initializer)(x_la)
    x_la = layers.Activation('relu')(x_la)
    
    output_la = layers.Conv2D(num_classes, (1, 1), padding='same',
                              kernel_initializer=kernel_initializer, name='output_la')(x_la)
    
    model = keras.Model([input_sa, input_la, input_sa_affine, input_la_affine],
                        [output_sa, output_la])
    
    return model

