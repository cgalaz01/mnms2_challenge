import tensorflow as tf
from tensorflow.keras.layers import Layer

from voxelmorph.tf.layers import SpatialTransformer


class TargetAffineLayer(Layer):
    
    def __init__(self, **kwargs):
        super(self.__class__, self).__init__(**kwargs)
    
    
    @tf.autograph.experimental.do_not_convert
    def _get_transformation(self, inputs):
        image_affine = inputs[0]
        target_affine = inputs[1]
        
        affine_transform = tf.cond(tf.reduce_all(tf.math.equal(target_affine, image_affine)),
                                   lambda: tf.eye(4, dtype=image_affine.dtype),
                                   lambda: tf.tensordot(tf.linalg.inv(image_affine),
                                                        target_affine, axes=1))
        
        return affine_transform
        
    
    def get_config(self):
        config = super().get_config().copy()
        return config
    
    
    def call(self, inputs):
        """
        Parameters
            inputs: list with four entries
        """
        # check shapes
        assert len(inputs) == 2, "inputs has to be len 2, found: %d" % len(inputs)
        image_affine = tf.cast(inputs[0], dtype=tf.float32)
        target_affine = tf.cast(inputs[1], dtype=tf.float32)
        
        affine_transform = tf.map_fn(self._get_transformation,
                                     [image_affine, target_affine],
                                     dtype=tf.float32)
        
        return affine_transform
    


class TargetShapePad(Layer):
    
    def __init__(self, image_shape, target_shape, **kwargs):
        super(self.__class__, self).__init__(**kwargs)

        self.paddings = [(0, 0),
                         (0, 0),
                         (0, 0)]
        
        self.init_config = {'image_shape': image_shape, 'target_shape': target_shape, **kwargs}
    
    
    def get_config(self):
        return self.init_config
    
    
    def call(self, inputs):
        padded_image = tf.keras.layers.ZeroPadding3D(self.paddings)(inputs)

        return padded_image



class TargetShapeCrop(Layer):
    
    def __init__(self, image_shape, target_shape, **kwargs):
        super(self.__class__, self).__init__(**kwargs)
        
        self.cropping = [(0, 0),
                         (0, 0),
                         (0, 16)]
        
        self.init_config = {'image_shape': image_shape, 'target_shape': target_shape, **kwargs}
        
    
    def get_config(self):
        return self.init_config


    def call(self, inputs):
        cropped_image = tf.keras.layers.Cropping3D(self.cropping)(inputs)
        
        return cropped_image
    
    
def spatial_target_transformer(x, affine_matrix, target_affine_matrix,
                               image_shape, target_image_shape):
    affine = TargetAffineLayer()([affine_matrix, target_affine_matrix])
    
    x = TargetShapePad(image_shape, target_image_shape)(x)
    
    original_dtype = x.dtype
    x = tf.cast(x, dtype=tf.float32)
    x = SpatialTransformer(interp_method='linear',
                           indexing='ij',
                           add_identity=False,
                           shift_center=False,
                           fill_value=0.0,
                           dtype=tf.float32)([x, affine])
    x = tf.cast(x, dtype=original_dtype)
    
    x = TargetShapeCrop(image_shape, target_image_shape)(x)
    
    return x