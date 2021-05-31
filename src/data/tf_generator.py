from typing import Dict, Tuple, Union

import tensorflow as tf

from .loader import DataGenerator


class TensorFlowDataGenerator():
    
    @staticmethod
    def _prepare_generators(dg: DataGenerator, batch_size: int,
                            output_shapes: Tuple[Dict[str, tf.TesnorShape]],
                            output_types: Tuple[Dict[str, tf.dtypes]],
                            max_buffer_size: Union[int, None]=None,
                            floating_precision: str='32') -> Tuple[tf.data.Dataset]:
        
        buffer_size = len(dg.train_list) * 2
        if max_buffer_size is not None:
            buffer_size = min(buffer_size, max_buffer_size)    
        train_generator = tf.data.Dataset.from_generator(dg.train_generator,
                                                         output_types=output_types,
                                                         output_shapes=output_shapes)
        train_generator = train_generator.shuffle(buffer_size=buffer_size,
                                                  seed=4875,
                                                  reshuffle_each_iteration=True
                                                  ).batch(batch_size).prefetch(2)
        
        validation_generator = tf.data.Dataset.from_generator(dg.validation_generator,
                                                              output_types=output_types)
        validation_generator = validation_generator.batch(batch_size)
        
        test_generator = tf.data.Dataset.from_generator(dg.test_generator,
                                                        output_types=output_types)
        test_generator = test_generator.batch(batch_size)
        
        return train_generator, validation_generator, test_generator, dg
        
    
    @staticmethod
    def get_generators(batch_size: int, max_buffer_size: Union[int, None]=None,
                       floating_precision: str='32') -> Tuple[tf.data.Dataset]:
        dg = DataGenerator(floating_precision)
        
        output_shapes = ({'input_sa': tf.TensorShape(dg.sa_shape),
                          'input_la': tf.TensorShape(dg.la_shape)},
                         {'output_sa': tf.TensorShape(dg.sa_target_shape),
                          'output_la': tf.TensorShape(dg.la_target_shape)})
        
        if floating_precision == '16':
            float_type = tf.float16
        else:
            float_type = tf.float32
        # TODO: Change to dynamic input parameters
        output_types = ({'input_sa': float_type,
                         'input_la': float_type},
                        {'output_sa': tf.uint8,
                         'output_la': tf.uint8})

        return TensorFlowDataGenerator._prepare_generators(dg, batch_size,
                                                           output_shapes,
                                                           output_types,
                                                           max_buffer_size,
                                                           floating_precision)


    @staticmethod
    def get_affine_generators(batch_size: int, max_buffer_size: Union[int, None]=None,
                              floating_precision: str='32') -> Tuple[tf.data.Dataset]:
        dg = DataGenerator(floating_precision)
        
        output_shapes = ({'input_sa': tf.TensorShape(dg.sa_shape),
                          'input_la': tf.TensorShape(dg.la_shape),
                          'input_sa_affine': tf.TensorShape(dg.affine_shape),
                          'input_la_affine': tf.TensorShape(dg.affine_shape)},
                         {'output_sa': tf.TensorShape(dg.sa_target_shape),
                          'output_la': tf.TensorShape(dg.la_target_shape)})
        
        if floating_precision == '16':
            float_type = tf.float16
        else:
            float_type = tf.float32
        # TODO: Change to dynamic input parameters
        output_types = ({'input_sa': float_type,
                         'input_la': float_type,
                         'input_sa_affine': tf.float32,
                         'input_la_affine': tf.float32},
                        {'output_sa': tf.uint8,
                         'output_la': tf.uint8})

        return TensorFlowDataGenerator._prepare_generators(dg, batch_size,
                                                           output_shapes,
                                                           output_types,
                                                           max_buffer_size,
                                                           floating_precision)

