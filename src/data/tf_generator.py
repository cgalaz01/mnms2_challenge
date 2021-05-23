from typing import Union

import tensorflow as tf

from .loader import DataGenerator


class TensorFlowDataGenerator():
    
    @staticmethod
    def get_generators(batch_size: int, max_buffer_size: Union[int, None]=None):
        dg = DataGenerator()
        
        output_types = ({'input_sa': tf.float32,
                         'input_la': tf.float32,},
                        {'output_sa': tf.float32,
                         'output_la': tf.float32})

        buffer_size = len(dg.train_list) * 2
        if max_buffer_size is not None:
            buffer_size = min(buffer_size, max_buffer_size)    
        train_generator = tf.data.Dataset.from_generator(dg.train_generator,
                                                         output_types=output_types)
        train_generator = train_generator.shuffle(buffer_size=buffer_size,
                                                  seed=4875,
                                                  reshuffle_each_iteration=True).batch(batch_size)
        
        validation_generator = tf.data.Dataset.from_generator(dg.validation_generator,
                                                              output_types=output_types)
        validation_generator = validation_generator.batch(batch_size)
        
        test_generator = tf.data.Dataset.from_generator(dg.test_generator,
                                                        output_types=output_types)
        test_generator = test_generator.batch(batch_size)
        
        return train_generator, validation_generator, test_generator, dg


