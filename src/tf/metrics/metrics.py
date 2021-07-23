import tensorflow as tf
from tensorflow.keras import backend as K


@tf.autograph.experimental.do_not_convert
def soft_dice(y_true, y_pred):
    epsilon = 1e-6
    
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)
    
    # Expected y_pred to be 'logits'
    y_pred = tf.sigmoid(y_pred)
    
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    intersection = K.sum(y_true * y_pred)
    dice_coef = (2. * intersection + epsilon) / (K.sum(y_true) + K.sum(y_pred) + epsilon)

    return dice_coef

    