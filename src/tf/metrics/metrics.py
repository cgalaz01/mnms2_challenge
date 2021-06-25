import tensorflow as tf


@tf.autograph.experimental.do_not_convert
def dice(y_true, y_pred, ignore_background=True, right_ventricle_only=True):
    epsilon = 1e-6
    
    y_true = tf.cast(y_true, dtype=tf.float32)
    y_pred = tf.cast(y_pred, dtype=tf.float32)
    if right_ventricle_only:
        # Expected one-hot encoding format, with background in index 0
        y_true = y_true[..., 3]
        y_pred = y_pred[..., 3]
    elif ignore_background:
        # Expected one-hot encoding format, with background in index 0
        y_true = y_true[..., 1:]
        y_pred = y_pred[..., 1:]
    
    # Expected y_pred to be 'logits'
    y_pred = tf.nn.softmax(y_pred)
    
    dim = tf.reduce_prod(tf.shape(y_true)[1:])
    y_true_flatten = tf.reshape(y_true, [-1, dim])
    y_pred_flatten = tf.reshape(y_pred, [-1, dim])

    intersection = tf.math.reduce_sum(y_true_flatten * y_pred_flatten, axis=-1)
    
    union = tf.math.reduce_sum(y_true_flatten, axis=-1) + \
        tf.math.reduce_sum(y_pred_flatten, axis=-1)
    
    dice_coef = tf.math.reduce_mean((2. * intersection + epsilon) / (union + epsilon))

    return dice_coef

    