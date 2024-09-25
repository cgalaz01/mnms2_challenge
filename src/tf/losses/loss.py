import tensorflow as tf
    
    
    
class SoftDiceLoss(tf.keras.losses.Loss):
    """Implements the Dice loss for classification problems.
    """

    def __init__(self,
                 reduction=tf.keras.losses.Reduction.AUTO,
                 name=None):
        """Initializes `DiceLoss`.
        Args:
          reduction: (Optional) Type of `tf.keras.losses.Reduction` to apply to
            loss. Default value is `AUTO`. `AUTO` indicates that the reduction
            option will be determined by the usage context. For almost all cases
            this defaults to `SUM_OVER_BATCH_SIZE`. When used with
            `tf.distribute.Strategy`, outside of built-in training loops such as
            `tf.keras` `compile` and `fit`, using `AUTO` or `SUM_OVER_BATCH_SIZE`
            will raise an error. Please see this custom training [tutorial](
              https://www.tensorflow.org/tutorials/distribute/custom_training) for
                more details.
          name: Optional name for the op.
        """
        super(SoftDiceLoss, self).__init__(reduction=reduction, name=name)
  
  
    def call(self, y_true, y_pred):
        """Invokes the `DiceLoss`.
        Args:
          y_true: A tensor of size [batch, ..., num_classes]
          y_pred: A tensor of size [batch, ..., num_classes]
        Returns:
          Summed loss float `Tensor`.
        """
        with tf.name_scope('dice_loss'):
            epsilon = 1e-4
            y_true = tf.cast(y_true, tf.float32)
            y_pred = tf.cast(y_pred, tf.float32)
            
            y_pred = tf.sigmoid(y_pred)
            
            numerator = 2 * tf.reduce_sum(y_true * y_pred) + epsilon
            denominator = tf.reduce_sum(y_true + y_pred) + epsilon
        
            dice_score = numerator / denominator
            loss = 1 - dice_score
        
        return loss
  
  
    def get_config(self):
        base_config = super(SoftDiceLoss, self).get_config()
        return base_config
    
    
def get_focal_loss():
    return tf.keras.losses.BinaryFocalCrossentropy(
            apply_class_balancing=False,
            alpha=0.25,
            gamma=2.0,
            from_logits=True,
            label_smoothing=1e-4,
            axis=-1,
            reduction='sum_over_batch_size')
 

def get_tversky_loss():
    return tf.keras.losses.Tversky(
        alpha=0.5,
        beta=0.5,
        reduction='sum_over_batch_size',
        name='tversky')


def combined_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
            
    focal_loss = get_focal_loss()
    dice_loss = SoftDiceLoss(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
    
    total_loss = focal_loss(y_true, y_pred) + dice_loss(y_true, y_pred)
    
    return total_loss
