import tensorflow as tf
import tensorflow_addons as tfa


# Loss taken from here:
#    https://github.com/tensorflow/models/blob/master/official/vision/keras_cv/losses/focal_loss.py
class FocalLoss(tf.keras.losses.Loss):
    """Implements a Focal loss for classification problems.
    Reference:
      [Focal Loss for Dense Object Detection](https://arxiv.org/abs/1708.02002).
    """

    def __init__(self,
                 alpha,
                 gamma,
                 reduction=tf.keras.losses.Reduction.AUTO,
                 name=None):
        """Initializes `FocalLoss`.
        Args:
          alpha: The `alpha` weight factor for binary class imbalance.
          gamma: The `gamma` focusing parameter to re-weight loss.
          reduction: (Optional) Type of `tf.keras.losses.Reduction` to apply to
            loss. Default value is `AUTO`. `AUTO` indicates that the reduction
            option will be determined by the usage context. For almost all cases
            this defaults to `SUM_OVER_BATCH_SIZE`. When used with
            `tf.distribute.Strategy`, outside of built-in training loops such as
            `tf.keras` `compile` and `fit`, using `AUTO` or `SUM_OVER_BATCH_SIZE`
            will raise an error. Please see this custom training [tutorial](
              https://www.tensorflow.org/tutorials/distribute/custom_training) for
                more details.
          name: Optional name for the op. Defaults to 'retinanet_class_loss'.
        """
        self._alpha = alpha
        self._gamma = gamma
        super(FocalLoss, self).__init__(reduction=reduction, name=name)
    
    
    def call(self, y_true, y_pred):
        """Invokes the `FocalLoss`.
        Args:
          y_true: A tensor of size [batch, num_anchors, num_classes]
          y_pred: A tensor of size [batch, num_anchors, num_classes]
        Returns:
          Summed loss float `Tensor`.
        """
        with tf.name_scope('focal_loss'):
            y_true = tf.cast(y_true, dtype=tf.float32)
            y_pred = tf.cast(y_pred, dtype=tf.float32)
            positive_label_mask = tf.equal(y_true, 1.0)
            cross_entropy = (
                tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred))
            probs = tf.sigmoid(y_pred)
            probs_gt = tf.where(positive_label_mask, probs, 1.0 - probs)
            # With small gamma, the implementation could produce NaN during back prop.
            modulator = tf.pow(1.0 - probs_gt, self._gamma)
            loss = modulator * cross_entropy
            weighted_loss = tf.where(positive_label_mask, self._alpha * loss,
                                     (1.0 - self._alpha) * loss)
        
        return weighted_loss
    
    
    def get_config(self):
        config = {
            'alpha': self._alpha,
            'gamma': self._gamma,
        }
        base_config = super(FocalLoss, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))



class TverskyLoss(tf.keras.losses.Loss):
    """Implements a Tversky loss for classification problems.
    Reference:
      [Tversky loss function for image segmentation using 3D fully convolutional
       deep networks](https://arxiv.org/abs/1706.05721).
      
      'In the case of α=β=0.5 the Tversky index simplifies to be the same as
       the Dice coefficient, which is also equal to the F1 score. With α=β=1,
       Equation 2 produces Tanimoto coefficient, and setting α+β=1 produces
       the set of Fβ scores. Larger βs weigh recall higher than precision (by
       placing more emphasis on false negatives)'
    """

    def __init__(self,
                 alpha,
                 beta,
                 reduction=tf.keras.losses.Reduction.AUTO,
                 name=None):
        """Initializes `TverskyLoss`.
        Args:
          alpha: The `alpha` weight factor for binary class imbalance.
          gamma: The `gamma` focusing parameter to re-weight loss.
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
        self._alpha = alpha
        self._beta = beta
        super(TverskyLoss, self).__init__(reduction=reduction, name=name)
  
  
    def call(self, y_true, y_pred):
        """Invokes the `TverskyLoss`.
        Args:
          y_true: A tensor of size [batch, ..., num_classes]
          y_pred: A tensor of size [batch, ..., num_classes]
        Returns:
          Summed loss float `Tensor`.
        """
        with tf.name_scope('tversky_loss'):
            epsilon = 1e-6
            y_true = tf.cast(y_true, dtype=tf.float32)
            y_pred = tf.cast(y_pred, dtype=tf.float32)
            
            # TODO: softmax is unstable
            y_pred = tf.nn.softmax(y_pred, axis=-1)
            
            dim = tf.reduce_prod(tf.shape(y_true)[1:])
            y_true_flatten = tf.reshape(y_true, [-1, dim])
            y_pred_flatten = tf.reshape(y_pred, [-1, dim])
            
            tp = tf.math.reduce_sum(y_true_flatten * y_pred_flatten, axis=-1)
            fp = tf.math.reduce_sum((1.0 - y_true_flatten) * y_pred_flatten, axis=-1)
            fn = tf.math.reduce_sum(y_true_flatten * (1.0 - y_pred_flatten), axis=-1)
            
            tversky = (tp + epsilon) / (tp + self._alpha * fp + self._beta * fn + epsilon)
            
            loss = 1 - tf.reduce_mean(tversky)
    
        return loss
  
  
    def get_config(self):
        config = {
            'alpha': self._alpha,
            'beta': self._beta
        }
        base_config = super(TverskyLoss, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    
    
class DiceLoss(tf.keras.losses.Loss):
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
        super(DiceLoss, self).__init__(reduction=reduction, name=name)
  
  
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
        base_config = super(DiceLoss, self).get_config()
        return base_config
    
    
def combined_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
            
    focal_loss = tfa.losses.SigmoidFocalCrossEntropy(from_logits=True,
                                                     alpha=0.25,
                                                     gamma=2.0,
                                                     reduction=tf.keras.losses.Reduction.SUM)
    dice_loss = DiceLoss(reduction=tf.keras.losses.Reduction.SUM)
    
    total_loss = tf.math.add(focal_loss(y_true, y_pred), dice_loss(y_true, y_pred))
    
    return total_loss
