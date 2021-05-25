import tensorflow as tf


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