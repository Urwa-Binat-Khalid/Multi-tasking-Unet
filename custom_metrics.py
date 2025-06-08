import tensorflow as tf
from keras import backend as K

def precision(y_true, y_pred):
    y_pred = tf.round(y_pred)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    y_pred = tf.round(y_pred)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def f1_score(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * ((p * r) / (p + r + K.epsilon()))

def mcc(y_true, y_pred):
    y_pred = tf.round(y_pred)
    y_true = tf.round(y_true)

    tp = K.sum(K.cast(y_true * y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1 - y_true) * (1 - y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1 - y_true) * y_pred, 'float'), axis=0)
    fn = K.sum(K.cast(y_true * (1 - y_pred), 'float'), axis=0)

    numerator = (tp * tn) - (fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = numerator / (denominator + K.epsilon())
    return mcc
# Define Mean Intersection over Union (mIoU) metric
def mean_iou(y_true, y_pred, num_classes=3, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(tf.argmax(y_pred, axis=-1), tf.float32)  # Convert to class labels
    iou_scores = []

    for i in range(num_classes):
        y_true_i = tf.cast(y_true == i, tf.float32)
        y_pred_i = tf.cast(y_pred == i, tf.float32)
        intersection = tf.reduce_sum(y_true_i * y_pred_i, axis=[1, 2])
        union = tf.reduce_sum(y_true_i + y_pred_i, axis=[1, 2]) - intersection
        iou = (intersection + smooth) / (union + smooth)
        iou_scores.append(iou)

    return tf.reduce_mean(tf.stack(iou_scores))

# Define Dice coefficient metric for multi-class segmentation
def dice_coefficient(y_true, y_pred, num_classes=3, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(tf.argmax(y_pred, axis=-1), tf.float32)  # Convert to class labels
    dice_scores = []

    for i in range(num_classes):
        y_true_i = tf.cast(y_true == i, tf.float32)
        y_pred_i = tf.cast(y_pred == i, tf.float32)
        intersection = tf.reduce_sum(y_true_i * y_pred_i, axis=[1, 2])
        union = tf.reduce_sum(y_true_i + y_pred_i, axis=[1, 2])
        dice = (2. * intersection + smooth) / (union + smooth)
        dice_scores.append(dice)

    return tf.reduce_mean(tf.stack(dice_scores))
  
