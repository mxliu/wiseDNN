from keras import backend as K
def mean_squared_error_weight(y_true, y_pred):
    mask_true = K.cast(K.not_equal(y_true, -1), K.floatx())
    masked_squared_error = K.square(mask_true * (y_true - y_pred))
    masked_mse = K.sum(masked_squared_error, axis=-1) / K.sum(mask_true, axis=-1)
    return masked_mse
