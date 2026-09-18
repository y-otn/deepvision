from tensorflow import keras
from tensorflow.keras import layers


def drop_path(x, drop_prob, training):
    if not drop_prob or not training:
        return x
    keep_prob = 1 - drop_prob
    mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep_prob)
    return x * mask / keep_prob


def same_padding(kernel_size, stride, dilation=1):
    padding = ((stride - 1) + dilation * (kernel_size - 1)) // 2
    return padding


def parse_model_inputs(backend, input_shape, input_tensor):
    if backend == "tensorflow":
        if input_tensor is None:
            return layers.Input(shape=input_shape)
        else:
            if not keras.backend.is_keras_tensor(input_tensor):
                return layers.Input(tensor=input_tensor, shape=input_shape)
            else:
                return input_tensor
    elif backend == "pytorch":
        return input_tensor
    else:
        raise ValueError(f"Backend not supported: {backend}")
