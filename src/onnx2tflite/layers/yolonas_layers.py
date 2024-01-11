import tensorflow as tf
from ..utils.op_registry import OPERATOR


@OPERATOR.register_operator("DFLReshape")
class DFLReshape:
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs) -> None:
        self.num_regs = node_attribute["num_regs"]
        self.anchor_size = node_attribute["anchor_size"]

    def __call__(self, inputs):
        return tf.reshape(inputs, shape=(inputs.shape[0], self.anchor_size, 4, self.num_regs))


@OPERATOR.register_operator("ClsReshape")
class ClsReshape:
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs) -> None:
        pass

    def __call__(self, inputs):
        return tf.reshape(inputs, shape=(inputs.shape[0], inputs.shape[1] * inputs.shape[2], inputs.shape[3]))


@OPERATOR.register_operator("Neg")
class TFNeg:
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs) -> None:
        pass

    def __call__(self, inputs):
        return -inputs
