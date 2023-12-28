import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import tensorflow as tf
from tensorflow import keras
from onnx import numpy_helper
from .op_registry import OPERATOR

from ..layers import conv_layers
from ..layers import deformation_layers


# copy from https://github.com/gmalivenko/onnx2keras
def decode_node_attribute(node) -> dict:
    """
    Parse ONNX attributes to Python dictionary
    :param args: ONNX attributes object
    :return: Python dictionary
    """

    def onnx_attribute_to_dict(onnx_attr):
        """
        Parse ONNX attribute
        :param onnx_attr: ONNX attribute
        :return: Python data type
        """
        if onnx_attr.HasField("t"):
            return numpy_helper.to_array(getattr(onnx_attr, "t"))

        for attr_type in ["f", "i"]:
            if onnx_attr.HasField(attr_type):
                return getattr(onnx_attr, attr_type)

        # s need to be decode, bytes to string
        if onnx_attr.HasField("s"):
            return getattr(onnx_attr, "s").decode()

        for attr_type in ["floats", "ints", "strings"]:
            if getattr(onnx_attr, attr_type):
                return list(getattr(onnx_attr, attr_type))

    return {arg.name: onnx_attribute_to_dict(arg) for arg in node.attribute}


def keras_builder(
    onnx_model,
    new_input_nodes: list = None,
    new_output_nodes: list = None,
    native_groupconv: bool = False,
    tflite_compat: bool = False,
):
    conv_layers.USE_NATIVE_GROUP_CONV = native_groupconv

    deformation_layers.TRANSPOSE_WITH_TFLITE_COMPAT = tflite_compat
    model_graph = onnx_model.graph

    """
        init onnx model's build-in tensors
    """
    onnx_weights = dict()
    for initializer in model_graph.initializer:
        onnx_weights[initializer.name] = numpy_helper.to_array(initializer)

    """
        build input nodes
    """
    tf_tensor, input_shape = {}, []
    for inp in model_graph.input:
        input_shape = [x.dim_value for x in inp.type.tensor_type.shape.dim]
        if input_shape == []:
            continue
        batch_size = 1 if input_shape[0] <= 0 else input_shape[0]
        input_shape = input_shape[2:] + input_shape[1:2]
        tf_tensor[inp.name] = keras.Input(shape=input_shape, batch_size=batch_size)

    """
        build model inline node by iterate onnx nodes.
    """
    input_node_names, outputs_node_names = [], []
    for node in model_graph.node:
        op_name, node_inputs, node_outputs, node_name = node.op_type, node.input, node.output, node.name
        op_attr = decode_node_attribute(node)

        tf_operator = OPERATOR.get(op_name)
        if tf_operator is None:
            raise KeyError(f"{op_name} not implemented yet")

        _inputs = None
        if len(node_inputs) > 0:
            _inputs = tf_tensor[node_inputs[0]] if node_inputs[0] in tf_tensor else onnx_weights[node_inputs[0]]

        for index in range(len(node_outputs)):
            tf_tensor[node_outputs[index]] = tf_operator(tf_tensor, onnx_weights, node_inputs, op_attr, index=index)(
                _inputs
            )

        """
            reorganize input and output nodes
        """
        if new_input_nodes is not None and node_name in new_input_nodes:
            input_node_names.append(node_outputs[0])
        # TODO for nodes with multiply outputs.
        if new_output_nodes is not None and node_name in new_output_nodes:
            outputs_node_names.append(node_outputs[0])
        if new_output_nodes is not None and len(outputs_node_names) == len(new_output_nodes):
            break

    """
        process input and output nodes
    """
    input_nodes = []
    if new_input_nodes is None:
        input_nodes = [tf_tensor[x.name] for x in model_graph.input]
    else:
        for node in model_graph.input:
            if node.name in new_input_nodes:
                input_node_names.append(node.name)
        input_nodes = [tf_tensor[x] for x in input_node_names]
    outputs_nodes = []
    if new_output_nodes is None:
        outputs_nodes = [tf_tensor[x.name] for x in model_graph.output]
    else:
        for node in model_graph.output:
            if node.name in new_output_nodes:
                outputs_node_names.append(node.name)
        outputs_nodes = [tf_tensor[x] for x in outputs_node_names]

    """
        build keras model
    """
    keras_model = keras.Model(inputs=input_nodes, outputs=outputs_nodes)
    keras_model.trainable = False
    # keras_model.summary()

    return keras_model
