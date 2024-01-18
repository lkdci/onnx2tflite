from typing import List
import struct

import onnx
import tensorflow as tf
import numpy as np

from onnx2tflite.utils.builder import keras_builder
from onnx2tflite.utils.onnx_sequence import IONNXSequence


class DeciFormerAttentionSRSequence(IONNXSequence):
    @classmethod
    def op_types(cls) -> List[str]:
        return [
            'Conv', 'Reshape', 'Transpose', 'Conv', 'Conv', 'Reshape', 'Gather', 'Gather', 'Transpose', 'MatMul', 'Mul',
            'Softmax', 'MatMul', 'Transpose', 'Reshape', 'Conv'
        ]


class DeciFormerAttentionSequence(IONNXSequence):
    @classmethod
    def op_types(cls) -> List[str]:
        return [
            'Conv', 'Reshape', 'Gather', 'Transpose', 'Gather', 'Gather', 'Transpose', 'MatMul', 'Mul', 'Softmax',
            'MatMul', 'Transpose', 'Reshape', 'Conv'
        ]


if __name__ == '__main__':
    onnx_path = "/Users/liork/Downloads/cityscapes_models/deciformer_v3.onnx"
    tflite_path = onnx_path.replace(".onnx", ".tflite")

    onnx_model = onnx.load_model(onnx_path)

    value_dict = {attr.name: attr for attr in onnx_model.graph.value_info}
    initializer_dict = {attr.name: attr for attr in onnx_model.graph.initializer}
    nodes = onnx_model.graph.node
    i = 0
    while i < len(nodes):
        if DeciFormerAttentionSRSequence.is_sequence(nodes, i, strict_order=True):
            num_ops = DeciFormerAttentionSRSequence.num_ops()
            attn_nodes = nodes[i: i + num_ops]
            input_node = attn_nodes[0]
            output_node = attn_nodes[-1]

            # Find Conv weights
            conv_nodes = [n for n in attn_nodes if n.op_type == "Conv"]
            q_conv_node = [n for n in conv_nodes if "/q/Conv" in n.name][0]
            embed_dim = initializer_dict[q_conv_node.input[1]].dims[1]
            sr_conv_node = [n for n in conv_nodes if "/spatial_reduction/" in n.name][0]
            sr_ratio = {attr.name: attr for attr in sr_conv_node.attribute}["strides"].ints[0]
            kv_conv_node = [n for n in conv_nodes if "/kv/Conv" in n.name][0]
            proj_conv_node = [n for n in conv_nodes if "/proj/" in n.name][0]
            # Find QK scale scalar weight
            qk_scale_node = [n for n in attn_nodes if n.op_type == "Mul"][0]
            qk_scale = struct.unpack("f", initializer_dict[qk_scale_node.input[1]].raw_data)[0]
            # Find num_heads by examining the input to Softmax
            softmax_node = [n for n in attn_nodes if n.op_type == "Softmax"][0]
            num_heads = value_dict[softmax_node.input[0]].type.tensor_type.shape.dim[1].dim_value

            new_node = onnx.helper.make_node(
                inputs=[
                    input_node.input[0],        # block input tensor
                    q_conv_node.input[1],       # q conv weights
                    sr_conv_node.input[1],      # sr conv weights
                    sr_conv_node.input[2],      # sr conv bias
                    kv_conv_node.input[1],      # kv conv weights
                    proj_conv_node.input[1],    # proj conv weights
                    proj_conv_node.input[2],    # proj conv bias
                ],
                outputs=list(output_node.output),
                name=f"/DeciFormerAttentionSR_{i}",
                op_type="DeciFormerAttentionSR",
                embed_dim=embed_dim,
                num_heads=num_heads,
                sr_ratio=sr_ratio,
                qk_scale=qk_scale,
                talking_heads=False,
            )
            onnx_model.graph.node.insert(i, new_node)
            for j in list(reversed(range(i + 1, i + 1 + num_ops))):
                del onnx_model.graph.node[j]
        elif DeciFormerAttentionSequence.is_sequence(nodes, i, strict_order=False):
            num_ops = DeciFormerAttentionSequence.num_ops()
            attn_nodes = nodes[i: i + num_ops]
            input_node = attn_nodes[0]
            output_node = attn_nodes[-1]

            # Find Conv weights
            conv_nodes = [n for n in attn_nodes if n.op_type == "Conv"]
            qkv_conv_node = [n for n in conv_nodes if "/qkv/Conv" in n.name][0]
            embed_dim = initializer_dict[qkv_conv_node.input[1]].dims[1]
            proj_conv_node = [n for n in conv_nodes if "/proj/" in n.name][0]
            # Find QK scale scalar weight
            qk_scale_node = [n for n in attn_nodes if n.op_type == "Mul"][0]
            qk_scale = struct.unpack("f", initializer_dict[qk_scale_node.input[1]].raw_data)[0]
            # Find num_heads by examining the input to Softmax
            softmax_node = [n for n in attn_nodes if n.op_type == "Softmax"][0]
            num_heads = value_dict[softmax_node.input[0]].type.tensor_type.shape.dim[1].dim_value

            new_node = onnx.helper.make_node(
                inputs=[
                    input_node.input[0],        # block input tensor
                    qkv_conv_node.input[1],       # q conv weights
                    proj_conv_node.input[1],    # proj conv weights
                    proj_conv_node.input[2],    # proj conv bias
                ],
                outputs=list(output_node.output),
                name=f"/DeciFormerAttention_{i}",
                op_type="DeciFormerAttention",
                embed_dim=embed_dim,
                num_heads=num_heads,
                qk_scale=qk_scale,
                talking_heads=False,
            )
            onnx_model.graph.node.insert(i, new_node)
            for j in list(reversed(range(i + 1, i + 1 + num_ops))):
                del onnx_model.graph.node[j]
        i += 1

    onnx.save_model(onnx_model, onnx_path)

    keras_model = keras_builder(
        onnx_model=onnx_model, native_groupconv=True, tflite_compat=False
    )

    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS] #, tf.lite.OpsSet.SELECT_TF_OPS]
    converter._experimental_disable_batchmatmul_unfold = True

    tflite_model = converter.convert()

    with open(tflite_path, "wb") as fp:
        fp.write(tflite_model)

    tf.lite.experimental.Analyzer.analyze(
        model_path=tflite_path, model_content=None, gpu_compatibility=True
    )

    # Load the TFLite model and allocate tensors
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test the model on random input data
    input_shape = input_details[0]['shape']
    interpreter.set_tensor(input_details[0]['index'], np.ones((1, 3, 1024, 2048), dtype=np.float32))

    interpreter.invoke()

    # get_tensor() returns a copy of the tensor data
    # use tensor() in order to get a pointer to the tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print(output_data.shape)
