import onnx
import torch
import tensorflow as tf
import numpy as np
from super_gradients.training import models
from super_gradients.training.models.conversion import onnx_simplify

from onnx2tflite.utils.builder import keras_builder
from examples.yolonas_tflite_compat import YoloNAS_S_TFLite


class RandomDatasetGenerator:
    def __init__(self, input_size, num_samples: int = 1):
        self.input_size = input_size
        self.counter = 0
        self.num_samples = num_samples

    def __iter__(self):
        self.counter = 0
        return self

    def iterator(self):
        return self.__iter__()

    def __next__(self):
        self.counter += 1
        if self.counter <= self.num_samples:
            return [np.random.rand(*self.input_size).astype(np.float32)]
        raise StopIteration()

if __name__ == '__main__':
    torch_model = models.get(model_name="YoloNAS_S_TFLite", num_classes=80).eval()
    # load state dict
    yolo_nas_model_weights = models.get(model_name="yolo_nas_s", num_classes=80, pretrained_weights="coco").eval()
    torch_model.load_state_dict(yolo_nas_model_weights.state_dict())

    input_size = [1, 3, 640, 640]
    channel_last_size = [input_size[0], input_size[2], input_size[3], input_size[1]]
    x_nchw = torch.randn(*input_size)
    quantize = True

    # state dict sanity check
    with torch.no_grad():
        (x1, x2), _ = torch_model(x_nchw)
        (y1, y2), _ = yolo_nas_model_weights(x_nchw)
        print(f"{'=' * 20} DIFF sanity check")
        diff_cls = torch.abs(y2 - x2.permute(0, 2, 1))
        print(f"DIFF cls preds: mean = {diff_cls.mean()}, max = {diff_cls.max()}")
        diff_reg = torch.abs(y1 - x1.squeeze(1))
        print(f"DIFF cls preds: mean = {diff_reg.mean()}, max = {diff_reg.max()}")
        print(f"{'=' * 20}")

    torch_model.prep_model_for_conversion([1, 3, 640, 640])
    onnx_path = "/Users/liork/Downloads/yolonas_s_for_tflite_b.onnx"
    torch.onnx.export(torch_model, x_nchw, onnx_path, opset_version=13)
    onnx_simplify(onnx_path, onnx_path)

    # Edit onnx model with custom ops
    onnx_model = onnx.load_model(onnx_path)

    i = 0
    counter = 0
    value_dict = {attr.name: attr for attr in onnx_model.graph.value_info}
    initializer_dict = {attr.name: attr for attr in onnx_model.graph.initializer}
    while i < len(onnx_model.graph.node):
        if "/heads/" in onnx_model.graph.node[i].name and onnx_model.graph.node[i].op_type == "Reshape" and \
                onnx_model.graph.node[i + 1].op_type == "Transpose" and onnx_model.graph.node[
            i + 2].op_type == "Softmax":
            output_edge = onnx_model.graph.node[i + 1].output[0]
            dims = [d.dim_value for d in value_dict[output_edge].type.tensor_type.shape.dim]
            num_regs = dims[1]
            anchor_size = dims[2]
            new_node = onnx.helper.make_node(
                inputs=list(onnx_model.graph.node[i].input),
                outputs=list(onnx_model.graph.node[i + 1].output),
                name=f"/DFL_Reshape{counter}",
                op_type="DFLReshape",
                num_regs=num_regs,
                anchor_size=anchor_size
            )
            onnx_model.graph.node.insert(i, new_node)
            del onnx_model.graph.node[i + 2]
            del onnx_model.graph.node[i + 1]
        elif "/heads/" in onnx_model.graph.node[i].name and onnx_model.graph.node[i].op_type == "Reshape":
            output_edge = onnx_model.graph.node[i].output[0]
            dims = [d.dim_value for d in value_dict[output_edge].type.tensor_type.shape.dim]
            new_node = onnx.helper.make_node(
                inputs=list(onnx_model.graph.node[i].input),
                outputs=list(onnx_model.graph.node[i].output),
                name=f"/Cls_Reshape{counter}",
                op_type="ClsReshape",
            )
            onnx_model.graph.node.insert(i, new_node)
            del onnx_model.graph.node[i + 1]
        i += 1

    onnx.save_model(onnx_model, onnx_path)

    # Convert to tflite
    tflite_path = onnx_path.replace(".onnx", "_quant.tflite" if quantize else ".tflite")

    onnx_model = onnx.load_model(onnx_path)
    keras_model = keras_builder(
        onnx_model=onnx_model, native_groupconv=True, tflite_compat=True
    )

    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS] #, tf.lite.OpsSet.SELECT_TF_OPS]

    if quantize:
        converter.representative_dataset = RandomDatasetGenerator(input_size=channel_last_size, num_samples=5).iterator
        converter._experimental_disable_per_channel = True
        converter.experimental_new_converter = True
        converter.experimental_new_quantizer = True
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.target_spec.supported_types = []

        try:
            tflite_model = converter.convert()
        except Exception as e:
            print("======================== Turn off `experimental_new_quantizer`, and try again")
            print(e)
            converter.experimental_new_quantizer = False
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
    interpreter.set_tensor(input_details[0]['index'], x_nchw.permute(0, 2, 3, 1).numpy())

    interpreter.invoke()

    # get_tensor() returns a copy of the tensor data
    # use tensor() in order to get a pointer to the tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print(output_data.shape)

    # with torch.no_grad():
    #     torch_out = module(x_nchw).numpy()
    #
    # np.testing.assert_allclose(output_data, torch_out, rtol=1e-4)




