import onnx
import torch
from onnx2tflite.utils.builder import keras_builder
import tensorflow as tf
import numpy as np


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
    input_size = [1, 3, 640, 640]
    channel_last_size = [input_size[0], input_size[2], input_size[3], input_size[1]]
    x_nchw = torch.randn(*input_size)
    quantize = True

    onnx_path = "/Users/liork/Downloads/yolonas_s_for_tflite.onnx"
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




