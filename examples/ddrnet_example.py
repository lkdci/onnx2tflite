import onnx
import tensorflow as tf
import numpy as np

from onnx2tflite.utils.builder import keras_builder


if __name__ == '__main__':
    onnx_path = "/Users/liork/Downloads/ddrnet23.onnx"
    tflite_path = onnx_path.replace(".onnx", ".tflite")

    onnx_model = onnx.load_model(onnx_path)
    keras_model = keras_builder(
        onnx_model=onnx_model, native_groupconv=True, tflite_compat=False
    )

    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS] #, tf.lite.OpsSet.SELECT_TF_OPS]

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
    interpreter.set_tensor(input_details[0]['index'], np.ones((1, 1024, 2048, 3), dtype=np.float32))

    interpreter.invoke()

    # get_tensor() returns a copy of the tensor data
    # use tensor() in order to get a pointer to the tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print(output_data.shape)
