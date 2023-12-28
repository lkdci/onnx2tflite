import os
import logging
import argparse
from .utils import load_onnx_modelproto, keras_builder

LOG = logging.getLogger("converter running:")


def onnx_converter(
    onnx_model_path: str,
    output_path: str = None,
    input_node_names: list = None,
    output_node_names: list = None,
    native_groupconv: bool = False,
):
    model_proto = load_onnx_modelproto(onnx_model_path)

    keras_model = keras_builder(model_proto, input_node_names, output_node_names, native_groupconv)

    onnx_path, model_name = os.path.split(onnx_model_path)
    if output_path is None:
        output_path = onnx_path
    output_path = os.path.join(output_path, model_name.split(".")[0])

    keras_model.save(output_path + ".h5")
    LOG.info(f"keras model saved in {output_path}.h5")


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, required=True, help="onnx model path")
    parser.add_argument("--outpath", type=str, default=None, help="tflite model save path")
    parser.add_argument(
        "--input-node-names",
        nargs="+",
        default=None,
        help="which inputs is you want, support middle layers, None will using onnx orignal inputs",
    )
    parser.add_argument(
        "--output-node-names",
        nargs="+",
        default=None,
        help="which outputs is you want, support middle layers, None will using onnx orignal outputs",
    )
    parser.add_argument(
        "--native-groupconv",
        default=False,
        action="store_true",
        help="using native method for groupconv, only support for tflite version >= 2.9",
    )
    opt = parser.parse_args()
    return opt


def run():
    opt = parse_opt()
    onnx_converter(
        onnx_model_path=opt.weights,
        input_node_names=opt.input_node_names,
        output_node_names=opt.output_node_names,
        output_path=opt.outpath,
        native_groupconv=opt.native_groupconv,
    )


if __name__ == "__main__":
    run()
