import os
import onnx
import logging

LOG = logging.getLogger("onnx_loader running:")
LOG.setLevel(logging.INFO)


def clean_model_input(model_proto):
    inputs = model_proto.graph.input
    name_to_input = {}
    for input in inputs:
        name_to_input[input.name] = input

    names = []
    for initializer in model_proto.graph.initializer:
        if initializer.name in name_to_input:
            inputs.remove(name_to_input[initializer.name])
            names.append(initializer.name)

    if len(names) > 0:
        LOG.warning(
            f"[{len(names)}] redundant input nodes are removed.\n \
            nodes name : {','.join(names)}"
        )


def load_onnx_modelproto(onnx_model_path: str):
    if not os.path.exists(onnx_model_path):
        LOG.error(f"{onnx_model_path} is not exists.")
        raise FileExistsError(f"{onnx_model_path} is not exists.")
    model_proto = onnx.load(onnx_model_path)

    return model_proto
