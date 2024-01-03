import hydra
import numpy as np
import onnx
import super_gradients
import tensorflow as tf
import torch
from omegaconf import DictConfig
from super_gradients.training.datasets.detection_datasets import COCODetectionDataset
from super_gradients.training.metrics import DetectionMetrics
from super_gradients.training.models.detection_models.pp_yolo_e import PPYoloEPostPredictionCallback
from super_gradients.training.utils import get_param
from tqdm import tqdm
from super_gradients.training import dataloaders
from src.onnx2tflite.utils.builder import keras_builder
from super_gradients.training import models
from super_gradients.training.models.conversion import onnx_simplify
from examples.yolonas_tflite_compat import YoloNAS_S_TFLite


def create_onnx_model_example():
    from torchvision.models import resnet50, ResNet50_Weights
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    model.eval()

    dummy_input = torch.zeros(1, 3, 224, 224)  # BCHW
    torch.onnx.export(model,
                      dummy_input,
                      'resnet_50.onnx',
                      verbose=False, opset_version=15,
                      training=torch.onnx.TrainingMode.EVAL,
                      do_constant_folding=True,
                      input_names=['input'],
                      output_names=['output'],
                      dynamic_axes=None)


class COCODatasetGenerator:
    def __init__(self, dataset, input_size, num_samples: int = 1):
        self.input_size = input_size
        self.counter = 0
        self.num_samples = num_samples
        self.dataset = dataset

    def __iter__(self):
        self.counter = 0
        return self

    def iterator(self):
        return self.__iter__()

    def __next__(self):
        self.counter += 1
        if self.counter <= self.num_samples and self.counter <= len(self.dataset):
            print(f"num samples is {self.counter}")
            sample = self.dataset[self.counter]
            img = np.transpose(np.expand_dims(sample[0], axis=0), (0, 2, 3, 1))
            return [img]

        raise StopIteration()


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
        if self.counter <= self.num_samples :
            print(f"num samples is {self.counter}")
            # Random input
            return [np.random.rand(*self.input_size).astype(np.float32)]
        raise StopIteration()


def eval_tflite_model(tflite_path, x_nchw, dataset):
    print(f'Start evaluate tflite model with path {tflite_path}')
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
    output_data_bbox = interpreter.get_tensor(output_details[0]['index'])
    output_data_cls = interpreter.get_tensor(output_details[1]['index'])
    print(output_data_bbox.shape)
    print(output_data_cls.shape)

    post_prediction_callback = PPYoloEPostPredictionCallback(score_threshold=0.1, max_predictions=300, nms_top_k=1000,
                                                             nms_threshold=0.7)
    metric = DetectionMetrics(score_thres=0.1, top_k_predictions=300, num_cls=80, normalize_targets=True,
                              post_prediction_callback=post_prediction_callback)

    for i, data in tqdm(enumerate(dataset)):
        label = torch.tensor(data[1])
        preds = []
        img = np.transpose(data[0], (0, 2, 3, 1))
        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()

        output_data_bbox = torch.from_numpy(interpreter.get_tensor(output_details[0]['index'])).squeeze(dim=3)
        output_data_cls = torch.from_numpy(interpreter.get_tensor(output_details[1]['index']))

        preds.append(output_data_bbox)
        preds.append(output_data_cls)

        metric.update(preds=[preds], target=label, inputs=data[0], device='cuda')

        if i % 1000 == 0:
            detection_metric = metric.compute()
            print(f"detection_metric result:\t {detection_metric}")

    detection_metric = metric.compute()
    print(f"detection_metric result:\t {detection_metric}")


def eval_torch_model(eval_dataset):
    print(f'Start evaluate torch model')
    torch_model = models.get(model_name="YoloNAS_S_TFLite", num_classes=80).eval()
    # load state dict
    yolo_nas_model_weights = models.get(model_name="yolo_nas_s", num_classes=80, pretrained_weights="coco").eval()
    torch_model.load_state_dict(yolo_nas_model_weights.state_dict())

    post_prediction_callback = PPYoloEPostPredictionCallback(score_threshold=0.1, max_predictions=300, nms_top_k=1000,
                                                             nms_threshold=0.7)
    metric = DetectionMetrics(score_thres=0.1, top_k_predictions=300, num_cls=80, normalize_targets=True,
                              post_prediction_callback=post_prediction_callback)

    for i, data in tqdm(enumerate(eval_dataset)):
        label = torch.tensor(data[1])
        preds = []

        pred = torch_model(data[0])

        preds.append(pred[0][0].squeeze(dim=0))
        preds.append(pred[0][1].permute(0, 2, 1))

        metric.update(preds=[preds], target=label, inputs=data[0], device='cuda')

        if i % 1000 == 0:
            detection_metric = metric.compute()
            print(f"detection_metric result:\t {detection_metric}")

    detection_metric = metric.compute()
    print(f"detection_metric result:\t {detection_metric}")


def get_yolonas_onnx(onnx_path):
    torch_model = models.get(model_name="YoloNAS_S_TFLite", num_classes=80).eval()
    # load state dict
    yolo_nas_model_weights = models.get(model_name="yolo_nas_s", num_classes=80, pretrained_weights="coco").eval()
    torch_model.load_state_dict(yolo_nas_model_weights.state_dict())

    input_size = [1, 3, 640, 640]
    x_nchw = torch.randn(*input_size)

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
                onnx_model.graph.node[i + 1].op_type == "Transpose" and onnx_model.graph.node[i + 2].op_type == "Softmax":
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
        i += 1

    onnx.save_model(onnx_model, onnx_path)

def calc_quantization_stats(converter):
    result_file = "result_tflite_quant.csv"
    debugger = tf.lite.experimental.QuantizationDebugger(
        converter=converter, debug_dataset=converter.representative_dataset)
    debugger.run()

    with open(result_file, 'w') as f:
        debugger.layer_statistics_dump(f)
def print_keras_model_layers(keras_mnodel):
    for i, layer in enumerate(keras_mnodel.layers):
        print(f"Layer {i}: {layer.name}")


@hydra.main(config_path="../recipes/", config_name="coco2017_yolo_nas", version_base="1.2.0")
def run(cfg: DictConfig):
    cfg = hydra.utils.instantiate(cfg)
    eval = True
    compile = True
    quantize = True
    input_size = [1, 3, 640, 640]
    channel_last_size = [input_size[0], input_size[2], input_size[3], input_size[1]]
    x_nchw = torch.randn(*input_size)

    onnx_path = "yolonas_s_for_tflite.onnx"
    tflite_path = onnx_path.replace(".onnx", "_quant.tflite" if quantize else ".tflite")

    if compile:
        get_yolonas_onnx(onnx_path)
        onnx_model = onnx.load_model(onnx_path)
        keras_model = keras_builder(
            onnx_model=onnx_model, native_groupconv=True, tflite_compat=True
        )

        # print_keras_model_layers(keras_model)

        converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]  # , tf.lite.OpsSet.SELECT_TF_OPS]

        if quantize:
            dataset_val = COCODetectionDataset(**cfg.dataset_params.val_dataset_params)
            converter.representative_dataset = COCODatasetGenerator(input_size=channel_last_size, num_samples=500,
                                                                      dataset=dataset_val).iterator

            converter._experimental_disable_per_channel = True
            converter.experimental_new_converter = True
            converter.experimental_new_quantizer = True
            converter.optimizations = [tf.lite.Optimize.DEFAULT]  # in their code is [tf.lite.Optimize.DEFAULT, tf.lite.OpsSet.SELECT_TF_OPS]

            # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.target_spec.supported_types = []

            try:

                # calc_quantization_stats(converter=converter)
                # tflite_model = converter.convert()
                tflite_path = "yolonas_s_for_tflite_selctive_quant.tflite"
                suspected_layers = ['StatefulPartitionedCall:0', 'model/tf.math.subtract/Sub',
                                    'model/tf.concat_15/concat','model/tf.__operators__.add_20/AddV2']
                debug_options = tf.lite.experimental.QuantizationDebugOptions(
                    denylisted_nodes=suspected_layers)

                debugger = tf.lite.experimental.QuantizationDebugger(
                    converter=converter,
                    debug_dataset=converter.representative_dataset,
                    debug_options=debug_options)

                print(f"Compile model to tflite with Selective Quantization, emitted layers: "
                      f" {suspected_layers}, saves in path {tflite_path}")

                tflite_model = debugger.get_nondebug_quantized_model()

            except Exception as e:
                print("======================== Turn off `experimental_new_quantizer`, and try again ======================")
                print(e)
                converter.experimental_new_quantizer = False
                tflite_model = converter.convert()
        else:
            tflite_model = converter.convert()

        with open(tflite_path, "wb") as fp:
            fp.write(tflite_model)

    if eval:
        val_dataloader = dataloaders.get(
            name=get_param(cfg, "val_dataloader"),
            dataset_params=cfg.dataset_params.val_dataset_params,
            dataloader_params=cfg.dataset_params.val_dataloader_params)

        # eval_torch_model(eval_dataset=val_dataloader)
        eval_tflite_model(tflite_path=tflite_path, x_nchw=x_nchw, dataset=val_dataloader)


if __name__ == '__main__':
    super_gradients.init_trainer()
    run()
