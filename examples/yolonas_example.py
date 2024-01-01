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


class RandomDatasetGenerator:
    def __init__(self, input_size, num_samples: int = 1, dataset=None):
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
            if self.dataset:
                sample = self.dataset[self.counter]
                img = np.transpose(np.expand_dims(sample[0], axis=0), (0, 2, 3, 1))
                return [img]

            # Random input
            return [np.random.rand(*self.input_size).astype(np.float32)]
        raise StopIteration()


def inference_tflite_model(tflite_path, x_nchw, dataset):
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
        output_data_cls = torch.from_numpy(interpreter.get_tensor(output_details[1]['index'])).permute(0, 2, 1)

        preds.append(output_data_bbox)
        preds.append(output_data_cls)

        metric.update(preds=[preds], target=label, inputs=data[0], device='cuda')

    detection_metric = metric.compute()
    print(f"detection_metric result:\t {detection_metric}")

    # with torch.no_grad():
    #     torch_out = module(x_nchw).numpy()
    #
    # np.testing.assert_allclose(output_data, torch_out, rtol=1e-4)


@hydra.main(config_path="../recipes/", config_name="coco2017_yolo_nas", version_base="1.2.0")
def run(cfg: DictConfig):
    cfg = hydra.utils.instantiate(cfg)
    inference = True
    compile = False
    input_size = [1, 3, 640, 640]
    channel_last_size = [input_size[0], input_size[2], input_size[3], input_size[1]]
    x_nchw = torch.randn(*input_size)
    quantize = True

    onnx_path = "/home/daniel.afrimi/pycharm_projects/dso_tflite/examples/yolonas_s_for_tflite.onnx"
    tflite_path = onnx_path.replace(".onnx", "_quant.tflite" if quantize else ".tflite")
    dataset_val = COCODetectionDataset(**cfg.dataset_params.val_dataset_params)

    if compile:
        onnx_model = onnx.load_model(onnx_path)
        keras_model = keras_builder(
            onnx_model=onnx_model, native_groupconv=True, tflite_compat=True
        )

        converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]  # , tf.lite.OpsSet.SELECT_TF_OPS]

        if quantize:
            # todo do we want samples from train/val?
            converter.representative_dataset = RandomDatasetGenerator(input_size=channel_last_size, num_samples=500,
                                                                      dataset=dataset_val).iterator
            converter._experimental_disable_per_channel = True
            converter.experimental_new_converter = True
            converter.experimental_new_quantizer = True
            converter.optimizations = [
                tf.lite.Optimize.DEFAULT]  # in their code is [tf.lite.Optimize.DEFAULT, tf.lite.OpsSet.SELECT_TF_OPS]

            # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
            converter.target_spec.supported_types = []

            try:
                tflite_model = converter.convert()
            except Exception as e:
                print(
                    "======================== Turn off `experimental_new_quantizer`, and try again ======================")
                print(e)
                converter.experimental_new_quantizer = False
                tflite_model = converter.convert()

            with open(tflite_path, "wb") as fp:
                fp.write(tflite_model)

    if inference:
        val_dataloader = dataloaders.get(
            name=get_param(cfg, "val_dataloader"),
            dataset_params=cfg.dataset_params.val_dataset_params,
            dataloader_params=cfg.dataset_params.val_dataloader_params,
        )
        inference_tflite_model(tflite_path=tflite_path, x_nchw=x_nchw, dataset=val_dataloader)


if __name__ == '__main__':
    super_gradients.init_trainer()
    run()
