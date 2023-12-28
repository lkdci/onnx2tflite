import logging
import tensorflow as tf

from ..utils.op_registry import OPERATOR
from . import dimension_utils

LOG = logging.getLogger("deformation_layers :")

TRANSPOSE_WITH_TFLITE_COMPAT = False


def tflite_compat_transpose(tensor, perm):
    """
    Apply transpose that is compatible with TFLite.
    This only works for tensors with static shapes.
    """
    # builtin tflite supports transpose of up to 5 dimensions
    TFLITE_MAX_TRANSPOSE_RANK = 5
    if tensor.shape.rank <= TFLITE_MAX_TRANSPOSE_RANK or not TRANSPOSE_WITH_TFLITE_COMPAT:
        return tf.transpose(tensor, perm)

    perm = tf.convert_to_tensor(perm)
    # unstack along the new zeroth axis
    new_first_axis = perm[0]
    new_perm = perm[1:] - tf.cast(perm[1:] > new_first_axis, tf.int32)
    tensors = tf.unstack(tensor, axis=new_first_axis)
    # recursively transpose lower-ranked tensors
    tensors = [tflite_compat_transpose(t, new_perm) for t in tensors]
    # restack on axis 0
    return tf.stack(tensors, axis=0)


@OPERATOR.register_operator("Transpose")
class TFTranspose():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs)->None:
        super().__init__()
        self.trans_in, self.trans_out = None, None
        if kwargs.get("perm_list"):
            self.perm_list = kwargs.get("perm_list")
        elif len(node_attribute['perm']) > 4:
            self.perm_list = []
            for axis in node_attribute['perm']:
                new_axis = dimension_utils.channel_to_last_dimension(axis)
                if new_axis == -1:
                    new_axis = max(node_attribute['perm'])
                self.perm_list.append(new_axis)
            self.perm_list = dimension_utils.shape_NCD_to_NDC_format(self.perm_list)
        else:
            self.perm_list = [i for i in node_attribute['perm']]
            LOG.info("Transpose will process tensor after change back to NCHW format.")
            shape_len = len(tensor_grap[node_inputs[0]].shape)
            self.trans_in = [0, shape_len-1] + [n for n in range(1, shape_len-1)]
            self.trans_out = [0] + [n for n in range(2, len(self.perm_list))] + [1]

    def __call__(self, inputs):
        if self.trans_in and self.trans_out:
            inputs = tflite_compat_transpose(inputs, perm=self.trans_in)
            inputs = tflite_compat_transpose(inputs, perm=self.perm_list)
            inputs = tflite_compat_transpose(inputs, perm=self.trans_out)
            return inputs
        else:
            return tflite_compat_transpose(inputs, perm=self.perm_list)

@OPERATOR.register_operator("Slice")
class TFSlice():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs) -> None:
        super().__init__()
        if len(node_inputs) == 1:
            self.starts = node_attribute['starts'][0]
            self.ends = node_attribute['ends'][0]
            self.axis = dimension_utils.channel_to_last_dimension(node_attribute['axes'][0])
            self.steps = 1
        else:
            self.starts = node_weights[node_inputs[1]][0] if node_inputs[1] in node_weights else tensor_grap[node_inputs[1]][0]
            self.axis = node_weights[node_inputs[3]][0] if node_inputs[3] in node_weights else tensor_grap[node_inputs[3]][0]
            self.axis = dimension_utils.channel_to_last_dimension(self.axis)
            self.ends = node_weights[node_inputs[2]][0] if node_inputs[2] in node_weights else tensor_grap[node_inputs[2]][0]
            self.ends = min(self.ends, tensor_grap[node_inputs[0]].shape[self.axis])
            if len(node_inputs) < 5:
                self.steps = 1
            else:
                self.steps = node_weights[node_inputs[4]][0] if node_inputs[4] in node_weights else tensor_grap[node_inputs[4]][0]

    def __call__(self, inputs):
        indices = tf.keras.backend.arange(self.starts, self.ends, step=self.steps)
        return tf.gather(inputs, indices, axis=self.axis)

@OPERATOR.register_operator("Gather")
class TFGather():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs) -> None:
        super().__init__()
        self.axis = dimension_utils.channel_to_last_dimension(node_attribute.get('axis', 0))
        self.indices = tensor_grap[node_inputs[1]] if node_inputs[1] in tensor_grap else node_weights[node_inputs[1]]

    def __call__(self, inputs):
        return tf.gather(inputs, self.indices, axis=self.axis)

@OPERATOR.register_operator("Concat")
class TFConcat():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs):
        super().__init__()
        self._axis = dimension_utils.channel_to_last_dimension(node_attribute['axis'])
        self._gather = [tensor_grap[x] if x in tensor_grap else dimension_utils.tensor_NCD_to_NDC_format(node_weights[x]) for x in node_inputs]

    def __call__(self, *args, **kwargs):
        return tf.concat(self._gather, axis=self._axis)

@OPERATOR.register_operator("Reshape")
class TFReshape():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs):
        super().__init__()
        shape_name = node_inputs[1]
        if shape_name in node_weights:
            self.out_shape = node_weights[shape_name]
        else:
            self.out_shape = tensor_grap[shape_name]
        self.trans_in, self.trans_out = None, None
        LOG.info("Reshape will process tensor after change back to NCHW format.")
        shape_len = len(tensor_grap[node_inputs[0]].shape)
        self.trans_in = [0, shape_len - 1] + [n for n in range(1, shape_len - 1)]
        self.trans_out = [0] + [n for n in range(2, self.out_shape.shape[0])] + [1]

    def __call__(self, inputs):
        inputs = tflite_compat_transpose(inputs, perm=self.trans_in)
        inputs = tf.reshape(inputs, shape=self.out_shape)
        inputs = tflite_compat_transpose(inputs, perm=self.trans_out)
        return inputs
        
@OPERATOR.register_operator("Flatten")
class TFFlatten():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs)->None:
        super().__init__()

        self.axis = node_attribute.get("axis", 1)
        if self.axis < 0:
            raise ValueError("Negative axis not supported yet")

    def __call__(self, inputs):
        # transpose to channels-first
        axes = tuple(range(len(inputs.shape)))
        axes_perm = axes[0:1] + axes[-1:] + axes[1:-1]
        inputs = tflite_compat_transpose(inputs, perm=axes_perm)

        # flatten
        inputs_shape = tf.shape(inputs)
        new_shape = tf.concat(
            [inputs_shape[0 : self.axis], tf.math.reduce_prod(inputs_shape[self.axis :], keepdims=True)], axis=0
        )
        inputs = tf.reshape(inputs, new_shape)
        return inputs


@OPERATOR.register_operator("Split")
class TFSplit():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs)->None:
        super().__init__()
        split = node_weights[node_inputs[1]]
        index = kwargs.get("index", 0)
        start = 0
        for i in range(index):
            start += int(split[i])
        end = start + split[index]
        self.indices = tf.keras.backend.arange(start, end, 1)
        self.axis = dimension_utils.channel_to_last_dimension(node_attribute.get("axis", 0))

    def __call__(self, inputs):
        return tf.gather(inputs, indices=self.indices, axis=self.axis)

@OPERATOR.register_operator("Expand")
class TFExpand():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs)->None:
        super().__init__()
        self.shape = dimension_utils.shape_NCD_to_NDC_format(node_weights[node_inputs[1]])

    def __call__(self, inputs):
        for i in range(len(self.shape)):
            if int(self.shape[i]//inputs.shape[i]) > 1:
                inputs = tf.repeat(inputs, repeats=int(self.shape[i]//inputs.shape[i]), axis=i)
            elif self.shape[i] < inputs.shape[i] and self.shape[i] != 1:
                inputs = tf.repeat(inputs, repeats=int(self.shape[i]), axis=i)
        return inputs

@OPERATOR.register_operator("Unsqueeze")
class TFUnsqueeze():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs)->None:
        super().__init__()
        self.axis = dimension_utils.channel_to_last_dimension(node_attribute['axes'][0])

    def __call__(self, inputs):
        return tf.expand_dims(inputs, self.axis)

@OPERATOR.register_operator("Squeeze")
class TFSqueeze():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs)->None:
        super().__init__()
        axis = node_attribute.get('axes')
        if axis is None and len(node_inputs) > 1:
            axis = node_weights[node_inputs[1]]

        self.axis = dimension_utils.channel_to_last_dimension(axis[0])

    def __call__(self, inputs):
        return tf.squeeze(inputs, self.axis)

@OPERATOR.register_operator("DepthToSpace")
class TFDepthToSpace():
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs)->None:
        super().__init__()
        self.block_size = node_attribute.get("blocksize", 2)
        self.mode = node_attribute.get("mode", "DCR")

    def __call__(self, inputs):
        if self.mode == "DCR":
            return tf.nn.depth_to_space(inputs, self.block_size)
        elif self.mode == "CRD":
            # help want, native tensorflow is not support CRD mode, this way will generate 5 dims op.
            b, h, w, c = inputs.shape
            tmp = tf.reshape(
                inputs, [b, h, w, c // (self.block_size * self.block_size), self.block_size, self.block_size]
            )
            tmp = tflite_compat_transpose(tmp, perm=[0, 1, 4, 2, 5, 3])
            tmp = tf.reshape(
                tmp, [b, h * self.block_size, w * self.block_size, c // (self.block_size * self.block_size)]
            )
            return tmp
        else:
            raise KeyError(f"For DepthToSpace, mode must be [DCR, CRD], not {self.mode}")