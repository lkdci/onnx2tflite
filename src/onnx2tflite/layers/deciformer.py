import tensorflow as tf
from ..utils.op_registry import OPERATOR
import keras


@OPERATOR.register_operator("DeciFormerAttentionSR")
class DeciFormerAttentionSR:
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs) -> None:
        self.embed_dim = node_attribute["embed_dim"]
        self.num_heads = node_attribute["num_heads"]
        self.sr_ratio = node_attribute["sr_ratio"]
        self.talking_heads = bool(node_attribute["talking_heads"])
        self.qk_scale = node_attribute["qk_scale"]

        if self.talking_heads:
            raise ValueError("talking_heads is not yet supported.")

        # Unpack weights
        q_weight = node_weights[node_inputs[1]].transpose(2, 3, 1, 0)
        sr_weight = node_weights[node_inputs[2]].transpose(2, 3, 1, 0)
        sr_bias = node_weights[node_inputs[3]]
        kv_weight = node_weights[node_inputs[4]].transpose(2, 3, 1, 0)
        proj_weight = node_weights[node_inputs[5]].transpose(2, 3, 1, 0)
        proj_bias = node_weights[node_inputs[6]]

        self.q = keras.layers.Conv2D(
            self.embed_dim, kernel_size=1, padding="SAME", use_bias=False, bias_initializer="zeros",
            kernel_initializer=keras.initializers.Constant(q_weight)
        )

        self.sr = keras.layers.Conv2D(
            self.embed_dim, kernel_size=self.sr_ratio, strides=self.sr_ratio, padding="SAME", use_bias=True,
            kernel_initializer=keras.initializers.Constant(sr_weight),
            bias_initializer=keras.initializers.Constant(sr_bias)
        )

        self.kv = keras.layers.Conv2D(
            2 * self.embed_dim, kernel_size=1, padding="SAME", use_bias=False, bias_initializer="zeros",
            kernel_initializer=keras.initializers.Constant(kv_weight)
        )

        self.proj = keras.layers.Conv2D(
            self.embed_dim, kernel_size=1, padding="SAME", use_bias=True,
            kernel_initializer=keras.initializers.Constant(proj_weight),
            bias_initializer=keras.initializers.Constant(proj_bias)
        )

    def __call__(self, inputs):
        B, H, W, C = inputs.shape
        # Q tensor
        q = self.q(inputs)
        q = tf.reshape(q, (B, H * W, self.num_heads, C // self.num_heads))
        q = tf.transpose(q, perm=[0, 2, 1, 3])

        # KV tensors
        kv = self.sr(inputs)
        kv = self.kv(kv)
        _, Hr, Wr, _ = kv.shape
        kv = tf.reshape(kv, (B, Hr * Wr, 2 * self.num_heads, C // self.num_heads))

        k = kv[:, :, :self.num_heads]
        v = kv[:, :, self.num_heads:]

        k = tf.transpose(k, perm=[0, 2, 3, 1])
        v = tf.transpose(v, perm=[0, 2, 1, 3])
        # Attention Q x K then attn x V
        attn = tf.matmul(q, k)
        attn = attn * self.qk_scale
        attn = keras.activations.softmax(attn, axis=-1)
        attn = tf.matmul(attn, v)

        # Projection
        attn = tf.transpose(attn, perm=[0, 2, 1, 3])
        attn = tf.reshape(attn, (B, H, W, C))
        attn = self.proj(attn)
        return attn


@OPERATOR.register_operator("DeciFormerAttention")
class DeciFormerAttention:
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs) -> None:
        self.embed_dim = node_attribute["embed_dim"]
        self.num_heads = node_attribute["num_heads"]
        self.talking_heads = bool(node_attribute["talking_heads"])
        self.qk_scale = node_attribute["qk_scale"]

        if self.talking_heads:
            raise ValueError("talking_heads is not yet supported.")

        # Unpack weights
        qkv_weight = node_weights[node_inputs[1]].transpose(2, 3, 1, 0)
        proj_weight = node_weights[node_inputs[2]].transpose(2, 3, 1, 0)
        proj_bias = node_weights[node_inputs[3]]

        self.qkv = keras.layers.Conv2D(
            self.embed_dim * 3, kernel_size=1, padding="SAME", use_bias=False, bias_initializer="zeros",
            kernel_initializer=keras.initializers.Constant(qkv_weight)
        )

        self.proj = keras.layers.Conv2D(
            self.embed_dim, kernel_size=1, padding="SAME", use_bias=True,
            kernel_initializer=keras.initializers.Constant(proj_weight),
            bias_initializer=keras.initializers.Constant(proj_bias)
        )

    def __call__(self, inputs):
        B, H, W, C = inputs.shape
        # Q tensor
        qkv = self.qkv(inputs)
        qkv = tf.reshape(qkv, (B, H * W, 3 * self.num_heads, C // self.num_heads))
        q = qkv[:, :, :self.num_heads]
        k = qkv[:, :, self.num_heads: 2 * self.num_heads]
        v = qkv[:, :, 2 * self.num_heads:]

        q = tf.transpose(q, perm=[0, 2, 1, 3])
        k = tf.transpose(k, perm=[0, 2, 3, 1])
        v = tf.transpose(v, perm=[0, 2, 1, 3])

        # Attention Q x K then attn x V
        attn = tf.matmul(q, k)
        attn = attn * self.qk_scale
        attn = keras.activations.softmax(attn, axis=-1)
        attn = tf.matmul(attn, v)

        # Projection
        attn = tf.transpose(attn, perm=[0, 2, 1, 3])
        attn = tf.reshape(attn, (B, H, W, C))
        attn = self.proj(attn)
        return attn
