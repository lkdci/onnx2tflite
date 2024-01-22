import tensorflow as tf
from ..utils.op_registry import OPERATOR
import keras


@OPERATOR.register_operator("SeaAttention")
class SeaAttention:
    def __init__(self, tensor_grap, node_weights, node_inputs, node_attribute, *args, **kwargs) -> None:
        self.num_heads = node_attribute["num_heads"]
        self.q_dims = node_attribute["q_dims"]
        self.v_dims = node_attribute["v_dims"]
        self.activation = node_attribute["activation"]
        self.qk_scale = node_attribute["qk_scale"]
        self.spatial_h = node_attribute["spatial_h"]
        self.spatial_w = node_attribute["spatial_w"]

        # Unpack weights
        qkv_weight = node_weights[node_inputs[1]].transpose(2, 3, 1, 0)
        qkv_bias = node_weights[node_inputs[2]]
        dwconv_weight = node_weights[node_inputs[3]].transpose(2, 3, 1, 0)
        dwconv_bias = node_weights[node_inputs[4]]
        pwconv_weight = node_weights[node_inputs[5]].transpose(2, 3, 1, 0)
        pwconv_bias = node_weights[node_inputs[6]]
        proj_row_weight = node_weights[node_inputs[7]].transpose(2, 3, 1, 0)
        proj_row_bias = node_weights[node_inputs[8]]
        proj_col_weight = node_weights[node_inputs[9]].transpose(2, 3, 1, 0)
        proj_col_bias = node_weights[node_inputs[10]]
        proj_weight = node_weights[node_inputs[11]].transpose(2, 3, 1, 0)
        proj_bias = node_weights[node_inputs[12]]

        # Positional embeddings
        pos_emb_rowq = node_weights[node_inputs[13]].transpose(0, 2, 1)
        pos_emb_rowq = pos_emb_rowq.reshape(pos_emb_rowq.shape[0], pos_emb_rowq.shape[1], 1, pos_emb_rowq.shape[2])
        self.pos_emb_rowq = tf.squeeze(
            tf.image.resize(pos_emb_rowq, (self.spatial_h, 1), method=tf.image.ResizeMethod.BILINEAR),
            axis=2
        )

        pos_emb_columnq = node_weights[node_inputs[14]].transpose(0, 2, 1)
        pos_emb_columnq = pos_emb_columnq.reshape(pos_emb_columnq.shape[0], pos_emb_columnq.shape[1], 1, pos_emb_columnq.shape[2])
        self.pos_emb_columnq = tf.squeeze(
            tf.image.resize(pos_emb_columnq, (self.spatial_w, 1), method=tf.image.ResizeMethod.BILINEAR),
            axis=2
        )

        pos_emb_rowk = node_weights[node_inputs[15]].transpose(0, 2, 1)
        pos_emb_rowk = pos_emb_rowk.reshape(pos_emb_rowk.shape[0], pos_emb_rowk.shape[1], 1, pos_emb_rowk.shape[2])
        self.pos_emb_rowk = tf.squeeze(
            tf.image.resize(pos_emb_rowk, (self.spatial_h, 1), method=tf.image.ResizeMethod.BILINEAR),
            axis=2
        )

        pos_emb_columnk = node_weights[node_inputs[16]].transpose(0, 2, 1)
        pos_emb_columnk = pos_emb_columnk.reshape(pos_emb_columnk.shape[0], pos_emb_columnk.shape[1], 1, pos_emb_columnk.shape[2])
        self.pos_emb_columnk = tf.squeeze(
            tf.image.resize(pos_emb_columnk, (self.spatial_w, 1), method=tf.image.ResizeMethod.BILINEAR),
            axis=2
        )

        self.to_qkv = keras.layers.Conv2D(
            qkv_weight.shape[3], kernel_size=1, padding="SAME", use_bias=True,
            kernel_initializer=keras.initializers.Constant(qkv_weight),
            bias_initializer=keras.initializers.Constant(qkv_bias),
        )
        self.dwconv = keras.layers.Conv2D(
            dwconv_weight.shape[3], kernel_size=dwconv_weight.shape[0], padding="SAME", use_bias=True,
            groups=dwconv_weight.shape[3],
            kernel_initializer=keras.initializers.Constant(dwconv_weight),
            bias_initializer=keras.initializers.Constant(dwconv_bias),
            activation=keras.activations.relu
        )
        self.pwconv = keras.layers.Conv2D(
            pwconv_weight.shape[3], kernel_size=1, padding="SAME", use_bias=True,
            kernel_initializer=keras.initializers.Constant(pwconv_weight),
            bias_initializer=keras.initializers.Constant(pwconv_bias),
        )
        self.proj_row = keras.layers.Conv2D(
            proj_row_weight.shape[3], kernel_size=1, padding="SAME", use_bias=True,
            kernel_initializer=keras.initializers.Constant(proj_row_weight),
            bias_initializer=keras.initializers.Constant(proj_row_bias),
        )
        self.proj_col = keras.layers.Conv2D(
            proj_col_weight.shape[3], kernel_size=1, padding="SAME", use_bias=True,
            kernel_initializer=keras.initializers.Constant(proj_col_weight),
            bias_initializer=keras.initializers.Constant(proj_col_bias),
        )
        self.proj = keras.layers.Conv2D(
            proj_weight.shape[3], kernel_size=1, padding="SAME", use_bias=True,
            kernel_initializer=keras.initializers.Constant(proj_weight),
            bias_initializer=keras.initializers.Constant(proj_bias),
        )

    def __call__(self, inputs):
        B, H, W, C = inputs.shape

        qkv = self.to_qkv(inputs)
        q = qkv[..., :self.q_dims]
        k = qkv[..., self.q_dims: 2 * self.q_dims]
        v = qkv[..., 2 * self.q_dims:]

        # detail enhance
        qkv = self.dwconv(qkv)
        qkv = self.pwconv(qkv)

        # squeeze axial attention
        # squeeze row
        qrow = tf.math.reduce_mean(q, axis=2, keepdims=False)
        krow = tf.math.reduce_mean(k, axis=2, keepdims=False)
        vrow = tf.math.reduce_mean(v, axis=2, keepdims=False)

        # squeeze column
        qcol = tf.math.reduce_mean(q, axis=1, keepdims=False)
        kcol = tf.math.reduce_mean(k, axis=1, keepdims=False)
        vcol = tf.math.reduce_mean(v, axis=1, keepdims=False)

        # Row attention branch
        qrow += self.pos_emb_rowq
        krow += self.pos_emb_rowk

        qrow = tf.reshape(qrow, (B, H, self.num_heads, self.q_dims // self.num_heads))
        qrow = tf.transpose(qrow, perm=[0, 2, 1, 3])

        krow = tf.reshape(krow, (B, H, self.num_heads, self.q_dims // self.num_heads))
        krow = tf.transpose(krow, perm=[0, 2, 3, 1])

        vrow = tf.reshape(vrow, (B, H, self.num_heads, self.v_dims // self.num_heads))
        vrow = tf.transpose(vrow, perm=[0, 2, 1, 3])

        attn_row = tf.matmul(qrow, krow)
        attn_row = attn_row * self.qk_scale
        attn_row = keras.activations.softmax(attn_row, axis=-1)
        attn_row = tf.matmul(attn_row, vrow)

        attn_row = tf.transpose(attn_row, perm=[0, 2, 1, 3])
        attn_row = tf.reshape(attn_row, (B, H, 1, self.v_dims))
        attn_row = keras.activations.relu(attn_row)
        attn_row = self.proj_row(attn_row)

        # Col attention branch
        qcol += self.pos_emb_columnq
        kcol += self.pos_emb_columnk

        qcol = tf.reshape(qcol, (B, W, self.num_heads, self.q_dims // self.num_heads))
        qcol = tf.transpose(qcol, perm=[0, 2, 1, 3])

        kcol = tf.reshape(kcol, (B, W, self.num_heads, self.q_dims // self.num_heads))
        kcol = tf.transpose(kcol, perm=[0, 2, 3, 1])

        vcol = tf.reshape(vcol, (B, W, self.num_heads, self.v_dims // self.num_heads))
        vcol = tf.transpose(vcol, perm=[0, 2, 1, 3])

        attn_col = tf.matmul(qcol, kcol)
        attn_col = attn_col * self.qk_scale
        attn_col = keras.activations.softmax(attn_col, axis=-1)
        attn_col = tf.matmul(attn_col, vcol)

        attn_col = tf.transpose(attn_col, perm=[0, 2, 1, 3])
        attn_col = tf.reshape(attn_col, (B, 1, W, self.v_dims))
        attn_col = keras.activations.relu(attn_col)
        attn_col = self.proj_col(attn_col)

        attn = attn_row + attn_col
        attn = v + attn
        attn = keras.activations.relu(attn)
        attn = self.proj(attn)
        # hard sigmoid
        attn = tf.clip_by_value(attn / 6 + 0.5, 0, 1)
        attn = attn * qkv
        return attn
