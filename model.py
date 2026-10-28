import tensorflow as tf
from tensorflow.keras import layers

from config import CFG


def width_multiplier(expert_id):
    return 1 + expert_id // CFG["EXPERTS_PER_WIDTH"]


def stage_channels(expert_id, stage):
    return int(16 * width_multiplier(expert_id) * (2 ** stage))


class ExpertNetwork(tf.keras.Model):

    def __init__(self, expert_id=0):
        super().__init__()
        self.expert_id = expert_id
        kernel = CFG["KERNEL_SIZE"]

        self.encoder_convs = [
            self._conv_block(stage_channels(expert_id, s), kernel) for s in range(4)
        ]
        self.encoder_pools = [layers.MaxPooling1D(2, padding='same') for _ in range(4)]

        lstm_units = int(CFG["LSTM_BASE"] * width_multiplier(expert_id))
        self.bilstm = layers.Bidirectional(
            layers.LSTM(lstm_units, return_sequences=True, return_state=True)
        )
        self.attention = layers.MultiHeadAttention(
            num_heads=CFG["ATTN_HEADS"], key_dim=CFG["ATTN_KEY_DIM"]
        )

        self.decoder_convs = [
            self._deconv_block(stage_channels(expert_id, s), kernel) for s in reversed(range(4))
        ]
        self.final_conv = layers.Conv1D(1, 1, activation='linear')

    def _conv_block(self, filters, kernel_size):
        return tf.keras.Sequential([
            layers.Conv1D(filters, kernel_size, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv1D(filters, kernel_size, padding='same', activation='relu'),
            layers.BatchNormalization(),
        ])

    def _deconv_block(self, filters, kernel_size):
        return tf.keras.Sequential([
            layers.Conv1DTranspose(filters, kernel_size, strides=2,
                                   padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv1D(filters, kernel_size, padding='same', activation='relu'),
            layers.BatchNormalization(),
        ])

    def call(self, inputs, training=None):
        x = inputs
        skips = []

        for conv, pool in zip(self.encoder_convs, self.encoder_pools):
            x = conv(x, training=training)
            skips.append(x)
            x = pool(x)

        bilstm_out, *states = self.bilstm(x)
        attn_out = self.attention(bilstm_out, bilstm_out)
        x = layers.Concatenate(axis=-1)([bilstm_out, attn_out])

        for i, deconv in enumerate(self.decoder_convs):
            x = deconv(x, training=training)
            skip = skips[-(i + 1)]

            current_length = x.shape[1]
            target_length = skip.shape[1]
            if current_length != target_length:
                if current_length < target_length:
                    x = layers.UpSampling1D(2)(x)
                    if x.shape[1] < target_length:
                        pad = target_length - x.shape[1]
                        x = layers.ZeroPadding1D((0, pad))(x)
                else:
                    x = layers.Cropping1D((0, current_length - target_length))(x)

            x = layers.Concatenate()([x, skip])

        return self.final_conv(x)


def gate_entropy(weights, mode):
    eps = 1e-8
    if mode == "per_sample":
        return tf.reduce_mean(-tf.reduce_sum(weights * tf.math.log(weights + eps), axis=-1))
    if mode == "batch_mean":
        mean_gate = tf.reduce_mean(weights, axis=0)
        return -tf.reduce_sum(mean_gate * tf.math.log(mean_gate + eps))
    raise ValueError('CFG["ENTROPY_MODE"] must be "per_sample" or "batch_mean"')


class MOE(tf.keras.Model):

    def __init__(self, num_experts=3):
        super().__init__()
        self.experts = [ExpertNetwork(i) for i in range(num_experts)]
        self.gating_network = self.build_gating()

    def build_gating(self):
        return tf.keras.Sequential([
            tf.keras.Input(shape=(CFG["WIN"], 1)),
            layers.Conv1D(64, 16, activation='relu', padding='same'),
            layers.MaxPooling1D(4),
            layers.Conv1D(128, 8, activation='relu', padding='same'),
            layers.MaxPooling1D(4),
            layers.Conv1D(256, 4, activation='relu', padding='same'),
            layers.GlobalMaxPool1D(),
            layers.Dense(128, activation='relu'),
            layers.Dense(len(self.experts), activation='softmax'),
        ], name="GatingNetwork")

    def call(self, inputs, training=None):
        expert_outputs = [expert(inputs, training=training) for expert in self.experts]
        weights = self.gating_network(inputs, training=training)
        if training and CFG["ENTROPY_LAMBDA"] > 0:
            self.add_loss(-CFG["ENTROPY_LAMBDA"] * gate_entropy(weights, CFG["ENTROPY_MODE"]))
        weights_expanded = weights[:, tf.newaxis, tf.newaxis, :]
        stacked_outputs = tf.stack(expert_outputs, axis=-1)
        return tf.reduce_sum(stacked_outputs * weights_expanded, axis=-1)