import tensorflow as tf
from tensorflow.keras import layers, models


class TokenPruner(layers.Layer):
    def __init__(self, keep_ratio=0.7):
        super(TokenPruner, self).__init__()
        self.keep_ratio = keep_ratio

    def build(self, input_shape):
        self.score_layer = layers.Dense(1)

    def call(self, x):
        scores = tf.squeeze(self.score_layer(x), axis=-1)
        k = tf.cast(tf.round(tf.shape(scores)[1] * self.keep_ratio), tf.int32)
        _, idx = tf.math.top_k(scores, k=k, sorted=False)

        batch = tf.range(tf.shape(x)[0])[:, None]
        batch = tf.tile(batch, [1, k])
        gather_idx = tf.stack([batch, idx], axis=-1)

        return tf.gather_nd(x, gather_idx)


def transformer_block(x, dim, heads):
    attn = layers.MultiHeadAttention(num_heads=heads, key_dim=dim)(x, x)
    x = layers.Add()([x, attn])
    x = layers.LayerNormalization()(x)
    ff = layers.Dense(dim * 2, activation="gelu")(x)
    ff = layers.Dense(dim)(ff)
    x = layers.Add()([x, ff])
    return layers.LayerNormalization()(x)


def build_model(shape=(32, 32, 3), classes=10, keep_ratio=0.7):
    inputs = layers.Input(shape=shape)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)

    x = layers.Reshape((-1, 64))(x)
    x = TokenPruner(keep_ratio)(x)
    x = transformer_block(x, dim=64, heads=4)
    x = layers.GlobalAveragePooling1D()(x)
    outputs = layers.Dense(classes, activation="softmax")(x)
    return models.Model(inputs, outputs)


model = build_model()
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model.fit(
    x_train, y_train,
    epochs=20,
    batch_size=64,
    validation_data=(x_test, y_test),
    verbose=1
)
