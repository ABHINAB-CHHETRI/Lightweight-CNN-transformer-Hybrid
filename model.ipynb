import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.layers import Input, Reshape, MultiHeadAttention, LayerNormalization, GlobalAveragePooling1D

# Load CIFAR-10
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.cifar10.load_data()
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

inputs = Input(shape=(32, 32, 3))

# Deeper CNN Backbone
x = Conv2D(64, (3,3), activation='relu', padding='same')(inputs)
x = Conv2D(64, (3,3), activation='relu', padding='same')(x)
x = MaxPooling2D((2,2))(x)

x = Conv2D(128, (3,3), activation='relu', padding='same')(x)
x = Conv2D(128, (3,3), activation='relu', padding='same')(x)
x = MaxPooling2D((2,2))(x)

x = Conv2D(256, (3,3), activation='relu', padding='same')(x)
x = Conv2D(256, (3,3), activation='relu', padding='same')(x)
x = MaxPooling2D((2,2))(x)

# Reshape for transformer
shape = x.shape
x = Reshape((shape[1]*shape[2], shape[3]))(x)

# Transformer Encoder block (2 blocks)
for _ in range(2):
    attn_output = MultiHeadAttention(num_heads=8, key_dim=shape[3])(x, x)
    x = LayerNormalization(epsilon=1e-6)(x + attn_output)
    ffn_output = Dense(512, activation='relu')(x)
    ffn_output = Dense(shape[3])(ffn_output)
    x = LayerNormalization(epsilon=1e-6)(x + ffn_output)

# Pool and classify
x = GlobalAveragePooling1D()(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(10, activation='softmax')(x)

model = Model(inputs=inputs, outputs=outputs)

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.1)
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")
