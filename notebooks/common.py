import tensorflow as tf
from tensorflow import keras
from keras import layers

def build_model(image_size):
    # Inputs to the model
    input_img = layers.Input(
        shape=(image_size[0], image_size[1], 1), name="image", dtype="float32"
    )

    # First conv block
    x = layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv1",
    )(input_img)
    x = layers.MaxPooling2D((2, 2), name="pool1")(x)

    # Second conv block
    x = layers.Conv2D(
        64,
        (3, 3),
        activation="relu",
        kernel_initializer="he_normal",
        padding="same",
        name="Conv2",
    )(x)
    x = layers.MaxPooling2D((2, 2), name="pool2")(x)

    # We have used two max pool with pool size and strides 2.
    # Hence, downsampled feature maps are 4x smaller. The number of
    # filters in the last layer is 64. Reshape accordingly before
    # passing the output to the RNN part of the model
    new_shape = ((image_size[0] // 4), (image_size[1] // 4) * 64)
    x = layers.Reshape(target_shape=new_shape, name="reshape")(x)
    x = layers.Dense(64, activation="linear", name="dense1")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Flatten()(x)

    x = layers.Dense(32, activation="relu")(x)
    # Output layer
    x = layers.Dense(
        12, activation="sigmoid", name="dense2"
    )(x)

    # Add CTC layer for calculating CTC loss at each step
    output = x

    # Define the model
    model = keras.models.Model(
        inputs=input_img, outputs=output, name="keypoint_model_v1"
    )
    # Optimizer
    # Compile the model and return
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model
