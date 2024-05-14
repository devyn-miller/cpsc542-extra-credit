from keras.models import Sequential
from keras.layers import Dense

def build_model(input_dim):
    """
    Constructs and compiles a neural network model with a binary classification output.

    This function builds a Sequential model with three Dense layers. The first two layers use ReLU activation,
    and the final layer uses sigmoid activation suitable for binary classification tasks. The model is compiled
    with the Adam optimizer and binary cross-entropy loss function.

    Parameters:
    input_dim (int): The number of input features for the model.

    Returns:
    keras.engine.sequential.Sequential: A compiled Keras Sequential model ready for training.
    """
    model = Sequential()
    model.add(Dense(units=256, activation='relu', input_dim=input_dim))
    model.add(Dense(units=64, activation='relu'))
    model.add(Dense(units=1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model
