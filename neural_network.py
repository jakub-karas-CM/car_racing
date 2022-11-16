import tensorflow as tf

class NeuralNetwork(tf.keras.models.Sequential):
    def __init__(self, state_size, action_size, optimizer):
        super.__init__([
            tf.keras.Input(shape=(state_size,)),
            tf.keras.layers.Dense(10, activation='relu', name = 'hidden_1'),
            tf.keras.layers.Dense(10, activation='relu', name = 'hidden_2'),
            tf.keras.layers.Dense(action_size, activation='linear', name = 'ouput')
        ])
        self.compile(loss='mse', optimizer=optimizer)