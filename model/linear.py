import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense

class Linear(Layer):
    def __init__(self):
        super(Linear, self).__init__()
        self.linear = tf.keras.Sequential([
            Dense(2048, activation='relu'),
            Dense(512, activation='relu'),
            Dense(1, activation='sigmoid')
        ])


    def call(self, x):
        """
        Args:
            x (tensor): Input with shape (..., num_cells, d_model)
        Returns:
            out (tensor): Output with shape (..., num_cells)
        """

        num_cells = tf.shape(x)[-2]
        
        out = self.linear(x)  # shape (..., num_cells, 1)
        out = tf.reshape(out, (-1, num_cells))  # shape (..., num_cells)

        return out