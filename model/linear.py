import tensorflow as tf
from tensorflow.keras.layers import Layer, Dense, Dropout, BatchNormalization

class Linear(Layer):
    def __init__(self, d_ff, dropout):
        """
        Args:
            d_ff (int): Dimension of the feed forward layer.
            dropout (float): Dropout rate.
        """
        
        super(Linear, self).__init__()
        self.ff = Dense(d_ff, activation='relu')
        self.dropout = Dropout(dropout)
        self.top = Dense(1, activation='sigmoid')
        self.batch_norm = BatchNormalization()


    def call(self, x, cell_mask, is_training):
        """
        Args:
            x (tensor): Input with shape (..., num_cells, d_model)
            cell_mask (tensor): Cell mask with shape (..., num_cells)
            is_training (bool): Whether the model is being trained.
        Returns:
            out (tensor): Output with shape (..., num_cells)
        """

        num_cells = tf.shape(x)[-2]
        
        x = self.batch_norm(x)

        out = self.ff(x)
        out = self.dropout(out, training=is_training)
        out = self.top(out)

        out = tf.reshape(out, (-1, num_cells))  # shape (..., num_cells)

        out = tf.math.multiply(out, cell_mask)  # shape (..., num_cells)

        return out
        