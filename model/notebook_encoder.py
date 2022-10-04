import tensorflow as tf
from tensorflow.keras.layers import Layer
from transformers import TFAutoModel

class NotebookEncoder(Layer):
    def __init__(self, model_path, d_model):
        """
        Args:
            model_path (str): Path of the pre-trained model.
            d_model (int): Dimension of the model.
        """
        
        super(NotebookEncoder, self).__init__()
        self.bert = TFAutoModel.from_pretrained(model_path)
        self.d_model = d_model

    
    def call(self, input_ids, attention_mask):
        """
        Args:
            input_ids (tensor): List of the input IDs of the tokens with shape (..., num_cells, max_len)
            attention_mask (tensor): List of the attention masks of the tokens with shape (..., num_cells, max_len)
        Returns:
            out (tensor): Cell token embeddings with shape (..., num_cells, max_len, d_model)
        """
        
        num_cells = tf.shape(input_ids)[1]
        max_len = tf.shape(input_ids)[2]

        input_ids = tf.reshape(input_ids, shape=(-1, max_len))
        attention_mask = tf.reshape(attention_mask, shape=(-1, max_len))

        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)[0]
        out = tf.reshape(out, shape=(-1, num_cells, max_len, self.d_model))
        return out