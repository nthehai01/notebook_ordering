import tensorflow as tf

from model.notebook_encoder import NotebookEncoder
from model.attention_pooling import AttentionPooling
from model.positional_encoder import PositionalEncoder
from model.decoder_layer import DecoderLayer
from model.linear import Linear


class Model(tf.keras.Model):
    def __init__(self, model_path, d_model, 
                max_cells, dropout_pos,
                num_decoder_layers,
                n_heads, dropout_trans, eps, d_ff_trans, ff_activation, n_layers):
        """
        Args:
            model_path (str): Path of the pre-trained model.
            d_model (int): Dimension of the model.
            dropout_pos (float): Dropout rate for the PositionEncoder.
            n_heads (int): The number of heads for the multi-head attention.
            max_cells (int): Maximum number of cells allowed for a notebook.
            dropout_trans (float): Dropout rate for the NotebookTransformer.
            eps (float): Epsilon for layer normalization.
            d_ff_trans (int): Dimension of the feed forward layer.
            ff_activation (str): Activation function of the feed forward layer.
            n_layers (int): Number of transformer encoder blocks to be stacked.
            d_ff_pointwise (int): Dimension of the feed forward layer for the PointwiseHead.
        """
        
        super(Model, self).__init__()
        self.code_encoder = NotebookEncoder(model_path, d_model)
        self.md_encoder = NotebookEncoder(model_path, d_model)
        self.code_attention_pooling = AttentionPooling(d_model)
        self.md_attention_pooling = AttentionPooling(d_model)
        self.positional_encoder = PositionalEncoder(d_model, max_cells, dropout_pos)
        self.decoder = DecoderLayer(num_decoder_layers, d_model, n_heads, dropout_trans, eps, d_ff_trans, ff_activation, n_layers)
        self.linear = Linear()


    def call(self, code_input_ids, code_attention_mask, 
            md_input_ids, md_attention_mask,
            is_training=False):
        """
        Args:
            input_ids (tensor): List of the input IDs of the tokens with shape (..., max_cells, max_len)
            attention_mask (tensor): List of the attention masks of the tokens with shape (..., max_cells, max_len)
            cell_features (tensor): Cell features with shape (..., max_cells, 2)
            cell_mask (tensor): Cell mask with shape (..., max_cells)
            is_training (bool): Whether the model is being trained. Default: False.
        Returns:
            out (tensor): Output with shape (..., max_cells)
        """
        

        # CODE ENCODING
        code_embeddings = self.code_encoder(code_input_ids, code_attention_mask)  # shape (..., max_cells, max_len, d_model)
        code_embeddings = self.code_attention_pooling(code_embeddings)  # shape (..., max_cells, d_model)
        code_embeddings = self.positional_encoder(code_embeddings, is_training)  # shape (..., max_cells, d_model)

        
        # MARKDOWN ENCODING
        md_embeddings = self.md_encoder(md_input_ids, md_attention_mask)  # shape (..., max_cells, max_len, d_model)
        md_embeddings = self.md_attention_pooling(md_embeddings)  # shape (..., max_cells, d_model)


        # TRANSFORMER DECODER
        out = self.decoder(code_embeddings, md_embeddings, is_training)  # shape (..., max_cells, d_model)


        # LINEAR
        out = self.linear(out)  # shape (..., max_cells)
        
        return out
        