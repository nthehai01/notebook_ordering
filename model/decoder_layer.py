from tensorflow.keras.layers import Layer

from model.transformer.transformer_decoder import TransformerDecoder

class DecoderLayer(Layer):
    def __init__(self, num_decoder_layers, d_model, n_heads, dropout, eps, d_ff, ff_activation, n_layers):
        """
        Args:
            num_decoder_layers (int): Number of decoder layers.
            d_model (int): Dimension of the model.
            n_heads (int): The number of heads for the multi-head attention.
            dropout (float): Dropout rate for the NotebookTransformer.
            eps (float): Epsilon for layer normalization.
            d_ff (int): Dimension of the feed forward layer.
            ff_activation (str): Activation function of the feed forward layer.
            n_layers (int): Number of transformer encoder blocks to be stacked.
        """
        
        super(DecoderLayer, self).__init__()
        self.num_decoder_layers = num_decoder_layers
        self.code_decoders = [TransformerDecoder(d_model, n_heads, dropout, eps, d_ff, ff_activation, n_layers) for _ in range(num_decoder_layers)]
        self.md_decoders = [TransformerDecoder(d_model, n_heads, dropout, eps, d_ff, ff_activation, n_layers) for _ in range(num_decoder_layers)]


    def call(self, code_embeddings, md_embeddings, is_training):
        """
        Args:
            code_embeddings (tensor): Code embedding with shape (..., max_code_cells, d_model)
            md_embeddings (tensor): Markdown embedding with shape (..., max_md_cells, d_model)
            is_training (bool): Whether the model is being trained.
        Returns:
            x_md (tensor): Output with shape (..., max_md_cells, d_model)
        """

        x_code = code_embeddings
        x_md = md_embeddings

        for step in range(self.num_decoder_layers):
            x_code = self.code_decoders[step](x_code, x_md, is_training)
            x_md = self.md_decoders[step](x_md, x_code, is_training)

        return x_md
