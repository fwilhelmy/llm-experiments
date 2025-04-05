"""
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import numpy as np
import math

from typing import Tuple, List, Dict, Union

from data import get_arithmetic_dataset
from plots import visualize_attention_matshow

########################################################################################
########################################################################################

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-5):
        super(LayerNorm, self).__init__()
        self.hidden_size = hidden_size
        self.eps = eps

        self.weight = nn.Parameter(torch.Tensor(hidden_size))
        self.bias = nn.Parameter(torch.Tensor(hidden_size))

        self.reset_parameters()

    def forward(self, inputs):
        """
        Layer Normalization.

        This module applies Layer Normalization, with rescaling and shift,
        only on the last dimension.

        Parameters
        ----------
        inputs (`torch.FloatTensor` of shape `(*dims, hidden_size)`)
            The input tensor. This tensor can have an arbitrary number N of
            dimensions, as long as `inputs.shape[N-1] == hidden_size`. The
            leading N - 1 dimensions `dims` can be arbitrary.

        Returns
        -------
        outputs (`torch.FloatTensor` of shape `(*dims, hidden_size)`)
            The output tensor, having the same shape as `inputs`.
        """

        mean = inputs.mean(dim=-1, keepdim=True)
        var = (inputs - mean).pow(2).mean(dim=-1, keepdim=True)
        # var = inputs.var(dim=-1, unbiased=False, keepdim=True)
        normalized = (inputs - mean) / torch.sqrt(var + self.eps)
        return normalized * self.weight + self.bias

    def reset_parameters(self):
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)

########################################################################################
########################################################################################

class MultiHeadedAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, bias: bool = True) -> None:
        super(MultiHeadedAttention, self).__init__()

        assert d_model % num_heads == 0
        self.head_size = d_model // num_heads
        self.num_heads = num_heads
      
        self.W_Q = nn.Linear(d_model, d_model, bias=bias)
        self.W_K = nn.Linear(d_model, d_model, bias=bias)
        self.W_V = nn.Linear(d_model, d_model, bias=bias)
        self.W_O = nn.Linear(d_model, d_model, bias=bias)

    def get_attention_weights(self, queries, keys):
        """
        Compute the attention weights.

        This computes the attention weights for all the sequences and all the
        heads in the batch. For a single sequence and a single head (for simplicity), 
        if Q are the queries (matrix of size `(sequence_length, head_size)`),
        and K are the keys (matrix of size `(sequence_length, head_size)`), then
        the attention weights are computed as

            weights = softmax(Q * K^{T} / sqrt(head_size))

        Here "*" is the matrix multiplication. Your attention weights must
        take into account the fact that we have a causal language model, i.e.
        there should be no influence from the future, attention is only computed on the past. 

        Parameters
        ----------
        queries (`torch.FloatTensor` of shape `(batch_size, num_heads, sequence_length, head_size)`)
            Tensor containing the queries for all the positions in the sequences
            and all the heads. For example, `queries[1, 3, 5]` is the query of
            the 4th head (index 3) for the 6th token (index 5) in the 2nd
            sequence (index 1) in the batch (it is a vector of size `head_size`).

        keys (`torch.FloatTensor` of shape `(batch_size, num_heads, sequence_length, head_size)`)
            Tensor containing the keys for all the positions in the sequences
            and all the heads. For example, `keys[0, 2, 4]` is the key of the
            3rd head (index 2) for the 5th token (index 4) in the 1st sequence
            (index 0) in the batch (it is a vector of size `head_size`).

        Returns
        -------
        attention_weights (`torch.FloatTensor` of shape `(batch_size, num_heads, sequence_length, sequence_length)`)
            Tensor containing the attention weights for all the heads and all
            the sequences in the batch. For example, `attention_weights[1, 3, 5, 7]`
            is the attention weights from the 8th token (index 7) on the 6th
            token (index 5) of the 4th head (index 3) in the 2nd sequence
            (index 1) in the batch. Note that because we have a causal language
            model here, `attention_weights[1, 3, 5, 7] == 0`, since the 8th token
            should not influence on the 6th token (7 > 5).
        """

        scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(self.head_size) # (B, num_heads, S, S)
        mask = torch.triu(torch.ones(scores.size(-2), scores.size(-1)), diagonal=1).to(scores.device) # (S, S)
        masked_scores = scores.masked_fill(mask == 1, float('-inf')) # (B, num_heads, S, S)
        return F.softmax(masked_scores, dim=-1)

    def apply_attention(self, queries, keys, values):
        """
        Apply the attention.

        This computes the output of the attention, for all the sequences and
        all the heads in the batch. For a single sequence and a single head
        (for simplicity), if Q are the queries (matrix of size `(sequence_length, head_size)`),
        K are the keys (matrix of size `(sequence_length, head_size)`), and V are
        the values (matrix of size `(sequence_length, head_size)`), then the ouput
        of the attention is given by

            weights = softmax(Q * K^{T} / sqrt(head_size))
            attended_values = weights * V
            outputs = concat(attended_values)

        Here "*" is the matrix multiplication, and "concat" is the operation
        that concatenates the attended values of all the heads (see the
        `merge_heads` function). See Lecture 06, slides 19-24.

        Parameters
        ----------
        queries (`torch.FloatTensor` of shape `(batch_size, num_heads, sequence_length, head_size)`)
            Tensor containing the queries for all the positions in the sequences
            and all the heads. For example, `queries[1, 3, 5]` is the query of
            the 4th head (index 3) for the 6th token (index 5) in the 2nd
            sequence (index 1) in the batch (it is a vector of size `head_size`).

        keys (`torch.FloatTensor` of shape `(batch_size, num_heads, sequence_length, head_size)`)
            Tensor containing the keys for all the positions in the sequences
            and all the heads. For example, `keys[0, 2, 4]` is the key of the
            3rd head (index 2) for the 5th token (index 4) in the 1st sequence
            (index 0) in the batch (it is a vector of size `head_size`).

        values (`torch.FloatTensor` of shape `(batch_size, num_heads, sequence_length, head_size)`)
            Tensor containing the values for all the positions in the sequences
            and all the heads. For example, `values[1, 3, 5]` is the key of the
            4th head (index 3) for the 6th token (index 5) in the 2nd sequence
            (index 1) in the batch (it is a vector of size `head_size`).

        Returns
        -------
        context (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_heads * head_size)`)
            Tensor containing the concatenated outputs of the attention for all
            the sequences in the batch, and all positions in each sequence. For
            example, `context[0, 2]` contains the output of the attention
            (concatenated for all heads) for the 3rd token (index 2) of the 1st
            sequence in the batch (index 0).

        attn_weights (`torch.FloatTensor` of shape `(batch_size, num_heads, sequence_length, sequence_length)`)
            Tensor containing the attention weights for all the heads and all
            the sequences in the batch. This tensor is used for visualization
            purposes only, and it is not used in the computation graph for
            backpropagation. It is returned here for debugging purposes.
            
        """

        attn_weights = self.get_attention_weights(queries, keys) # (B, num_heads, S, S)
        outputs = torch.matmul(attn_weights, values) # (B, num_heads, S, head_size)
        context = self.merge_heads(outputs) # (B, S, num_heads * head_size)
        
        # Use clone().detach() to detach attn_weights from the computation graph
        # Since we don't need to backpropagate through them, we can detach them from the graph
        return context, attn_weights.clone().detach()


    def split_heads(self, tensor):
        """
        Split the head vectors.

        This function splits the head vectors that have been concatenated (e.g. through the `merge_heads` function) 
        into a separate dimension. This function also transposes the `sequence_length` and `num_heads` axes.
        It only reshapes and transposes the input tensor, and it does not apply any further transformation to the tensor. 
        The function `split_heads` is the inverse of the function `merge_heads`.

        Parameters
        ----------
        tensor (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_heads * dim)`)
            Input tensor containing the concatenated head vectors (each having a size `dim`, which can be arbitrary).

        Returns
        -------
        output (`torch.FloatTensor` of shape `(batch_size, num_heads, sequence_length, dim)`)
            Reshaped and transposed tensor containing the separated head vectors. 
            Here `dim` is the same dimension as the one in the definition of the input `tensor` above.
        """

        batch_size, seq_size, _ = tensor.size() # Batch size, sequence length, hidden size
        tensor = tensor.view(batch_size, seq_size, self.num_heads, self.head_size)
        return tensor.transpose(1, 2) # (batch_size, num_heads, seq_size, head_size)

    def merge_heads(self, tensor):
        """
        Merge the head vectors.

        This function concatenates the head vectors in a single vector. This
        function also transposes the `sequence_length` and the newly created
        "merged" dimension. It only reshapes and transposes the input tensor,
        and it does not apply any further transformation to the tensor. The
        function `merge_heads` is the inverse of the function `split_heads`.

        Parameters
        ----------
        tensor (`torch.FloatTensor` of shape `(batch_size, num_heads, sequence_length, dim)`)
            Input tensor containing the separated head vectors (each having a size `dim`, which can be arbitrary).

        Returns
        -------
        output (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_heads * dim)`)
            Reshaped and transposed tensor containing the concatenated head vectors. 
            Here `dim` is the same dimension as the one in the definition of the input `tensor` above.
        """

        batch_size, num_heads, seq_size, head_size = tensor.size() # (batch_size, num_heads, seq_size, head_size)
        tensor = tensor.transpose(1, 2) # (batch_size, seq_size, num_heads, head_size)
        return tensor.reshape(batch_size, seq_size, num_heads * head_size)

    def forward(self,  queries: Tensor, keys: Tensor, values: Tensor):
        """
        Multi-headed attention.

        This applies the multi-headed attention on the input tensors `queries`,
        `keys`, and `values`. For self-attention, these three tensors are the
        same, but they can be different for cross-attention. For self-attention,
        these are the hidden states from the previous layer (a matrix of size 
        `(sequence_length, num_heads * head_size)` containing the concatenated head vectors). 
        For cross-attention, the keys and values are the hidden states from the decoder, 
        while the queries are the hidden states from the encoder.

        The output of multi-headed attention is given by

            Q = queries * W_{Q} + b_{Q}        # Queries
            K = keys * W_{K} + b_{K}           # Keys
            V = values * W_{V} + b_{V}         # Values

            context = attention(Q, K, V)              # Attended values (concatenated for all heads)
            outputs = context * W_{O} + b_{O}         # Linear projection

        Here "*" is the matrix multiplication.
        
        The function also returns the attention scores (return by apply_attention) 
        and the values (V) for visualization purposes only. These tensors are not 
        used in the computation graph for backpropagation. They are returned here for debugging purposes.

        Parameters
        ----------
        queries (`torch.FloatTensor` of shape `(batch_size, sequence_length, d_model)`)
            Tensor containing the queries for all the positions in the sequences. This
            is, for example, the tensor returned by the previous layer.

        keys (`torch.FloatTensor` of shape `(batch_size, sequence_length, d_model)`)
            Tensor containing the keys for all the positions in the sequences.

        values (`torch.FloatTensor` of shape `(batch_size, sequence_length, d_model)`)
            Tensor containing the values for all the positions in the sequences.

        
        Returns
        -------
        output (`torch.FloatTensor` of shape `(batch_size, sequence_length, num_heads * head_size)`)
            Tensor containing the output of multi-headed attention for all the
            sequences in the batch, and all positions in each sequence.
        
        attn_weights (`torch.FloatTensor` of shape `(batch_size, num_heads, sequence_length, sequence_length)`)
            Tensor containing the attention weights for all the heads and all
            the sequences in the batch. This tensor is used for visualization
            purposes only, and it is not used in the computation graph for
            backpropagation. It is returned here for debugging purposes.
        """

        Q = self.W_Q(queries)
        Q = self.split_heads(Q)

        K = self.W_K(keys)
        K = self.split_heads(K)

        V = self.W_V(values)
        V = self.split_heads(V)
        
        context, attn_weights = self.apply_attention(Q, K, V)

        return self.W_O(context), attn_weights
        
########################################################################################
########################################################################################

class Block(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        multiplier: int,
        dropout: float,
        non_linearity: str = "gelu",
        bias: bool = True
    ) -> None:
        """
        This module combines a Multi-headed Attention module and an MLP to
        create a layer of the transformer, with normalization and skip-connections.
        """
        super().__init__()

        assert non_linearity in ["relu", "gelu"]
        non_linearities = {"relu":nn.ReLU, "gelu":nn.GELU}

        self.self_attn = MultiHeadedAttention(d_model, num_heads, bias=bias)   
        self.self_attn_norm = LayerNorm(d_model)

        d_ff = int(multiplier * d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            non_linearities[non_linearity](),
            nn.Linear(d_ff, d_model, bias=False),
        )
        self.ffn_drop = nn.Dropout(p=dropout)
        self.ffn_norm = LayerNorm(d_model)

    def forward(self, x: Tensor) -> Tuple[Tensor, List[Tensor], List[Tensor]]:
        """
        Forward pass of the block.
        x: (batch_size, sequence_length, d_model)
        """
        a1, layer_attns = self.self_attn(x, x, x)
        a1 = self.self_attn_norm(x + a1)

        a2 = self.ffn(a1)
        a2 = self.ffn_drop(a2)
        a2 = self.ffn_norm(a1 + a2)

        # (B, S, d_model), 
        # (B, num_heads, S, S)
        return a2, layer_attns

########################################################################################
########################################################################################

class Decoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_blocks: int,
        multiplier: int,
        dropout: float,
        non_linearity: str = "gelu",
        bias: bool = True
    ) -> None:
        """
        A decoder layer.

        This module combines multiple blocks to create the decoder of the transformer
        """
        super().__init__()

        self.blocks = nn.ModuleList(
            [
                Block(d_model, num_heads, multiplier, dropout, non_linearity,  bias=bias)
                for _ in range(num_blocks)
            ]
        )

    def forward(self, x: Tensor) -> Tuple[Tensor, List[List[Tensor]], List[List[Tensor]]]:
        """
        Forward pass of the decoder.
        x: (batch_size, sequence_length, d_model)
        """
        a = x
        hidden_states = [a]
        attentions = []
        for block in self.blocks:
            a, layer_attentions = block(a)
            hidden_states.append(a)
            attentions.append(layer_attentions)
        
        # # (B, S, d_model),
        # # ( (B, S, d_model) x num_layers, (B, num_heads, S, S) x num_layers )
        # return a, (hidden_states, attentions)

        # (B, S, d_model), ( (B, num_layers, S, d_model), (B, num_layers, num_heads, S, S))
        return a, (torch.stack(hidden_states, dim=0).transpose(0, 1), torch.stack(attentions, dim=0).transpose(0, 1))


########################################################################################
########################################################################################

class GPTEmbedding(nn.Module):
    def __init__(
        self, 
        vocabulary_size,
        embedding_size,
        n_max_positions,
        padding_index:int=None,
    ):
        """
        Embedding module for GPT.
        This module combines token and positional embeddings

        Parameters
        ----------
        vocabulary_size : int
            The size of the vocabulary, i.e., the number of tokens.
        embedding_size : int
            The size of the token embeddings.
        n_max_positions : int
            The maximum number of positions to consider.
        padding_index : int, optional 
            The index of the padding token in the vocabulary. Default is None.
        """
        super(GPTEmbedding, self).__init__()

        self.tokens = nn.Embedding(vocabulary_size, embedding_size, padding_idx=padding_index)  # type: ignore

        self.register_buffer(
            "position_encoding", 
            self.create_sinusoidal_embeddings(
                n_positions=n_max_positions, dimension=embedding_size
            ) # (n_max_positions, embedding_size)
        )

    @classmethod
    def create_sinusoidal_embeddings(self, n_positions: int, dimension: int) -> torch.FloatTensor:
        """
        Create the sinusoidal embeddings.
        PE(pos, i) = sin(pos / 10000^((i//2)/dimension)) if i is even
                   = cos(pos / 10000^((i//2)/dimension)) if i is odd

        Parameters
        ----------
        n_positions : int
            The number of positions to consider.
        dimension : int
            The size of the embeddings.
        
        Returns
        -------
        position_embeddings (`torch.FloatTensor` of shape `(n_positions, dimension)`)
            The tensor containing the positional embeddings.
        """

        exponants = (torch.arange(dimension, dtype=torch.float32)//2)*2/dimension # (dimension,)
        positions = torch.arange(n_positions, dtype=torch.float32).unsqueeze(1) # (n_positions, 1)
        angles = positions / torch.pow(10000, exponants) # (n_positions, dimension)

        position_embeddings = torch.zeros(n_positions, dimension)
        position_embeddings[:, 0::2] = torch.sin(angles[:, 0::2]) # Apply sin to even indices
        position_embeddings[:, 1::2] = torch.cos(angles[:, 1::2]) # Apply cos to odd indices
    
        return position_embeddings


    def forward(self, tokens: Tensor) -> Tensor:
        """
        Return the embeddings from a sequence of input tokens 
        
        Parameters
        ----------
        tokens (`torch.LongTensor` of shape `(batch_size, sequence_length)`)
            The input tensor containing the token sequences. All the tokens
            must be integers in [0, vocabulary_size).

        Returns
        -------
        embeddings (`torch.FloatTensor` of shape `(batch_size, sequence_length, embedding_size)`)
            The tensor containing the embeddings. For example, `embeddings[0, 2]`
            is the embedding vector for the token in 3rd position (index 2)
            of the 1st sequence in the batch (index 0).
        
        """
       
        token_embeddings = self.tokens(tokens) # (batch_size, sequence_length, embedding_size)
        position_embeddings = self.position_encoding[:tokens.size(1), :].unsqueeze(0) # (1, sequence_length, embedding_size)
        return token_embeddings + position_embeddings # (batch_size, sequence_length, embedding_size)

########################################################################################
########################################################################################
  
class GPT(nn.Module):
    def __init__(
        self,
        num_heads: int, 
        num_layers: int,
        embedding_size : int,
        vocabulary_size : int,
        sequence_length : int,
        multiplier: float = 4,
        dropout: float = 0.0,
        non_linearity: str = "gelu",
        padding_index:int = None,
        bias_attention:bool=True,
        bias_classifier:bool=True,
        share_embeddings:bool=False
    ) -> None:
        super().__init__()

        self.num_heads = num_heads
        self.num_layers = num_layers
        self.embedding_size = embedding_size
        self.vocabulary_size = vocabulary_size
        self.sequence_length = sequence_length
        self.non_linearity = non_linearity

        self.embedding = GPTEmbedding(
            vocabulary_size = vocabulary_size,
            embedding_size = embedding_size,
            n_max_positions = sequence_length,
            padding_index = padding_index
        ) 

        
        self.decoder = Decoder(
            d_model = embedding_size,
            num_heads = num_heads,
            num_blocks = num_layers,
            multiplier = multiplier,
            dropout = dropout,
            non_linearity = non_linearity,
            bias = bias_attention
        )

        self.classifier = nn.Linear(embedding_size, vocabulary_size, bias=bias_classifier)

        # Tying classifier and embedding weights
        if share_embeddings:            
            self.classifier.weight = self.embedding.tokens.weight
  
    def forward(self, x: Tensor) -> Tuple[Tensor, Union[Tensor, None], Union[Tensor, None]]:
        """
        parameters:
            x : (batch_size, sequence_length,)
            vocab indices of decoder input token sequence

        returns:
            logits : (batch_size, sequence_length, vocab_size)
            (hidden_states, attentions) :  
                (batch_size, sequence_length, embedding_dimension) 
                and 
                (batch_size, num_layers, num_heads, sequence_length, sequence_length)
        """

        embeddings = self.embedding(x) # (batch_size, sequence_length, embedding_size)
        outputs, (hidden_states, attentions) = self.decoder(embeddings)
        logits = self.classifier(outputs) # (batch_size, sequence_length, vocab_size)
        return logits, (hidden_states, attentions)

########################################################################################
########################################################################################

if __name__ == "__main__":
    # Data
    p = 31
    operator = "+"
    operation_orders = [2]

    (dataset, _), tokenizer, MAX_LENGTH, padding_index = get_arithmetic_dataset(p, p, operator, 1.0, operation_orders, is_symmetric=False, shuffle=True, seed=42)
    inputs = dataset[:1]

    # Model 
    vocabulary_size=len(tokenizer)
    batch_size, sequence_length = 2, MAX_LENGTH
    embedding_size=8
    num_heads=4
    num_layers=2
    padding_index=padding_index
    assert embedding_size % num_heads == 0

    model = GPT(
        num_heads, 
        num_layers,
        embedding_size,
        vocabulary_size,
        sequence_length,
        multiplier = 4,
        dropout = 0.0,
        non_linearity = "gelu",
        padding_index = padding_index,
        bias_attention=True,
        bias_classifier=False,
        share_embeddings=False
    )

    logits, (hidden_states, attentions) = model(inputs[0])
    print(logits.shape, hidden_states.shape, attentions.shape)

    visualize_attention_matshow(model, inputs, tokenizer, "plots", "visualize_attention_matshow")