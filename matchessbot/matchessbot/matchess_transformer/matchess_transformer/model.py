"""
The content of this file is based on the implementation of a single-agent transformer model for chess: 

The following list of classes in this file have been directly reused from:
https://github.com/sgrvinod/chess-transformers/blob/efe16dbf163c0c515a39afd79564e6449396d2eb/chess_transformers/transformers/modules.py
- LabelSmoothedCE: https://github.com/sgrvinod/chess-transformers/blob/efe16dbf163c0c515a39afd79564e6449396d2eb/chess_transformers/transformers/criteria.py#L9
- PositionWiseFCNetwork module from: https://github.com/sgrvinod/chess-transformers/blob/efe16dbf163c0c515a39afd79564e6449396d2eb/chess_transformers/transformers/modules.py#L260
- MultiHeadAttention module is almost identical to the implementation found at: https://github.com/sgrvinod/chess-transformers/blob/efe16dbf163c0c515a39afd79564e6449396d2eb/chess_transformers/transformers/modules.py#L10
    - Small modificaitons has been made to enable extracting the attention weights for visualization of the model's attention scores for specific input queries.
- The implemented ChessTransformerEncoder and MATChessTransformerEncoder module is based on the implementation of the single-agent Chess Transformer encoder only model from:
https://github.com/sgrvinod/chess-transformers/blob/efe16dbf163c0c515a39afd79564e6449396d2eb/chess_transformers/transformers/models.py#L223
    - Small changes have been made throughout the implemented ChessTransformerEncoder module to adapt it to the configuration file format used in this project.
    - Other changes include adapting the initialization of the weights and the format of the input/output of the original module to the desired parameters, input, and output of the desired model for this project
    - The implementation of those two modules still closely resemble the original code that they are based on.

        
The following list of modules are also based on that implementation of a single-agent transformer model for chess, but have been heavily modified in order to convert it to be used in a multi-agent reinforment learning setting:
- The BoardEncoder implementation in this file is based on the BoardEncoder module from: https://github.com/sgrvinod/chess-transformers/blob/efe16dbf163c0c515a39afd79564e6449396d2eb/chess_transformers/transformers/modules.py#L338
    - The implementation of the BoardEncoder module in this file has been heavily modified to convert it to be used in a multi-agent reinforcement learning setting.
"""

import math
import torch
import argparse
from torch import nn
import torch.nn.functional as F

from .config import import_config
#from config import import_config

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

"""
The LabelSmoothedCE module implementation for computing the Label-smoothed Cross Entropy Loss was copied from:
https://github.com/sgrvinod/chess-transformers/blob/efe16dbf163c0c515a39afd79564e6449396d2eb/chess_transformers/transformers/criteria.py#L9
"""
class LabelSmoothedCE(torch.nn.Module):
    """
    Cross Entropy loss with label-smoothing as a form of regularization.

    See "Rethinking the Inception Architecture for Computer Vision",
    https://arxiv.org/abs/1512.00567
    """

    def __init__(self, eps, n_predictions):
        """
        Init.

        Args:

            eps (float): Smoothing co-efficient. 

            n_predictions (int): Number of predictions expected per
            datapoint, or length of the predicted sequence.
        """
        super(LabelSmoothedCE, self).__init__()
        self.eps = eps
        self.indices = torch.arange(n_predictions).unsqueeze(0).to(DEVICE)  # (1, n_predictions)
        self.indices.requires_grad = False

    def forward(self, predicted, targets, lengths):
        """
        Forward prop.

        Args:

            predicted (torch.FloatTensor): The predicted probabilities,
            of size (N, n_predictions, vocab_size).

            targets (torch.LongTensor): The actual targets, of size (N,
            n_predictions).

            lengths (torch.LongTensor): The true lengths of the
            prediction sequences, not including special tokens, of size
            (N, 1).

        Returns:

            torch.Tensor: The mean label-smoothed cross-entropy loss, a
            scalar.
        """
        # print(f'\npredicted.shape: {predicted.shape}')
        # print(f'targets.shape: {targets.shape}')
        # print(f'lengths.shape: {lengths.shape}')

        # target_tensor_shape = targets.shape
        
        # Remove pad-positions and flatten
        predicted = predicted[
            self.indices < lengths
        ]  # (sum(lengths), vocab_size)
        targets = targets[self.indices < lengths]  # (sum(lengths))

        # "Smoothed" one-hot vectors for the gold sequences
        target_vector = (
            torch.zeros_like(predicted)
            .scatter(dim=1, index=targets.unsqueeze(1), value=1.0)
            .to(DEVICE)
        )  # (sum(lengths), vocab_size), one-hot
        target_vector = target_vector * (
            1.0 - self.eps
        ) + self.eps / target_vector.size(
            1
        )  # (sum(lengths), vocab_size), "smoothed" one-hot

        # Compute smoothed cross-entropy loss
        loss = (-1 * target_vector * F.log_softmax(predicted, dim=1)).sum(
            dim=1
        )  # (sum(lengths))

        # Compute mean loss
        loss = torch.mean(loss)

        return loss


class MultiHeadAttention(nn.Module):
    """
    The Multi-Head Attention sublayer.

    Reused from https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Machine-Translation.
    """

    def __init__(
        self, d_model, n_heads, d_queries, d_values, dropout, in_decoder=False
    ):
        """
        Init.

        Args:

            d_model (int): The size of vectors throughout the
            transformer model, i.e. input and output sizes for this
            sublayer.

            n_heads (int): The number of heads in the multi-head
            attention.

            d_queries (int): The size of query vectors (and also the
            size of the key vectors).

            d_values (int): The size of value vectors.

            dropout (float): The dropout probability.

            in_decoder (bool, optional): Is this Multi-Head Attention
            sublayer instance in the Decoder? Defaults to False.
        """
        super(MultiHeadAttention, self).__init__()

        self.d_model = d_model
        self.n_heads = n_heads

        self.d_queries = d_queries
        self.d_values = d_values
        self.d_keys = d_queries  # size of key vectors, same as of the query vectors to allow dot-products for similarity

        self.in_decoder = in_decoder

        # A linear projection to cast (n_heads sets of) queries from the
        # input query sequences
        self.cast_queries = nn.Linear(d_model, n_heads * d_queries)

        # A linear projection to cast (n_heads sets of) keys and values
        # from the input reference sequences
        self.cast_keys_values = nn.Linear(d_model, n_heads * (d_queries + d_values))

        # A linear projection to cast (n_heads sets of) computed
        # attention-weighted vectors to output vectors (of the same size
        # as input query vectors)
        self.cast_output = nn.Linear(n_heads * d_values, d_model)

        # Softmax layer
        self.softmax = nn.Softmax(dim=-1)

        # Layer-norm layer
        self.layer_norm = nn.LayerNorm(d_model)

        # Dropout layer
        self.apply_dropout = nn.Dropout(dropout)

    def forward(self, query_sequences, key_value_sequences, key_value_sequence_lengths):
        """
        Forward prop.

        Args:

            query_sequences (torch.FloatTensor): The input query
            sequences, of size (N, query_sequence_pad_length, d_model).

            key_value_sequences (torch.FloatTensor): The sequences to be
            queried against, of size (N, key_value_sequence_pad_length,
            d_model).

            key_value_sequence_lengths (torch.LongTensor): The true
            lengths of the key_value_sequences, to be able to ignore
            pads, of size (N).

        Returns:

            torch.FloatTensor: Attention-weighted output sequences for
            the query sequences, of size (N, query_sequence_pad_length,
            d_model).
        """
        batch_size = query_sequences.size(0)  # batch size (N) in number of sequences
        query_sequence_pad_length = query_sequences.size(1)
        key_value_sequence_pad_length = key_value_sequences.size(1)

        # Is this self-attention?
        self_attention = torch.equal(key_value_sequences, query_sequences)

        # Store input for adding later
        input_to_add = query_sequences.clone()

        # Apply layer normalization
        query_sequences = self.layer_norm(
            query_sequences
        )  # (N, query_sequence_pad_length, d_model)
        # If this is self-attention, do the same for the key-value
        # sequences (as they are the same as the query sequences) If
        # this isn't self-attention, they will already have been normed
        # in the last layer of the Encoder (from whence they came)
        if self_attention:
            key_value_sequences = self.layer_norm(
                key_value_sequences
            )  # (N, key_value_sequence_pad_length, d_model)

        # Project input sequences to queries, keys, values
        queries = self.cast_queries(
            query_sequences
        )  # (N, query_sequence_pad_length, n_heads * d_queries)
        keys, values = self.cast_keys_values(key_value_sequences).split(
            split_size=self.n_heads * self.d_keys, dim=-1
        )  # (N, key_value_sequence_pad_length, n_heads * d_keys), (N, key_value_sequence_pad_length, n_heads * d_values)

        # Split the last dimension by the n_heads subspaces
        queries = queries.contiguous().view(
            batch_size, query_sequence_pad_length, self.n_heads, self.d_queries
        )  # (N, query_sequence_pad_length, n_heads, d_queries)
        keys = keys.contiguous().view(
            batch_size, key_value_sequence_pad_length, self.n_heads, self.d_keys
        )  # (N, key_value_sequence_pad_length, n_heads, d_keys)
        values = values.contiguous().view(
            batch_size, key_value_sequence_pad_length, self.n_heads, self.d_values
        )  # (N, key_value_sequence_pad_length, n_heads, d_values)

        # Re-arrange axes such that the last two dimensions are the
        # sequence lengths and the queries/keys/values And then, for
        # convenience, convert to 3D tensors by merging the batch and
        # n_heads dimensions This is to prepare it for the batch matrix
        # multiplication (i.e. the dot product)
        queries = (
            queries.permute(0, 2, 1, 3)
            .contiguous()
            .view(-1, query_sequence_pad_length, self.d_queries)
        )  # (N * n_heads, query_sequence_pad_length, d_queries)
        keys = (
            keys.permute(0, 2, 1, 3)
            .contiguous()
            .view(-1, key_value_sequence_pad_length, self.d_keys)
        )  # (N * n_heads, key_value_sequence_pad_length, d_keys)
        values = (
            values.permute(0, 2, 1, 3)
            .contiguous()
            .view(-1, key_value_sequence_pad_length, self.d_values)
        )  # (N * n_heads, key_value_sequence_pad_length, d_values)

        # Perform multi-head attention

        # Perform dot-products
        attention_weights = torch.bmm(
            queries, keys.permute(0, 2, 1)
        )  # (N * n_heads, query_sequence_pad_length, key_value_sequence_pad_length)

        # Scale dot-products
        attention_weights = (
            1.0 / math.sqrt(self.d_keys)
        ) * attention_weights  # (N * n_heads, query_sequence_pad_length, key_value_sequence_pad_length)

        # Before computing softmax weights, prevent queries from
        # attending to certain keys

        # MASK 1: keys that are pads
        not_pad_in_keys = (
            torch.LongTensor(range(key_value_sequence_pad_length))
            .unsqueeze(0)
            .unsqueeze(0)
            .expand_as(attention_weights)
            .to(DEVICE)
        )  # (N * n_heads, query_sequence_pad_length, key_value_sequence_pad_length)
        not_pad_in_keys = (
            not_pad_in_keys
            < key_value_sequence_lengths.repeat_interleave(self.n_heads)
            .unsqueeze(1)
            .unsqueeze(2)
            .expand_as(attention_weights)
        )  # (N * n_heads, query_sequence_pad_length, key_value_sequence_pad_length)
        # Note: PyTorch auto-broadcasts singleton dimensions in
        # comparison operations (as well as arithmetic operations)

        # Mask away by setting such weights to a large negative number,
        # so that they evaluate to 0 under the softmax
        attention_weights = attention_weights.masked_fill(
            ~not_pad_in_keys, -float("inf")
        )  # (N * n_heads, query_sequence_pad_length, key_value_sequence_pad_length)

        # MASK 2: if this is self-attention in the Decoder, keys
        # chronologically ahead of queries
        if self.in_decoder and self_attention:
            # Therefore, a position [n, i, j] is valid only if j <= i
            # torch.tril(), i.e. lower triangle in a 2D matrix, sets j >
            # i to 0
            not_future_mask = (
                torch.ones_like(attention_weights).tril().bool().to(DEVICE)
            )  # (N * n_heads, query_sequence_pad_length, key_value_sequence_pad_length)

            # Mask away by setting such weights to a large negative
            # number, so that they evaluate to 0 under the softmax
            attention_weights = attention_weights.masked_fill(
                ~not_future_mask, -float("inf")
            )  # (N * n_heads, query_sequence_pad_length, key_value_sequence_pad_length)

        # Compute softmax along the key dimension
        attention_weights = self.softmax(
            attention_weights
        )  # (N * n_heads, query_sequence_pad_length, key_value_sequence_pad_length)

        # Apply dropout
        attention_weights = self.apply_dropout(
            attention_weights
        )  # (N * n_heads, query_sequence_pad_length, key_value_sequence_pad_length)

        # Calculate sequences as the weighted sums of values based on
        # these softmax weights
        sequences = torch.bmm(
            attention_weights, values
        )  # (N * n_heads, query_sequence_pad_length, d_values)

        # Unmerge batch and n_heads dimensions and restore original
        # order of axes
        sequences = (
            sequences.contiguous()
            .view(batch_size, self.n_heads, query_sequence_pad_length, self.d_values)
            .permute(0, 2, 1, 3)
        )  # (N, query_sequence_pad_length, n_heads, d_values)

        # Concatenate the n_heads subspaces (each with an output of size
        # d_values)
        sequences = sequences.contiguous().view(
            batch_size, query_sequence_pad_length, -1
        )  # (N, query_sequence_pad_length, n_heads * d_values)

        # Transform the concatenated subspace-sequences into a single
        # output of size d_model
        sequences = self.cast_output(
            sequences
        )  # (N, query_sequence_pad_length, d_model)

        # Apply dropout and residual connection
        sequences = (
            self.apply_dropout(sequences) + input_to_add
        )  # (N, query_sequence_pad_length, d_model)

        ##########################################
        self.attention_scores = attention_weights
        ##########################################

        return sequences


class PositionWiseFCNetwork(nn.Module):
    """
    The Position-Wise Feed Forward Network sublayer.

    Reused from https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Machine-Translation.
    """

    def __init__(self, d_model, d_inner, dropout):
        """
        Init.

        Args:

            d_model (int): The size of vectors throughout the
            transformer model, i.e. input and output sizes for this
            sublayer.

            d_inner (int): An intermediate size.

            dropout (float): The dropout probability.
        """
        super(PositionWiseFCNetwork, self).__init__()

        self.d_model = d_model
        self.d_inner = d_inner

        # Layer-norm layer
        self.layer_norm = nn.LayerNorm(d_model)

        # A linear layer to project from the input size to an
        # intermediate size
        self.fc1 = nn.Linear(d_model, d_inner)

        # ReLU
        self.relu = nn.ReLU()

        # A linear layer to project from the intermediate size to the
        # output size (same as the input size)
        self.fc2 = nn.Linear(d_inner, d_model)

        # Dropout layer
        self.apply_dropout = nn.Dropout(dropout)

    def forward(self, sequences):
        """
        Forward prop.

        Args:

            sequences (torch.FloatTensor): The input sequences, of size
            (N, pad_length, d_model).

        Returns:

            torch.FloatTensor: The transformed output sequences, of size
            (N, pad_length, d_model).
        """

        # Store input for adding later
        input_to_add = sequences.clone()  # (N, pad_length, d_model)

        # Apply layer-norm
        sequences = self.layer_norm(sequences)  # (N, pad_length, d_model)

        # Transform position-wise
        sequences = self.apply_dropout(
            self.relu(self.fc1(sequences))
        )  # (N, pad_length, d_inner)
        sequences = self.fc2(sequences)  # (N, pad_length, d_model)

        # Apply dropout and residual connection
        sequences = (
            self.apply_dropout(sequences) + input_to_add
        )  # (N, pad_length, d_model)

        return sequences







class BoardEncoder(nn.Module):
    """
    The Board Encoder.

    Adapted from https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Machine-Translation.
    """

    def __init__(
        self,
        vocab_sizes,
        d_model,
        n_heads,
        d_queries,
        d_values,
        d_inner,
        n_layers,
        dropout,
    ):
        """
        Init.

        Args:

            vocab_sizes (dict): The vocabulary sizes of input sequence
            components.

            d_model (int): The size of vectors throughout the
            transformer model, i.e. input and output sizes for the
            Encoder.

            n_heads (int): The number of heads in the multi-head
            attention.

            d_queries (int): The size of query vectors (and also the
            size of the key vectors) in the multi-head attention.

            d_values (int): The size of value vectors in the multi-head
            attention.

            d_inner (int): An intermediate size in the position-wise FC.

            n_layers (int): The number of [multi-head attention +
            position-wise FC] layers in the Encoder.

            dropout (float): The dropout probability.
        """
        super(BoardEncoder, self).__init__()

        self.vocab_sizes = vocab_sizes
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_queries = d_queries
        self.d_values = d_values
        self.d_inner = d_inner
        self.n_layers = n_layers
        self.dropout = dropout

        # # Embedding layers
        # self.turn_embeddings = nn.Embedding(vocab_sizes["turn"], d_model)
        # self.white_kingside_castling_rights_embeddings = nn.Embedding(
        #     vocab_sizes["white_kingside_castling_rights"], d_model
        # )
        # self.white_queenside_castling_rights_embeddings = nn.Embedding(
        #     vocab_sizes["white_queenside_castling_rights"], d_model
        # )
        # self.black_kingside_castling_rights_embeddings = nn.Embedding(
        #     vocab_sizes["black_kingside_castling_rights"], d_model
        # )
        # self.black_queenside_castling_rights_embeddings = nn.Embedding(
        #     vocab_sizes["black_queenside_castling_rights"], d_model
        # )
        # self.board_position_embeddings = nn.Embedding(
        #     vocab_sizes["board_position"], d_model
        # )

        # Embedding layers
        # 'board_position': board_posiiton,
        # "kingside_castling_rights": kingside_castling_rights,
        # "queenside_castling_rights": queenside_castling_rights,
        # "opponent_castling_rights_kingside": opponent_castling_rights_kingside,
        # "opponent_castling_rights_queenside": opponent_castling_rights_queenside,
        # "agent_ids": agent_ids,
        # "agents_pos": agents_pos,
        # "reward_weights": reward_weights,
        # "moves": moves,
        # "rewards": weighted_target_rewards,
        
        self.board_position_embeddings = nn.Embedding(
            vocab_sizes["board_position"], d_model
        )
        self.kingside_castling_rights_embeddings = nn.Embedding(
            vocab_sizes["kingside_castling_rights"], d_model
        )
        self.queenside_castling_rights_embeddings = nn.Embedding(
            vocab_sizes["queenside_castling_rights"], d_model
        )
        self.opponent_kingside_castling_rights_embeddings = nn.Embedding(
            vocab_sizes["opponent_castling_rights_kingside"], d_model
        )
        self.opponent_queenside_castling_rights_embeddings = nn.Embedding(
            vocab_sizes["opponent_castling_rights_queenside"], d_model
        )
        
        self.agent_ids_embeddings = nn.Embedding(
            vocab_sizes["agent_ids"], vocab_sizes["agent_ids"]
        )
        self.agents_pos_embeddings = nn.Embedding(
            vocab_sizes["agents_pos"], vocab_sizes["agents_pos"],
        )
        
        # # # self.agent_reward_weights_embeddings = nn.Embedding(
        # # #     vocab_sizes["agents_rewards"], d_model
        # # # )
        # # self.agent_reward_weights_embeddings = nn.Linear(
        # #     vocab_sizes["agents_rewards"], d_model
        # # )

        self.per_agent_input_channels = vocab_sizes["agent_ids"] + vocab_sizes["agents_pos"] + vocab_sizes["agents_rewards"]
        # self.per_agent_state_linear = nn.Linear(
        #     self.per_agent_input_channels, d_model
        # )
        self.per_agent_state_embeddings = nn.Linear(
            self.per_agent_input_channels, d_model
        )

        # # self.per_agent_state_embeddings = nn.Embedding(
        # #     d_model , d_model
        # # )

        # print(f"\nper-agent state input feature channels: {self.per_agent_input_channels}")


        # per_agent_reward_weights = [1.0] * self.per_agent_input_channels
        # self.per_agent_state_embeddings = nn.Embedding(
        #     torch.tensor(per_agent_reward_weights, requires_grad=True).long(), d_model
        # )
        
        # print(f"\nshape of self.agent_reward_weights_embeddings input features={self.agent_reward_weights_embeddings.in_features},output features={self.agent_reward_weights_embeddings.out_features}\n")
        # print(f"shape of per-agent embeddings:\tagent_ids_embeddings.shape={self.agent_ids_embeddings.shape}")

        #torch.tensor(per_agent_reward_weights, requires_grad=True)

        # # Positional embedding layer
        # self.positional_embeddings = nn.Embedding(
        #     vocab_sizes["agent_ids"] + 68,
        #     d_model,
        # )
        self.model_input_sequence_length = 64 + 4 + 16
        # print(f"BOARD_STATUS_LENGTH={self.model_input_sequence_length}?")


        # Positional embedding layer
        self.positional_embeddings = nn.Embedding(
            self.model_input_sequence_length,
            d_model,
        )

        # Encoder layers
        self.encoder_layers = nn.ModuleList(
            [self.make_encoder_layer() for i in range(n_layers)]
        )

        # Dropout layer
        self.apply_dropout = nn.Dropout(dropout)

        # Layer-norm layer
        self.layer_norm = nn.LayerNorm(d_model)

    def make_encoder_layer(self):
        """
        Creates a single layer in the Encoder by combining a multi-head
        attention sublayer and a position-wise FC sublayer.
        """
        # A ModuleList of sublayers
        encoder_layer = nn.ModuleList(
            [
                MultiHeadAttention(
                    d_model=self.d_model,
                    n_heads=self.n_heads,
                    d_queries=self.d_queries,
                    d_values=self.d_values,
                    dropout=self.dropout,
                    in_decoder=False,
                ),
                PositionWiseFCNetwork(
                    d_model=self.d_model, d_inner=self.d_inner, dropout=self.dropout
                ),
            ]
        )

        return encoder_layer

    def forward(
        self,
        # turns,
        # white_kingside_castling_rights,
        # white_queenside_castling_rights,
        # black_kingside_castling_rights,
        # black_queenside_castling_rights,
        board_positions,
        kingside_castling_rights,
        queenside_castling_rights,
        opponent_kingside_castling_rights,
        opponent_queen_castling_rights,
        agent_ids,
        agents_pos,
        agent_reward_weights,
    ):
        """
        Forward prop.

        Args:

            turns (torch.LongTensor): The current turn (w/b), of size
            (N, 1).

            white_kingside_castling_rights (torch.LongTensor): Whether
            white can castle kingside, of size (N, 1).

            white_queenside_castling_rights (torch.LongTensor): Whether
            white can castle queenside, of size (N, 1).

            black_kingside_castling_rights (torch.LongTensor): Whether
            black can castle kingside, of size (N, 1).

            black_queenside_castling_rights (torch.LongTensor): Whether
            black can castle queenside, of size (N, 1).

            board_positions (torch.LongTensor): The current board
            positions, of size (N, 64).

        Returns:

            torch.FloatTensor: The encoded board, of size (N,
            BOARD_STATUS_LENGTH, d_model).
        """
        batch_size = board_positions.size(0)  # N
        # print(f"batch_size = {batch_size}")

        # print(f"\nIn BoardEncoder.forward(): shape of agent_reward_weights={agent_reward_weights.shape}\n")

        # # reward_type_0_weight_embedding_input = agent_reward_weights[:,:,0]
        # # print(f'Example of rewardtype_0 reward weights output:\ttype{type(reward_type_0_weight_embedding_input)} \tshape:\t{reward_type_0_weight_embedding_input.shape}')

        # print(f"\nShape of agent_ids input = {agent_ids.shape}")
        # print(f"\nShape of agents_pos input = {agents_pos.shape}")

        # agent_id_embedded_output = self.agent_ids_embeddings(agent_ids)
        # agent_pos_embedded_output = self.agents_pos_embeddings(agents_pos)


        # print(f"\nShape of agent_ids_embeddings(agent_ids) output = {agent_id_embedded_output.shape}")
        # print(f"\nShape of agents_pos_embeddings(agents_pos) output = {agent_pos_embedded_output.shape}")

        # # agent_ids_example_datapoint = agent_id_embedded_output[0]
        # # print(f"\n\nagent_ids_example_datapoint={agent_ids_example_datapoint}")

        # # agent_pos_example_datapoint = agent_pos_embedded_output[0]
        # # print(f"\n\nagent_pos_example_datapoint={agent_pos_example_datapoint}")

        per_agent_input_features = torch.cat(
            [
                self.agent_ids_embeddings(
                    agent_ids
                ),
                self.agents_pos_embeddings(
                    agents_pos
                ),
                agent_reward_weights
            ],
            dim=2
        )

        # print(f"shape of board_position input channels={board_positions.shape}")
        # print(f"\nshape of board embeddings output={self.board_position_embeddings(board_positions).shape}\n")
        
        # print(f"\nshape of kingside_castling_rights_embeddings output={self.kingside_castling_rights_embeddings(kingside_castling_rights).shape}\n")
        # print(f"Castling rights input example: {type(kingside_castling_rights)}: {kingside_castling_rights.shape}")


        # print(f"\n\nshape of per_agent_input_features after torch.cat={per_agent_input_features.shape}\n\n")
        # # print(f"\nshape of per-agent state embeddings output={self.per_agent_state_embeddings(per_agent_input_features).size()}\n")

        # Embeddings
        embeddings = torch.cat(
            [
                #self.turn_embeddings(turns),
                self.board_position_embeddings(board_positions.reshape(batch_size, board_positions.size(2))),
                self.kingside_castling_rights_embeddings(
                    kingside_castling_rights
                ),
                self.queenside_castling_rights_embeddings(
                    queenside_castling_rights
                ),
                self.opponent_kingside_castling_rights_embeddings(
                    opponent_kingside_castling_rights
                ),
                self.opponent_queenside_castling_rights_embeddings(
                    opponent_queen_castling_rights
                ),
                self.per_agent_state_embeddings(per_agent_input_features)
            ],
            dim=1,
        )  # (N, BOARD_STATUS_LENGTH, d_model)
        # print(f"\nshape of board embeddings={embeddings.shape} - BOARD EMBEDDING WORKS!!!\n")

        # Add positional embeddings
        boards = embeddings + self.positional_embeddings.weight.unsqueeze(
            0
        )  # (N, BOARD_STATUS_LENGTH, d_model)
        boards = boards * math.sqrt(self.d_model)  # (N, BOARD_STATUS_LENGTH, d_model)

        # Dropout
        boards = self.apply_dropout(boards)  # (N, BOARD_STATUS_LENGTH, d_model)

        # Encoder layers
        for encoder_layer in self.encoder_layers:
            # Sublayers
            boards = encoder_layer[0](
                query_sequences=boards,
                key_value_sequences=boards,
                key_value_sequence_lengths=torch.LongTensor([self.model_input_sequence_length] * batch_size).to(
                    DEVICE
                ),
            )  # (N, BOARD_STATUS_LENGTH, d_model)
            boards = encoder_layer[1](
                sequences=boards
            )  # (N, BOARD_STATUS_LENGTH, d_model)

        # Apply layer-norm
        boards = self.layer_norm(boards)  # (N, BOARD_STATUS_LENGTH, d_model)

        return boards



class ChessTransformerEncoder(nn.Module):
    """
    The Chess Transformer (Encoder only), for predicting the next move.
    """

    def __init__(
        self,
        CONFIG,
    ):
        """
        Init.

        Args:

            CONFIG (dict): The configuration, containing the following
            parameters for the model:

                VOCAB_SIZES (dict): The sizes of the vocabularies of the
                Encoder sequence components.

                D_MODEL (int): The size of vectors throughout the
                transformer model, i.e. input and output sizes for the
                Encoder.

                N_HEADS (int): The number of heads in the multi-head
                attention.

                D_QUERIES (int): The size of query vectors (and also the
                size of the key vectors) in the multi-head attention.

                D_VALUES (int): The size of value vectors in the
                multi-head attention.

                D_INNER (int): An intermediate size in the position-wise
                FC.

                N_LAYERS (int): The number of [multi-head attention +
                multi-head attention + position-wise FC] layers in the
                Encoder.

                DROPOUT (int): The dropout probability.
        """
        super(ChessTransformerEncoder, self).__init__()

        self.code = "E"

        self.vocab_sizes = CONFIG['VOCAB_SIZES'] # CONFIG.VOCAB_SIZES
        self.d_model = CONFIG['D_MODEL']
        self.n_heads = CONFIG['N_HEADS']
        self.d_queries = CONFIG['D_QUERIES']
        self.d_values = CONFIG['D_VALUES']
        self.d_inner = CONFIG['D_INNER']
        self.n_layers = CONFIG['N_LAYERS']
        self.dropout = CONFIG['DROPOUT']

        # Encoder
        self.board_encoder = BoardEncoder(
            vocab_sizes=self.vocab_sizes,
            d_model=self.d_model,
            n_heads=self.n_heads,
            d_queries=self.d_queries,
            d_values=self.d_values,
            d_inner=self.d_inner,
            n_layers=self.n_layers,
            dropout=self.dropout,
        )

        # # Output linear layer that will compute logits for the
        # # vocabulary
        # self.fc = nn.Linear(self.d_model, self.vocab_sizes["moves"])

        # Output linear layers - for the "From" square and "To" square
        self.policy_head = nn.Linear(self.d_model, self.vocab_sizes["moves"])
        self.value_head = nn.Linear(self.d_model, self.vocab_sizes["agents_rewards"])

        # Initialize weights
        self.init_weights()

    def init_weights(self):
        """
        Initialize weights in the transformer model.
        """
        # Glorot uniform initialization with a gain of 1.
        for p in self.parameters():
            # Glorot initialization needs at least two dimensions on the
            # tensor
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=1.0)

        # For the embeddings, normal initialization with 0 mean and
        # 1/sqrt(d_model) S.D.
        nn.init.normal_(
            self.board_encoder.board_position_embeddings.weight,
            mean=0.0,
            std=math.pow(self.d_model, -0.5),
        )
        # nn.init.normal_(
        #     self.board_encoder.turn_embeddings.weight,
        #     mean=0.0,
        #     std=math.pow(self.d_model, -0.5),
        # )
        nn.init.normal_(
            self.board_encoder.kingside_castling_rights_embeddings.weight,
            mean=0.0,
            std=math.pow(self.d_model, -0.5),
        )
        nn.init.normal_(
            self.board_encoder.queenside_castling_rights_embeddings.weight,
            mean=0.0,
            std=math.pow(self.d_model, -0.5),
        )
        nn.init.normal_(
            self.board_encoder.opponent_kingside_castling_rights_embeddings.weight,
            mean=0.0,
            std=math.pow(self.d_model, -0.5),
        )
        nn.init.normal_(
            self.board_encoder.opponent_queenside_castling_rights_embeddings.weight,
            mean=0.0,
            std=math.pow(self.d_model, -0.5),
        )


        nn.init.normal_(
            self.board_encoder.agent_ids_embeddings.weight,
            mean=0.0,
            std=math.pow(self.d_model, -0.5),
        )
        nn.init.normal_(
            self.board_encoder.agents_pos_embeddings.weight,
            mean=0.0,
            std=math.pow(self.d_model, -0.5),
        )
        # nn.init.normal_(
        #     self.board_encoder.agent_reward_weights_embeddings.weight,
        #     mean=0.0,
        #     std=math.pow(self.d_model, -0.5),
        # )
        nn.init.normal_(
            self.board_encoder.per_agent_state_embeddings.weight,
            mean=0.0,
            std=math.pow(self.d_model, -0.5),
        )
        
        nn.init.normal_(
            self.board_encoder.positional_embeddings.weight,
            mean=0.0,
            std=math.pow(self.d_model, -0.5),
        )

    def forward(self, batch):
        """
        Forward prop.

        Args:

            batch (dict): A single batch, containing the following keys:

                turns (torch.LongTensor): The current turn (w/b), of
                size (N, 1).

                white_kingside_castling_rights (torch.LongTensor):
                Whether white can castle kingside, of size (N, 1).

                white_queenside_castling_rights (torch.LongTensor):
                Whether white can castle queenside, of size (N, 1).

                black_kingside_castling_rights (torch.LongTensor):
                Whether black can castle kingside, of size (N, 1).

                black_queenside_castling_rights (torch.LongTensor):
                Whether black can castle queenside, of size (N, 1).

                board_positions (torch.LongTensor): The current board
                positions, of size (N, 64).

        Returns:

            torch.FloatTensor: The next-move logits, of size (N, 1,
            vocab_size).
        """
        # Encoder
        boards = self.board_encoder(
            batch["board_positions"],
            batch["kingside_castling_rights"],
            batch["queenside_castling_rights"],
            batch["opponent_castling_rights_kingside"],
            batch["opponent_castling_rights_queenside"],
            batch["agent_ids"],
            batch["agents_pos"],
            batch["reward_weights"]
            
        )  # (N, BOARD_STATUS_LENGTH, d_model)

        # # Find logits over vocabulary at the "turn" token
        # moves = self.fc(boards[:, :1, :])  # (N, 1, vocab_size)

        # Find logits over vocabulary at the "agent_id" token:
        agents_move_preds = (
            self.policy_head(boards[:, 68:, :]) #.squeeze(2).unsqueeze(1)
        )  # (N, 16, 1971)
        # agents_reward_preds = (
        #     self.value_head(boards[:, 68:, :]).squeeze(2).unsqueeze(1)
        # )  # (N, 1, 64)

        return agents_move_preds #, agents_reward_preds



############################## MATChess Transformer (encoder only) ##############################
def huber_loss(e, d):
    a = (abs(e) <= d).float()
    b = (e > d).float()
    return a*e**2/2 + b*d*(abs(e)-d/2)


class MATChessTransformerEncoder(nn.Module):
    """
    The MATChess Transformer (Encoder only) for decision-making for a team of heterogeneous chess piece agents.

    Adapted from the 
    """

    def __init__(
        self,
        CONFIG,
    ):
        """
        Init.

        Args:

            CONFIG (dict): The configuration, containing the following
            parameters for the model:

                VOCAB_SIZES (dict): The sizes of the vocabularies of the
                Encoder sequence components.

                D_MODEL (int): The size of vectors throughout the
                transformer model, i.e. input and output sizes for the
                Encoder.

                N_HEADS (int): The number of heads in the multi-head
                attention.

                D_QUERIES (int): The size of query vectors (and also the
                size of the key vectors) in the multi-head attention.

                D_VALUES (int): The size of value vectors in the
                multi-head attention.

                D_INNER (int): An intermediate size in the position-wise
                FC.

                N_LAYERS (int): The number of [multi-head attention +
                multi-head attention + position-wise FC] layers in the
                Encoder.

                DROPOUT (int): The dropout probability.
        """
        super(MATChessTransformerEncoder, self).__init__()

        self.code = "E"

        self.vocab_sizes = CONFIG['VOCAB_SIZES']
        self.d_model = CONFIG['D_MODEL']
        self.n_heads = CONFIG['N_HEADS']
        self.d_queries = CONFIG['D_QUERIES']
        self.d_values = CONFIG['D_VALUES']
        self.d_inner = CONFIG['D_INNER']
        self.n_layers = CONFIG['N_LAYERS']
        self.dropout = CONFIG['DROPOUT']

        # Encoder
        self.board_encoder = BoardEncoder(
            vocab_sizes=self.vocab_sizes,
            d_model=self.d_model,
            n_heads=self.n_heads,
            d_queries=self.d_queries,
            d_values=self.d_values,
            d_inner=self.d_inner,
            n_layers=self.n_layers,
            dropout=self.dropout,
        )

        # Output linear layer that will compute logits for the corresponding vocabularies:
        self.policy_head = nn.Linear(self.d_model, self.vocab_sizes["moves"])
        self.value_head = nn.Linear(self.d_model, self.vocab_sizes["agents_rewards"])

        # Initialize weights
        self.init_weights()

    def init_weights(self):
        """
        Initialize weights in the transformer model.
        """
        # Glorot uniform initialization with a gain of 1.
        for p in self.parameters():
            # Glorot initialization needs at least two dimensions on the
            # tensor
            if p.dim() > 1:
                nn.init.xavier_uniform_(p, gain=1.0)

        # For the embeddings, normal initialization with 0 mean and
        # 1/sqrt(d_model) S.D.
        nn.init.normal_(
            self.board_encoder.board_position_embeddings.weight,
            mean=0.0,
            std=math.pow(self.d_model, -0.5),
        )
        nn.init.normal_(
            self.board_encoder.kingside_castling_rights_embeddings.weight,
            mean=0.0,
            std=math.pow(self.d_model, -0.5),
        )
        nn.init.normal_(
            self.board_encoder.queenside_castling_rights_embeddings.weight,
            mean=0.0,
            std=math.pow(self.d_model, -0.5),
        )
        nn.init.normal_(
            self.board_encoder.opponent_kingside_castling_rights_embeddings.weight,
            mean=0.0,
            std=math.pow(self.d_model, -0.5),
        )
        nn.init.normal_(
            self.board_encoder.opponent_queenside_castling_rights_embeddings.weight,
            mean=0.0,
            std=math.pow(self.d_model, -0.5),
        )
        nn.init.normal_(
            self.board_encoder.agent_ids_embeddings.weight,
            mean=0.0,
            std=math.pow(self.d_model, -0.5),
        )
        nn.init.normal_(
            self.board_encoder.agents_pos_embeddings.weight,
            mean=0.0,
            std=math.pow(self.d_model, -0.5),
        )
        nn.init.normal_(
            self.board_encoder.per_agent_state_embeddings.weight,
            mean=0.0,
            std=math.pow(self.d_model, -0.5),
        )
        nn.init.normal_(
            self.board_encoder.positional_embeddings.weight,
            mean=0.0,
            std=math.pow(self.d_model, -0.5),
        )

    def forward(self, batch):
        """
        Forward prop.

        Args:

            batch (dict): A single batch, containing the following keys:

                turns (torch.LongTensor): The current turn (w/b), of
                size (N, 1).

                white_kingside_castling_rights (torch.LongTensor):
                Whether white can castle kingside, of size (N, 1).

                white_queenside_castling_rights (torch.LongTensor):
                Whether white can castle queenside, of size (N, 1).

                black_kingside_castling_rights (torch.LongTensor):
                Whether black can castle kingside, of size (N, 1).

                black_queenside_castling_rights (torch.LongTensor):
                Whether black can castle queenside, of size (N, 1).

                board_positions (torch.LongTensor): The current board
                positions, of size (N, 64).

        Returns:

            torch.FloatTensor: The next-move logits, of size (N, 1,
            vocab_size).
        """
        # Encoder
        boards = self.board_encoder(
            batch["board_positions"],
            batch["kingside_castling_rights"],
            batch["queenside_castling_rights"],
            batch["opponent_castling_rights_kingside"],
            batch["opponent_castling_rights_queenside"],
            batch["agent_ids"],
            batch["agents_pos"],
            batch["reward_weights"]
            
        )  # (N, BOARD_STATUS_LENGTH, d_model)

        # # Find logits over vocabulary at the "turn" token
        # moves = self.fc(boards[:, :1, :])  # (N, 1, vocab_size)

        # Find logits over the move vocabulary at the 16 tokens representing the state of each agent:
        agents_move_preds = (
            self.policy_head(boards[:, 68:, :])
        )  # (N, 16, 1971)

        # Predict the eleven rewards for each of the 16 agents on the team at the corresponding per-agent state input tokens:
        agents_reward_preds = (
            self.value_head(boards[:, 68:, :])
        )  # (N, 16, 11)

        return agents_move_preds, agents_reward_preds




if __name__ == "__main__":
    # # Get configuration
    # parser = argparse.ArgumentParser()
    # parser.add_argument("config_name", type=str, help="Name of configuration file.", default="CT-E-20")
    # args = parser.parse_args()
    # CONFIG = import_config(args.config_name)
    CONFIG = import_config()
    

    # Model
    model = ChessTransformerEncoder(CONFIG).to(DEVICE)
    print(
        "There are %d learnable parameters in this model."
        % sum([p.numel() for p in model.parameters()])
    )