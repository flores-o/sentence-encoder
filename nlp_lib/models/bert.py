import math
from collections import OrderedDict
import torch
from torch import nn

# Activating function


def gelu(x):
    """
    Compute the Gaussian Error Linear Unit (GELU) activation function.
    :param x: Input tensor.
    :return: Tensor after applying the GELU function.
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class Config(object):
    def __init__(self,
                 vocab_size,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02):

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = dropout_prob
        self.attention_probs_dropout_prob = dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range

    @classmethod
    def from_dict(cls, dict_object):
        """
        Creates a Config object from a dictionary.

        :param dict_object: Dictionary containing configuration parameters.
        :return: Config object with set parameters.
        """
        config = Config(vocab_size=None)
        for (key, value) in dict_object.items():
            config.__dict__[key] = value
        return config


class LayerNorm(nn.Module):
    """
    Layer normalization module.

    Normalizes input tensor along its last dimension.
    """

    def __init__(self, hidden_size, variance_epsilon=1e-12):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = variance_epsilon

    # Bug Fix: The normalization was being applied across the wrong dimension.
    # It should be applied across the hidden dimension.

    # def forward(self, x):
    #   u = x.mean(0, keepdim=True)
    #   s = (x + u).pow(2).mean(0, keepdim=True)
    #   x = (x + u) / torch.sqrt(s + self.variance_epsilon)
    #   return self.gamma * x + self.beta

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta


class MLP(nn.Module):
    """
    Multi-Layer Perceptron module.

    Contains two dense layers.
    """

    def __init__(self, hidden_size, intermediate_size):
        super(MLP, self).__init__()
        self.dense_expansion = nn.Linear(hidden_size, intermediate_size)
        self.dense_contraction = nn.Linear(intermediate_size, hidden_size)

    def forward(self, x):
        x = self.dense_expansion(x)
        x = self.dense_contraction(gelu(x))
        return x


class Layer(nn.Module):
    """
    Core transformer layer used in BERT.

    Contains multi-head self-attention mechanism followed by a feed-forward network.
    """

    def __init__(self, config):
        super(Layer, self).__init__()

        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(
            config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

        self.attn_out = nn.Linear(config.hidden_size, config.hidden_size)
        self.ln1 = LayerNorm(config.hidden_size)

        self.mlp = MLP(config.hidden_size, config.intermediate_size)
        self.ln2 = LayerNorm(config.hidden_size)

    # Bug Fix: The split_heads method was reshaping the tensor incorrectly.

    # def split_heads(self, tensor, num_heads, attention_head_size):
    #     new_shape = tensor.size()[:-1] + (num_heads, attention_head_size)
    #     tensor = tensor.view(*new_shape)
    #     return tensor.permute(0, 2, 1, 3)

    def split_heads(self, tensor, num_heads, attention_head_size):
        """
        Reshapes the input tensor to split it into multiple heads.
        """
        batch_size = tensor.size(0)
        tensor = tensor.view(batch_size, -1, num_heads, attention_head_size)
        return tensor.permute(0, 2, 1, 3)

    def merge_heads(self, tensor, num_heads, attention_head_size):
        """
        Reshapes the input tensor to merge multiple heads.
        """
        tensor = tensor.permute(0, 2, 1, 3).contiguous()
        new_shape = tensor.size()[:-2] + (num_heads * attention_head_size,)
        return tensor.view(new_shape)

    def attn(self, q, k, v, attention_mask):
        """
        Compute the attention weights and apply them to values.
        """
        mask = attention_mask == 1
        mask = mask.unsqueeze(1).unsqueeze(2)

        # Bug Fix Incorrect matrix multiplication for attention scores
        # s = torch.matmul(q, k)
        s = torch.matmul(q, k.transpose(-1, -2))

        s = s / math.sqrt(self.attention_head_size)

        # Bug Fix Negative infinity used for masked values.
        # s = torch.where(mask, s, torch.tensor(float('inf')))
        s = torch.where(mask, s, torch.tensor(-float('inf')))

        # Bug Fix: Softmax applied on the correct dimension.
        # p = s
        p = torch.nn.functional.softmax(s, dim=-1)

        p = self.dropout(p)

        a = torch.matmul(p, v)
        return a

    def forward(self, x, attention_mask):
        q, k, v = self.query(x), self.key(x), self.value(x)

        q = self.split_heads(q, self.num_attention_heads,
                             self.attention_head_size)
        k = self.split_heads(k, self.num_attention_heads,
                             self.attention_head_size)
        v = self.split_heads(v, self.num_attention_heads,
                             self.attention_head_size)

        a = self.attn(q, k, v, attention_mask)
        a = self.merge_heads(a, self.num_attention_heads,
                             self.attention_head_size)
        a = self.attn_out(a)
        a = self.dropout(a)
        # Bug Fix: Residual connections
        # a = self.ln1(a)
        a = self.ln1(a + x)

        m = self.mlp(a)
        m = self.dropout(m)
        # Bug Fix: Residual connections
        # m = self.ln2(m)
        m = self.ln2(m + a)

        return m


class Bert(nn.Module):
    """
    BERT model class.

    Contains token, position, and token type embeddings, multiple transformer layers, and a pooler.
    """

    def __init__(self, config_dict):
        print("This is the custom Bert model being used")
        super(Bert, self).__init__()
        self.config = Config.from_dict(config_dict)
        self.embeddings = nn.ModuleDict({
            'token': nn.Embedding(self.config.vocab_size, self.config.hidden_size, padding_idx=0),
            'position': nn.Embedding(self.config.max_position_embeddings, self.config.hidden_size),
            'token_type': nn.Embedding(self.config.type_vocab_size, self.config.hidden_size),
        })

        self.ln = LayerNorm(self.config.hidden_size)
        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)

        self.layers = nn.ModuleList([
            Layer(self.config) for _ in range(self.config.num_hidden_layers)
        ])

        self.pooler = nn.Sequential(OrderedDict([
            ('dense', nn.Linear(self.config.hidden_size, self.config.hidden_size)),
            ('activation', nn.Tanh()),
        ]))

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, ):
        """
        Computes the forward pass of the BERT model.

        Returns the sequence of hidden states and the pooled output.
        """
        position_ids = torch.arange(input_ids.size(
            1), dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # Bug Fix: The embeddings were being concatenated instead of added.
        # x = torch.cat((self.embeddings.token(input_ids),
        #                self.embeddings.position(position_ids),
        #                self.embeddings.token_type(token_type_ids)),
        #               dim=-1)
        x = self.embeddings.token(input_ids) + self.embeddings.position(
            position_ids) + self.embeddings.token_type(token_type_ids)

        # Bug Fix?/ Design Choice: Applying layer norm before dropout.
        # x = self.dropout(self.ln(x))
        x = self.ln(self.dropout(x))

        for layer in self.layers:
            x = layer(x, attention_mask)

        o = self.pooler(x[:, 0])
        return (x, o)

    def load_model(self, path):
        """
        Loads a pre-trained model from a given path.

        :param path: Path to the saved model.
        :return: Model with loaded weights.
        """
        saved_params = torch.load(path)
        self.load_state_dict(saved_params)
        return self
