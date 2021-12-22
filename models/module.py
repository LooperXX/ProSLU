import math
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.utils.rnn import pad_packed_sequence
from transformers import BertConfig, BertModel
from transformers import XLNetConfig, XLNetModel
from transformers import ElectraModel, ElectraConfig
from transformers import AlbertConfig, AlbertModel
from utils.config import *


class BertEncoder(nn.Module):
    def __init__(self, dropout_rate):
        super(BertEncoder, self).__init__()
        model_config = BertConfig.from_pretrained(args.model_type_path)
        self.model = BertModel.from_pretrained(args.model_type_path, config=model_config)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input):
        outputs = self.model(input, attention_mask=(input != 0).float())
        hidden = self.dropout(outputs.last_hidden_state)
        return hidden


class XLNetEncoder(nn.Module):
    def __init__(self, dropout_rate):
        super(XLNetEncoder, self).__init__()
        model_config = XLNetConfig.from_pretrained(args.model_type_path)
        self.model = XLNetModel.from_pretrained(args.model_type_path, config=model_config)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input):
        outputs = self.model(input, attention_mask=(input != 0).float())
        hidden = self.dropout(outputs.last_hidden_state)
        return hidden


class ElectraEncoder(nn.Module):
    def __init__(self, dropout_rate):
        super(ElectraEncoder, self).__init__()
        model_config = ElectraConfig.from_pretrained(args.model_type_path)
        self.model = ElectraModel.from_pretrained(args.model_type_path, config=model_config)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input):
        outputs = self.model(input, attention_mask=(input != 0).float())
        hidden = self.dropout(outputs.last_hidden_state)
        return hidden


class AlbertEncoder(nn.Module):
    def __init__(self, dropout_rate):
        super(AlbertEncoder, self).__init__()
        model_config = AlbertConfig.from_pretrained(args.model_type_path)
        self.model = AlbertModel.from_pretrained(args.model_type_path, config=model_config)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, input):
        outputs = self.model(input, attention_mask=(input != 0).float())
        hidden = self.dropout(outputs.last_hidden_state)
        return hidden


class LSTMEncoder(nn.Module):
    """
    Encoder structure based on bidirectional LSTM.
    """

    def __init__(self, embedding_dim, hidden_dim, dropout_rate):
        super(LSTMEncoder, self).__init__()

        # Parameter recording.
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim // 2
        self.dropout_rate = dropout_rate

        # Network attributes.
        self.dropout_layer = nn.Dropout(self.dropout_rate)
        self.lstm_layer = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_dim,
            batch_first=True,
            bidirectional=True,
            dropout=self.dropout_rate,
            num_layers=1
        )

    def forward(self, embedded_text, seq_lens, enforce_sorted=True):
        """ Forward process for LSTM Encoder.

        (batch_size, max_sent_len)
        -> (batch_size, max_sent_len, word_dim)
        -> (batch_size, max_sent_len, hidden_dim)
        -> (total_word_num, hidden_dim)

        :param embedded_text: padded and embedded input text.
        :param seq_lens: is the length of original input text.
        :return: is encoded word hidden vectors.
        """

        # Padded_text should be instance of LongTensor.
        dropout_text = self.dropout_layer(embedded_text)

        # Pack and Pad process for input of variable length.
        packed_text = pack_padded_sequence(
            dropout_text, seq_lens, batch_first=True, enforce_sorted=enforce_sorted)
        lstm_hiddens, (h_last, c_last) = self.lstm_layer(packed_text)
        padded_hiddens, _ = pad_packed_sequence(lstm_hiddens, batch_first=True)
        return padded_hiddens


class LSTMDecoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate, embedding_dim=None, extra_dim=None,
                 info_dim=None, args=None):
        super(LSTMDecoder, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.embedding_dim = embedding_dim
        self.extra_dim = extra_dim
        self.args = args

        # If embedding_dim is not None, the output and input
        # of this structure is relevant.
        if self.embedding_dim is not None:
            self.embedding_layer = nn.Embedding(output_dim, embedding_dim)
            self.init_tensor = nn.Parameter(
                torch.randn(1, self.embedding_dim),
                requires_grad=True
            )

        # Make sure the input dimension of iterative LSTM.
        lstm_input_dim = self.input_dim
        if self.extra_dim is not None:
            lstm_input_dim += self.extra_dim
        if self.embedding_dim is not None:
            lstm_input_dim += self.embedding_dim
        if info_dim != 0:
            lstm_input_dim += info_dim

        # Network parameter definition.
        self.dropout_layer = nn.Dropout(self.dropout_rate)
        self.lstm_layer = nn.LSTM(
            input_size=lstm_input_dim,
            hidden_size=self.hidden_dim,
            batch_first=True,
            bidirectional=False,
            dropout=self.dropout_rate,
            num_layers=1
        )

        self.linear_layer = nn.Linear(
            self.hidden_dim,
            self.output_dim
        )

    def _cuda(self, x):
        if args.gpu:
            return x.cuda()
        else:
            return x

    def forward(self, encoded_hiddens, seq_lens, extra_input=None, forced_input=None, info_emb=None):
        # Concatenate information tensor if possible.
        if info_emb is not None:
            encoded_hiddens = torch.cat([encoded_hiddens, info_emb], dim=-1)
        if extra_input is not None:
            input_tensor = torch.cat([encoded_hiddens, extra_input], dim=-1)
        else:
            input_tensor = encoded_hiddens
        output_tensor_list = []
        if self.embedding_dim is not None and forced_input is not None:
            forced_tensor = self.embedding_layer(forced_input)[:, :-1]
            prev_tensor = torch.cat((self.init_tensor.unsqueeze(0).repeat(len(forced_tensor), 1, 1),
                                     forced_tensor), dim=1)
            combined_input = torch.cat([input_tensor, prev_tensor], dim=2)
            dropout_input = self.dropout_layer(combined_input)
            packed_input = pack_padded_sequence(dropout_input, seq_lens, batch_first=True)
            lstm_out, _ = self.lstm_layer(packed_input)
            lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
            # flatten output
            for sent_i in range(0, len(seq_lens)):
                lstm_out_i = lstm_out[sent_i][:seq_lens[sent_i]]
                linear_out = self.linear_layer(lstm_out_i)
                output_tensor_list.append(linear_out)
        else:
            prev_tensor = self.init_tensor.unsqueeze(0).repeat(len(seq_lens), 1, 1)
            last_h, last_c = None, None
            for word_i in range(seq_lens[0]):
                combined_input = torch.cat((input_tensor[:, word_i].unsqueeze(1), prev_tensor), dim=2)
                dropout_input = self.dropout_layer(combined_input)
                if last_h is None and last_c is None:
                    lstm_out, (last_h, last_c) = self.lstm_layer(dropout_input)
                else:
                    lstm_out, (last_h, last_c) = self.lstm_layer(dropout_input, (last_h, last_c))
                lstm_out = self.linear_layer(lstm_out.squeeze(1))
                output_tensor_list.append(lstm_out)

                _, index = lstm_out.topk(1, dim=1)
                prev_tensor = self.embedding_layer(index.squeeze(1)).unsqueeze(1)
            # flatten output
            output_tensor = torch.stack(output_tensor_list)
            output_tensor_list = [output_tensor[:length, i] for i, length in enumerate(seq_lens)]
        return torch.cat(output_tensor_list, dim=0)


class QKVAttention(nn.Module):
    """
    Attention mechanism based on Query-Key-Value architecture. And
    especially, when query == key == value, it's self-attention.
    """

    def __init__(self, query_dim, key_dim, value_dim, hidden_dim, output_dim, dropout_rate):
        super(QKVAttention, self).__init__()

        # Record hyper-parameters.
        self.query_dim = query_dim
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate

        # Declare network structures.
        self.query_layer = nn.Linear(self.query_dim, self.hidden_dim)
        self.key_layer = nn.Linear(self.key_dim, self.hidden_dim)
        self.value_layer = nn.Linear(self.value_dim, self.output_dim)
        self.dropout_layer = nn.Dropout(p=self.dropout_rate)

    def forward(self, input_query, input_key, input_value, seq_lens=None):
        """ The forward propagation of attention.

        Here we require the first dimension of input key
        and value are equal.

        :param input_query: is query tensor, (n, d_q)
        :param input_key:  is key tensor, (m, d_k)
        :param input_value:  is value tensor, (m, d_v)
        :return: attention based tensor, (n, d_h)
        """

        # Linear transform to fine-tune dimension.
        linear_query = self.query_layer(input_query)
        linear_key = self.key_layer(input_key)
        linear_value = self.value_layer(input_value)

        score_tensor = torch.matmul(
            linear_query,
            linear_key.transpose(-2, -1)
        ) / math.sqrt(self.hidden_dim)

        if seq_lens is not None:
            max_len = max(seq_lens)
            for i, l in enumerate(seq_lens):
                if l < max_len:
                    score_tensor.data[i, l:] = -1e9

        score_tensor = F.softmax(score_tensor, dim=-1)

        forced_tensor = torch.matmul(score_tensor, linear_value)
        forced_tensor = self.dropout_layer(forced_tensor)

        return forced_tensor


class Hierarchical_Fusion(nn.Module):
    def __init__(self, query_dim, hidden_dim, dropout_rate):
        super(Hierarchical_Fusion, self).__init__()
        # Record parameters.
        self.query_dim = query_dim
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate

        # Record network parameters.
        self.dropout_layer = nn.Dropout(self.dropout_rate)
        self.linear_layer = nn.Linear(query_dim, hidden_dim)
        self.softmax_layer = nn.Softmax(dim=-1)

    def forward(self, query, info):
        dropout_query = self.dropout_layer(query)
        dropout_info = self.dropout_layer(info)
        scores = self.softmax_layer(torch.matmul(self.linear_layer(dropout_query), dropout_info.transpose(-2, -1)))
        scores_tensor = torch.matmul(scores, dropout_info)
        return scores_tensor


class SelfAttention(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate):
        super(SelfAttention, self).__init__()

        # Record parameters.
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate

        # Record network parameters.
        self.dropout_layer = nn.Dropout(self.dropout_rate)
        self.attention_layer = QKVAttention(
            self.input_dim, self.input_dim, self.input_dim,
            self.hidden_dim, self.output_dim, self.dropout_rate
        )

    def forward(self, input_x, seq_lens):
        dropout_x = self.dropout_layer(input_x)
        attention_x = self.attention_layer(
            dropout_x, dropout_x, dropout_x, seq_lens
        )
        return attention_x


class UnflatSelfAttention(nn.Module):
    """
    scores each element of the sequence with a linear layer and uses the normalized scores to compute a context over the sequence.
    """

    def __init__(self, d_hid, dropout=0.):
        super().__init__()
        self.scorer = nn.Linear(d_hid, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inp, lens):
        batch_size, seq_len, d_feat = inp.size()
        inp = self.dropout(inp)
        scores = self.scorer(inp.contiguous().view(-1, d_feat)).view(batch_size, seq_len)
        max_len = max(lens)
        for i, l in enumerate(lens):
            if l < max_len:
                scores.data[i, l:] = -1e9
        scores = F.softmax(scores, dim=1)
        context = scores.unsqueeze(2).expand_as(inp).mul(inp).sum(1)
        return context
