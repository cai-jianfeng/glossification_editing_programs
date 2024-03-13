# _*_ coding:utf-8 _*_
"""
@Project Name: glossification
@FileName: transformer.py
@Begin Date: 2023/11/2 16:31
@Author: caijianfeng
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import utils

def initialize_weight(x):
    nn.init.xavier_uniform_(x.weight)
    if x.bias is not None:
        nn.init.constant_(x.bias, 0)


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, inner_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, inner_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.layer2 = nn.Linear(inner_size, hidden_size)

        initialize_weight(self.layer1)
        initialize_weight(self.layer2)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, dropout_rate, head_num=8):
        super(MultiHeadAttention, self).__init__()

        self.head_size = head_num

        self.att_size = att_size = hidden_size // head_num
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, head_num * att_size, bias=False)
        self.linear_k = nn.Linear(hidden_size, head_num * att_size, bias=False)
        self.linear_v = nn.Linear(hidden_size, head_num * att_size, bias=False)
        initialize_weight(self.linear_q)
        initialize_weight(self.linear_k)
        initialize_weight(self.linear_v)

        self.att_dropout = nn.Dropout(dropout_rate)

        self.output_layer = nn.Linear(head_num * att_size,
                                      hidden_size, bias=False)
        initialize_weight(self.output_layer)

    def forward(self, q, k, v, mask=None, cache=None):
        """
        multi-head attention
        :param q: Tensor(b, q_len, d_model), dtype = float
        :param k: Tensor(b, k_len, d_model), dtype = float
        :param v: Tensor(b, v_len, d_model), dtype = float
        :param mask: Tensor(q_len, k_len), dtype = bool
        :param cache: dict
        :return: Tensor(b, q_len, d_model)
        """
        orig_q_size = q.size()  # [b, q_len, d_model]

        d_k = self.att_size
        d_v = self.att_size
        batch_size = q.size(0)

        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        q = self.linear_q(q).view(batch_size, -1, self.head_size, d_k)  # (b, q_len, h, d_k)
        if cache is not None and 'encdec_k' in cache:
            k, v = cache['encdec_k'], cache['encdec_v']
        else:
            k = self.linear_k(k).view(batch_size, -1, self.head_size, d_k)  # (b, k_len, h, d_k)
            v = self.linear_v(v).view(batch_size, -1, self.head_size, d_v)  # (b, v_len, h, d_v)

            if cache is not None:
                cache['encdec_k'], cache['encdec_v'] = k, v

        q = q.transpose(1, 2)                  # [b, h, q_len, d_k]
        v = v.transpose(1, 2)                  # [b, h, v_len, d_v]
        k = k.transpose(1, 2).transpose(2, 3)  # [b, h, k_len, d_k] -> [b, h, d_k, k_len]

        # Scaled Dot-Product Attention
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q.mul_(self.scale)
        x = torch.matmul(q, k)  # [b, h, q_len, k_len]
        x.masked_fill_(mask.unsqueeze(1), -1e9)
        x = torch.softmax(x, dim=3)  # softmax in k_len
        x = self.att_dropout(x)
        x = x.matmul(v)  # [b, h, q_len, k_len] * [b, h, v_len, d_v] -> [b, h, q_len, attn(d_v)]  (k_len = v_len)

        x = x.transpose(1, 2).contiguous()  # [b, q_len, h, attn(d_v)]
        x = x.view(batch_size, -1, self.head_size * d_v)  # [b, q_len, h * attn]

        x = self.output_layer(x)  # [b, q_len, d_model]

        assert x.size() == orig_q_size
        return x  # [b, q_len, d_model]


class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, inner_size, dropout_rate, head_num):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.self_attention = MultiHeadAttention(hidden_size, dropout_rate, head_num)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.ffn = FeedForwardNetwork(hidden_size, inner_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, mask):
        # multi-head attention
        y = self.self_attention_norm(x)
        y = self.self_attention(y, y, y, mask)
        y = self.self_attention_dropout(y)
        x = x + y

        # feed forward network
        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y

        return x


class DecoderLayer(nn.Module):
    def __init__(self, hidden_size, inner_size, dropout_rate, head_num):
        super(DecoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.self_attention = MultiHeadAttention(hidden_size, dropout_rate, head_num)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.enc_dec_attention_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.enc_dec_attention = MultiHeadAttention(hidden_size, dropout_rate, head_num)
        self.enc_dec_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size, eps=1e-6)
        self.ffn = FeedForwardNetwork(hidden_size, inner_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, enc_output, self_mask, i_mask, cache):
        # masked multi-head attention
        y = self.self_attention_norm(x)
        y = self.self_attention(y, y, y, self_mask)
        y = self.self_attention_dropout(y)
        x = x + y

        # cross multi-head attention
        if enc_output is not None:
            y = self.enc_dec_attention_norm(x)
            y = self.enc_dec_attention(y, enc_output, enc_output, i_mask,
                                       cache)
            y = self.enc_dec_attention_dropout(y)
            x = x + y

        # feed forward network
        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y

        return x


class Encoder(nn.Module):
    def __init__(self, hidden_size, inner_size, dropout_rate, n_layers, head_num):
        super(Encoder, self).__init__()

        encoders = [EncoderLayer(hidden_size, inner_size, dropout_rate, head_num)
                    for _ in range(n_layers)]
        self.layers = nn.ModuleList(encoders)

        self.last_norm = nn.LayerNorm(hidden_size, eps=1e-6)

    def forward(self, inputs, mask):
        encoder_output = inputs
        for enc_layer in self.layers:
            encoder_output = enc_layer(encoder_output, mask)
        return self.last_norm(encoder_output)
        # return encoder_output


class Decoder(nn.Module):
    def __init__(self, hidden_size, inner_size, dropout_rate, n_layers, head_num):
        super(Decoder, self).__init__()

        decoders = [DecoderLayer(hidden_size, inner_size, dropout_rate, head_num)
                    for _ in range(n_layers)]
        self.layers = nn.ModuleList(decoders)

        self.last_norm = nn.LayerNorm(hidden_size, eps=1e-6)

    def forward(self, targets, enc_output, i_mask, t_self_mask, cache=None):

        decoder_output = targets
        for i, dec_layer in enumerate(self.layers):
            layer_cache = None
            if cache is not None:
                if i not in cache:
                    cache[i] = {}
                layer_cache = cache[i]
            decoder_output = dec_layer(decoder_output, enc_output,
                                       t_self_mask, i_mask, layer_cache)
        return self.last_norm(decoder_output)
        # return decoder_output


class Transformer(nn.Module):
    def __init__(self, i_vocab_size, t_vocab_size,
                 n_layers=6,
                 head_num=10,
                 hidden_size=512,
                 inner_size=2048,
                 dropout_rate=0.1,
                 share_target_embedding=True,
                 has_inputs=True,
                 src_pad_idx=None,
                 trg_pad_idx=None):
        """
        :param i_vocab_size: int, the vocabulary size of the inputs
        :param t_vocab_size: int, the vocabulary size of the outputs (in this project, i_vocab_size = t_vocab_size)
        :param n_layers: int, the layer number of the transformer encoder and decoder
        :param head_num: int, the head attention number of the transformer encoder and decoder
        :param hidden_size: int, d_model in the transformer
        :param inner_size: int, inner-layer dimensionality in the feed forward network
        :param dropout_rate: int, the dropout rate in the transformer
        :param share_target_embedding: bool, whether the inputs and outputs embedding is the same
        :param has_inputs: bool, whether has the input
        :param src_pad_idx: int, the index of '<pad>' in the input vocabulary
        :param trg_pad_idx: int, the index of '<pad>' in the output vocabulary (in this project, src_pad_idx = trg_pad_idx)
        """
        super(Transformer, self).__init__()

        self.hidden_size = hidden_size
        self.emb_scale = hidden_size ** 0.5
        self.has_inputs = has_inputs
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx

        # TODO: 将 t_vocab_embedding 转化为 Fasttext.zip 中已经训练好的 embedding
        self.t_vocab_embedding = nn.Embedding(t_vocab_size, hidden_size)
        nn.init.normal_(self.t_vocab_embedding.weight, mean=0,
                        std=hidden_size**-0.5)
        self.t_emb_dropout = nn.Dropout(dropout_rate)
        self.decoder = Decoder(hidden_size, inner_size,
                               dropout_rate, n_layers, head_num)

        # TODO: 将 i_vocab_embedding 转化为 Fasttext.zip 中已经训练好的 embedding
        if has_inputs:
            if not share_target_embedding:
                self.i_vocab_embedding = nn.Embedding(i_vocab_size,
                                                      hidden_size)
                nn.init.normal_(self.i_vocab_embedding.weight, mean=0,
                                std=hidden_size**-0.5)
            else:
                self.i_vocab_embedding = self.t_vocab_embedding

            self.i_emb_dropout = nn.Dropout(dropout_rate)

            self.encoder = Encoder(hidden_size, inner_size,
                                   dropout_rate, n_layers, head_num)

        # For positional encoding
        num_timescales = self.hidden_size // 2
        max_timescale = 10000.0
        min_timescale = 1.0
        log_timescale_increment = (
            math.log(float(max_timescale) / float(min_timescale)) /
            max(num_timescales - 1, 1))
        inv_timescales = min_timescale * torch.exp(
            torch.arange(num_timescales, dtype=torch.float32) *
            -log_timescale_increment)
        self.register_buffer('inv_timescales', inv_timescales)

    def forward(self, inputs, targets):
        # inputs.shape = [b, i_len]
        # targets.shape = [b, t_len]
        enc_output, i_mask = None, None
        if self.has_inputs:
            i_mask = utils.create_pad_mask(inputs, self.src_pad_idx)  # [b, 1, i_len]
            # print('i_mask shape: ', i_mask.shape)
            enc_output = self.encode(inputs, i_mask)

        t_mask = utils.create_pad_mask(targets, self.trg_pad_idx)  # [b, 1, t_len]
        # print('t_mask shape: ', t_mask.shape)
        target_size = targets.size()[1]  # t_len
        t_self_mask = utils.create_trg_self_mask(target_size,
                                                 device=targets.device)  # [1, t_len, t_len]
        # print('t_self_mask shape: ', t_self_mask.shape)
        return self.decode_origin(targets, enc_output, i_mask, t_self_mask, t_mask)

    def encode(self, inputs, i_mask):
        # Input embedding
        input_embedded = self.i_vocab_embedding(inputs)  # [b, i_len, d_model]
        # i_mask.squeeze(1).unsqueeze(-1) -> [b, i_len, 1]
        input_embedded.masked_fill_(i_mask.squeeze(1).unsqueeze(-1), 0)  # [b, i_len, d_model]
        input_embedded *= self.emb_scale  # [b, i_len, d_model]
        input_embedded += self.get_position_encoding(inputs)  # [b, i_len, d_model]
        input_embedded = self.i_emb_dropout(input_embedded)  # [b, i_len, d_model]

        return self.encoder(input_embedded, i_mask)

    def decode(self, targets, enc_output, i_mask, t_self_mask, t_mask,
               cache=None):
        # target embedding
        target_embedded = self.t_vocab_embedding(targets)  # [b, t_len, d_model]
        target_embedded.masked_fill_(t_mask.squeeze(1).unsqueeze(-1), 0)  # [b, t_len, d_model]

        # Shifting
        # 将 target_embedded 整体右移一个，最右边多出来的舍弃，最左边空出来的 pad 0
        target_embedded = target_embedded[:, :-1]  # [b, t_len-1, d_model]
        target_embedded = F.pad(target_embedded, (0, 0, 1, 0))  # [b, t_len, d_model]

        target_embedded *= self.emb_scale
        target_embedded += self.get_position_encoding(targets)
        target_embedded = self.t_emb_dropout(target_embedded)

        # decoder
        decoder_output = self.decoder(target_embedded, enc_output, i_mask,
                                      t_self_mask, cache)  # [b, t_len, d_model]

        return decoder_output

    def decode_origin(self, targets, enc_output, i_mask, t_self_mask, t_mask,
                      cache=None):
        decoder_output = self.decode(targets, enc_output, i_mask, t_self_mask, t_mask, cache)
        # linear
        # [b, t_len, d_model] * [d_model, t_vocab_size]
        output = torch.matmul(decoder_output,
                              self.t_vocab_embedding.weight.transpose(0, 1))  # [b, t_len, t_vocab_size]
        return output

    def get_position_encoding(self, x):
        max_length = x.size()[1]  # i_len
        position = torch.arange(max_length, dtype=torch.float32,
                                device=x.device)  # [i_len,]
        scaled_time = position.unsqueeze(1) * self.inv_timescales.unsqueeze(0)  # [i_len, d_model // 2]
        signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)],
                           dim=1)  # [i_len, d_model]
        # F.pad(·) 的第二个参数表示每个维度的 pad 数(一个维度占 2 个位置)
        # 例如：F.pad(signal, (1, 1, 2, 2)) 表示将 signal 的最后一维左右各 pad 1 个；倒数第二维左右各 pad 2 个
        signal = F.pad(signal, (0, self.hidden_size % 2, 0, 0))
        signal = signal.view(1, max_length, self.hidden_size)  # [1, i_len, d_model]
        return signal


if __name__ == '__main__':
    i_vocab_size = 100
    t_vocab_size = 200
    src_pad_idx = 99
    trg_pad_idx = 199
    model = Transformer(i_vocab_size=i_vocab_size,
                        t_vocab_size=t_vocab_size,
                        src_pad_idx=src_pad_idx,
                        trg_pad_idx=trg_pad_idx)
    i_len, t_len = 10, 20
    batch_size = 5
    inputs = torch.randint(0, i_vocab_size, [batch_size, i_len])
    target = torch.randint(0, t_vocab_size, [batch_size, t_len])
    output = model(inputs, target)
    # print(output.shape)  # [batch_size, t_len, t_vocab_size]