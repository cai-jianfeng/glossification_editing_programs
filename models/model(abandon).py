# _*_ coding:utf-8 _*_
"""
@Software: (已读)glossification
@FileName: model.py
@Date: 2023/11/1 19:25
@Author: caijianfeng
"""
import math

import torch
from torch import nn
import torch.nn.functional as F

import utils
from transformer import EncoderLayer, DecoderLayer, MultiHeadAttention

"""
the architecture of the generator and executor, both of them form the model glossificaiton
"""


class Generator(nn.Module):
    def __init__(self,
                 i_vocab_size,
                 p_vocab_size,
                 src_pad_idx,
                 trg_pad_idx,
                 head_num,
                 hidden_size=512,
                 inner_size=2048,
                 dropout_rate=0.1,
                 encoder_n_layers=3,
                 decoder_n_layers=1,
                 share_target_embeddings=True):
        super(Generator, self).__init__()

        self.hidden_size = hidden_size
        self.emb_scale = hidden_size ** 0.5
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx

        self.i_vocab_embedding = nn.Embedding(i_vocab_size, hidden_size)
        if not share_target_embeddings:
            self.p_vocab_embedding = nn.Embedding(p_vocab_size, hidden_size)
        else:
            self.p_vocab_embedding = self.i_vocab_embedding
        self.i_emb_dropout = nn.Dropout(dropout_rate)
        self.p_emb_dropout = nn.Dropout(dropout_rate)

        # Generator Encoder
        encoders = [EncoderLayer(hidden_size, inner_size, dropout_rate, head_num) for _ in range(encoder_n_layers)]
        self.encoder = nn.ModuleList(encoders)
        # Generator Decoder
        decoders = [DecoderLayer(hidden_size, inner_size, dropout_rate, head_num) for _ in range(decoder_n_layers)]
        self.decoder = nn.ModuleList(decoders)

        # position embedding
        num_timescales = self.hidden_size // 2
        max_timescale = 10000.0
        min_timescale = 1.0
        log_timescale_increment = (math.log(float(max_timescale) / float(min_timescale)) / max(num_timescales - 1, 1))
        inv_timescales = min_timescale * torch.exp(torch.arange(num_timescales, dtype=torch.float32) * -log_timescale_increment)
        self.register_buffer('inv_timescales', inv_timescales)

    def forward(self, inputs, programs):
        # inputs.shape = [b, i_len]
        # programs.shape = [b, p_len]
        # encoder
        i_mask = utils.create_pad_mask(inputs, self.src_pad_idx)  # [b, 1, i_len]
        enc_output = self.encode(inputs, i_mask)  # [b, i_len, d_model]
        # decoder
        p_mask = utils.create_pad_mask(programs, self.trg_pad_idx)  # [b, 1, p_len]
        program_size = programs.size()[1]  # p_len
        p_self_mask = utils.create_trg_self_mask(program_size, device=programs.device)  # [p_len, p_len]
        dec_output = self.decode(programs, enc_output, i_mask, p_self_mask, p_mask)  # [b, p_len, d_model]
        return dec_output  # [b, p_len, d_model]

    def encode(self, inputs, i_mask):
        input_embedded = self.i_vocab_embedding(inputs)  # [b, i_len, d_model]
        # i_mask.squeeze(1).unsqueeze(-1) -> [b, i_len, 1]
        input_embedded.masked_fill_(i_mask.squeeze(1).unsqueeze(-1), 0)  # [b, i_len, d_model]
        input_embedded *= self.emb_scale  # [b, i_len, d_model]
        input_embedded += self.get_position_encoding(inputs)  # [b, i_len, d_model]
        input_embedded = self.i_emb_dropout(input_embedded)  # [b, i_len, d_model]

        encoder_output = input_embedded  # [b, i_len, d_model]
        for enc_layer in self.encoder:
            encoder_output = enc_layer(encoder_output, i_mask)  # [b, i_len, d_model]

        return encoder_output  # [b, i_len, d_model]

    def decode(self, programs, enc_output, i_mask, t_self_mask, t_mask):
        # programs embedding
        program_embedded = self.p_vocab_embedding(programs)  # [b, p_len, d_model]
        program_embedded.masked_fill_(t_mask.squeeze(1).unsqueeze(-1), 0)  # [b, p_len, d_model]

        # Shifting
        # 将 program_embedded 整体右移一个，最右边多出来的舍弃，最左边空出来的 pad 0
        program_embedded = program_embedded[:, :-1]  # [b, p_len-1, d_model]
        program_embedded = F.pad(program_embedded, (0, 0, 1, 0))  # [b, p_len, d_model]

        program_embedded *= self.emb_scale  # [b, p_len, d_model]
        program_embedded += self.get_position_encoding(programs)  # [b, p_len, d_model]
        program_embedded = self.p_emb_dropout(program_embedded)  # [b, p_len, d_model]

        # decoder
        decoder_output = program_embedded  # [b, p_len, d_model]
        for dec_layer in self.decoder:
            decoder_output = dec_layer(decoder_output, enc_output, t_self_mask, i_mask)  # [b, p_len, d_model]
        # linear
        # [b, p_len, d_model] * [d_model, p_vocab_size]
        # output = torch.matmul(decoder_output, self.p_vocab_embedding.weight.transpose(0, 1))  # [b, p_len, p_vocab_size]

        return decoder_output  # [b, p_len, d_model]

    def get_position_encoding(self, x):
        max_length = x.size()[1]  # i_len
        position = torch.arange(max_length, dtype=torch.float32, device=x.device)  # [i_len,]
        scaled_time = position.unsqueeze(1) * self.inv_timescales.unsqueeze(0)  # [i_len, d_model // 2]
        signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)  # [i_len, d_model]

        # F.pad(·) 的第二个参数表示每个维度的 pad 数(一个维度占 2 个位置)
        # 例如：F.pad(signal, (1, 1, 2, 2)) 表示将 signal 的最后一维左右各 pad 1 个；倒数第二维左右各 pad 2 个
        signal = F.pad(signal, (0, self.hidden_size % 2, 0, 0))
        signal = signal.view(1, max_length, self.hidden_size)
        return signal

    def editing_causal_attention(self, inputs, mask):
        output = inputs * self.hidden_size
        return output


class Executor(nn.Module):
    def __init__(self,
                 t_vocab_size,
                 pro_pad_idx,
                 head_num,
                 hidden_size=512,
                 inner_size=2048,
                 dropout_rate=0.1,
                 encoder_n_layers=1,
                 ):
        super(Executor, self).__init__()
        self.hidden_size = hidden_size
        self.emb_scale = hidden_size ** 0.5
        self.pro_pad_idx = pro_pad_idx

        self.t_vocab_embedding = nn.Embedding(t_vocab_size, hidden_size)
        self.t_emb_dropout = nn.Dropout(dropout_rate)

        # Executor Encoder
        encoders = [EncoderLayer(hidden_size, inner_size, dropout_rate, head_num) for _ in range(encoder_n_layers)]
        self.encoder = nn.ModuleList(encoders)
        # position embedding
        num_timescales = self.hidden_size // 2
        max_timescale = 10000.0
        min_timescale = 1.0
        log_timescale_increment = (math.log(float(max_timescale) / float(min_timescale)) / max(num_timescales - 1, 1))
        inv_timescales = min_timescale * torch.exp(
            torch.arange(num_timescales, dtype=torch.float32) * -log_timescale_increment)
        self.register_buffer('inv_timescales', inv_timescales)

    def forward(self, inputs):
        # inputs.shape = [b, t_len]
        # encoder
        i_mask = utils.create_pad_mask(inputs, self.pro_pad_idx)  # [b, 1, t_len]
        enc_output = self.encode(inputs, i_mask)  # [b, t_len, d_model]
        return enc_output  # [b, t_len, d_model]

    def encode(self, inputs, i_mask):
        # inputs.shape = [b, i_len]
        input_embedded = self.i_vocab_embedding(inputs)  # [b, i_len, d_model]
        # i_mask.squeeze(1).unsqueeze(-1) -> [b, i_len, 1]
        input_embedded.masked_fill_(i_mask.squeeze(1).unsqueeze(-1), 0)  # [b, i_len, d_model]
        input_embedded *= self.emb_scale  # [b, i_len, d_model]
        input_embedded += self.get_position_encoding(inputs)  # [b, i_len, d_model]
        input_embedded = self.i_emb_dropout(input_embedded)  # [b, i_len, d_model]

        encoder_output = input_embedded  # [b, i_len, d_model]
        for enc_layer in self.encoder:
            encoder_output = enc_layer(encoder_output, i_mask)  # [b, i_len, d_model]

        return encoder_output  # [b, i_len, d_model]

    def get_position_encoding(self, x):
        max_length = x.size()[1]  # i_len
        position = torch.arange(max_length, dtype=torch.float32, device=x.device)  # [i_len,]
        scaled_time = position.unsqueeze(1) * self.inv_timescales.unsqueeze(0)  # [i_len, d_model // 2]
        signal = torch.cat([torch.sin(scaled_time), torch.cos(scaled_time)], dim=1)  # [i_len, d_model]

        # F.pad(·) 的第二个参数表示每个维度的 pad 数(一个维度占 2 个位置)
        # 例如：F.pad(signal, (1, 1, 2, 2)) 表示将 signal 的最后一维左右各 pad 1 个；倒数第二维左右各 pad 2 个
        signal = F.pad(signal, (0, self.hidden_size % 2, 0, 0))
        signal = signal.view(1, max_length, self.hidden_size)
        return signal


class Glossification(nn.Module):
    def __init__(self,
                 i_vocab_size,
                 p_vocab_size,
                 t_vocab_size,
                 src_pad_idx,
                 trg_pad_idx,
                 pro_pad_idx,
                 head_num=10,
                 hidden_size=512,
                 inner_size=2048,
                 dropout_rate=0.1,
                 generator_encoder_n_layers=3,
                 generator_decoder_n_layers=1,
                 executor_encoder_n_layers=1,
                 share_target_embeddings=True):
        """
        :param i_vocab_size: int, the vocabulary size of the inputs
        :param p_vocab_size: int, the vocabulary size of the programs
        :param t_vocab_size: int, the vocabulary size of the outputs (in this project, i_vocab_size = t_vocab_size)
        :param src_pad_idx: int, the index of '<pad>' in the input vocabulary
        :param trg_pad_idx: int, the index of '<pad>' in the output vocabulary (in this project, src_pad_idx = trg_pad_idx)
        :param pro_pad_idx: int, the index of '<pad>' in the program vocabulary
        :param head_num: int, head number in the glossification
        :param hidden_size: int, d_model in the glossification
        :param inner_size: int, inner-layer dimensionality in the feed forward network
        :param dropout_rate: int, the dropout rate in the glossification
        :param generator_encoder_n_layers: int, the layer number of the generator encoder
        :param generator_decoder_n_layers: int, the layer number of the generator decoder
        :param executor_encoder_n_layers: int, the layer number of the executor encoder
        :param share_target_embeddings: bool, whether the inputs and outputs embedding is the same
        """
        super(Glossification, self).__init__()
        self.generator = Generator(i_vocab_size,
                                   p_vocab_size,
                                   src_pad_idx,
                                   trg_pad_idx,
                                   head_num,
                                   hidden_size,
                                   inner_size,
                                   dropout_rate,
                                   generator_encoder_n_layers,
                                   generator_decoder_n_layers,
                                   share_target_embeddings)
        self.executor = Executor(t_vocab_size,
                                 pro_pad_idx,
                                 head_num,
                                 hidden_size,
                                 inner_size,
                                 dropout_rate,
                                 executor_encoder_n_layers)
        self.edit_attn = MultiHeadAttention(hidden_size,
                                            dropout_rate,
                                            head_num)
        self.linear = nn.Linear(hidden_size, p_vocab_size)

    def forward(self, inputs, targets, programs):
        # inputs.shape = [b, i_len]
        # targets.shape = [b, t_len]
        # programs.shape = [b, p_len]
        gen_outputs = self.generator(inputs, programs)  # [b, p_len, d_model]
        exc_outputs = self.executor(targets)  # [b, t_len, d_model]
        d_mask = self.execute(inputs, programs)  # [b, p_len, t_len]
        edit_outputs = self.editing_causal_attention(gen_outputs, exc_outputs, d_mask)  # [b, p_len, d_model]
        outputs = self.linear(edit_outputs)
        return outputs

    def editing_causal_attention(self, inputs, targets, d_mask):
        # inputs.shape = [b, p_len, d_model]
        # targets.shape = [b, t_len, d_model]
        # d_mask.shape = [b, p_len, t_len]
        outputs = self.edit_attn(inputs, targets, targets, d_mask)  # [b, p_len, d_model]
        return outputs  # [b, p_len, d_model]
