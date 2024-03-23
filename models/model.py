# _*_ coding:utf-8 _*_
"""
@Project Name: glossification
@FileName: model.py
@Begin Date: 2024/3/12 16:28
@Author: caijianfeng
"""

from models.transformer import Encoder, Decoder, MultiHeadAttention
import torch
from torch import nn
import torch.nn.functional as F

import utils
import math


class Generator(nn.Module):
    def __init__(self,
                 i_vocab_size,
                 p_vocab_size,
                 head_num,
                 hidden_size=512,
                 inner_size=2048,
                 dropout_rate=0.1,
                 encoder_n_layers=3,
                 decoder_n_layers=1,
                 src_pad_idx=None,
                 pro_pad_idx=None,
                 share_target_embeddings=False,
                 use_pre_trained_embedding=False,
                 pre_trained_embedding=None):
        super(Generator, self).__init__()
        self.hidden_size = hidden_size
        self.emb_scale = hidden_size ** 0.5
        self.src_pad_idx = src_pad_idx
        self.pro_pad_idx = pro_pad_idx

        # self.i_vocab_embedding = nn.Embedding(i_vocab_size,
        #                                       hidden_size)
        # nn.init.normal_(self.i_vocab_embedding.weight,
        #                 mean=0,
        #                 std=hidden_size ** -0.5)
        # self.i_emb_dropout = nn.Dropout(dropout_rate)
        # if not share_target_embeddings:
        #     self.p_vocab_embedding = nn.Embedding(p_vocab_size,
        #                                           hidden_size)
        # else:
        #     self.p_vocab_embedding = self.i_vocab_embedding
        # self.p_emb_dropout = nn.Dropout(dropout_rate)
        if use_pre_trained_embedding:
            assert pre_trained_embedding is not None, 'If use_pre_trained_embedding is True, you muse offer pre_trained_embedding !'
            assert pre_trained_embedding.weight.shape == torch.Size([i_vocab_size, hidden_size]),  'If use_pre_trained_embedding is True, you muse offer the special i_vocab_size and hidden_size which are same as pre_trained_embedding !'
            self.i_vocab_embedding = pre_trained_embedding
        else:
            self.i_vocab_embedding = nn.Embedding(i_vocab_size,
                                                  hidden_size)
            nn.init.normal_(self.i_vocab_embedding.weight, mean=0,
                            std=hidden_size ** -0.5)

        self.i_emb_dropout = nn.Dropout(dropout_rate)

        if share_target_embeddings:
            self.pro_pad_idx = self.src_pad_idx
            self.p_vocab_embedding = self.i_vocab_embedding
        else:
            self.p_vocab_embedding = nn.Embedding(p_vocab_size, hidden_size)
            nn.init.normal_(self.p_vocab_embedding.weight, mean=0,
                            std=hidden_size ** -0.5)

        self.p_emb_dropout = nn.Dropout(dropout_rate)

        self.encoder = Encoder(hidden_size=self.hidden_size,
                               inner_size=inner_size,
                               dropout_rate=dropout_rate,
                               n_layers=encoder_n_layers,
                               head_num=head_num)
        self.decoder = Decoder(hidden_size=self.hidden_size,
                               inner_size=inner_size,
                               dropout_rate=dropout_rate,
                               n_layers=decoder_n_layers,
                               head_num=head_num)

        # For positional encoding
        self.position_embedding()

    def forward(self, inputs, programs):
        # inputs.shape = [b, i_len]
        # programs.shape = [b, p_len]
        # encoder
        i_mask = utils.create_pad_mask(inputs, self.src_pad_idx)  # [b, 1, i_len]
        enc_output = self.encode(inputs, i_mask)  # [b, i_len, d_model]
        # decoder
        p_mask = utils.create_pad_mask(programs, self.pro_pad_idx)  # [b, 1, p_len]
        program_size = programs.size()[1]  # p_len
        p_self_mask = utils.create_trg_self_mask(program_size, device=programs.device)  # [1, p_len, p_len]
        dec_output = self.decode(programs, enc_output, i_mask, p_self_mask, p_mask)  # [b, p_len, d_model]
        return dec_output  # [b, p_len, d_model]

    def encode(self, inputs, i_mask):
        input_embedded = self.i_vocab_embedding(inputs)  # [b, i_len, d_model]
        # i_mask.squeeze(1).unsqueeze(-1) -> [b, i_len, 1]
        input_embedded.masked_fill_(i_mask.squeeze(1).unsqueeze(-1), 0)  # [b, i_len, d_model]
        input_embedded *= self.emb_scale  # [b, i_len, d_model]
        input_embedded += self.get_position_encoding(inputs)  # [b, i_len, d_model]
        input_embedded = self.i_emb_dropout(input_embedded)  # [b, i_len, d_model]

        # encoder_output = input_embedded  # [b, i_len, d_model]
        encoder_output = self.encoder(input_embedded, i_mask)  # [b, i_len, d_model]

        return encoder_output  # [b, i_len, d_model]

    def decode(self, programs, enc_output, i_mask, p_self_mask, p_mask):
        # programs embedding
        program_embedded = self.p_vocab_embedding(programs)  # [b, p_len, d_model]
        program_embedded.masked_fill_(p_mask.squeeze(1).unsqueeze(-1), 0)  # [b, p_len, d_model]

        # Shifting
        # 将 program_embedded 整体右移一个，最右边多出来的舍弃，最左边空出来的 pad 0
        program_embedded = program_embedded[:, :-1]  # [b, p_len-1, d_model]
        program_embedded = F.pad(program_embedded, (0, 0, 1, 0))  # [b, p_len, d_model]

        program_embedded *= self.emb_scale  # [b, p_len, d_model]
        program_embedded += self.get_position_encoding(programs)  # [b, p_len, d_model]
        program_embedded = self.p_emb_dropout(program_embedded)  # [b, p_len, d_model]

        # decoder
        # decoder_output = program_embedded  # [b, p_len, d_model]
        decoder_output = self.decoder(program_embedded, enc_output, i_mask, p_self_mask)  # [b, p_len, d_model]

        return decoder_output

    def position_embedding(self):
        num_timescales = self.hidden_size // 2
        max_timescale = 10000.0
        min_timescale = 1.0
        log_timescale_increment = (
                math.log(float(max_timescale) /
                         float(min_timescale)) /
                         max(num_timescales - 1, 1))
        inv_timescales = min_timescale * torch.exp(
            torch.arange(num_timescales, dtype=torch.float32) *
            -log_timescale_increment)
        self.register_buffer('inv_timescales', inv_timescales)

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


class Executor(nn.Module):
    def __init__(self,
                 t_vocab_size,
                 trg_pad_idx,
                 head_num,
                 hidden_size=512,
                 inner_size=2048,
                 dropout_rate=0.1,
                 encoder_n_layers=1,
                 use_pre_trained_embedding=False,
                 pre_trained_embedding=None
                 ):
        super(Executor, self).__init__()
        self.hidden_size = hidden_size
        self.emb_scale = hidden_size ** 0.5
        self.trg_pad_idx = trg_pad_idx

        if use_pre_trained_embedding:
            assert pre_trained_embedding is not None, 'If use_pre_trained_embedding is True, you muse offer pre_trained_embedding !'
            assert pre_trained_embedding.weight.shape == torch.Size([t_vocab_size, hidden_size]), 'If use_pre_trained_embedding is True, you muse offer the special t_vocab_size and hidden_size which are same as pre_trained_embedding !'

            self.t_vocab_embedding = pre_trained_embedding
        else:
            self.t_vocab_embedding = nn.Embedding(t_vocab_size, hidden_size)
        self.t_emb_dropout = nn.Dropout(dropout_rate)

        # Executor Encoder
        self.encoder = Encoder(hidden_size=self.hidden_size,
                               inner_size=inner_size,
                               dropout_rate=dropout_rate,
                               n_layers=encoder_n_layers,
                               head_num=head_num)
        # position embedding
        self.position_embedding()

    def forward(self, inputs):
        # inputs.shape = [b, t_len]
        # encoder
        i_mask = utils.create_pad_mask(inputs, self.trg_pad_idx)  # [b, 1, t_len]
        enc_output = self.encode(inputs, i_mask)  # [b, t_len, d_model]
        return enc_output  # [b, t_len, d_model]

    def position_embedding(self):
        num_timescales = self.hidden_size // 2
        max_timescale = 10000.0
        min_timescale = 1.0
        log_timescale_increment = (
                math.log(float(max_timescale) /
                         float(min_timescale)) /
                         max(num_timescales - 1, 1))
        inv_timescales = min_timescale * torch.exp(
            torch.arange(num_timescales, dtype=torch.float32) *
            -log_timescale_increment)
        self.register_buffer('inv_timescales', inv_timescales)

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

    def encode(self, inputs, i_mask):
        # inputs.shape = [b, i_len]
        input_embedded = self.t_vocab_embedding(inputs)  # [b, i_len, d_model]
        # i_mask.squeeze(1).unsqueeze(-1) -> [b, i_len, 1]
        input_embedded.masked_fill_(i_mask.squeeze(1).unsqueeze(-1), 0)  # [b, i_len, d_model]
        input_embedded *= self.emb_scale  # [b, i_len, d_model]
        input_embedded += self.get_position_encoding(inputs)  # [b, i_len, d_model]
        input_embedded = self.t_emb_dropout(input_embedded)  # [b, i_len, d_model]

        encoder_output = input_embedded  # [b, i_len, d_model]
        encoder_output = self.encoder(encoder_output, i_mask)  # [b, i_len, d_model]

        return encoder_output  # [b, i_len, d_model]


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
                 share_target_embeddings=True,
                 use_pre_trained_embedding=False,
                 pre_trained_embedding=None
                 ):
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
        :param dropout_rate: float, the dropout rate in the glossification
        :param generator_encoder_n_layers: int, the layer number of the generator encoder
        :param generator_decoder_n_layers: int, the layer number of the generator decoder
        :param executor_encoder_n_layers: int, the layer number of the executor encoder
        :param share_target_embeddings: bool, whether the inputs and outputs embedding is the same
        :param use_pre_trained_embedding: bool, whether using the pre-trained embedding table
        :param pre_trained_embedding: torch.nn.modules.sparse.Embedding, the pre-trained embedding table. It must be offered if use_pre_trained_embedding is True.
        """
        super(Glossification, self).__init__()
        self.generator = Generator(i_vocab_size,
                                   p_vocab_size,
                                   head_num,
                                   hidden_size,
                                   inner_size,
                                   dropout_rate,
                                   generator_encoder_n_layers,
                                   generator_decoder_n_layers,
                                   src_pad_idx,
                                   pro_pad_idx,
                                   share_target_embeddings,
                                   use_pre_trained_embedding,
                                   pre_trained_embedding)
        self.executor = Executor(t_vocab_size,
                                 trg_pad_idx,
                                 head_num,
                                 hidden_size,
                                 inner_size,
                                 dropout_rate,
                                 executor_encoder_n_layers,
                                 use_pre_trained_embedding,
                                 pre_trained_embedding)
        self.edit_attn = MultiHeadAttention(hidden_size,
                                            dropout_rate,
                                            head_num)
        self.linear = nn.Linear(hidden_size, p_vocab_size)

    def forward_origin(self, inputs, targets, programs):
        # inputs.shape = [b, i_len]
        # targets.shape = [b, t_len]
        # programs.shape = [b, p_len]
        gen_outputs = self.generator(inputs, programs)  # [b, p_len, d_model]
        exc_outputs = self.executor(targets)  # [b, t_len, d_model]
        # TODO: parallel 生成 editing casual attention 的 mask
        d_mask = self.execute(targets, programs)  # [b, p_len, t_len]
        edit_outputs = self.editing_causal_attention(gen_outputs, exc_outputs, d_mask)  # [b, p_len, d_model]
        outputs = self.linear(edit_outputs)
        # linear
        # [b, p_len, d_model] * [d_model, p_vocab_size]
        output = torch.matmul(outputs,
                              self.generator.t_vocab_embedding.weight.transpose(0, 1))  # [b, p_len, p_vocab_size]
        return output  # [b, p_len, p_vocab_size]

    def forward(self, inputs, targets, programs, editing_casual_mask):
        # inputs.shape = [b, i_len]
        # targets.shape = [b, t_len]
        # programs.shape = [b, p_len]
        gen_outputs = self.generator(inputs, programs)  # [b, p_len, d_model]
        # print('gen_outputs shape: ', gen_outputs.shape)
        exc_outputs = self.executor(targets)  # [b, t_len, d_model]
        # print('exc_outputs shape: ', exc_outputs.shape)
        # print(gen_outputs.shape, '; ', exc_outputs.shape, '; ', editing_casual_mask.shape)
        # gen_outputs.shape = [b, p_len, d_model]
        # exc_outputs.shape = [b, t_len, d_model]
        # editing_casual_mask.shape = [b, p_len, t_len]
        edit_outputs = self.editing_causal_attention(gen_outputs, exc_outputs, editing_casual_mask)  # [b, p_len, d_model]
        # print('edit_outputs shape: ', edit_outputs.shape)
        # 原论文是在 editing casual attention 之后添加一个 linear 层将输出映射到 p_vocab_size 维度
        # 这里使用推荐的与目标输出的 embedding table 相乘将输出映射到 p_vocab_size 维度, 无需添加 softmax !
        # edit_outputs = self.linear(edit_outputs)  # [b, p_len, p_vocab_size]
        # linear
        # [b, p_len, d_model] * [d_model, p_vocab_size]
        output = torch.matmul(edit_outputs,
                              self.generator.p_vocab_embedding.weight.transpose(0, 1))  # [b, p_len, p_vocab_size]
        return output  # [b, p_len, p_vocab_size]

    def editing_causal_attention(self, inputs, targets, d_mask):
        # inputs.shape = [b, p_len, d_model]
        # targets.shape = [b, t_len, d_model]
        # d_mask.shape = [b, p_len, t_len]
        outputs = self.edit_attn(inputs, targets, targets, d_mask)  # [b, p_len, d_model]
        return outputs  # [b, p_len, d_model]

    def execute(self, targets, programs):
        """
        通过执行 programs 生成 editing casual attention 的 mask
        :param targets: glosses -> shape = [b, t_len]
        :param programs: min editing programs -> shape = [b, p_len]
        :return: editing casual attention mask -> shape = [b, p_len, t_len]
        """
        mask = torch.zeros(programs.shape[0], programs.shape[1], targets.shape[1])
        for i, program in enumerate(programs):
            generator_pointer = 0
            for j, edit in enumerate(program):
                if 'Add' or 'Copy' in edit:
                    index = int(edit.split(' ')[1])
                    mask[i][generator_pointer: generator_pointer + index] = -1e-8
                    generator_pointer += index
        return mask


if __name__ == '__main__':
    i_vocab_size = 100
    t_vocab_size = 200
    p_vocab_size = 300
    src_pad_idx = 99
    trg_pad_idx = 199
    pro_pad_idx = 299
    head_num = 10
    model = Glossification(i_vocab_size=i_vocab_size,
                           p_vocab_size=p_vocab_size,
                           t_vocab_size=t_vocab_size,
                           src_pad_idx=src_pad_idx,
                           trg_pad_idx=trg_pad_idx,
                           pro_pad_idx=pro_pad_idx,
                           head_num=head_num,
                           share_target_embeddings=False)
    i_len, t_len, p_len = 10, 20, 30
    batch_size = 5
    inputs = torch.randint(0, i_vocab_size, [batch_size, i_len])
    target = torch.randint(0, t_vocab_size, [batch_size, t_len])
    program = torch.randint(0, p_vocab_size, [batch_size, p_len])
    editing_casual_mask = torch.triu(torch.ones([batch_size, p_len, t_len], dtype=torch.uint8), diagonal=1) == 1
    outputs = model(inputs, target, program, editing_casual_mask)
    print(outputs.shape)  # [batch_size, p_len, p_vocab_size]