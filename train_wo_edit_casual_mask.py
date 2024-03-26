# _*_ coding:utf-8 _*_
"""
@Project Name: glossification
@FileName: train.py
@Begin Date: 2024/3/14 11:08
@End Date: 
@Author: caijianfeng
"""
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from optimizer import LRScheduler
from transformers import AutoModel

from tensorboardX import SummaryWriter
from rouge import Rouge
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

from models.model import Glossification
from data.dataset import CSL_Dataset

from utils import set_proxy, get_loss, get_accuracy, save_checkpoint

import argparse
import os
from tqdm import tqdm
import time
import json

# set_proxy()


def summarize_train(writer, global_step, last_time, opt,
                    inputs, targets, programs, optimizer, loss, pred_edit_op, ans_edit_op, pred_edit_num, ans_edit_num):
    writer.add_scalar('input_stats/batch_size',
                      targets.size(0), global_step)

    writer.add_scalar('input_stats/input_length',
                      inputs.size(1), global_step)
    i_nonpad = (inputs != opt.src_pad_idx).view(-1).type(torch.float32)
    writer.add_scalar('input_stats/inputs_nonpadding_frac',
                      i_nonpad.mean(), global_step)

    writer.add_scalar('input_stats/target_length',
                      targets.size(1), global_step)
    t_nonpad = (targets != opt.trg_pad_idx).view(-1).type(torch.float32)
    writer.add_scalar('input_stats/target_nonpadding_frac',
                      t_nonpad.mean(), global_step)

    writer.add_scalar('input_stats/program_length',
                      programs.size(1), global_step)
    p_nonpad = (programs != opt.pro_pad_idx).view(-1).type(torch.float32)
    writer.add_scalar('input_stats/program_nonpadding_frac',
                      p_nonpad.mean(), global_step)

    writer.add_scalar('optimizer/learning_rate',
                      optimizer.learning_rate(), global_step)

    writer.add_scalar('loss', loss.item(), global_step)

    acc = (get_accuracy(pred_edit_num, ans_edit_num, opt.pro_pad_idx) +
           get_accuracy(pred_edit_op, ans_edit_op, opt.edit_num)) / 2
    writer.add_scalar('training/accuracy',
                      acc, global_step)

    steps_per_sec = 100.0 / (time.time() - last_time)
    writer.add_scalar('global_step/sec', steps_per_sec,
                      global_step)


def train(model, dataloader, optimizer, opt, device, writer, global_step):
    torch.autograd.set_detect_anomaly(True)
    model = model.to(device)
    model.train()
    last_time = time.time()
    pbar = tqdm(total=len(dataloader.dataset), ascii=True)
    for batch_data in dataloader:
        # {
        #     'src': self.data_sentence_token[item],
        #     'trg': self.data_gloss_token[item],
        #     'pro': self.data_editing_program_token[item],
        #     'editing_casual_mask': self.editing_casual_mask[item]
        # }
        src = batch_data['src'].to(device)
        trg = batch_data['trg'].to(device)
        pro = batch_data['pro'].to(device)

        # print(pro.dtype)
        # editing_casual_mask = batch_data['editing_casual_mask'].to(device)
        pred_edit_op, pred_edit_num = model(src, pro)

        pred_edit_op = pred_edit_op.view(-1, pred_edit_op.size(-1))  # [b * p_len/2, edit_op_num]
        pred_edit_num = pred_edit_num.view(-1, pred_edit_num.size(-1))  # [b * p_len/2, p_vocab_size]
        ans_edit_op = pro[:, 1::2].contiguous().view(-1)  # [b * p_len/2]
        # 将 ans_edit_op由 tokenize 得到的 p_vocab_size 转化为 edit_num
        ans_edit_op[ans_edit_op == opt.pro_pad_idx] = opt.edit_num
        for i in range(opt.edit_num):
            ans_edit_op[ans_edit_op == opt.edit_op[i]] = i

        ans_edit_num = pro[:, ::2].contiguous().view(-1)  # [b * p_len/2]
        loss = get_loss(pred_edit_num, ans_edit_num, opt.p_vocab_size, opt.label_smoothing, opt.pro_pad_idx) + \
               F.cross_entropy(pred_edit_op, ans_edit_op, ignore_index=opt.edit_num)
        # print(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if global_step % 100 == 0:
            summarize_train(writer, global_step, last_time, opt,
                            src, trg, pro, optimizer, loss, pred_edit_op, ans_edit_op, pred_edit_num, ans_edit_num)
            last_time = time.time()

        pbar.set_description('[Loss: {:.4f}]'.format(loss.item()))

        global_step += 1
        pbar.update(trg.size(0))
    pbar.close()
    return global_step


def validation(dataloader, model, global_step, val_writer, opt, device):
    model.eval()
    total_loss = 0.0
    total_cnt = 0
    total_bleu3_score = 0
    total_bleu4_score = 0
    rouge = Rouge()
    total_rouge_l = 0
    total_acc = 0
    for batch_data in dataloader:
        src = batch_data['src'].to(device)
        trg = batch_data['trg'].to(device)
        pro = batch_data['pro'].to(device)
        # editing_casual_mask = batch_data['editing_casual_mask'].to(device)

        with torch.no_grad():
            pred_edit_op, pred_edit_num = model(src, pro)  # [b, p_len/2, edit_num] and [b, p_len/2, p_vocab_size]
            pred_edit_op_index = torch.argmax(pred_edit_op, dim=-1)  # [b, p_len/2]
            pred_edit_num_index = torch.argmax(pred_edit_num, dim=-1)  # [b, p_len/2]
            pred_index = torch.zeros_like(pro)
            pred_index[:, 1::2] = pred_edit_op_index
            pred_index[:, ::2] = pred_edit_num_index
            pred_index = pred_index.tolist()
            ans_edit_op = pro[:, 1::2]  # [b, p_len/2]
            ans_edit_op[ans_edit_op == opt.pro_pad_idx] = opt.edit_num
            for i in range(opt.edit_num):
                ans_edit_op[ans_edit_op == opt.edit_op[i]] = i
            ans_edit_num = pro[:, ::2]
            pro_index = torch.zeros_like(pro)  # [b, p_len]
            pro_index[:, 1::2] = ans_edit_op
            pro_index[:, ::2] = ans_edit_num
            pro_index = pro_index.tolist()
            rouge_score = rouge.get_scores([" ".join(str(index) for index in pre_index) for pre_index in pred_index],
                                            [" ".join(str(index) for index in pr_index) for pr_index in pro_index])
            assert len(rouge_score) == len(pro_index), \
                f'the length of rouge_score is {len(rouge_score)}, ' \
                f'and the length of batch data is {len(batch_data)},' \
                f'the length of program is {len(pro_index)}'
            rouge_l = sum([score['rouge-l']['f'] for score in rouge_score])
            pro_index = [[p] for p in pro_index]
            # print(pred.shape, '; ', pro.shape)
            # dataset = dataloader.dataset
            # pred_index = [p.append(dataset.token_to_id('.')) for p in pred_index]
            # pred_str = dataset.decode(pred_index)  # [b , str(len=p_len)]
            # program_str = dataset.decode(pro.tolist())  # [b , str(len=p_len)]
            # print(pred_str, '; ', program_str)
            # programs_str = [[pro] for pro in program_str]  # [b, 1, str(len=p_len)]
            # predictions is list[str]; references is list[list[str]]
            # total_bleu_score += results['bleu']
            # each_bleu_score = [origin + new for origin, new in zip(each_bleu_score, results['precisions'])]
            smooth_fun = SmoothingFunction()
            bleu3 = corpus_bleu(pro_index, pred_index, weights=[0.5, 0.5, 0.5], smoothing_function=smooth_fun.method1)
            bleu4 = corpus_bleu(pro_index, pred_index, smoothing_function=smooth_fun.method1)
            pred_edit_op = pred_edit_op.view(-1, pred_edit_op.size(-1))  # [b * p_len/2, edit_op_num]
            pred_edit_num = pred_edit_num.view(-1, pred_edit_num.size(-1))  # [b * p_len/2, p_vocab_size]
            ans_edit_op = pro[:, 1::2].reshape(-1)  # [b * p_len/2]
            # 将 ans_edit_op由 tokenize 得到的 p_vocab_size 转化为 edit_num
            ans_edit_op[ans_edit_op == opt.pro_pad_idx] = opt.edit_num
            for i in range(opt.edit_num):
                ans_edit_op[ans_edit_op == opt.edit_op[i]] = i

            ans_edit_num = pro[:, ::2].reshape(-1)  # [b * p_len/2]

            loss = get_loss(pred_edit_num, ans_edit_num, opt.p_vocab_size, opt.label_smoothing, opt.pro_pad_idx) + \
                   F.cross_entropy(pred_edit_op, ans_edit_op, ignore_index=opt.edit_num)
            acc = (get_accuracy(pred_edit_num, ans_edit_num, opt.pro_pad_idx) +
                   get_accuracy(pred_edit_op, ans_edit_op, opt.edit_num)) / 2
        total_bleu3_score += bleu3 * len(pro_index)
        total_bleu4_score += bleu4 * len(pro_index)
        total_rouge_l += rouge_l
        total_loss += loss.item() * len(pro_index)
        total_acc += acc * len(pro_index)
        total_cnt += len(pro_index)

    val_loss = total_loss / total_cnt
    total_bleu3_score = total_bleu3_score / total_cnt
    total_bleu4_score = total_bleu4_score / total_cnt
    total_rouge_l = total_rouge_l / total_cnt
    total_acc = total_acc / total_cnt
    print("Validation Loss: ", val_loss)
    print("total bleu3 score: ", total_bleu3_score)
    print("total bleu4 score: ", total_bleu4_score)
    print("total rouge-l score: ", total_rouge_l)
    print("total accuracy score: ", total_acc)
    val_writer.add_scalar('loss', val_loss, global_step)
    val_writer.add_scalar('bleu3 score', total_bleu3_score, global_step)
    val_writer.add_scalar('bleu4 score', total_bleu4_score, global_step)
    val_writer.add_scalar('rouge-l score', total_rouge_l, global_step)
    val_writer.add_scalar('accuracy score', total_acc, global_step)
    return val_loss, total_bleu3_score, total_bleu4_score, total_rouge_l, total_acc


def main():
    arg = argparse.ArgumentParser()
    arg.add_argument('--dataset', type=str, default='CSL')
    arg.add_argument('--dataset_path', type=str, default='./CSL_data/')
    arg.add_argument('--tokenizer', type=str, default='bert-base-chinese')
    arg.add_argument('--batch_size', type=int, default=64)
    arg.add_argument('--head_num', type=int, default=10)
    arg.add_argument('--inner_size', type=int, default=1024)
    arg.add_argument('--dropout_rate', type=float, default=0.1)
    arg.add_argument('--generator_encoder_n_layers', type=int, default=3)
    arg.add_argument('--generator_decoder_n_layers', type=int, default=1)
    arg.add_argument('--executor_encoder_n_layers', type=int, default=1)
    arg.add_argument('--share_target_embeddings', action='store_true')
    arg.add_argument('--use_pre_trained_embedding', action='store_true')
    arg.add_argument('--label_smoothing', type=float, default=0.05)
    arg.add_argument('--warmup', type=int, default=100)
    arg.add_argument('--no_cuda', action='store_true')
    arg.add_argument("--output_dir", type=str, default='./output')
    arg.add_argument('--train_epochs', type=int, default=150)
    opt = arg.parse_args()
    if opt.dataset == 'CSL':
        dataset_file = os.path.join(opt.dataset_path, 'CSL-Daily_editing_chinese.txt')
    else:
        assert True, 'Currently only the CSL datasets is supported !'

    device = torch.device('cpu' if opt.no_cuda else 'cuda')

    tokenizer_model_name = opt.tokenizer

    CSL_dataset = CSL_Dataset(dataset_file=dataset_file,
                              # editing_casual_mask_file=editing_casual_mask_file,
                              pre_trained_tokenizer=True,
                              tokenizer_name=tokenizer_model_name)

    batch_size = opt.batch_size
    CSL_dataloader = DataLoader(CSL_dataset, batch_size, shuffle=True)

    # tokenizer = CSL_dataset.tokenizer
    # Load the pre-trained model and tokenizer
    tokenizer_model = AutoModel.from_pretrained(tokenizer_model_name)
    # Get the word embeddings layer
    embeddings_table = tokenizer_model.get_input_embeddings()

    head_num = opt.head_num
    inner_size = opt.inner_size
    dropout_rate = opt.dropout_rate
    generator_encoder_n_layers = opt.generator_encoder_n_layers
    generator_decoder_n_layers = opt.generator_decoder_n_layers
    executor_encoder_n_layers = opt.executor_encoder_n_layers
    share_target_embeddings = opt.share_target_embeddings
    use_pre_trained_embedding = opt.use_pre_trained_embedding
    edit_op_num = 4
    opt.i_vocab_size = CSL_dataset.get_vocab_size()
    opt.src_pad_idx = CSL_dataset.get_pad_id()
    opt.t_vocab_size = CSL_dataset.get_vocab_size()
    opt.trg_pad_idx = CSL_dataset.get_pad_id()
    opt.p_vocab_size = CSL_dataset.get_vocab_size()
    opt.pro_pad_idx = CSL_dataset.get_pad_id()
    opt.edit_num = edit_op_num
    opt.edit_op = [CSL_dataset.get_token_id('加'), CSL_dataset.get_token_id('删'), CSL_dataset.get_token_id('贴'),
                   CSL_dataset.get_token_id('过')]
    model = Glossification(CSL_dataset.get_vocab_size(),
                           CSL_dataset.get_vocab_size(),
                           CSL_dataset.get_vocab_size(),
                           src_pad_idx=CSL_dataset.get_pad_id(),
                           trg_pad_idx=CSL_dataset.get_pad_id(),
                           pro_pad_idx=CSL_dataset.get_pad_id(),
                           head_num=head_num,
                           hidden_size=embeddings_table.weight.shape[1],
                           inner_size=inner_size,
                           dropout_rate=dropout_rate,
                           generator_encoder_n_layers=generator_encoder_n_layers,
                           generator_decoder_n_layers=generator_decoder_n_layers,
                           executor_encoder_n_layers=executor_encoder_n_layers,
                           edit_op_num=edit_op_num,
                           share_target_embeddings=share_target_embeddings,
                           use_pre_trained_embedding=use_pre_trained_embedding,
                           pre_trained_embedding=embeddings_table,
                           with_editing_casual_mask=False)
    global_step = 0

    writer = SummaryWriter(logdir=os.path.join(opt.output_dir, 'last'))
    val_writer = SummaryWriter(logdir=os.path.join(opt.output_dir, 'last/val'))

    params_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("# of parameters: {}".format(params_num))
    writer.add_scalar('model parameter number', params_num)

    warmup = opt.warmup
    Adam_optimizer = LRScheduler(parameters=model.parameters(),
                                 hidden_size=embeddings_table.weight.shape[1],
                                 warmup=warmup,
                                 step=global_step)

    if not os.path.exists(opt.output_dir):
        os.mkdir(opt.output_dir)

    best_val_loss = float('inf')
    best_total_bleu3 = 0
    best_total_bleu4 = 0
    best_total_rouge_l = 0
    best_total_acc = 0
    print('train begin !')
    for epoch in range(opt.train_epochs):
        print('Epoch: ', epoch)
        start_epoch_time = time.time()
        global_step = train(model, CSL_dataloader, Adam_optimizer, opt, device, writer, global_step)
        print("Epoch Time: {:.2f} sec".format(time.time() - start_epoch_time))

        val_loss, total_bleu3, total_bleu4, total_rouge_l, total_acc = validation(CSL_dataloader, model, global_step,
                                                                                  val_writer, opt, device)
        save_checkpoint(model, opt.output_dir + '/last/models',
                        global_step, val_loss < best_val_loss)
        best_val_loss = min(val_loss, best_val_loss)
        best_total_bleu3 = max(best_total_bleu3, total_bleu3)
        best_total_bleu4 = max(best_total_bleu3, total_bleu4)
        best_total_rouge_l = max(best_total_rouge_l, total_rouge_l)
        best_total_acc = max(best_total_acc, total_acc)

    writer.close()
    val_writer.close()

    train_args = opt.__dict__

    with open(os.path.join(opt.output_dir, 'train.json'), 'w', encoding='utf-8') as f:
        json.dump(train_args, f)
    print('save train args succeed !')

    print('total best bleu3: ', best_total_bleu3)
    print('total best bleu4: ', best_total_bleu4)
    print('total best rouge-l: ', best_total_rouge_l)
    print('total best accuracy: ', best_total_acc)

    with open(os.path.join(opt.output_dir, 'result.txt'), 'w', encoding='utf-8') as f:
        f.write(f'total best bleu3: {best_total_bleu3}\n')
        f.write(f'total best bleu4: {best_total_bleu4}\n')
        f.write(f'total best rouge-l: {best_total_rouge_l}\n')
        f.write(f'total best accuracy: {best_total_acc}\n')


if __name__ == '__main__':
    main()
