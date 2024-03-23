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
from optimizer import LRScheduler
from transformers import AutoModel

from tensorboardX import SummaryWriter

from models.model import Glossification
from data.dataset import CSL_Dataset

from utils import set_proxy, get_loss, get_accuracy, save_checkpoint

import argparse
import os
from tqdm import tqdm
import time

# set_proxy()


def summarize_train(writer, global_step, last_time, opt,
                    inputs, targets, programs, optimizer, loss, pred, ans):
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

    acc = get_accuracy(pred, ans, opt.pro_pad_idx)
    writer.add_scalar('training/accuracy',
                      acc, global_step)

    steps_per_sec = 100.0 / (time.time() - last_time)
    writer.add_scalar('global_step/sec', steps_per_sec,
                      global_step)


def train(model, dataloader, optimizer, opt, device, writer, global_step):
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
        editing_casual_mask = batch_data['editing_casual_mask'].to(device)
        pred = model(src, trg, pro, editing_casual_mask)

        pred = pred.view(-1, pred.size(-1))  # [b * p_len, p_vocab_size]
        ans = pro.view(-1)  # [b * p_len]

        loss = get_loss(pred, ans, opt.p_vocab_size,
                        opt.label_smoothing, opt.pro_pad_idx)
        # print(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if global_step % 100 == 0:
            summarize_train(writer, global_step, last_time, opt,
                            src, trg, pro, optimizer, loss, pred, ans)
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
    for batch_data in dataloader:
        src = batch_data['src'].to(device)
        trg = batch_data['trg'].to(device)
        pro = batch_data['pro'].to(device)
        editing_casual_mask = batch_data['editing_casual_mask'].to(device)

        with torch.no_grad():
            pred = model(src, trg, pro, editing_casual_mask)

            pred = pred.view(-1, pred.size(-1))
            ans = pro.view(-1)
            loss = get_loss(pred, ans, opt.p_vocab_size, 0, opt.pro_pad_idx)
        total_loss += loss.item() * len(batch_data)
        total_cnt += len(batch_data)

    val_loss = total_loss / total_cnt
    print("Validation Loss: ", val_loss)
    val_writer.add_scalar('loss', val_loss, global_step)
    return val_loss


def main():
    arg = argparse.ArgumentParser()
    arg.add_argument('--dataset', type=str, default='CSL')
    arg.add_argument('--dataset_path', type=str, default='./data/')
    arg.add_argument('--tokenizer', type=str, default='bert-base-chinese')
    arg.add_argument('--batch_size', type=int, default=4)
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
    arg.add_argument('--train_epochs', type=int, default=10)
    opt = arg.parse_args()
    if opt.dataset == 'CSL':
        dataset_file = os.path.join(opt.dataset_path, 'CSL-Daily_editing_chinese.txt')
        editing_casual_mask_file = os.path.join(opt.dataset_path, 'editing_casual_mask_CSL_174_40.npy')
    else:
        assert True, 'Currently only the CSL datasets is supported !'

    device = torch.device('cpu' if opt.no_cuda else 'cuda')

    tokenizer_model_name = opt.tokenizer

    CSL_dataset = CSL_Dataset(dataset_file=dataset_file,
                              editing_casual_mask_file=editing_casual_mask_file,
                              pre_trained_tokenizer=True,
                              tokenizer_name=tokenizer_model_name)

    batch_size = opt.batch_size
    CSL_dataloader = DataLoader(CSL_dataset, batch_size)

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

    opt.src_pad_idx = CSL_dataset.get_pad_id()
    opt.trg_pad_idx = CSL_dataset.get_pad_id()
    opt.p_vocab_size = CSL_dataset.get_vocab_size()
    opt.pro_pad_idx = CSL_dataset.get_pad_id()

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
                           share_target_embeddings=share_target_embeddings,
                           use_pre_trained_embedding=use_pre_trained_embedding,
                           pre_trained_embedding=embeddings_table)
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
    print('train begin !')
    for epoch in range(opt.train_epochs):
        print('Epoch: ', epoch)
        start_epoch_time = time.time()
        global_step = train(model, CSL_dataloader, Adam_optimizer, opt, device, writer, global_step)
        print("Epoch Time: {:.2f} sec".format(time.time() - start_epoch_time))

        val_loss = validation(CSL_dataloader, model, global_step, val_writer, opt, device)
        save_checkpoint(model, opt.output_dir + '/last/models',
                        global_step, val_loss < best_val_loss)
        best_val_loss = min(val_loss, best_val_loss)

    writer.close()
    val_writer.close()


if __name__ == '__main__':
    main()
