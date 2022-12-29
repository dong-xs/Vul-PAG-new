# -*- coding: utf-8 -*-
# @Time    : 2021-10-24 18:54
# @Author  : lldzyshwjx
import os

import fire
import warnings
import torch

import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

from torch.autograd import Variable
from config import Config
from model import BERT_LSTM_CRF
from utils import (load_vocab, read_corpus_bert_wordpiece, load_model,read_corpus_bert,
                   save_model, load_vocab_reverse,
                   compute_eval_metrics, build_data_loader)
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from tqdm import tqdm

warnings.filterwarnings('ignore', category=UserWarning)

random_seed = 1024
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)


# bert+lstm+crf
def train(**kwargs):
    config = Config()
    config.update(**kwargs)
    tokenizer = BertTokenizer.from_pretrained(config.bert_path, do_lower_case=False)   #tokenizer是BERT默认的分词器，即为wordpiece的结果
    print('*' * 80)
    print('config file is:\n', config)
    print('*' * 80)
    if config.use_cuda:
        torch.cuda.set_device(config.gpu)

    print('data loading...')
    label_dic = load_vocab(config.label_file)

    tagset_size = len(label_dic)

    # get data with wordpiece
    data = read_corpus_bert_wordpiece(config.data_file, max_length=config.max_length, label_dic=label_dic,tokenizer=tokenizer)
    # get data without wordpiece
    data_unk = read_corpus_bert(config.data_file, max_length=config.max_length, label_dic=label_dic, tokenizer=tokenizer)

    # shuffle data...
    random.shuffle(data)
    random.shuffle(data_unk)

    train_loader, dev_loader = build_data_loader(data, config, random_seed)  # training set and test set construction，result with wordpiece
    train_loader_unk, dev_loader_unk = build_data_loader(data_unk, config, random_seed)  # training set and test set construction，result without wordpiece

    print('train_data:{}'.format(len(train_loader.dataset)))
    print('dev_data:{}'.format(len(dev_loader.dataset)))
    print('train_data_unk:{}'.format(len(train_loader_unk.dataset)))
    print('dev_data_unk:{}'.format(len(dev_loader_unk.dataset)))
    print('data loading finished!!!')
    print('*' * 80)

    # model initizlization
    model = BERT_LSTM_CRF(config.bert_path, tagset_size, config.bert_embedding, config.char_embedding,
                          config.hidden_dim, config.lstm_layers, dropout_ratio=config.dropout_ratio,
                          dropout1=config.dropout1, use_cuda=config.use_cuda)
    if config.load_model:
        assert config.load_path is not None
        model = load_model(model, name=config.load_path)

    # torch.backends.cudnn.enabled=False    #test code on cpu only

    if config.use_cuda:
        model.cuda()
    model.train()
    optimizer = getattr(optim, config.optim)
    optimizer = optimizer(model.parameters(), lr=config.lr,
                          weight_decay=config.weight_decay)
    eval_loss = 10000
    eval_f1 = 0
    best_epoch = 0
    best_model_metric = [0, 0, 0, 0, 0]

    for epoch in range(config.base_epoch):
        step, length = 0, 0

        for batch, batch_unk in tqdm(zip(train_loader, train_loader_unk)):
            model.zero_grad()
            inputs, masks, wordpiece_masks, tags, start_ids,  dp_ids, char_embeddings, word2vec_embedding, \
            fasttext_embeddings, input_unk, masks_unk, tags_unk = batch

            inputs, masks, tags, start_ids, wordpiece_masks, dp_ids, char_embeddings, word2vec_embedding, \
            fasttext_embeddings = Variable(inputs), Variable(masks), Variable(tags),\
                                                                  Variable(start_ids), Variable(wordpiece_masks), \
                                                                  Variable(dp_ids), Variable(char_embeddings), \
                                                                  Variable(word2vec_embedding),Variable(fasttext_embeddings)

            input_unk, masks_unk, tags_unk = Variable(batch_unk[-3]), Variable(batch_unk[-2]), Variable(batch_unk[-1])


            if config.use_cuda:
                inputs, masks, tags, start_ids, wordpiece_masks, dp_ids, char_embeddings, \
                word2vec_embedding, fasttext_embeddings, input_unk, masks_unk, tags_unk =\
                    inputs.cuda(), masks.cuda(), tags.cuda(), start_ids.cuda(), wordpiece_masks.cuda(), \
                    dp_ids.cuda(), char_embeddings.cuda(), word2vec_embedding.cuda(),fasttext_embeddings.cuda(), \
                    input_unk.cuda(), masks_unk.cuda(), tags_unk.cuda()

            torch.cuda.empty_cache()

            # Passing parameters to the model
            feats = model(inputs, input_unk, masks_unk, tags_unk, masks, start_ids, wordpiece_masks, char_embeddings)  # fc输出
            loss = model.loss(feats, masks, tags)  # 自定义crf loss 返回每个batch的平均损失  feats shape：batch_size*seq_length*dim
            loss.backward()

            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            step += 1
            length += inputs.size(0)
            accuary, precision, recall, f1, classification_report = val(model, feats, masks, tags, config)
            print('Train Epoch: {} [{}/{} ({:.2f}%)]\tloss: {:.6f}\taccuary: {:.6f}\tprecision: {:.6f}\trecall: {:.6f}\tf1: {:.6f}'.format(
                    epoch, length, len(train_loader.dataset), 100. * step / len(train_loader), loss.item(), accuary,
                    precision, recall, f1))

        # testing after a epoch on dev set
        loss_temp, accuary, precision, recall, f1, classification_report = dev(model, dev_loader, dev_loader_unk, epoch, config)

        # if loss_temp < eval_loss:
        #     save_model(model, epoch)
        #     eval_loss = loss_temp
        #     best_model_metric = [accuary, precision, recall, f1, classification_report]
        if f1 > eval_f1:
            save_model(model, epoch)
            eval_f1 = f1
            eval_loss = loss_temp
            best_epoch = epoch
            best_model_metric = [accuary, precision, recall, f1, classification_report]
        print("*" * 50)
        print('best epoch:{}'.format(best_epoch))
        print('best loss:{}'.format(eval_loss))
        print("best accuary: ", best_model_metric[0])
        print("best precision: ", best_model_metric[1])
        print("best recall: ", best_model_metric[2])
        print("best f1: ", best_model_metric[3])
        print("classification report: ")
        print(best_model_metric[4])


# word2vec+lstm+crf
# def train_word2vec(**kwargs):
#     config = Config()
#     config.update(**kwargs)
#     print('*' * 80)
#     print('当前配置文件为:\n', config)
#     print('*' * 80)
#     if config.use_cuda:
#         torch.cuda.set_device(config.gpu)
#
#     print('加载数据中...')
#
#     label_dic = load_vocab(config.label_file)  # 得到标签到索引的映射字典
#     tagset_size = len(label_dic) - 2  # 去除<start>、<eos>后，还剩余18个标签
#
#     data = read_corpus_word2vec(config.data_file, max_length=config.max_length,
#                                 label_dic=label_dic)  # 读取数据，输出为InputFeature对象，里面包含四个值
#
#     # 对数据进行随机打乱
#     random.shuffle(data)
#
#     train_loader, dev_loader = build_data_loader(data, config, random_seed)  # 构造训练集和测试集
#     print('train_data:{}条'.format(len(train_loader.dataset)))
#     print('dev_data:{}条'.format(len(dev_loader.dataset)))
#     print('数据加载完毕!')
#     print('*' * 80)
#     model = WORE2VEC_LSTM_CRF(tagset_size, config.word2vec_embedding, config.char_embedding, config.hidden_dim,
#                               config.lstm_layers, dropout_ratio=config.dropout_ratio, dropout1=config.dropout1,
#                               use_cuda=config.use_cuda)
#     if config.load_model:
#         assert config.load_path is not None
#         model = load_model(model, name=config.load_path)
#     if config.use_cuda:
#         model.cuda()
#     model.train()
#     optimizer = getattr(optim, config.optim)
#     optimizer = optimizer(model.parameters(), lr=config.lr,
#                           weight_decay=config.weight_decay)  # weight_decay权重衰减（L2惩罚）（默认: 0）
#     eval_loss = 10000
#     eval_f1 = 0
#     best_epoch = 0
#     best_model_metric = [0, 0, 0, 0, 0]
#     # early_stopping = EarlyStopping(patience=patience, verbose=True)
#
#     for epoch in range(config.base_epoch):
#         step, length = 0, 0
#         # train_loader 长度为总样本数 / batch_size
#         for batch in tqdm(train_loader):
#             model.zero_grad()
#             # ids, masks, wordpiece_masks, tags, start_ids, dp_ids, char_embeddings, word2vec_embeddings, fasttext_embeddings
#             inputs, masks, wordpiece_masks, tags, start_ids, dp_ids, char_embeddings, word2vec_embeddings, fasttext_embeddings = batch
#             inputs, masks, tags, start_ids, wordpiece_masks, dp_ids, char_embeddings, word2vec_embeddings, fasttext_embeddings = \
#                 Variable(inputs), Variable(masks), Variable(tags), Variable(start_ids), Variable(wordpiece_masks), \
#                 Variable(dp_ids), Variable(char_embeddings), Variable(word2vec_embeddings), Variable(fasttext_embeddings)
#             if config.use_cuda:
#                 inputs, masks, tags, start_ids, wordpiece_masks, dp_ids, char_embeddings, word2vec_embeddings, fasttext_embeddings = inputs.cuda(), masks.cuda(), tags.cuda(), start_ids.cuda(), wordpiece_masks.cuda(), dp_ids.cuda(), char_embeddings.cuda(), word2vec_embeddings.cuda(), fasttext_embeddings.cuda()
#             feats = model(word2vec_embeddings, char_embeddings,
#                           masks)  # fc输出            feats = model(word2vec_embeddings, char_embeddings, masks)  # fc输出
#             loss = model.loss(feats, masks, tags)  # 自定义crf loss 返回每个batch的平均损失  feats shape：batch_size*seq_length*dim
#             loss.backward()
#             # # 梯度截断，防止梯度爆炸
#             # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#
#             optimizer.step()
#             step += 1
#             length += word2vec_embeddings.size(0)
#             accuary, precision, recall, f1, classification_report = val(model, feats, masks, tags, config)
#             # if step % 10 == 0:
#             # train_loader.dataset得到数据总样本数  # len(train_loader) 长度为总样本数 / batch_size
#             print(
#                 'Train Epoch: {} [{}/{} ({:.2f}%)]\tloss: {:.6f}\taccuary: {:.6f}\tprecision: {:.6f}\trecall: {:.6f}\tf1: {:.6f}'.format(
#                     epoch, length, len(train_loader.dataset), 100. * step / len(train_loader), loss.item(), accuary,
#                     precision, recall, f1))
#         # 训练一个epoch后就在开发集上进行测试
#         loss_temp, accuary, precision, recall, f1, classification_report = dev(model, dev_loader, epoch, config)
#
#         # # early_stopping needs the validation loss to check if it has decresed,
#         # # and if it has, it will make a checkpoint of the current model
#         # early_stopping(loss_temp, model)
#         #
#         # if early_stopping.early_stop:
#         #     print("Early stopping")
#         #     break
#
#         # if loss_temp < eval_loss:
#         #     save_model(model, epoch)
#         #     eval_loss = loss_temp
#         #     best_model_metric = [accuary, precision, recall, f1, classification_report]
#         if f1 > eval_f1:
#             save_model(model, epoch)
#             eval_f1 = f1
#             eval_loss = loss_temp
#             best_epoch = epoch
#             best_model_metric = [accuary, precision, recall, f1, classification_report]
#         print("*" * 50)
#         print('best epoch:{}'.format(best_epoch))
#         print('best loss:{}'.format(eval_loss))
#         print("best accuary: ", best_model_metric[0])
#         print("best precision: ", best_model_metric[1])
#         print("best recall: ", best_model_metric[2])
#         print("best f1: ", best_model_metric[3])
#         print("classification report: ")
#         print(best_model_metric[4])


# fasttext+lstm+crf
# def train_fasttext(**kwargs):
#     config = Config()
#     config.update(**kwargs)
#     print('*' * 80)
#     print('当前配置文件为:\n', config)
#     print('*' * 80)
#     if config.use_cuda:
#         torch.cuda.set_device(config.gpu)
#
#     print('加载数据中...')
#
#     label_dic = load_vocab(config.label_file)  # 得到标签到索引的映射字典
#     tagset_size = len(label_dic) - 2  # 去除<start>、<eos>后，还剩余18个标签
#
#     data = read_corpus_fasttext(config.data_file, max_length=config.max_length,
#                                 label_dic=label_dic)  # 读取数据，输出为InputFeature对象，里面包含四个值
#
#     # 对数据进行随机打乱
#     random.shuffle(data)
#
#     train_loader, dev_loader = build_data_loader(data, config, random_seed)  # 构造训练集和测试集
#     print('train_data:{}条'.format(len(train_loader.dataset)))
#     print('dev_data:{}条'.format(len(dev_loader.dataset)))
#     print('数据加载完毕!')
#     print('*' * 80)
#     model = FASTTEXT_LSTM_CRF(tagset_size, config.fasttext_embedding, config.char_embedding, config.hidden_dim,
#                               config.lstm_layers, dropout_ratio=config.dropout_ratio, dropout1=config.dropout1,
#                               use_cuda=config.use_cuda)
#     if config.load_model:
#         assert config.load_path is not None
#         model = load_model(model, name=config.load_path)
#     if config.use_cuda:
#         model.cuda()
#     model.train()
#     optimizer = getattr(optim, config.optim)
#     optimizer = optimizer(model.parameters(), lr=config.lr,
#                           weight_decay=config.weight_decay)  # weight_decay权重衰减（L2惩罚）（默认: 0）
#     eval_loss = 10000
#     eval_f1 = 0
#     best_epoch = 0
#     best_model_metric = [0, 0, 0, 0, 0]
#     # early_stopping = EarlyStopping(patience=patience, verbose=True)
#
#     for epoch in range(config.base_epoch):
#         step, length = 0, 0
#         # train_loader 长度为总样本数 / batch_size
#         for batch in tqdm(train_loader):
#             model.zero_grad()
#
#             inputs, masks, tags, start_ids, wordpiece_masks, dp_ids, char_embeddings, word2vec_embeddings, fasttext_embeddings = batch
#             inputs, masks, tags, start_ids, wordpiece_masks, dp_ids, char_embeddings, word2vec_embeddings, fasttext_embeddings = Variable(
#                 inputs), Variable(masks), Variable(tags), Variable(start_ids), Variable(wordpiece_masks), Variable(
#                 dp_ids), Variable(char_embeddings), Variable(word2vec_embeddings), Variable(fasttext_embeddings)
#             if config.use_cuda:
#                 inputs, masks, tags, start_ids, wordpiece_masks, dp_ids, char_embeddings, word2vec_embeddings, fasttext_embeddings = inputs.cuda(), masks.cuda(), tags.cuda(), start_ids.cuda(), wordpiece_masks.cuda(), dp_ids.cuda(), char_embeddings.cuda(), word2vec_embeddings.cuda(), fasttext_embeddings.cuda()
#             feats = model(fasttext_embeddings, char_embeddings, masks)  # fc输出
#
#             loss = model.loss(feats, masks, tags)  # 自定义crf loss 返回每个batch的平均损失  feats shape：batch_size*seq_length*dim
#             loss.backward()
#
#             optimizer.step()
#             step += 1
#             length += fasttext_embeddings.size(0)
#             accuary, precision, recall, f1, classification_report = val(model, feats, masks, tags, config)
#             # if step % 10 == 0:
#             # train_loader.dataset得到数据总样本数  # len(train_loader) 长度为总样本数 / batch_size
#             print(
#                 'Train Epoch: {} [{}/{} ({:.2f}%)]\tloss: {:.6f}\taccuary: {:.6f}\tprecision: {:.6f}\trecall: {:.6f}\tf1: {:.6f}'.format(
#                     epoch, length, len(train_loader.dataset), 100. * step / len(train_loader), loss.item(), accuary,
#                     precision, recall, f1))
#         # 训练一个epoch后就在开发集上进行测试
#         loss_temp, accuary, precision, recall, f1, classification_report = dev(model, dev_loader, epoch, config)
#
#         # # early_stopping needs the validation loss to check if it has decresed,
#         # # and if it has, it will make a checkpoint of the current model
#         # early_stopping(loss_temp, model)
#         #
#         # if early_stopping.early_stop:
#         #     print("Early stopping")
#         #     break
#
#         # if loss_temp < eval_loss:
#         #     save_model(model, epoch)
#         #     eval_loss = loss_temp
#         #     best_model_metric = [accuary, precision, recall, f1, classification_report]
#         if f1 > eval_f1:
#             save_model(model, epoch)
#             eval_f1 = f1
#             eval_loss = loss_temp
#             best_epoch = epoch
#             best_model_metric = [accuary, precision, recall, f1, classification_report]
#         print("*" * 50)
#         print('best epoch:{}'.format(best_epoch))
#         print('best loss:{}'.format(eval_loss))
#         print("best accuary: ", best_model_metric[0])
#         print("best precision: ", best_model_metric[1])
#         print("best recall: ", best_model_metric[2])
#         print("best f1: ", best_model_metric[3])
#         print("classification report: ")
#         print(best_model_metric[4])


# word2vec+lstm(无crf)
# def train_word2vec_lstm(**kwargs):
#     config = Config()
#     config.update(**kwargs)
#     print('*' * 80)
#     print('当前配置文件为:\n', config)
#     print('*' * 80)
#     if config.use_cuda:
#         torch.cuda.set_device(config.gpu)
#
#     print('加载数据中...')
#
#     label_dic = load_vocab(config.label_file)  # 得到标签到索引的映射字典
#     tagset_size = len(label_dic) - 2  # 去除<start>、<eos>后，还剩余18个标签
#
#     data = read_corpus_word2vec(config.data_file, max_length=config.max_length,
#                                 label_dic=label_dic)  # 读取数据，输出为InputFeature对象，里面包含四个值
#
#     # 对数据进行随机打乱
#     random.shuffle(data)
#
#     train_loader, dev_loader = build_data_loader(data, config, random_seed)  # 构造训练集和测试集
#     print('train_data:{}条'.format(len(train_loader.dataset)))
#     print('dev_data:{}条'.format(len(dev_loader.dataset)))
#     print('数据加载完毕!')
#     print('*' * 80)
#     model = WORE2VEC_LSTM(tagset_size, config.word2vec_embedding, config.hidden_dim,
#                           config.lstm_layers, dropout_ratio=config.dropout_ratio, dropout1=config.dropout1,
#                           use_cuda=config.use_cuda)
#     if config.load_model:
#         assert config.load_path is not None
#         model = load_model(model, name=config.load_path)
#     if config.use_cuda:
#         model.cuda()
#     model.train()
#     optimizer = getattr(optim, config.optim)
#     optimizer = optimizer(model.parameters(), lr=config.lr,
#                           weight_decay=config.weight_decay)  # weight_decay权重衰减（L2惩罚）（默认: 0）
#     eval_loss = 10000
#     eval_f1 = 0
#     best_epoch = 0
#     best_model_metric = [0, 0, 0, 0, 0]
#     loss_function = nn.CrossEntropyLoss()  # 使用crossentropy作为损失函数，因为当前目标是衡量两个序列间的分布情况，使用交叉熵更加适合。
#
#     for epoch in range(config.base_epoch):
#         step, length = 0, 0
#         # train_loader 长度为总样本数 / batch_size
#         for batch in tqdm(train_loader):
#             model.zero_grad()
#
#             inputs, masks, tags, start_ids, wordpiece_masks, dp_ids, char_embeddings, word2vec_embeddings, fasttext_embeddings = batch
#             inputs, masks, tags, start_ids, wordpiece_masks, dp_ids, char_embeddings, word2vec_embeddings, fasttext_embeddings = Variable(
#                 inputs), Variable(masks), Variable(tags), Variable(start_ids), Variable(wordpiece_masks), Variable(
#                 dp_ids), Variable(char_embeddings), Variable(word2vec_embeddings), Variable(fasttext_embeddings)
#             if config.use_cuda:
#                 inputs, masks, tags, start_ids, wordpiece_masks, dp_ids, char_embeddings, word2vec_embeddings, fasttext_embeddings = inputs.cuda(), masks.cuda(), tags.cuda(), start_ids.cuda(), wordpiece_masks.cuda(), dp_ids.cuda(), char_embeddings.cuda(), word2vec_embeddings.cuda(), fasttext_embeddings.cuda()
#             feats = model(
#                 word2vec_embeddings)  # fc输出            feats = model(word2vec_embeddings, char_embeddings, masks)  # fc输出
#             # feats输出大小为：[32, 256, 18]，即为[batch_size,seqlength,tagsize]，标签数量为18，
#             # 接下来需要构建预测值与真实值间的损失函数，首先应该使用softmax将当前18个标签展示出来，其中每个tag是以其索引的形式出现的
#
#             # pred_tags=torch.empty(tags.size(0),tags.size(1))           #用于记录每个token的最大标签值，即预测标签值
#             # for i in range(0,tags.size(0)):
#             #     pred_tags[i]=feats[i].argmax(dim=1)
#             # loss=loss_function(pred_tags,tags)
#             # 交叉熵损失函数自带一个softmax功能，因此不需要执行softmax操作，但无法时对一个batch执行损失
#             loss = torch.tensor(0.0).cuda()
#             for i in range(feats.size(0)):
#                 loss += loss_function(feats[i], tags[i])
#
#             loss /= feats.size(0)
#             loss.backward()
#             # # 梯度截断，防止梯度爆炸
#             # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
#
#             optimizer.step()
#             step += 1
#             length += word2vec_embeddings.size(0)
#             accuary, precision, recall, f1, classification_report = val_without_crf(model, feats, tags, config)
#             # if step % 10 == 0:
#             # train_loader.dataset得到数据总样本数  # len(train_loader) 长度为总样本数 / batch_size
#             print(
#                 'Train Epoch: {} [{}/{} ({:.2f}%)]\tloss: {:.6f}\taccuary: {:.6f}\tprecision: {:.6f}\trecall: {:.6f}\tf1: {:.6f}'.format(
#                     epoch, length, len(train_loader.dataset), 100. * step / len(train_loader), loss.item(), accuary,
#                     precision, recall, f1))
#         # 训练一个epoch后就在开发集上进行测试
#         loss_temp, accuary, precision, recall, f1, classification_report = dev_without_crf(model, dev_loader, epoch,
#                                                                                            config)
#
#         # # early_stopping needs the validation loss to check if it has decresed,
#         # # and if it has, it will make a checkpoint of the current model
#         # early_stopping(loss_temp, model)
#         #
#         # if early_stopping.early_stop:
#         #     print("Early stopping")
#         #     break
#
#         # if loss_temp < eval_loss:
#         #     save_model(model, epoch)
#         #     eval_loss = loss_temp
#         #     best_model_metric = [accuary, precision, recall, f1, classification_report]
#         if f1 > eval_f1:
#             save_model(model, epoch)
#             eval_f1 = f1
#             eval_loss = loss_temp
#             best_epoch = epoch
#             best_model_metric = [accuary, precision, recall, f1, classification_report]
#         print("*" * 50)
#         print('best epoch:{}'.format(best_epoch))
#         print('best loss:{}'.format(eval_loss))
#         print("best accuary: ", best_model_metric[0])
#         print("best precision: ", best_model_metric[1])
#         print("best recall: ", best_model_metric[2])
#         print("best f1: ", best_model_metric[3])
#         print("classification report: ")
#         print(best_model_metric[4])

#此val是针对于未使用crf层的
def val_without_crf(model, feats, tags, config):
    model.eval()  # model.eval表示不更新参数
    true = []  # 真实序列
    pred = []  # 预测序列

    pred_tags = torch.empty(tags.size(0), tags.size(1))  # 用于记录每个token的最大标签值，即预测标签值
    for i in range(0, tags.size(0)):
        pred_tags[i] = feats[i].argmax(
            dim=1)  # pred_tag存储的就是预测序列,序列大小为32*256;feats的大小为32*256*18;tags存储的是一个batch的真实序列，序列大小也为32*256

    true.extend([t for t in tags])
    pred.extend([t for t in pred_tags])
    index_to_token = load_vocab_reverse(config.label_file)  # 构建索引值与标签间的映射关系
    y_pred = []  # 用于记录每一个句子的预测标签
    y_true = []  # 用于记录每一个句子的真实标签
    for true_labels, pred_labels in zip(true, pred):  # true_label和pred_label均表示一个句子的序列，为256
        for tl, pl in zip(true_labels, pred_labels):
            y_true.append(index_to_token[int(tl)])
            y_pred.append(index_to_token[int(pl)])
    # print('pred:', y_pred)
    # print('true:', y_true)
    with torch.no_grad():
        model.train()
    return compute_eval_metrics(y_true=[y_true], y_pred=[y_pred])

#此dev是针对于未使用crf层的
def dev_without_crf(model, dev_loader, epoch, config):
    model.eval()  # model.eval表示不更新参数
    eval_loss = 0
    true = []  # 真实序列
    pred = []  # 预测的序列
    length = 0
    loss_function = nn.CrossEntropyLoss()
    for batch in tqdm(dev_loader):
        inputs, masks, tags, start_ids, wordpiece_masks, dp_ids, char_embeddings, word2vec_embedding, fasttext_embeddings = batch  # input为一个batch的数据
        length += inputs.size(0)  # length表示有多少条样本
        inputs, masks, tags, start_ids, wordpiece_masks, dp_ids, char_embeddings, word2vec_embedding, fasttext_embeddings = Variable(
            inputs), Variable(masks), Variable(tags), Variable(start_ids), Variable(wordpiece_masks), Variable(
            dp_ids), Variable(char_embeddings), Variable(word2vec_embedding), Variable(fasttext_embeddings)
        if config.use_cuda:
            inputs, masks, tags, wordpiece_masks, dp_ids, char_embeddings, word2vec_embedding, fasttext_embeddings = inputs.cuda(), masks.cuda(), tags.cuda(), wordpiece_masks.cuda(), dp_ids.cuda(), char_embeddings.cuda(), word2vec_embedding.cuda(), fasttext_embeddings.cuda()
        feats = model(word2vec_embedding)

        loss = torch.tensor(0.0).cuda()
        pred_tags = torch.empty(tags.size(0), tags.size(1))  # 用于记录每个token的最大标签值，即预测标签值

        for i in range(feats.size(0)):
            loss += loss_function(feats[i], tags[i])
            pred_tags[i] = feats[i].argmax(
                dim=1)  # pred_tag存储的就是预测序列,序列大小为32*256;feats的大小为32*256*18;tags存储的是一个batch的真实序列，序列大小也为32*256

        loss /= feats.size(0)

        eval_loss += loss.item() * inputs.size(0)
        pred.extend([t for t in pred_tags])
        true.extend([t for t in tags])

    print('eval  epoch: {}|  loss: {}'.format(epoch, eval_loss / length))  # 输出所有样本的平均损失

    # 计算准确率----不计算padding的值和开始、结尾符号
    # 加载索引到token的字典
    index_to_token = load_vocab_reverse(config.label_file)
    y_pred = []
    y_true = []
    for true_labels, pred_labels in zip(true, pred):
        for tl, pl in zip(true_labels, pred_labels):
            y_true.append(index_to_token[int(tl)])
            y_pred.append(index_to_token[int(pl)])
    # print('y_pred: ', y_pred)
    # print('y_true: ', y_true)

    accuary, precision, recall, f1, classification_report = compute_eval_metrics(y_true=[y_true], y_pred=[y_pred])
    print("accuary: ", accuary)
    print("precision: ", precision)
    print("recall: ", recall)
    print("f1: ", f1)
    print("classification report: ")
    print(classification_report)
    with torch.no_grad():
        model.train()
    return eval_loss / length, accuary, precision, recall, f1, classification_report


#此val是针对于使用了crf层的
def val(model, feats, masks, tags, config):
    model.eval()  # model.eval表示不更新参数
    true = []  # 真实序列
    pred = []  # 预测的序列
    path_score, best_path = model.crf(feats, masks.bool())  # best_path为预测的序列结果
    pred.extend([t for t in best_path])
    true.extend([t for t in tags])
    index_to_token = load_vocab_reverse(config.label_file)
    y_pred = []
    y_true = []
    for true_labels, pred_labels in zip(true, pred):
        for tl, pl in zip(true_labels, pred_labels):
            if index_to_token[tl.item()] == '<start>':
                continue
            if index_to_token[tl.item()] == '<eos>':
                break
            y_true.append(index_to_token[tl.item()])
            y_pred.append(index_to_token[pl.item()])
    # print('pred:', y_pred)
    # print('true:', y_true)
    with torch.no_grad():
        model.train()
    return compute_eval_metrics(y_true=[y_true], y_pred=[y_pred])

#此dev是针对于使用了crf层的
def dev(model, dev_loader, dev_loader_unk, epoch, config):
    model.eval()  # model.eval表示不更新参数
    eval_loss = 0
    true = []  # 真实序列
    pred = []  # 预测的序列
    length = 0
    for batch, batch_unk in tqdm(zip(dev_loader, dev_loader_unk)):
        inputs, masks, wordpiece_masks, tags, start_ids,  dp_ids, char_embeddings, word2vec_embedding, fasttext_embeddings, \
        input_unk, masks_unk, tags_unk = batch  # input为一个batch的数据
        input_unk, masks_unk, tags_unk = Variable(batch_unk[-3]), Variable(batch_unk[-2]), Variable(batch_unk[-1])

        length += inputs.size(0)  # length表示有多少条样本
        inputs, masks, tags, start_ids, wordpiece_masks, dp_ids, char_embeddings, word2vec_embedding, fasttext_embeddings = \
            Variable(inputs), Variable(masks), Variable(tags), Variable(start_ids), Variable(wordpiece_masks),\
            Variable(dp_ids), Variable(char_embeddings), Variable(word2vec_embedding), Variable(fasttext_embeddings)
        if config.use_cuda:
            inputs, masks, tags, wordpiece_masks, dp_ids, char_embeddings, word2vec_embedding, fasttext_embeddings, input_unk, masks_unk, tags_unk = \
                inputs.cuda(), masks.cuda(), tags.cuda(), wordpiece_masks.cuda(), dp_ids.cuda(), char_embeddings.cuda(), \
                word2vec_embedding.cuda(), fasttext_embeddings.cuda(), input_unk.cuda(), masks_unk.cuda(), tags_unk.cuda()
        #所有feats中只要带了mask的，一定是接了crf层的，如果是以input为输入，则
        feats = model(inputs, input_unk, masks_unk, tags_unk, masks, start_ids, wordpiece_masks,
                      char_embeddings)  # fc输出

        #下面这两个feats肯定是使用到了char_embedding的
        # feats = model(word2vec_embedding, masks=masks, wordpiece_masks=wordpiece_masks)
        # feats = model(fasttext_embeddings, char_embeddings, masks)

        path_score, best_path = model.crf(feats, masks.bool())  # best_path为预测的结果
        loss = model.loss(feats, masks, tags)  # 返回每个batch中样本的平均损失
        eval_loss += loss.item() * inputs.size(0)
        pred.extend([t for t in best_path])
        true.extend([t for t in tags])

    print('eval  epoch: {}|  loss: {}'.format(epoch, eval_loss / length))  # 输出所有样本的平均损失

    # 计算准确率----不计算padding的值和开始、结尾符号
    # 加载索引到token的字典
    index_to_token = load_vocab_reverse(config.label_file)
    y_pred = []
    y_true = []
    for true_labels, pred_labels in zip(true, pred):
        for tl, pl in zip(true_labels, pred_labels):
            if index_to_token[tl.item()] == '<start>':
                continue
            if index_to_token[tl.item()] == '<eos>':
                break

            y_true.append(index_to_token[tl.item()])
            y_pred.append(index_to_token[pl.item()])
    # print('y_pred: ', y_pred)
    # print('y_true: ', y_true)

    accuary, precision, recall, f1, classification_report = compute_eval_metrics(y_true=[y_true], y_pred=[y_pred])
    print("accuary: ", accuary)
    print("precision: ", precision)
    print("recall: ", recall)
    print("f1: ", f1)
    print("classification report: ")
    print(classification_report)
    with torch.no_grad():
        model.train()
    return eval_loss / length, accuary, precision, recall, f1, classification_report


if __name__ == '__main__':
    fire.Fire(train)
    # fire.Fire(train_word2vec)  # 命令行带参数启动命令：python -u train.py -gpu=1 -max_length=128
    # fire.Fire(train_fasttext)
    # fire.Fire(train_word2vec_lstm)
    # train()
