# -*- coding: utf-8 -*-
# @Time    : 2021-11-09 16:24
# @Author  : lldzyshwjx
import fire
import warnings
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import pandas as pd

from torch.autograd import Variable
from config import Config
from model import BERT_LSTM_CRF
from utils import (load_vocab, read_predict_data, load_model, load_vocab_reverse)
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from tqdm import tqdm  # 注意不要直接 import tqdm

warnings.filterwarnings('ignore', category=UserWarning)

random_seed = 1024
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)

def predict(**kwargs):
    config = Config()
    config.update(**kwargs)
    tokenizer = BertTokenizer.from_pretrained(config.bert_path, do_lower_case=False)
    print('*' * 80)
    print('当前配置文件为:\n', config)
    print('*' * 80)
    if config.use_cuda:
        torch.cuda.set_device(config.gpu)

    print('加载数据中...')
    vocab = load_vocab(config.vocab)  # 得到词表对应的索引映射字典
    label_dic = load_vocab(config.label_file)  # 得到标签到索引的映射字典
    tagset_size = len(label_dic)

    # 进行 wordpiece
    result, data = read_predict_data(config.predict_data_file, max_length=config.max_length, tokenizer=tokenizer)

    ids = torch.LongTensor([temp.input_id for temp in data])
    masks = torch.LongTensor([temp.input_mask for temp in data])
    wordpiece_masks = torch.LongTensor([temp.wordpiece_mask for temp in data])
    tags = torch.LongTensor([temp.label_id for temp in data])
    start_ids = torch.LongTensor([temp.token_start_ids for temp in data])
    dp_ids = torch.LongTensor([temp.dp_ids for temp in data])
    char_embeddings = torch.FloatTensor([temp.char_embeddings for temp in data])  # 因为char_embeddings值为小数，所以需要使用Float

    datasets = TensorDataset(ids, masks, tags, start_ids, wordpiece_masks, dp_ids, char_embeddings)
    data_loader = DataLoader(datasets, shuffle=False, batch_size=config.batch_size)  # shuffle设为False则不打乱顺序

    print('data: {}条'.format(len(data_loader.dataset)))
    print('数据加载完毕!')
    print('*' * 80)

    model = BERT_LSTM_CRF(config.bert_path, tagset_size, config.bert_embedding, config.char_embedding, config.hidden_dim, config.lstm_layers, dropout_ratio=config.dropout_ratio, dropout1=config.dropout1, use_cuda=config.use_cuda, predict=config.predict)
    if config.load_model:
        assert config.load_path is not None
        model = load_model(model, name=config.load_path)
    if config.use_cuda:
        model.cuda()

    # train_loader 长度为总样本数 / batch_size
    preds = []
    sentence_lens = []
    embedding_vecs = []
    for batch in tqdm(data_loader):
        inputs, masks, tags, start_ids, wordpiece_masks, dp_ids, char_embeddings = batch
        inputs, masks, tags, start_ids, wordpiece_masks, dp_ids, char_embeddings = Variable(inputs), Variable(masks), Variable(tags), Variable(start_ids), Variable(wordpiece_masks), Variable(dp_ids), Variable(char_embeddings)
        sentence_lens.extend([sum(list(mask)) for mask in masks])

        if config.use_cuda:
            inputs, masks, tags, start_ids, wordpiece_masks, dp_ids, char_embeddings = inputs.cuda(), masks.cuda(), tags.cuda(), start_ids.cuda(), wordpiece_masks.cuda(), dp_ids.cuda(), char_embeddings.cuda()
        # feats, embedding_vec = model(inputs, masks, start_ids, wordpiece_masks, dp_ids, char_embeddings)
        feats, embedding_vec = model(inputs, masks, start_ids, wordpiece_masks, char_embeddings)
        path_score, best_path = model.crf(feats, masks.bool())
        preds.extend([t for t in best_path])
        embedding_vecs.extend(embedding_vec)

    index_to_token = load_vocab_reverse(config.label_file)
    y_preds = []
    for pred_labels, length in zip(preds, sentence_lens):
        temp = []
        for i in range(1, length - 1):
            # if index_to_token[pred_labels[i].item()] == '<start>':
            #     continue
            # if index_to_token[pred_labels[i].item()] == '<eos>':
            #     break
            temp.append(index_to_token[pred_labels[i].item()])
        y_preds.append(temp)
    # result['cnt_pred'] = pd.Series([len(y) for y in y_preds])
    # result['cnt_tokens'] = pd.Series([len(token) for token in result['tokens'].to_list()])
    # result['embedding_cnt'] = pd.Series([len(w) for w in embedding_vecs])
    # result['embedding_dim'] = pd.Series([len(w[0]) for w in embedding_vecs])

    result['predict_label'] = pd.Series(y_preds)
    result['embedding_vector'] = pd.Series(embedding_vecs)
    result.to_csv('./data/predicts_syntactic_semantic_layers.csv', index=False)

if __name__ == '__main__':
    predict(load_model=True, predict=True)
