import torch
import os
import chars2vec
import re
import numpy as np
import pandas as pd
from transformers import BertTokenizer, BertModel
from torch.utils.data import TensorDataset, DataLoader
# from gensim.models import word2vec
# from gensim.models import FastText

from seqeval.metrics import f1_score, recall_score, accuracy_score, classification_report, precision_score

char_embedding_model = chars2vec.load_model('./data/chars2vec/eng_150')
# word2vec_embedding_model = word2vec.Word2Vec.load('./data/nvd-word2vec/pre_training_word2vec.model')   #加载模型
# fasttext_embedding_model= FastText.load('./data/nvd-fasttext/pre_training_fasttext.model')
# description_sentence = [sent.strip("\"") for sent in pd.read_csv('data/data.csv', index_col=False)['description'].to_list()]

class InputFeatures(object):
    def __init__(self, input_id=None, label_id=None, input_mask=None, wordpiece_mask=None, token_start_ids=None, dp_ids=None, char_embeddings=None, word2vec_embeddings=None,fasttext_embeddings=None,input_id_unk=None,input_mask_unk=None,label_id_unk=None):
        self.input_id = input_id if input_id else []
        self.label_id = label_id if label_id else []
        self.input_mask = input_mask if input_mask else []
        self.wordpiece_mask = wordpiece_mask if wordpiece_mask else []
        self.token_start_ids = token_start_ids if token_start_ids else []
        self.dp_ids = dp_ids if dp_ids else []
        self.char_embeddings = char_embeddings if isinstance(char_embeddings, np.ndarray) else np.array([[]])
        self.word2vec_embeddings = word2vec_embeddings if isinstance(word2vec_embeddings, np.ndarray) else np.array([[]])
        self.fasttext_embeddings = fasttext_embeddings if isinstance(fasttext_embeddings, np.ndarray) else np.array([[]])
        self.input_id_unk = input_id_unk if input_id_unk else []
        self.input_mask_unk = input_mask_unk if input_mask_unk else []
        self.label_id_unk = label_id_unk if label_id_unk else []

def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = {}
    index = 0
    with open(vocab_file, "r", encoding="utf-8") as reader:
        while True:
            token = reader.readline()
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab


def load_vocab_reverse(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = {}
    index = 0
    with open(vocab_file, "r", encoding="utf-8") as reader:
        while True:
            token = reader.readline()
            if not token:
                break
            token = token.strip()
            vocab[index] = token
            index += 1
    return vocab


def transform_data(content):
    data, temp = [], []
    content = [c.strip() for c in content] + ['']  # + [''] 是为了解决最后一个样本的问题
    for c in content:
        if c:
            temp.append(c)
        elif temp:
            data.append(temp)
            temp = []
        else:
            temp = []
    return data

def read_corpus_bert(path, max_length, label_dic, tokenizer):
    """
    构造无wordpiece的数据，将不再词典中的词标为[UNK]-----不需要拼接
    :param path:数据文件路径
    :param max_length: 最大长度
    :param label_dic: 标签字典
    :return:
    """
    file = open(path, encoding='utf-8')
    content = file.readlines()
    file.close()
    content = transform_data(content)
    result = []
    for line in content:
        tokens = [e.split('  ')[0] for e in line]
        label = [e.split('  ')[1] for e in line]

        if len(tokens) > max_length - 2:
            tokens = tokens[0:(max_length - 2)]
            label = label[0:(max_length - 2)]

        tokens_f = ['[CLS]'] + tokens + ['[SEP]']
        input_ids = tokenizer.convert_tokens_to_ids(tokens_f)

        label_f = ["<start>"] + label + ["<eos>"]
        label_ids = [label_dic[i] for i in label_f]

        input_mask = [1] * len(input_ids)

        while len(input_ids) < max_length:
            input_ids.append(0)
            input_mask.append(0)
            label_ids.append(label_dic['<pad>'])
        assert len(input_ids) == max_length
        assert len(input_mask) == max_length
        assert len(label_ids) == max_length
        feature = InputFeatures(input_id_unk=input_ids, input_mask_unk=input_mask, label_id_unk=label_ids)
        result.append(feature)
    return result

def read_corpus_bert_wordpiece(path, max_length, label_dic, tokenizer):
    """
    该函数是使用wordpiece后不进行拼接的情况，会引起句子的长度变长
    :param path:数据文件路径
    :param max_length: 最大长度
    :param label_dic: 标签字典
    :return:
    """
    file = open(path, encoding='utf-8')
    content = file.readlines()    #读取数据
    file.close()
    content = transform_data(content)
    result = []
    for i, line in enumerate(content):          #遍历每一条description
        tokens = [e.split('  ')[0] for e in line]      #一条description的所有token，不带起始位和终止位
        label = [e.split('  ')[1] for e in line]       #一条description的所有label

        if len(tokens) > max_length - 2:  # 如果token的长度大于254，则只取前254个，直接进行截断
            tokens = tokens[0:(max_length - 2)]    #原始句子的token
            label = label[0:(max_length - 2)]       #原始句子的label

        subword_tokens = list(map(tokenizer.tokenize, tokens))  # 对tokens中的每个token执行tokenize函数,返回每句话的subwords列表，这是一个二维列表

        tokens_f = ['[CLS]'] + [item for indices in subword_tokens for item in indices] + [
            '[SEP]']  # 加入cls和sep以后才进行padding，将subwords进行拉伸，这里得到的是所有subword_token得到的一维列表。

        # 如果wordpiece后的长度小于或等于max_length，则进行wordpiece，否则实际不进行wordpiece但按照wordpiece对待
        # 仔细思考，其实在网安领域存在很多需要wordpiece的词，而且这也将势必造成句子长度大于256，此时将随机初始化方法和wordpiece结合起来是一个有效的表示
        if len(tokens_f) <= max_length:  # 如果子词的长度大于最大长度
            input_ids = tokenizer.convert_tokens_to_ids(tokens_f)  # 将wordpiece token转化为索引
            subword_lengths = list(map(len, subword_tokens))  # 得到每个词wordpiece的长度

            # 基于cumsum方法对长度进行累加，获取词首index，整体+1，相当于加入了cls标记占位的影响 n + np.cumsum([n1, n2, n3])  ==> [n+n1, n+n1+n2, n+n1+n2+n3], 累加
            token_start_idxs = 1 + np.cumsum([0] + subword_lengths[:-1])  #这里加1的目的是，起始位置加了[cls]，为了保证索引与input_ids一致而加上了1

            token_start_idxs = list(
                np.append(token_start_idxs, len(tokens_f) - 1))  # 加上最后一个元素的索引是为了消除句尾加入的sep的影响，即该列表的最后一个值为[SEP]的索引

        else:
            # 长度超过256时，不需要wordpiece
            tokens_f = ['[CLS]'] + tokens + ['[SEP]']
            input_ids = tokenizer.convert_tokens_to_ids(tokens_f)  # 用上述代码得到的结果是一样的
            token_start_idxs = list(range(1, len(input_ids)))  # 最后一个值为SEP索引
        # 真实序列的mask
        input_mask = [1] * (len(tokens) + 2)  # 加2表示的是对CLS和SEP，表示真实字符的长度
        # wordpiece mask
        wordpiece_mask = [1] * len(input_ids)    # 这儿表示wordpiece后的字符长度

        # label
        label_f = ["<start>"] + label + ['<eos>']

        label_ids = [label_dic[i] for i in label_f]     #真实label的长度

        while len(input_mask) < max_length:    #将input_mask进行扩充至256
            input_mask.append(0)

        while len(input_ids) < max_length:      #将input_id进行扩充至256
            input_ids.append(0)
            wordpiece_mask.append(0)

        # 这里单独写是因为当进行wordpiece后，label与token长度不一致，使得其需要补充的数量也不同
        while len(label_ids) < max_length:
            label_ids.append(label_dic['<pad>'])

        while len(token_start_idxs) < max_length:
            token_start_idxs.append(-1)  # 将填充的元素的索引都设为-1

        assert len(input_ids) == max_length
        assert len(input_mask) == max_length
        assert len(label_ids) == max_length
        assert len(wordpiece_mask) == max_length

        feature = InputFeatures(input_id=input_ids, input_mask=input_mask, label_id=label_ids,
                                token_start_ids=token_start_idxs, wordpiece_mask=wordpiece_mask)
        #这里得到的所有记录长度均为256
        result.append(feature)
    return result  # result对象里包含了上述的值
#
# def read_corpus_bert_wordpiece_concate(path, max_length, label_dic, tokenizer):
#     """
#     :param path:数据文件路径
#     :param max_length: 最大长度
#     :param label_dic: 标签字典
#     :return:
#     """
#     file = open(path, encoding='utf-8')
#     content = file.readlines()    #读取数据
#     file.close()
#     content = transform_data(content)
#     result = []
#     for i, line in enumerate(content):          #遍历每一条description
#         tokens = [e.split('  ')[0] for e in line]      #一条description的所有token，不带起始位和终止位
#         label = [e.split('  ')[1] for e in line]       #一条description的所有label
#
#         if len(tokens) > max_length - 2:  # 如果token的长度大于254，则只取前254个，直接进行截断
#             tokens = tokens[0:(max_length - 2)]    #原始句子的token
#             label = label[0:(max_length - 2)]       #原始句子的label
#
#         # 进行wordpiece
#         subword_tokens = list(map(tokenizer.tokenize, tokens))  # 对tokens中的每个token执行tokenize函数,返回每句话的subwords列表，这是一个二维列表
#
#         tokens_f = ['[CLS]'] + [item for indices in subword_tokens for item in indices] + [
#             '[SEP]']  # 加入cls和sep以后才进行padding，将subwords进行拉伸，这里得到的是所有subword_token得到的一维列表。
#
#         if len(tokens_f) <= max_length:  # 如果子词的长度大于最大长度
#             input_ids = tokenizer.convert_tokens_to_ids(tokens_f)  # 将wordpiece token转化为索引
#             subword_lengths = list(map(len, subword_tokens))  # 得到每个词wordpiece的长度
#
#             # 基于cumsum方法对长度进行累加，获取词首index，整体+1，相当于加入了cls标记占位的影响 n + np.cumsum([n1, n2, n3])  ==> [n+n1, n+n1+n2, n+n1+n2+n3], 累加
#             token_start_idxs = 1 + np.cumsum([0] + subword_lengths[:-1])  #这里加1的目的是，起始位置加了[cls]，为了保证索引与input_ids一致而加上了1
#
#             token_start_idxs = list(
#                 np.append(token_start_idxs, len(tokens_f) - 1))  # 加上最后一个元素的索引是为了消除句尾加入的sep的影响，即该列表的最后一个值为[SEP]的索引
#
#         else:
#             # 长度超过256时，不需要wordpiece
#             tokens_f = ['[CLS]'] + tokens + ['[SEP]']
#             input_ids = tokenizer.convert_tokens_to_ids(tokens_f)  # 用上述代码得到的结果是一样的
#             token_start_idxs = list(range(1, len(input_ids)))  # 最后一个值为SEP索引
#         # 真实序列的mask
#         input_mask = [1] * (len(tokens) + 2)  # 加2表示的是对CLS和SEP，表示真实字符的长度
#         # wordpiece mask
#         wordpiece_mask = [1] * len(input_ids)    # 这儿表示wordpiece后的字符长度
#
#         # label
#         label_f = ["<start>"] + label + ['<eos>']
#
#         label_ids = [label_dic[i] for i in label_f]     #真实label的长度
#
#         while len(input_mask) < max_length:    #将input_mask进行扩充至256
#             input_mask.append(0)
#
#         while len(input_ids) < max_length:      #将input_id进行扩充至256
#             input_ids.append(0)
#             wordpiece_mask.append(0)
#
#         # 这里单独写是因为当进行wordpiece后，label与token长度不一致，使得其需要补充的数量也不同
#         while len(label_ids) < max_length:
#             label_ids.append(label_dic['<pad>'])
#
#         while len(token_start_idxs) < max_length:
#             token_start_idxs.append(-1)  # 将填充的元素的索引都设为-1
#
#         assert len(input_ids) == max_length
#         assert len(input_mask) == max_length
#         assert len(label_ids) == max_length
#         assert len(wordpiece_mask) == max_length
#
#         feature = InputFeatures(input_id=input_ids, input_mask=input_mask, label_id=label_ids,
#                                 token_start_ids=token_start_idxs, wordpiece_mask=wordpiece_mask)
#         #这里得到的所有记录长度均为256
#         result.append(feature)
#     return result  # result对象里包含了上述的值，此时的输出只是所有分词后的结果，而还没有给入到BERT中得到向量。

# word2vec先不管
# def read_corpus_word2vec(path,max_length,label_dic):
#     '''
#
#     :param path: 原始文件位置
#     :param max_length: 句子最大长度
#     :param label_dic: 标签对应字典
#     :return:
#     '''
#     file = open(path, encoding='utf-8')
#     content = file.readlines()
#     file.close()
#     # 转换content使其容易处理
#     content = transform_data(content)    #将按照原数据得到的
#     result = []
#     for line in content:  # 对每一个token序列做处理
#         tokens = [e.split('  ')[0] for e in line]
#         label = [e.split('  ')[1] for e in line]
#         if len(tokens) > max_length:
#             tokens = tokens[0:max_length]
#             label = label[0:max_length]
#         # print(word2vec_embedding_model.wv.key_to_index)    #查看当前word2vec中共有多少个词
#         # print(len(word2vec_embedding_model.wv.key_to_index))
#
#         word2vec_embeddings=word2vec_embedding_model.wv[tokens]     #当前的输出就是一个ndarray格式
#         word2vec_embeddings = np.pad(word2vec_embeddings, ((0, max_length - word2vec_embeddings.shape[0]), (0, 0)),
#                                      'constant', constant_values=0.0)     #将句子词向量扩充为max_length一样的长度
#
#         # char embedding---没有开始符和结束符，只有真实的值
#         char_embeddings = char_embedding_model.vectorize_words(tokens)
#         char_embeddings = np.pad(char_embeddings, ((0, max_length - char_embeddings.shape[0]), (0, 0)),
#                                      'constant', constant_values=0.0)       #将每个token对应字符向量总长度也扩充为max_length一样的长度
#
#         label_ids = [label_dic[i] for i in label]
#         input_mask = [1] * len(tokens)              #用于记录mask的位置，值为1的表示有真实值，值为0的表示为pad值
#         # 将每个句子填充成同等长度,需要对word2vec_embedding进行填充,与此同时还要对标签进行填充
#         while len(label_ids) < max_length:              #将标签长度扩充为max_length，填充位置使用<pad>
#             label_ids.append(label_dic['<pad>'])
#         while len(input_mask) < max_length:
#             input_mask.append(0)
#         assert len(word2vec_embeddings) == max_length
#         assert len(label_ids) == max_length
#         assert len(char_embeddings) == max_length
#         assert len(input_mask) == max_length
#         feature = InputFeatures(word2vec_embeddings=word2vec_embeddings, label_id=label_ids, char_embeddings=char_embeddings, input_mask=input_mask)   #仅返回嵌入向量和标签索引
#         result.append(feature)      #每个返回特征里包含4个值，分别是字符向量、词向量、标签序列、mask位置向量
#     return result

# def read_corpus_fasttext(path,max_length,label_dic):
#     '''
#
#     :param path: 原始文件位置
#     :param max_length: 句子最大长度
#     :param label_dic: 标签对应字典
#     :return:
#     '''
#     file = open(path, encoding='utf-8')
#     content = file.readlines()
#     file.close()
#     # 转换content使其容易处理
#     content = transform_data(content)    #将按照原数据得到的
#     result = []
#     for line in content:  # 对每一个token序列做处理
#         tokens = [e.split('  ')[0] for e in line]
#         label = [e.split('  ')[1] for e in line]
#         if len(tokens) > max_length:
#             tokens = tokens[0:max_length]
#             label = label[0:max_length]
#         # print(word2vec_embedding_model.wv.key_to_index)    #查看当前word2vec中共有多少个词
#         # print(len(word2vec_embedding_model.wv.key_to_index))
#
#         fasttext_embeddings=fasttext_embedding_model.wv[tokens]     #当前的输出就是一个ndarray格式
#
#         fasttext_embeddings = np.pad(fasttext_embeddings, ((0, max_length - fasttext_embeddings.shape[0]), (0, 0)),
#                                      'constant', constant_values=0.0)     #将句子词向量扩充为max_length一样的长度
#
#         # char embedding---没有开始符和结束符，只有真实的值
#         char_embeddings = char_embedding_model.vectorize_words(tokens)
#         char_embeddings = np.pad(char_embeddings, ((0, max_length - char_embeddings.shape[0]), (0, 0)),
#                                      'constant', constant_values=0.0)       #将每个token对应字符向量总长度也扩充为max_length一样的长度
#
#         label_ids = [label_dic[i] for i in label]
#         input_mask = [1] * len(tokens)              #用于记录mask的位置，值为1的表示有真实值，值为0的表示为pad值
#         # 将每个句子填充成同等长度,需要对word2vec_embedding进行填充,与此同时还要对标签进行填充
#         while len(label_ids) < max_length:              #将标签长度扩充为max_length，填充位置使用<pad>
#             label_ids.append(label_dic['<pad>'])
#         while len(input_mask) < max_length:
#             input_mask.append(0)
#         assert len(fasttext_embeddings) == max_length
#         assert len(label_ids) == max_length
#         assert len(char_embeddings) == max_length
#         assert len(input_mask) == max_length
#         feature = InputFeatures(fasttext_embeddings=fasttext_embeddings, label_id=label_ids, char_embeddings=char_embeddings, input_mask=input_mask)   #仅返回嵌入向量和标签索引
#         result.append(feature)      #每个返回特征里包含4个值，分别是字符向量、词向量、标签序列、mask位置向量
#     return result


def read_corpus_word2vec_pretrain(path):
    data=pd.read_csv(path,index_col=False)
    return [spilt_sentence(item) for item in data['description']]

def build_data_loader(train_data, config, random_seed):
    ids = torch.LongTensor([temp.input_id for temp in train_data])
    masks = torch.LongTensor([temp.input_mask for temp in train_data])
    wordpiece_masks = torch.LongTensor([temp.wordpiece_mask for temp in train_data])
    tags = torch.LongTensor([temp.label_id for temp in train_data])
    start_ids = torch.LongTensor([temp.token_start_ids for temp in train_data])
    dp_ids = torch.LongTensor([temp.dp_ids for temp in train_data])
    char_embeddings = torch.FloatTensor([temp.char_embeddings for temp in train_data])
    word2vec_embeddings = torch.FloatTensor([temp.word2vec_embeddings for temp in train_data])
    fasttext_embeddings = torch.FloatTensor([temp.fasttext_embeddings for temp in train_data])
    input_id_unk = torch.LongTensor([temp.input_id_unk for temp in train_data])
    input_mask_unk = torch.LongTensor([temp.input_mask_unk for temp in train_data])
    label_id_unk = torch.LongTensor([temp.label_id_unk for temp in train_data])

    datasets = TensorDataset(ids, masks, wordpiece_masks, tags, start_ids, dp_ids, char_embeddings, word2vec_embeddings,
                             fasttext_embeddings, input_id_unk, input_mask_unk,label_id_unk)

    # 如果需要切分数据--分2个
    length = len(datasets)
    train_length = int(length * config.train_data_ratio)
    dev_length = length - train_length
    train_dataset, dev_dataset = torch.utils.data.random_split(datasets, [train_length, dev_length],
                                                               generator=torch.Generator().manual_seed(random_seed))
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=config.batch_size)
    dev_loader = DataLoader(dev_dataset, shuffle=True, batch_size=config.batch_size)

    return train_loader, dev_loader

def save_model(model, epoch, path='result', **kwargs):
    """
    默认保留所有模型
    :param model: 模型
    :param path: 保存路径
    :param loss: 校验损失
    :param last_loss: 最佳epoch损失
    :param kwargs: every_epoch or best_epoch
    :return:
    """
    # if not os.path.exists(path):
    #     os.mkdir(path)
    if kwargs.get('name', None) is None:
        # cur_time = datetime.datetime.now().strftime('%Y-%m-%d#%H:%M:%S')
        name = 'epoch_{}'.format('best')
        full_name = os.path.join(path, name)
        # full_name = 'epoch_{}.pkl'.format(epoch)
        torch.save(model.state_dict(), full_name)
        print('Saved model at epoch {} successfully'.format(epoch))
        with open('{}/checkpoint'.format(path), 'w') as file:
            file.write('epoch_{}'.format(epoch))
            print('Write to checkpoint')


def load_model(model, path='result', **kwargs):
    if kwargs.get('name', None) is None:
        with open('{}/checkpoint'.format(path)) as file:
            content = file.read().strip()
            name = os.path.join(path, content)
    else:
        name = kwargs['name']
        name = os.path.join(path, name)
    model.load_state_dict(torch.load(name, map_location=lambda storage, loc: storage))
    print('load model {} successfully'.format(name))
    return model


def get_data_length(path):
    """
    获取数据集中的样本数
    :param path:
    :return:
    """
    file = open(path, encoding='utf-8')
    content = file.readlines()
    file.close()
    # 转换content使其容易处理
    content = transform_data(content)
    print(len(content))
    # result = []
    # for line in content:
    #     tokens = [e.split('  ')[0] for e in line]
    #     label = [e.split('  ')[1] for e in line]


def compute_eval_metrics(y_true, y_pred):
    # y_true = ['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O', 'B-PER', 'I-PER']
    # y_pred = ['O', 'O', 'B-MISC', 'I-MISC', 'B-MISC', 'I-MISC', 'O', 'B-PER', 'I-PER']
    # print("accuary: ", accuracy_score(y_true, y_pred))
    # print("precision: ", precision_score(y_true, y_pred))
    # print("recall: ", recall_score(y_true, y_pred))
    # print("f1: ", f1_score(y_true, y_pred))
    # print("classification report: ")
    # print(classification_report(y_true, y_pred))
    return accuracy_score(y_true, y_pred), precision_score(y_true, y_pred), recall_score(y_true, y_pred),  \
           f1_score(y_true, y_pred), classification_report(y_true, y_pred)


# ******************************以下为预测时所需的函数******************************
def spilt_sentence(text):
    re_tokens = [t for t in re.split(r'[\s+]', text.replace('"', '')) if t]  # 判断是否为空时为了处理句尾有空格的情况
    tokens = []
    for token in re_tokens:
        if token[-1] not in [',', '.'] and '(' not in token and ')' not in token:
            tokens.append(token)
            continue
        temp = []
        if token[-1] in [',', '.']:
            temp.append(token[-1])
            token = token[:-1]
        token = token.replace('(', '( ').replace(')', ' )')
        token = [t for t in re.split(r'[\s+]', token) if t]
        tokens.extend(token)
        tokens.extend(temp)
    return tokens


def read_predict_data(path, max_length, tokenizer):
    data = pd.read_csv(path, index_col=False)
    data['tokens'] = data['description'].apply(spilt_sentence)
    content = data['tokens'].to_list()
    result = []
    for i, tokens in enumerate(content):
        # 这里减2是因为后续会添加[CLS]和[SEP]分隔符
        if len(tokens) > max_length - 2:
            tokens = tokens[0:(max_length - 2)]
        # 进行wordpiece
        subword_tokens = list(map(tokenizer.tokenize, tokens))
        tokens_f = ['[CLS]'] + [item for indices in subword_tokens for item in indices] + ['[SEP]']  # 加入cls和sep以后才进行padding
        # 如果wordpiece后的长度小于或等于max_length，则进行wordpiece，否则实际不进行wordpiece但按照wordpiece对待
        if len(tokens_f) <= max_length:
            input_ids = tokenizer.convert_tokens_to_ids(tokens_f)  # 将wordpiece token转化为索引
            subword_lengths = list(map(len, subword_tokens))  # 得到每个词wordpiece的长度
            # 基于cumsum方法对长度进行累加，获取词首index，整体+1，相当于加入了cls标记占位的影响 n + np.cumsum([n1, n2, n3])  ==> [n+n1, n+n1+n2, n+n1+n2+n3], 累加
            token_start_idxs = 1 + np.cumsum([0] + subword_lengths[:-1])  #
            token_start_idxs = list(
                np.append(token_start_idxs, len(tokens_f) - 1))  # 加上最后一个元素的索引是为了消除句尾加入的sep的影响，即该列表的最后一个值为[SEP]的索引
        else:
            # 不需要wordpiece
            tokens_f = ['[CLS]'] + tokens + ['[SEP]']
            # input_ids = [int(vocab[i]) if i in vocab else int(vocab['[UNK]']) for i in tokens_f]
            input_ids = tokenizer.convert_tokens_to_ids(tokens_f)  # 用上述代码得到的结果是一样的
            token_start_idxs = list(range(1, len(input_ids)))  # 最后一个值为SEP索引
        # 真实序列的mask
        input_mask = [1] * (len(tokens) + 2)  # 加2表示的是对CLS和SEP
        # wordpiece mask
        wordpiece_mask = [1] * len(input_ids)


        # # dp信息
        # dp_ids = []
        # if data_type == 'train':
        #     dp_ids = get_spacy_dp_bert_result(train_description_sentence[i], max_length)

        # char embedding---没有开始符和结束符，只有真实的值
        char_embeddings = char_embedding_model.vectorize_words(tokens)
        char_embeddings = np.pad(char_embeddings, ((1, max_length - char_embeddings.shape[0] - 1), (0, 0)), 'constant',
                                 constant_values=(0, 0))
        while len(input_mask) < max_length:
            input_mask.append(0)

        while len(input_ids) < max_length:
            input_ids.append(0)
            wordpiece_mask.append(0)

        while len(token_start_idxs) < max_length:
            token_start_idxs.append(-1)  # 将填充的元素的索引都设为-1

        # while len(dp_ids) < max_length:
        #     dp_ids.append(-1)

        assert len(input_ids) == max_length
        assert len(input_mask) == max_length
        assert len(wordpiece_mask) == max_length
        # assert len(dp_ids) == max_length

        feature = InputFeatures(input_id=input_ids, input_mask=input_mask, label_id=[],
                                token_start_ids=token_start_idxs, wordpiece_mask=wordpiece_mask,
                                # dp_ids=dp_ids,
                                char_embeddings=char_embeddings,
                                )
        result.append(feature)
    return data, result


if __name__ == '__main__':
    pass
    # get_data_length('./data/test_data_zip.txt')
    # l = ['B-VN', 'O', 'O', 'I-VN', 'I-VN', 'I-VN', 'I-VN', 'O', 'I-VN', 'I-VN', 'I-VN', 'I-VN', 'B-VV', 'I-VV', 'I-VV', 'I-VV', 'O', 'B-VAT', 'I-VAT', 'O', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'O', 'I-VAV', 'I-VAV', 'I-VAV', 'O', 'O', 'O', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'O', 'O', 'B-VN', 'O', 'B-VN', 'I-VN', 'O', 'I-VV', 'O', 'B-VAT', 'I-VAT', 'O', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'O', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'O', 'B-VN', 'I-VN', 'I-VN', 'I-VN', 'I-VN', 'O', 'B-VAT', 'I-VAT', 'O', 'I-VR', 'I-VR', 'I-VR', 'O', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'O', 'O', 'O', 'O', 'O', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'O', 'O', 'O', 'O', 'O', 'O', 'B-VN', 'I-VN', 'I-VN', 'I-VN', 'O', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VR', 'I-VR', 'I-VR', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'O', 'B-VN', 'I-VN', 'I-VN', 'I-VN', 'O', 'B-VAT', 'I-VAT', 'O', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VAV', 'I-VAV', 'I-VAV', 'O', 'B-VN', 'I-VN', 'O', 'I-VN', 'I-VN', 'I-VN', 'O', 'B-VAT', 'I-VAT', 'O', 'I-VR', 'I-VR', 'I-VR', 'O', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'O', 'O', 'B-VN', 'I-VRC', 'I-VRC', 'O', 'I-VRC', 'I-VRC', 'I-VRC', 'O', 'O', 'O', 'B-VAT', 'I-VAT', 'O', 'I-VR', 'I-VR', 'I-VR', 'O', 'B-VN', 'I-VN', 'O', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'O', 'O', 'O', 'B-VAT', 'I-VAT', 'O', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'O', 'B-VN', 'I-VN', 'O', 'B-VAT', 'I-VAT', 'I-VR', 'I-VR', 'I-VR', 'O', 'I-VAV', 'I-VAV', 'I-VAV', 'O', 'B-VN', 'I-VN', 'O', 'O', 'B-VN', 'O', 'I-VV', 'O', 'O', 'O', 'B-VAT', 'I-VAT', 'O', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'O', 'B-VN', 'O', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'O', 'O', 'O', 'B-VAT', 'I-VAT', 'O', 'I-VR', 'O', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'O', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'O', 'O', 'I-VN', 'I-VN', 'O', 'B-VN', 'I-VN', 'I-VN', 'I-VN', 'I-VN', 'I-VN', 'I-VN', 'O', 'O', 'I-VAT', 'O', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'O', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'O', 'B-VN', 'I-VN', 'I-VN', 'I-VN', 'O', 'O', 'O', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'O', 'O', 'I-VRC', 'I-VN', 'O', 'O', 'O', 'O', 'I-VP', 'I-VP', 'O', 'I-VN', 'I-VN', 'B-VV', 'I-VV', 'I-VV', 'O', 'O', 'O', 'I-VAT', 'I-VAT', 'O', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VAV', 'I-VAV', 'I-VAV', 'O', 'O', 'B-VN', 'I-VN', 'I-VN', 'I-VN', 'I-VN', 'I-VN', 'I-VN', 'I-VN', 'I-VN', 'I-VV', 'O', 'I-VRC', 'I-VRC', 'O', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', '<eos>', 'O', 'O', 'O', 'O', 'B-VN', 'I-VN', 'I-VN', 'O', 'B-VAT', 'I-VAT', 'O', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'O', 'I-VAV', 'I-VAV', 'I-VAV', 'O', 'B-VN', 'I-VN', 'I-VN', 'O', 'I-VV', 'O', 'B-VAT', 'I-VAT', 'O', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'O', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VR', 'I-VR', 'I-VR', 'O', 'O', 'O', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'O', 'B-VN', 'I-VN', 'O', 'B-VN', 'I-VN', 'O', 'B-VN', 'I-VN', 'I-VN', 'I-VN', 'O', 'O', 'B-VAT', 'I-VAT', 'O', 'I-VR', 'I-VR', 'I-VR', 'O', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'O', 'B-VN', 'I-VN', 'I-VN', 'I-VN', 'O', 'B-VN', 'O', 'I-VN', 'I-VN', 'I-VN', 'O', 'O', 'I-VRC', 'I-VRC', 'I-VRC', 'O', 'B-VN', 'O', 'O', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'O', 'O', 'O', 'B-VAT', 'I-VAT', 'O', 'I-VR', 'I-VR', 'I-VR', 'O', 'O', 'O', 'O', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'O', 'B-VN', 'I-VN', 'O', 'B-VN', 'I-VN', 'I-VN', 'O', 'O', 'O', 'B-VAT', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'O', 'O', 'B-VN', 'I-VP', 'O', 'O', 'B-VN', 'I-VN', 'O', 'B-VN', 'B-VV', 'I-VV', 'I-VV', 'O', 'I-VV', 'I-VV', 'I-VV', 'I-VV', 'I-VV', 'I-VV', 'I-VV', 'I-VV', 'I-VV', 'I-VV', 'O', 'I-VV', 'I-VAV', 'I-VAV', 'O', 'O', 'I-VRC', 'O', 'O', 'O', 'B-VAT', 'I-VAT', 'O', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VAV', 'I-VAV', 'I-VAV', 'O', 'O', 'O', 'O', 'O', 'I-VAV', 'I-VAV', 'O', 'O', 'O', 'B-VN', 'I-VN', 'I-VN', 'I-VN', 'O', 'I-VRC', 'I-VRC', 'I-VRC', 'O', 'O', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'O', 'O', 'O', 'O', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'O', 'O', 'B-VN', 'I-VN', 'O', 'O', 'O', 'B-VN', 'I-VN', 'O', 'B-VV', 'I-VV', 'I-VV', 'O', 'I-VV', 'I-VV', 'O', 'O', 'B-VAT', 'I-VAT', 'O', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VAV', 'I-VAV', 'I-VAV', 'O', 'B-VN', 'O', 'B-VN', 'I-VN', 'O', 'B-VN', 'I-VN', 'I-VN', 'B-VV', 'O', 'I-VV', 'O', 'B-VAT', 'I-VAT', 'O', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'O', 'O', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'O', 'B-VN', 'I-VN', 'I-VN', 'I-VN', 'O', 'I-VN', 'I-VN', 'I-VN', 'I-VN', 'I-VN', 'O', 'O', 'O', 'O', 'I-VV', 'O', 'O', 'O', 'O', 'I-VAV', 'I-VAV', 'O', 'I-VRC', 'I-VR', 'O', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'O', 'O', 'O', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'O', 'O', 'I-VR', 'I-VR', 'O', 'I-VR', 'I-VR', 'I-VAV', 'I-VAV', 'I-VAV', 'O', 'O', 'B-VN', 'I-VN', 'O', 'I-VRC', 'I-VAV', 'I-VAV', 'O', 'I-VV', 'O', 'O', 'I-VAV', 'I-VAV', 'O', 'O', 'O', 'B-VAT', 'I-VAT', 'O', 'I-VR', 'I-VR', 'I-VAV', 'I-VAV', 'I-VAV', 'O', 'B-VN', 'O', 'B-VN', 'O', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'O', 'O', 'O', 'B-VAT', 'I-VAT', 'O', 'I-VR', 'I-VR', 'I-VR', 'O', 'I-VAV', 'I-VAV', 'I-VAV', 'O', 'B-VN', 'I-VN', 'I-VN', 'I-VN', 'I-VN', 'I-VN', 'O', 'B-VAT', 'I-VAT', 'O', 'I-VR', 'I-VR', 'I-VR', 'O', 'B-VN', 'I-VN', 'O', 'B-VN', 'I-VN', 'O', 'O', 'B-VAT', 'I-VAT', 'O', 'I-VR', 'I-VR', 'O', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'O', 'O', 'B-VN', 'I-VN', 'I-VN', 'O', 'O', 'O', 'O', 'B-VN', 'I-VN', 'O', 'O', 'O', 'B-VAT', 'I-VAT', 'O', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VAV', 'I-VAV', 'I-VAV', 'O', 'B-VN', 'I-VN', 'O', 'B-VN', 'I-VN', 'I-VN', 'O', 'B-VAT', 'I-VAT', 'O', 'I-VR', 'I-VR', 'O', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'O', 'B-VN', 'I-VN', 'O', 'B-VN', 'O', 'B-VN', 'B-VV', 'I-VV', 'I-VV', 'O', 'B-VAT', 'I-VAT', 'O', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'O', 'B-VN', 'I-VN', 'O', 'O', 'I-VN', 'O', 'O', 'O', 'B-VN', 'I-VN', 'I-VN', 'I-VN', 'I-VN', 'I-VN', 'I-VN', 'I-VN', 'O', 'O', 'I-VRC', 'I-VRC', 'O', 'I-VAV', 'O', 'O', 'O', 'I-VR', 'O', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'O', 'O', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'O', 'O', 'O', 'O', 'I-VAT', 'O', 'I-VR', 'I-VR', 'I-VR', 'O', 'B-VN', 'I-VN', 'I-VN', 'O', 'I-VN', 'I-VN', 'O', 'I-VV', 'O', 'I-VV', 'O', 'O', 'O', 'B-VAT', 'I-VAT', 'O', 'I-VR', 'I-VR', 'O', 'I-VAV', 'O', 'B-VN', 'O', 'O', 'B-VN', 'O', 'O', 'B-VN', 'I-VN', 'I-VN', 'O', 'B-VAT', 'I-VAT', 'O', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'O', 'B-VN', 'I-VN', 'O', 'I-VN', 'O', 'I-VN', 'I-VN', 'I-VN', 'I-VN', 'O', 'I-VAT', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'O', 'B-VN', 'O', 'O', 'O', 'B-VN', 'I-VN', 'B-VV', 'I-VV', 'I-VV', 'I-VV', 'O', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'O', 'O', 'O', 'O', 'I-VR', 'I-VR', 'I-VR', 'O', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'O', 'B-VN', 'I-VN', 'I-VN', 'O', 'B-VAT', 'I-VAT', 'O', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'O', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'O', 'O', 'O', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VN', 'O', 'O', 'B-VN', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VAV', 'I-VAV', 'I-VAV', 'O', 'O', 'O', 'I-VAT', 'O', 'I-VR', 'I-VR', 'I-VAV', 'I-VAV', 'O', '<start>', 'O', 'I-VN', 'I-VN', 'O', 'B-VN', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-VN', 'I-VN', 'I-VN', 'I-VV', 'I-VV', 'O', 'I-VV', 'O', 'B-VAT', 'I-VAT', 'I-VAT', 'O', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'O', 'B-VN', 'I-VN', 'I-VN', 'O', 'O', 'B-VN', 'I-VP', 'O', 'B-VN', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'O', 'O', 'O', 'O', 'I-VAT', 'O', 'I-VAV', 'I-VAV', 'I-VAV', 'O', 'B-VN', 'O', 'B-VN', 'O', 'B-VAT', 'I-VAT', 'O', 'I-VR', 'I-VR', 'O', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'O', 'O', 'I-VN', 'I-VN', 'I-VP', 'O', 'B-VN', 'I-VP', 'I-VN', 'O', 'I-VN', 'I-VN', 'I-VN', 'I-VN', 'O', 'B-VAT', 'I-VAT', 'O', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'O', 'I-VAV', 'I-VAV', 'I-VAV', 'O', 'B-VN', 'I-VN', 'O', 'B-VN', 'I-VN', 'O', 'B-VN', 'I-VN', 'I-VN', 'I-VN', 'I-VN', 'I-VN', 'I-VN', 'I-VN', 'B-VV', 'O', 'B-VV', 'I-VV', 'I-VV', 'I-VV', 'I-VN', 'I-VV', 'O', 'B-VAT', 'I-VAT', 'O', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'O', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'O', 'B-VN', 'I-VN', 'I-VN', 'B-VV', 'I-VV', 'I-VV', 'O', 'I-VV', 'I-VV', 'I-VV', 'O', 'I-VV', 'I-VV', 'I-VV', 'I-VV', 'I-VV', 'O', 'I-VV', 'I-VV', 'I-VV', 'I-VV', 'O', 'O', 'I-VV', 'I-VV', 'I-VV', 'I-VV', 'O', 'B-VAT', 'I-VAT', 'O', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'O', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'O', 'B-VN', 'I-VN', 'I-VN', 'O', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'O', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'O', 'O', 'O', 'O', 'B-VAT', 'I-VAT', 'O', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'O', 'B-VN', 'I-VN', 'O', 'B-VN', 'I-VN', 'I-VN', 'I-VN', 'O', 'B-VN', 'O', 'O', 'O', 'I-VAT', 'O', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'O', 'O', 'B-VN', 'I-VN', 'I-VN', 'O', 'O', 'B-VAT', 'I-VAT', 'O', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VAV', 'I-VAV', 'I-VAV', 'O', 'B-VN', 'I-VN', 'I-VN', 'O', 'B-VAT', 'I-VAT', 'O', 'I-VR', 'I-VR', 'I-VR', 'I-VAV', 'I-VAV', 'I-VAV', 'O', 'B-VN', 'I-VRC', 'I-VRC', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'O', 'I-VAV', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VAV', 'O']
    # l1 = ['B-VP', 'I-VP', 'I-VP', 'I-VP', 'I-VP', 'I-VP', 'I-VP', 'O', 'B-VN', 'I-VN', 'I-VN', 'B-VV', 'I-VV', 'I-VV', 'I-VV', 'I-VV', 'O', 'B-VAT', 'I-VAT', 'O', 'B-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'O', 'B-VAV', 'I-VAV', 'I-VAV', 'O', 'O', 'O', 'B-VT', 'I-VT', 'I-VT', 'I-VT', 'I-VT', 'I-VT', 'I-VT', 'O', 'O', 'B-VP', 'O', 'B-VN', 'B-VV', 'O', 'B-VV', 'O', 'B-VAT', 'I-VAT', 'O', 'B-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'O', 'B-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'O', 'B-VN', 'I-VN', 'I-VN', 'I-VN', 'B-VV', 'O', 'B-VAT', 'I-VAT', 'O', 'B-VR', 'I-VR', 'I-VR', 'O', 'B-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-VP', 'I-VP', 'I-VP', 'I-VP', 'O', 'B-VN', 'I-VN', 'B-VV', 'B-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'O', 'B-VN', 'B-VV', 'B-VP', 'I-VP', 'O', 'B-VAT', 'I-VAT', 'O', 'B-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'O', 'B-VAV', 'I-VAV', 'I-VAV', 'O', 'B-VP', 'I-VP', 'O', 'B-VN', 'I-VN', 'B-VV', 'O', 'B-VAT', 'I-VAT', 'O', 'B-VR', 'I-VR', 'I-VR', 'O', 'B-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'O', 'O', 'B-VN', 'B-VP', 'I-VP', 'B-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'O', 'O', 'O', 'B-VAT', 'I-VAT', 'O', 'B-VR', 'I-VR', 'I-VR', 'O', 'B-VN', 'B-VV', 'B-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'O', 'B-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'O', 'O', 'O', 'B-VAT', 'I-VAT', 'O', 'B-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'O', 'B-VN', 'B-VP', 'O', 'B-VAT', 'I-VAT', 'O', 'B-VR', 'I-VR', 'O', 'B-VAV', 'I-VAV', 'I-VAV', 'O', 'B-VN', 'I-VN', 'O', 'O', 'B-VC', 'I-VC', 'I-VC', 'I-VC', 'I-VC', 'O', 'B-VAT', 'I-VAT', 'O', 'B-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'O', 'B-VAV', 'I-VAV', 'I-VAV', 'O', 'B-VN', 'B-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'O', 'O', 'O', 'B-VAT', 'I-VAT', 'O', 'B-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'O', 'B-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'O', 'O', 'B-VT', 'I-VT', 'O', 'B-VN', 'I-VN', 'I-VN', 'I-VN', 'B-VP', 'I-VP', 'B-VV', 'O', 'O', 'B-VAT', 'O', 'B-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'O', 'B-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'O', 'B-VN', 'O', 'B-VP', 'I-VP', 'O', 'O', 'O', 'O', 'O', 'B-VT', 'I-VT', 'I-VT', 'O', 'O', 'B-VRC', 'I-VRC', 'I-VRC', 'O', 'O', 'O', 'B-VP', 'I-VP', 'O', 'B-VN', 'I-VN', 'I-VN', 'B-VV', 'I-VV', 'O', 'O', 'O', 'B-VAT', 'I-VAT', 'O', 'B-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'O', 'O', 'B-VN', 'I-VN', 'I-VN', 'I-VN', 'I-VN', 'I-VN', 'B-VV', 'I-VV', 'O', 'B-VN', 'B-VRC', 'I-VRC', 'I-VRC', 'O', 'B-VP', 'I-VP', 'I-VP', 'I-VP', 'O', 'O', 'B-VRC', 'I-VRC', 'O', 'B-VN', 'I-VN', 'I-VN', 'O', 'B-VAT', 'I-VAT', 'O', 'B-VR', 'I-VR', 'I-VR', 'I-VR', 'O', 'B-VAV', 'I-VAV', 'I-VAV', 'O', 'B-VN', 'I-VN', 'B-VV', 'I-VV', 'I-VV', 'O', 'B-VAT', 'I-VAT', 'O', 'B-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'O', 'B-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-VT', 'I-VT', 'O', 'B-VP', 'I-VP', 'O', 'B-VN', 'I-VN', 'I-VN', 'B-VV', 'O', 'B-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'O', 'B-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'O', 'B-VN', 'I-VN', 'I-VN', 'I-VN', 'I-VN', 'I-VN', 'I-VN', 'I-VN', 'B-VV', 'I-VV', 'O', 'B-VV', 'I-VV', 'I-VV', 'I-VV', 'I-VV', 'I-VV', 'O', 'B-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'O', 'O', 'O', 'B-VAT', 'I-VAT', 'O', 'B-VR', 'I-VR', 'I-VR', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-VT', 'I-VT', 'O', 'B-VN', 'O', 'B-VP', 'I-VP', 'O', 'B-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'O', 'O', 'B-VP', 'I-VP', 'I-VP', 'I-VP', 'I-VP', 'I-VP', 'O', 'B-VN', 'I-VN', 'B-VV', 'I-VV', 'O', 'O', 'B-VV', 'I-VV', 'I-VV', 'I-VV', 'O', 'B-VV', 'I-VV', 'I-VV', 'B-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'O', 'O', 'O', 'B-VAT', 'I-VAT', 'O', 'B-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'O', 'B-VAV', 'I-VAV', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-VN', 'I-VN', 'I-VN', 'B-VV', 'B-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'O', 'O', 'B-VP', 'I-VP', 'I-VP', 'I-VP', 'I-VP', 'O', 'O', 'O', 'O', 'B-VAT', 'I-VAT', 'O', 'B-VR', 'I-VR', 'O', 'O', 'B-VP', 'I-VP', 'I-VP', 'O', 'O', 'B-VN', 'I-VN', 'I-VN', 'I-VN', 'O', 'B-VN', 'I-VN', 'I-VN', 'I-VN', 'O', 'O', 'B-VAT', 'I-VAT', 'O', 'B-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'O', 'B-VAV', 'I-VAV', 'I-VAV', 'O', 'O', 'O', 'B-VN', 'B-VV', 'O', 'B-VN', 'I-VN', 'I-VN', 'B-VV', 'I-VV', 'I-VV', 'O', 'B-VAT', 'I-VAT', 'O', 'B-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'O', 'B-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'O', 'B-VP', 'O', 'B-VN', 'B-VV', 'O', 'B-VN', 'I-VN', 'I-VN', 'I-VN', 'B-VV', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'O', 'B-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'O', 'O', 'B-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'O', 'O', 'B-VAT', 'I-VAT', 'O', 'B-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'O', 'O', 'B-VN', 'I-VN', 'B-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'O', 'O', 'B-VP', 'I-VP', 'O', 'O', 'O', 'B-VAT', 'I-VAT', 'O', 'B-VR', 'I-VR', 'O', 'B-VAV', 'I-VAV', 'O', 'B-VN', 'I-VN', 'I-VN', 'B-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'O', 'O', 'O', 'B-VAT', 'I-VAT', 'O', 'B-VR', 'I-VR', 'I-VR', 'O', 'B-VAV', 'I-VAV', 'I-VAV', 'O', 'B-VT', 'O', 'B-VN', 'I-VN', 'B-VP', 'I-VP', 'O', 'B-VAT', 'I-VAT', 'O', 'B-VR', 'I-VR', 'I-VR', 'O', 'B-VT', 'I-VT', 'O', 'B-VN', 'B-VP', 'I-VP', 'O', 'B-VAT', 'I-VAT', 'O', 'B-VR', 'I-VR', 'O', 'B-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'O', 'O', 'B-VN', 'B-VP', 'I-VP', 'I-VP', 'O', 'B-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'O', 'O', 'O', 'B-VAT', 'I-VAT', 'O', 'B-VR', 'I-VR', 'O', 'B-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'O', 'B-VT', 'I-VT', 'O', 'B-VN', 'I-VN', 'B-VV', 'O', 'B-VAT', 'I-VAT', 'O', 'B-VR', 'I-VR', 'O', 'B-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'O', 'B-VT', 'I-VT', 'O', 'B-VP', 'O', 'B-VN', 'B-VV', 'I-VV', 'I-VV', 'O', 'B-VAT', 'I-VAT', 'O', 'B-VR', 'I-VR', 'I-VR', 'O', 'B-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'O', 'B-VP', 'I-VP', 'I-VP', 'I-VP', 'I-VP', 'I-VP', 'O', 'O', 'B-VN', 'I-VN', 'I-VN', 'I-VN', 'I-VN', 'I-VN', 'O', 'B-VV', 'B-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'O', 'O', 'O', 'O', 'B-VAT', 'O', 'B-VR', 'I-VR', 'I-VR', 'O', 'B-VN', 'I-VN', 'I-VN', 'B-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'O', 'O', 'O', 'B-VAT', 'I-VAT', 'O', 'B-VR', 'I-VR', 'O', 'B-VAV', 'O', 'B-VT', 'O', 'O', 'B-VP', 'I-VP', 'O', 'B-VN', 'I-VN', 'B-VV', 'B-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'O', 'B-VT', 'I-VT', 'O', 'B-VN', 'I-VN', 'I-VN', 'I-VN', 'O', 'B-VV', 'O', 'B-VAT', 'I-VAT', 'I-VAT', 'I-VAT', 'I-VAT', 'O', 'B-VR', 'I-VR', 'O', 'B-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'O', 'B-VP', 'O', 'O', 'O', 'B-VN', 'I-VN', 'B-VV', 'O', 'O', 'B-VV', 'B-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'O', 'O', 'O', 'B-VAT', 'I-VAT', 'I-VAT', 'I-VAT', 'O', 'B-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'O', 'B-VN', 'I-VN', 'B-VV', 'O', 'B-VAT', 'I-VAT', 'O', 'B-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'O', 'B-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'O', 'B-VP', 'I-VP', 'I-VP', 'O', 'O', 'B-VP', 'I-VP', 'I-VP', 'I-VP', 'I-VP', 'I-VP', 'I-VP', 'O', 'B-VN', 'B-VV', 'B-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'O', 'O', 'O', 'B-VAT', 'O', 'B-VR', 'I-VR', 'I-VR', 'I-VR', 'O', 'B-VT', 'I-VT', 'I-VT', 'I-VT', 'O', 'B-VT', 'O', 'O', 'O', 'O', 'B-VP', 'I-VP', 'O', 'O', 'B-VN', 'I-VN', 'I-VN', 'B-VV', 'I-VV', 'O', 'B-VN', 'O', 'B-VAT', 'I-VAT', 'I-VAT', 'O', 'B-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'O', 'B-VAV', 'I-VAV', 'I-VAV', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B-VN', 'B-VV', 'B-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'O', 'O', 'O', 'O', 'B-VAT', 'O', 'B-VR', 'I-VR', 'I-VR', 'O', 'B-VP', 'O', 'B-VN', 'O', 'B-VAT', 'I-VAT', 'O', 'B-VR', 'I-VR', 'O', 'B-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'O', 'O', 'B-VP', 'I-VP', 'I-VP', 'I-VP', 'I-VP', 'I-VP', 'I-VP', 'O', 'B-VN', 'I-VN', 'B-VV', 'I-VV', 'O', 'B-VAT', 'I-VAT', 'O', 'B-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'O', 'B-VAV', 'I-VAV', 'I-VAV', 'O', 'B-VT', 'I-VT', 'O', 'B-VP', 'I-VP', 'O', 'B-VN', 'I-VN', 'I-VN', 'I-VN', 'I-VN', 'I-VN', 'I-VN', 'O', 'B-VV', 'I-VV', 'I-VV', 'O', 'B-VN', 'I-VN', 'I-VN', 'B-VV', 'O', 'B-VAT', 'I-VAT', 'O', 'B-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'O', 'O', 'B-VR', 'I-VR', 'I-VR', 'I-VR', 'O', 'O', 'B-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'O', 'B-VN', 'I-VN', 'B-VV', 'I-VV', 'I-VV', 'I-VV', 'O', 'B-VV', 'I-VV', 'I-VV', 'O', 'B-VV', 'I-VV', 'I-VV', 'I-VV', 'I-VV', 'O', 'B-VV', 'I-VV', 'I-VV', 'I-VV', 'O', 'O', 'B-VV', 'I-VV', 'I-VV', 'I-VV', 'O', 'B-VAT', 'I-VAT', 'O', 'B-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'O', 'B-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'O', 'B-VN', 'I-VN', 'B-VV', 'B-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'I-VRC', 'O', 'O', 'O', 'O', 'B-VAT', 'I-VAT', 'O', 'B-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'O', 'B-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'O', 'B-VT', 'I-VT', 'O', 'B-VN', 'B-VP', 'I-VP', 'I-VP', 'I-VP', 'I-VP', 'I-VP', 'I-VP', 'O', 'B-VAT', 'O', 'B-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'O', 'B-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'O', 'O', 'B-VN', 'B-VP', 'I-VP', 'I-VP', 'O', 'B-VAT', 'I-VAT', 'O', 'B-VR', 'I-VR', 'I-VR', 'I-VR', 'O', 'B-VAV', 'I-VAV', 'I-VAV', 'O', 'B-VN', 'B-VP', 'I-VP', 'O', 'B-VAT', 'I-VAT', 'O', 'B-VR', 'I-VR', 'I-VR', 'O', 'B-VAV', 'I-VAV', 'O', 'B-VN', 'B-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'I-VR', 'O', 'B-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'I-VAV', 'O']
    # l = [l]
    # l1 = [l1]
    # print(len(l), len(l1))
    # a1, a2, a3, a4, a5 = compute_eval_metrics(y_true=l, y_pred=l1)
    # print(a5)
    #
    # l2 = [['B-VN', 'O', 'O', 'I-VN', 'I-VN', 'I-VN', 'I-VN', 'O', 'I-VN', 'I-VN', 'I-VN']]
    # l3 = [['B-VP', 'I-VP', 'I-VP', 'I-VP', 'I-VP', 'I-VP', 'I-VP', 'O', 'B-VN', 'I-VN', 'I-VN']]
    # a1, a2, a3, a4, a5 = compute_eval_metrics(y_true=l2, y_pred=l3)
    # print(a5)
    # res = get_spacy_dp_bert_result('I am a man')
    # print(res)

