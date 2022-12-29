# -*- coding: utf-8 -*-

import torch.nn as nn
from transformers import BertModel
from model import CRF
from torch.autograd import Variable
import torch

torch.cuda.current_device()

class BERT_LSTM_CRF(nn.Module):
    def __init__(self, bert_config, tagset_size, bert_embedding_dim, char_embedding_dim, hidden_dim, lstm_layers,
                 dropout_ratio, dropout1, use_cuda=True, predict=False):
        super(BERT_LSTM_CRF, self).__init__()
        self.predict = predict
        self.embedding_dim = bert_embedding_dim + char_embedding_dim
        self.char_embedding_dim = char_embedding_dim  # 150
        self.bert_embedding_dim = bert_embedding_dim  # 768
        self.hidden_dim = hidden_dim
        self.word_embeds = BertModel.from_pretrained(bert_config)
        self.lstm = nn.LSTM(self.bert_embedding_dim, hidden_dim, num_layers=lstm_layers, bidirectional=True,
                            dropout=dropout_ratio, batch_first=True)

        self.lstm_layers = lstm_layers
        self.dropout1 = nn.Dropout(p=dropout1)
        self.crf = CRF(target_size=tagset_size, average_batch=True, use_cuda=use_cuda)
        self.liner = nn.Linear(hidden_dim * 2, tagset_size + 2)  # 双向LSTM输出为将两个最终的h拼接得到，即维度为隐层维度的2倍
        self.tagset_size = tagset_size  # 20
        self.use_cuda = use_cuda
        self.attention = nn.MultiheadAttention(embed_dim=2 * self.hidden_dim, num_heads=1, dropout=0.1)
        self.bn = nn.BatchNorm1d(2 * self.hidden_dim)
        self.bert_embedding_liner = nn.Linear(bert_embedding_dim, char_embedding_dim)

    def rand_init_hidden(self, batch_size):
        """
        random initialize hidden variable  h0和c0 形状一样
        """
        return Variable(torch.randn(2 * self.lstm_layers, batch_size, self.hidden_dim)), \
               Variable(torch.randn(2 * self.lstm_layers, batch_size, self.hidden_dim))

    def forward(self, sentence, sentence_word, mask_unk=None, tags_unk=None, masks=None, start_ids=None,
                wordpiece_masks=None, char_embeddings=None):
        '''
        当wordpiece_masks为空时表示不进行wordpiece，不需要进行拼接
        args:
            sentence (word_seq_len, batch_size) : word-level representation of sentence
            hidden: initial hidden state

        return:
            crf output (word_seq_len, batch_size, tag_size, tag_size), hidden
        '''
        batch_size = sentence.size(0)
        seq_length = sentence.size(1)  # size为256
        attention_mask = wordpiece_masks if wordpiece_masks[
            0].tolist() else masks
        # the embedding result with wordpiece
        outputs = self.word_embeds(sentence, attention_mask=attention_mask, output_hidden_states=True,
                                   output_attentions=True)  # print(outputs.keys())

        embeds = []

        for i in range(batch_size):
            sentence_embedding = outputs['hidden_states'][-1][i]    # 维度为256*768。使用最后一层
            embed = [sentence_embedding[0].tolist()]                # 加入[CLS]对应的向量，当前维度为1*768，起始位置加进去
            indexs = [index for index in start_ids[i] if
                      index != -1]                                  # 最后一个元素为句子后加的[SEP]的索引，当前indexs里已经添加了[SEP]的索引位置

            # 这里拼接是为了解决bert wordpiece机制导致的词的长度与label不能一一对应的问题
            for j in range(len(indexs) - 1):
                res = sentence_embedding[start_ids[i][j].item(): start_ids[i][j + 1].item()]
                if res.size(0) > 1:                                     # 如果当前索引间的长度差大于1，则表示当前位置为wordpiece后的结果
                    # res = torch.mean(res, dim=0, keepdim=True)          # 拼接方式一：求整个值段的平均值
                    # res = (res[0]+res[-1])/2                              # 拼接方式二：使用左右两边的单词
                    # res = res[0]                                        # 拼接方式三：直接使用最左边的单词来代替
                    res = res[-1]                                       # 拼接方式四：直接使用最后一个位置上的单词代替

                    embed.append(torch.squeeze(res).tolist())           # squeeze用于对1*n的矩阵降低一个维度为n；
                else:
                    embed.append(torch.squeeze(res).tolist())

            embed.append(sentence_embedding[indexs[
                -1]].tolist())  # 加入[SEP]对应的向量，这里embed的size为句子的原始长度以及加上[sep]且包含[cls]，因为[cls]的向量在最上面已经添加进去了

            # sentence_lens.append(len(embed))  # 记录每个样本（句子）的长度（词的个数---这里包括了[CLS]和[SEP]）
            # 填充的词向量设为0

            while len(embed) < seq_length:
                embed.append([0] * self.bert_embedding_dim)
            embeds.append(embed)
        outputs = torch.FloatTensor(embeds)
        if self.use_cuda:
            outputs = outputs.cuda()

        # outputs_word是针对[unk]标签得到的结果
        outputs_word = self.word_embeds(sentence_word, output_hidden_states=True, output_attentions=True)

        '''
            outputs输出说明：
                outputs输出包括四个部分：['last_hidden_state', 'pooler_output', 'hidden_states', 'attentions']
                其中outputs['hidden_states']的size为：13*1*256*768，
                上述表示隐藏层的输出有13层、1个句子、每个句子长度为256、每个token长度为768，如果是批次处理，则第二个位置则为批次大小而不是1
        '''

        sentence_lens = [sum([m.item() for m in mask]) for mask in
                         masks]  # 真实句子长度--包括首尾分隔符,这儿的长度是16，batch_size,每个位置上的值是句子真实长度
        word_len = [sum([m.item() for m in mask]) for mask in mask_unk]
        embeds = []

        for i in range(batch_size):
            sentence_embedding_wordpiece = outputs[i]  # 获取wordpiece的最后一层，将该embedding结果喂入一个LSTM去学习上下文特征
            sentence_embedding_local = outputs_word['hidden_states'][-1][i]  # 获取最后一层作为语义信息，都是针对于[unk]标签的
            sentence_embedding_global = outputs_word['hidden_states'][-5][i]  # 获取第八层作为句法信息，都是针对于[unk]标签的
            q = 1.0
            k = 0.00
            sentence_embedding = q * sentence_embedding_wordpiece + k * sentence_embedding_local + (1 - q - k) * sentence_embedding_global

            embed = []
            for j in range(sentence_lens[i]):
                res = sentence_embedding[j]
                embed.append(torch.squeeze(res).tolist())
            # 填充的词向量设为0
            while len(embed) < seq_length:
                embed.append([0] * self.bert_embedding_dim)
            embeds.append(embed)
        embeds = torch.FloatTensor(embeds)

        hidden = self.rand_init_hidden(batch_size)  # 初始化lstm的输入h0和c0
        if self.use_cuda:
            # hidden = (i.cuda() for i in hidden)  # 该代码在centos上运行会报错TypeError: 'generator' object is not subscriptable
            hidden1 = hidden[0].cuda()
            hidden2 = hidden[1].cuda()
            hidden = (hidden1, hidden2)
            embeds = embeds.cuda()
        # # # 拼接字符级嵌入
        # final_embeds = torch.cat((embeds, char_embeddings), 2)

        # （将padding词向量设为0后）直接使用lstm---与上述对比
        # lstm_out, (hidden, c) = self.lstm(final_embeds, hidden)
        lstm_out, (hidden, c) = self.lstm(embeds,
                                          hidden)  # lstm_out为所有时间步的隐层输出，hidden为最后一个隐层输出，c为c的值（双向所以形状为2*batch_size*dim）
        # lstm的输出包括output,(hn,cn)两个部分，其中output的shape为：seq_len*batch*hidden_size

        # 因为是序列标注问题，故必须使用所有层的隐状态lstm_out
        out = lstm_out.contiguous().view(-1,
                                         self.hidden_dim * 2)  # -1表示在其它维度确定的情况下自动补齐缺失的维度,变换形状输入全连接层对其进行降维，这里将形状转化为全连接层的输出
        d_out = self.dropout1(out)  # 加入dropout层，防止过拟合（dropout加在）
        l_out = self.liner(d_out)  # 这里加入全连接层是为了给每个词的表示降维,降维成label的大小（种类）---因为后续会对每个词进行标注，而每个词可能的情况有label的种类那么多
        feats = l_out.contiguous().view(batch_size, seq_length,
                                        -1)  # 降维后将形状还原contiguous：view只能用在contiguous的variable上。如果在view之前用了transpose, permute等，需要用contiguous()来返回一个contiguous copy。
        # 此处feats的shape为seq_len*batch*tagset_size

        # 一种可能的解释是：有些tensor并不是占用一整块内存，而是由不同的数据块组成，而tensor的view()操作依赖于内存是整块的，这时只需要执行contiguous()这个函数，把tensor变成在内存中连续分布的形式。
        if self.predict:
            # 如果是预测，则返回
            embedding_vector = []
            for i in range(batch_size):
                bert_embedding = embeds[i]
                # char_embedding = char_embeddings[i]
                sample_vec = []
                # for j in range(1, sentence_lens[i] - 1):
                for j in range(sentence_lens[i]):
                    # sample_vec.append(bert_embedding[j].tolist() + char_embedding[j].tolist())
                    sample_vec.append(bert_embedding[j].tolist())
                embedding_vector.append(sample_vec)
            return feats, embedding_vector
        return feats  # batch_size*seq_length*target_dim,包括了起始位置和终止位置

    # 自定义损失函数--得到每个batch的平均损失--即每个样本的损失
    def loss(self, feats, mask, tags):
        """
        feats: size=(batch_size, seq_len, tag_size)
            mask: size=(batch_size, seq_len)
            tags: size=(batch_size, seq_len)
        :return:
        """
        loss_value = self.crf.neg_log_likelihood_loss(feats, mask, tags)
        batch_size = feats.size(0)
        loss_value /= float(batch_size)
        return loss_value
