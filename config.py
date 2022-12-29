import torch

class Config(object):
    def __init__(self):
        self.label_file = './data/tag.txt'
        self.data_file = 'data/data_1.txt'
        self.predict_data_file = 'data/data.csv'      #原始数据，包含cveid 和 description两个部分
        self.train_data_ratio = 0.9
        self.vocab = './data/bert-base-cased/vocab.txt'
        self.max_length = 256
        self.use_cuda = True if torch.cuda.is_available() else False
        self.gpu = 0
        self.batch_size = 16
        self.bert_path = './data/bert-base-cased'
        self.hidden_dim = 64
        self.bert_embedding = 768          #嵌入维度为768
        self.char_embedding = 150          #char embedding维度为150
        self.dropout1 = 0.1
        self.dropout_ratio = 0.1
        self.lstm_layers = 1
        self.lr = 0.01
        self.lr_decay = 0.001
        self.weight_decay = 0.0005
        self.checkpoint = 'result/'
        self.optim = 'Adam'
        self.load_model = False
        self.load_path = 'epoch_best'
        self.base_epoch = 100
        self.predict = False

    def update(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __str__(self):
        return '\n'.join(['%s:%s' % item for item in self.__dict__.items()])


if __name__ == '__main__':
    con = Config()
    con.update(gpu=8)
    print(con.gpu)
    print(con)
