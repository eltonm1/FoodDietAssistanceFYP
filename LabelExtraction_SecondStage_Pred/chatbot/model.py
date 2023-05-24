from torch import nn
import torch
import torch.nn.functional as F

class Model(nn.Module):

    def __init__(self, vocab_size, embed_dim, num_class, num_tags=3):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, sparse=True)
        # self.lstm = nn.LSTM(embed_dim, 32, num_layers = 2,
        #                     bidirectional=True, dropout = 0.2, batch_first=True)
        # self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=True)
        self.fc = nn.Linear(embed_dim, num_class)

        #the LSTM takens embedded sentence
        self.lstm = nn.LSTM(embed_dim, 64, batch_first=True, bidirectional=True)

        #fc layer transforms the output to give the final output layer
        self.fc2 = nn.Linear(128, num_tags)
        # self.init_weights()

    # def init_weights(self):
    #     initrange = 0.5
    #     self.embedding.weight.data.uniform_(-initrange, initrange)
    #     self.fc.weight.data.uniform_(-initrange, initrange)
    #     self.fc.bias.data.zero_()

    def forward(self, text):
        # embedded = self.embedding(text, offsets)
        x = self.embedding(text)
        x1 = torch.mean(self.fc(x), dim=1)
        x2, _ = self.lstm(x)
        x2 = x2.reshape(-1, x2.shape[2])
        x2 = self.fc2(x2)
        return x1, F.log_softmax(x2, dim=1)  