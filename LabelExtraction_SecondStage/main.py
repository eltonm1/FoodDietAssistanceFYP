import sys
from itertools import groupby
import torch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
import torchvision.transforms.functional as TF
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib as mpl
import matplotlib.pyplot as plt
from model import CRNN
from dataset import LabelDataset, collate_fn
device = torch.device('cuda:2' if torch.cuda.is_available() else "cpu")

class CTCLabelConverter(object):
    """ Convert between text-label and text-index """

    def __init__(self, character):
        # character (str): set of the possible characters.
        dict_character = list(character)

        self.dict = {}
        for i, char in enumerate(dict_character):
            # NOTE: 0 is reserved for 'CTCblank' token required by CTCLoss
            self.dict[char] = i + 1

        self.character = ['[CTCblank]'] + dict_character  # dummy '[CTCblank]' token for CTCLoss (index 0)

    def encode(self, text, batch_max_length=25):
        """convert text-label into text-index.
        input:
            text: text labels of each image. [batch_size]
            batch_max_length: max length of text label in the batch. 25 by default
        output:
            text: text index for CTCLoss. [batch_size, batch_max_length]
            length: length of each text. [batch_size]
        """
        length = [len(s) for s in text]

        # The index used for padding (=0) would not affect the CTC loss calculation.
        batch_text = torch.LongTensor(len(text), batch_max_length).fill_(0)
        for i, t in enumerate(text):
            text = list(t)
            text = [self.dict[char] for char in text]
            batch_text[i][:len(text)] = torch.LongTensor(text)
        return (batch_text.to(device), torch.IntTensor(length).to(device))

    def decode(self, text_index, length):
        """ convert text-index into text-label. """
        texts = []
        for index, l in enumerate(length):
            t = text_index[index, :]

            char_list = []
            for i in range(l):
                if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):  # removing repeated characters and blank.
                    char_list.append(self.character[t[i]])
            text = ''.join(char_list)

            texts.append(text)
        return texts

class Trainer():

    def __init__(self, saved_weight = None) -> None:
        self.total_epochs = 40
        self.saved_weight = saved_weight
        self.image_channel = 1
        self.batch_size = 32
        self.CTCLabelConverter = CTCLabelConverter(character = LabelDataset.CHARS)

    def train(self):

        image_height = 32
        
        

        label_dataset = LabelDataset()
        # train_set, val_set = torch.utils.data.random_split(label_dataset,
        #                                                 [int(len(label_dataset) * 0.8), int(len(label_dataset) * 0.2)])


        train_loader = torch.utils.data.DataLoader(label_dataset, batch_size=self.batch_size, shuffle=True, pin_memory=True, collate_fn=collate_fn, num_workers=4)
        # val_loader = torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=True)

        num_classes = len(LabelDataset.LABEL2CHAR) + 1
        model = CRNN(self.image_channel, image_height, 0, num_classes).to(device)
        if self.saved_weight is not None:
            model.load_state_dict(torch.load(self.saved_weight, map_location=device))

        criterion = nn.CTCLoss(reduction='mean', zero_infinity=True).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # ================================================ TRAINING MODEL ======================================================
        for epoch in range(self.total_epochs):
            # ============================================ TRAINING ============================================================
            train_correct = 0
            train_total = 0
            for train_data in (pbar := tqdm(train_loader,
                                        position=0, leave=True,
                                        file=sys.stdout)):
                model.train()
                # x_train, y_train, y_train_length = train_data
                x_train, labels = train_data
                batch_size = x_train.shape[0]  # x_train.shape == torch.Size([64, 28, 140])
                text, length = self.CTCLabelConverter.encode(labels, batch_max_length=40)

                preds = model(x_train.to(device))
                preds_size = torch.IntTensor([preds.size(1)] * batch_size)
                preds = preds.log_softmax(2).permute(1, 0, 2)

                loss = criterion(preds, text, preds_size, length)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5) # gradient clipping with 5
                optimizer.step()
                pbar.set_description("Loss: {}".format(loss.item()))
            torch.save(model.state_dict(), 'model.pt')
        # _, max_index = torch.max(y_pred, dim=2)  # max_index.shape == torch.Size([32, 64])
        # for i in range(batch_size):
        #     raw_prediction = list(max_index[:, i].detach().cpu().numpy())  # len(raw_prediction) == 32
        #     prediction = torch.IntTensor([c for c, _ in groupby(raw_prediction) if c != blank_label])
        #     if len(prediction) == len(y_train[i]) and torch.all(prediction.eq(y_train[i])):
        #         train_correct += 1
        #     train_total += 1
    # print('TRAINING. Correct: ', train_correct, '/', train_total, '=', train_correct / train_total)

    # ============================================ VALIDATION ==========================================================
    # val_correct = 0
    # val_total = 0
    # for x_val, y_val in tqdm(val_loader,
    #                          position=0, leave=True,
    #                          file=sys.stdout, bar_format="{l_bar}%s{bar}%s{r_bar}"):
    #     batch_size = x_val.shape[0]
    #     x_val = x_val.view(x_val.shape[0], 1, x_val.shape[1], x_val.shape[2])
    #     y_pred = model(x_val)
    #     y_pred = y_pred.permute(1, 0, 2)
    #     input_lengths = torch.IntTensor(batch_size).fill_(cnn_output_width)
    #     target_lengths = torch.IntTensor([len(t) for t in y_val])
    #     criterion(y_pred, y_val, input_lengths, target_lengths)
    #     _, max_index = torch.max(y_pred, dim=2)
    #     for i in range(batch_size):
    #         raw_prediction = list(max_index[:, i].detach().cpu().numpy())
    #         prediction = torch.IntTensor([c for c, _ in groupby(raw_prediction) if c != blank_label])
    #         if len(prediction) == len(y_val[i]) and torch.all(prediction.eq(y_val[i])):
    #             val_correct += 1
    #         val_total += 1
    
    # print('TESTING. Correct: ', val_correct, '/', val_total, '=', val_correct / val_total)
    
# ============================================ TESTING =================================================================
# number_of_test_imgs = 10
# test_loader = torch.utils.data.DataLoader(val_set, batch_size=number_of_test_imgs, shuffle=True)
# test_preds = []
# (x_test, y_test) = next(iter(test_loader))
# y_pred = model(x_test.view(x_test.shape[0], 1, x_test.shape[1], x_test.shape[2]))
# y_pred = y_pred.permute(1, 0, 2)
# _, max_index = torch.max(y_pred, dim=2)
# for i in range(x_test.shape[0]):
#     raw_prediction = list(max_index[:, i].detach().cpu().numpy())
#     prediction = torch.IntTensor([c for c, _ in groupby(raw_prediction) if c != blank_label])
#     test_preds.append(prediction)

# for j in range(len(x_test)):
#     mpl.rcParams["font.size"] = 8
#     plt.imshow(x_test[j], cmap='gray')
#     mpl.rcParams["font.size"] = 18
#     plt.gcf().text(x=0.1, y=0.1, s="Actual: " + str(y_test[j].numpy()))
#     plt.gcf().text(x=0.1, y=0.2, s="Predicted: " + str(test_preds[j].numpy()))
#     plt.show()

if __name__ == "__main__":
    trainer = Trainer()
    trainer.train()