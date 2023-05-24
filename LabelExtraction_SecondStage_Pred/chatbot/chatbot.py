import torch
from .model import Model
from torchtext.data.utils import get_tokenizer
import re
import numpy as np
import json

class ChatBot():
    def __init__(self):
        super().__init__()
        self.vocab = torch.load("chatbot/vocab.pt")
        with open ('chatbot/intents.json', 'r') as file:
            intnets_data = json.load(file)
        self.model = Model(len(self.vocab), 64, len(intnets_data["intents"]))
        #load weight
        self.model.load_state_dict(torch.load("chatbot/model.pt"))
        self.model.eval()
        self.intents = intnets_data['intents']
        self.tokenizer = get_tokenizer('basic_english')

    def process(self, question):
        question = re.sub(r"[^A-Za-z0-9. ]+", '', question)
        question = self.tokenizer(question)
        y = self.model(torch.IntTensor(self.vocab(question)).unsqueeze(0))
        intent_id, entities= y
        intent_id = intent_id.argmax(1).item()
        entities = entities.argmax(1)

        product_entity = ' '.join(np.ma.array(question, mask=(entities!=1)).compressed())
        date_entity = np.ma.array(question, mask=(entities!=2)).compressed()
        print(f"Intent: {intent_id}-{self.intents[intent_id]['tag']} | entity: {product_entity}, {date_entity}")
        return intent_id, self.intents[intent_id]['tag'], product_entity, date_entity
    
if __name__ == "__main__":
    chatbot = ChatBot()
    while True:
        question = input('>>>')
        chatbot.process(question)
    # chatbot.predict("what is the cost of coke")
    # chatbot.predict("what is the cost of milk shake yesterday")