

import torch
import torch.nn as nn
#from transformers import BertModel, BertTokenizer
from transformers import AutoTokenizer, AutoModel


class BERTClassifier(nn.Module):
    def __init__(self, bert_model_name, num_genres, tokenizer):
        super(BERTClassifier, self).__init__()
        # self.bert = BertModel.from_pretrained(bert_model_name)
        self.bert = AutoModel.from_pretrained(bert_model_name)
        self.bert.resize_token_embeddings(len(tokenizer)) # добавляем новый токен для переноса строк
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        self.linear = nn.Linear(self.bert.config.hidden_size + num_genres + 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask, genres, views):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs[1]  # [CLS] token
        x = torch.cat((cls_output, genres, views), dim=1)
        x = self.dropout(x)
        x = self.linear(x)
        x = self.sigmoid(x)

        return x


class BERTClassifier2(nn.Module):
    def __init__(self, bert_model_name, num_genres, tokenizer, hidden_dim=128):
        super(BERTClassifier2, self).__init__()
        # self.bert = BertModel.from_pretrained(bert_model_name)
        self.bert = AutoModel.from_pretrained(bert_model_name)
        self.bert.resize_token_embeddings(len(tokenizer)) # добавляем новый токен для переноса строк
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        self.linear = nn.Linear(self.bert.config.hidden_size + num_genres + 1, hidden_dim)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask, genres, views):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs[1]  # [CLS] token
        x = torch.cat((cls_output, genres, views), dim=1)
        x = self.dropout(x)
        x = self.linear(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.sigmoid(x)

        return x
