
import torch
from torch.utils.data import Dataset


class DatasetPoetry(Dataset):
    def __init__(self
                 , texts
                 , genres
                 , views
                 , labels
                 , tokenizer
                 , max_len
                 ):
        self.texts = texts
        self.genres = genres
        self.views = views
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = self.texts[item]
        genre = self.genres[item]
        views = self.views[item]
        label = self.labels[item]
        model_inputs = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'text': text,
            'input_ids': model_inputs['input_ids'].flatten(),
            'attention_mask': model_inputs['attention_mask'].flatten(),
            'genre': torch.tensor(genre, dtype=torch.long),
            'views': torch.tensor(views, dtype=torch.float),
            'labels': torch.tensor(label, dtype=torch.float).unsqueeze(0) # чтобы размеры labels и выходов модели совпадали
        }