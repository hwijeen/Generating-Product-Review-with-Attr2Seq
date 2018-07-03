# coding: utf-8

import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader

PAD_IDX = 0
SOS_TOKEN = 1
EOS_TOKEN = 2

class Data(Dataset):
    def __init__(self, csv_file, vocab_file, tag_vocab, rating_dict, category):
        self.data_df = pd.read_csv(csv_file, index_col=0)
        self.category = category
        self.word2idx = {'PAD':PAD_IDX, 'SOS':SOS_TOKEN, 'EOS':EOS_TOKEN}
        self.idx2word = {}    
        self.tag2idx = {'PAD':PAD_IDX}
        self.rating2idx = rating_dict
        self.category2idx = {category:idx for idx, category in enumerate(set(self.data_df[category]))}
        self.build_vocab(vocab_file)
        self.build_tag(tag_vocab)
        
    def build_vocab(self, vocab_file):
        for line in open(vocab_file, "r"):
            word, count = line.split(' ')
            if word not in self.word2idx:
                self.word2idx[word] = len(self.word2idx)
        self.idx2word = {idx:word for word, idx in self.word2idx.items()}
    
    def build_tag(self, tag_vocab):
        for tag in open(tag_vocab, "r"):
            if tag not in self.tag2idx:
                self.tag2idx[tag.strip()] = len(self.tag2idx)
                
    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        item = self.data_df.iloc[idx]
        rating = torch.tensor(self.rating2idx[item['Rating']])
        # category , subcat 구별 위해 self.category
        category = torch.tensor(self.category2idx[item[self.category]])
        tokens_ = item['Tags'].strip().split()
        tag = torch.tensor([self.tag2idx[tag] for tag in tokens_])
        review = torch.tensor(self.preprocess(item['Review']))
        return rating, category, tag, review
        
    def preprocess(self, review):
        tokens_ = review.strip().split()
        sequence = []
        sequence.append(self.word2idx['SOS'])
        sequence.extend([self.word2idx[word] for word in tokens_])
        sequence.append(self.word2idx['EOS'])
        return sequence              

def collate_fn(data):
    def merge(sequences):
        if sequences[0].dim() == 0:    # rating, category: fixed dim
            return torch.stack(sequences).view(-1, 1)    # model을 2차원 받도록 만듬..ㅎ
        else:    # tag, review: variable length
            lengths = [len(seq) for seq in sequences]
            padded_seqs = torch.zeros(len(sequences), max(lengths)).long()
            for i, seq in enumerate(sequences):
                end = lengths[i]
                padded_seqs[i, :end] = seq[:end]
            return padded_seqs    # ,lengths?
            
    ratings, categories, tags, reviews = zip(*data)
    ratings = merge(ratings)
    categories = merge(categories)
    tags = merge(tags)
    reviews = merge(reviews)
    
    return ratings, categories, tags, reviews
    
def get_dataset_loader(csv_file, vocab_file, tag_vocab, rating_dict, category, batch_size):
    dataset = Data(csv_file, vocab_file, tag_vocab, rating_dict, category)
    dataloader = DataLoader(dataset, batch_size, shuffle=True, collate_fn=collate_fn)
    return dataset, dataloader


