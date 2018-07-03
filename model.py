# coding: utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.emb_rating = nn.Embedding(self.config.rating_size, self.config.attribute_size)
        self.emb_category = nn.Embedding(self.config.category_size, self.config.attribute_size)
        self.emb_tag = nn.Embedding(self.config.tag_size, self.config.attribute_size, padding_idx=self.config.padding_idx)
        self.out = nn.Linear(self.config.attribute_size * 3, self.config.hidden_size*self.config.num_layers)
        self.init_hidden()

    def forward(self, rating, category, tag):
        """
        Inputs:
            rating: TENSOR of shape (batch_size, 1)
            category: TENSOR of shape (batch_size, 1)
            tag : 1) TENSOR of shape (batch_size, tag_MAXLEN)
        Returns:
            concatenated attr for attention, encoder_output
        """

        assert len(rating) == len(category) == len(tag)
        attr_rating = self.emb_rating(rating)        # N x 1 x attr_size
        attr_category = self.emb_category(category)  # N x 1 x attr_size
        tag_len = self.get_tag_len(tag)
        attr_tag = torch.sum(self.emb_tag(tag), 1, keepdim=True) / tag_len    # CBOW
                                                     # N x max_tag_len x attr_size
                                                     # N x 1 x attr_size*3
        attr = torch.cat((attr_rating, attr_category, attr_tag), 2)
        out = self.out(attr)    # N x 1 x hidden_size * num_layers(decoder)
        attr = attr.view(self.config.batch_size, self.config.num_attr, -1)  # N x 3 x 64
        encoder_output = F.tanh(out)
        return attr, encoder_output

    def get_tag_len(self, tag):
        """padding 제외한 token 개수"""
        return torch.sum(tag!=self.config.padding_idx, 1).unsqueeze(1).unsqueeze(1).type(torch.float)

    def init_hidden(self):
        for param in self.parameters():
            nn.init.uniform_(param, -0.08, 0.08)

            
class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.embedding = nn.Embedding(self.config.output_size, self.config.hidden_size)
        self.lstm = nn.LSTM(self.config.hidden_size, self.config.hidden_size,                             num_layers=self.config.num_layers, dropout=self.config.dropout,                            batch_first=True)
        self.out = nn.Linear(self.config.hidden_size, self.config.output_size)
        self.init_hidden()

    def forward(self, input_token, hidden):
        """
        Inputs:
            input_token: TENSOR of shape (batch_size, 1)
            hidden: from last hidden of encoder (h_0, c_0) batch first
                        h_0 - num_layers * num_direction X batch X hidden_size
                        c_0 - num_layers * num_direction X batch X hidden_size
        Returns:
        """
        # 가운데 1이니까 unroll방식으로만!  - 바꿀 수 있나?!
        output = self.embedding(input_token)          # N x 1(seq_len) x hidden_size
        # LSTM의 hidden은 (hx, cx)
        output, hidden = self.lstm(output, hidden)    # N x 1(seq_len) x hidden_size * num_dir
                                                      # num_layers * num_direction x N x hidden_size
        output = self.out(output)    # N x 1(seq_len) x output_size
        output = F.log_softmax(output, dim=2)
        return output, hidden

    def init_hidden(self):
        for param in self.parameters():
            nn.init.uniform_(param, -0.08, 0.08)


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.attn_score = nn.Linear(self.config.hidden_size + self.config.attribute_size, 1)
        self.init_hidden()

    def forward(self, last_hidden, attrs):
        # last_hidden : torch.Size([num_layers*num_direction, seq_len, hidden_dim])
        # attrs : torch.Size([batch_size, num_attr, attr_size])
        attn_energies = torch.zeros((self.config.batch_size, 1, self.config.num_attr))
        # B x 1(seq_len인가?) x 3
        for i in range(self.config.num_attr):
            attn_energies[:,:,i] = self.score(last_hidden.squeeze(), attrs[:,i,:])
        return F.softmax(attn_energies, dim=-1)#.unsqueeze(0).unsqueeze(0) # 1,1,3

    def score(self, last_hidden, attr):
        energy = self.attn_score(torch.cat((last_hidden, attr.squeeze()), -1))
        energy = F.tanh(energy)    # (batch, 1)
        return energy
    
    def init_hidden(self):
        for param in self.parameters():
            nn.init.uniform_(param, -0.08, 0.08)


class AttnDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(self.config.output_size, self.config.hidden_size)
        self.lstm = nn.LSTM(self.config.hidden_size, self.config.hidden_size, num_layers=self.config.num_layers, dropout=self.config.dropout,                            batch_first=True)
        self.attn_out = nn.Linear(self.config.hidden_size + self.config.attribute_size,
                                  self.config.hidden_size)
        self.out = nn.Linear(self.config.hidden_size, self.config.output_size)

        self.attn = Attention(self.config)
        self.init_hidden()

    def forward(self, input_token, hidden, attrs):

        word_embedded = self.embedding(input_token)
        output, hidden = self.lstm(word_embedded, hidden)

        attn_weights = self.attn(output, attrs)

        attrs = attrs.view(self.config.batch_size, self.config.num_attr, -1)
        context = attn_weights.bmm(attrs)
        output = F.tanh(self.attn_out(torch.cat((output, context), -1)))
        output = F.log_softmax(self.out(output), dim=-1)

        return output, hidden, attn_weights

    def init_hidden(self):
        for param in self.parameters():
            nn.init.uniform_(param, -0.08, 0.08)






