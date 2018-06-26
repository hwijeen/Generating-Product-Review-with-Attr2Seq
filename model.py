
# coding: utf-8

# ## TODO
# 1. Batching!   
#     -Encoder.Forward의 input 모양 어떻게 되지? / .view 인자 확인!
# 2. Attention  
# 3. Teacher Forcing  
# 4. Parameter(things to be updated) 등록 잘 됐나 확인

# #### NOTE
# 1. Decoder가 2 layer일때, initial hidden?  
#     - https://discuss.pytorch.org/t/understanding-output-of-lstm/12320/2

# In[1]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim


# In[2]:


class Config():
    # Encoder
    rating_size = 5
    category_size = 10
    tag_size = 3
    pretrained = False 
    # embedding_size = 300    # needed when not using pretrained vector
    attribute_size = 64
    hidden_size = 512 # fixed-vector size 
    # word-embedding = hidden_size
    
    # Decoder
    num_layers = 2
    output_size = 10


# In[3]:


class Encoder(nn.Module):
    def __init__(self, config):        
        super().__init__()
        self.config = config
        
        # Embedding instead of Linear for efficient indexing
        self.emb_rating = nn.Embedding(self.config.rating_size, self.config.attribute_size)   
        self.emb_category = nn.Embedding(self.config.category_size, self.config.attribute_size)
        self.emb_tag = nn.Embedding(self.config.tag_size, self.config.attribute_size)
        self.out = nn.Linear(self.config.attribute_size * 3, self.config.hidden_size*self.config.num_layers)
        self.init_hidden()
        
    def forward(self, rating, category, tag):
        """
        Inputs:
            rating: TENSOR of shape (1, )
            category: TENSOR of shape (1, )
            tag : 1) TENSOR of shape (tag_size, )
        Returns:
            concatenated attr for attention, encoder_output
        """
        # TODO: check if len(rating), len(category), len(tag) matches
        attr_rating = self.emb_rating(rating).view(1,1,-1)    # shape?!
        attr_category = self.emb_category(category).view(1,1,-1)
        attr_tag = torch.sum(self.emb_tag(tag), 0) / len(tag)    # embedding 평균
        attr_tag = attr_tag.view(1,1,-1)
        
        attr = torch.cat((attr_rating, attr_category, attr_tag), 2)    # specify dim?
        out = self.out(attr)
        encoder_output = F.tanh(out)
        return attr, encoder_output
    
    def init_hidden(self):
        for param in self.parameters():
            nn.init.uniform_(param, -0.08, 0.08)


# In[4]:


print("Testing encoder...\n")
config = Config()
encoder = Encoder(config)
rating = torch.tensor([3]).type(torch.long)    # idx of rating in tensor
category = torch.tensor([7]).type(torch.long)  # idx of category in tensor
tag = torch.tensor([1,2,1]).type(torch.long)    # CBOW of one-hot
attr, encoder_output = encoder(rating,category,tag)
print(attr.size())
print(encoder_output.size())


# In[5]:


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # TODO: if self.config.pretrained = True
        self.embedding = nn.Embedding(self.config.output_size, self.config.hidden_size)
        self.lstm = nn.LSTM(self.config.hidden_size, self.config.hidden_size,                             num_layers=self.config.num_layers, dropout=0.2)
        self.out = nn.Linear(self.config.hidden_size, self.config.output_size)
        
    def forward(self, input_token, hidden):
        """
        Inputs:
            input_token: TENSOR of shape (1,1,1)
            hidden: from last hidden of encoder
        Returns:
            concatenated attr for attention, encoder_output
        """
        output = self.embedding(input_token).view(len(input_token), 1, -1)
        # LSTM의 hidden은 (hx, cx)
        output, hidden = self.lstm(output, hidden)
        output = self.out(output)
        output = F.log_softmax(output, dim=2)
        return output, hidden

    def initHidden(self):
        pass


# In[6]:


print("testing decoder with encoder_output...\n")
decoder = Decoder(config)
# h_ = torch.split(encoder_output,[config.hidden_size, config.hidden_size], 2)
# h_ = torch.cat(h_, dim=0)
h_ = encoder_output.view(config.num_layers, 1, config.hidden_size)
c_ = encoder_output.view(config.num_layers, 1, config.hidden_size)
hidden = h_, c_

input_token = torch.tensor([0,0,0])
output, hidden = decoder(input_token, hidden)
print("input_token.size(): ", input_token.size())
print("hidden[0].size(): ", hidden[0].size())
print("hidden[1].size(): ", hidden[1].size())
print("output.size(): ", output.size())



# In[7]:


class Attr2Seq(nn.Module):
    def __init__(self, config, criterion):
        super().__init__()
        self.config = config
        self.criterion = criterion
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)
    
    def forward(self, rating, category, tag, target_tensor):
        # 함수 호출시 *[rating, category, tag]하기!
        # target_tensor도 1차원!!!!
        target_length = target_tensor.size(0)    
        attr, encoder_output = self.encoder(rating,category,tag)
        
        hidden = self.splitHidden(encoder_output)
        input_token = torch.zeros((1,1,1)).type(torch.long)    # SOS token
        
        decoder_outputs = []
        for idx in range(target_length):
            decoder_output, decoder_hidden = decoder(input_token, hidden)
            topv, topi = decoder_output.topk(1)
            input_token = topi.detach()            
            decoder_outputs.append(decoder_output.squeeze())
        decoder_outputs = torch.cat(decoder_outputs, 0).view(target_length, -1)
        loss = self.criterion(decoder_outputs, target_tensor)
        return loss
    
    def splitHidden(self, encoder_output):
        """
        Encoder의 ouput인 fixed size vector를 Decoder의 hidden으로 쪼개기
        """
        return encoder_output.view(self.config.num_layers, 1, self.config.hidden_size),                 encoder_output.view(self.config.num_layers, 1, self.config.hidden_size)    
    
    def inference(self):
        pass


# In[17]:


criterion = nn.NLLLoss()
model = Attr2Seq(config, criterion)
input_list = [torch.tensor([2]).type(torch.long), torch.tensor([7]).type(torch.long),
              torch.tensor([0,2,1])]
#target_tensor = torch.tensor([[3],[3],[4],[6]])
target_tensor = torch.tensor([9,2,1,3])
loss = model(*input_list, target_tensor)
print(loss)


# In[9]:


def train(input_list, target_tensor, model, optimizer):
    """
    perform a step(update)
    """
    optimizer.zero_grad()
    
    target_length = target_tensor.size(0)
    
    loss = model(*input_list, target_tensor)

    loss.backward()
    optimizer.step()
    return loss.item() / target_length


# In[18]:


optimizer = optim.SGD(model.parameters(), lr=0.002)
avg_loss = train(input_list, target_tensor, model, optimizer)
print(avg_loss)


# In[13]:


def trainIters():
    pass


# In[9]:


def evaluate():
    pass


# In[10]:


def evaluateRandomly():
    pass


# In[190]:


print(model)

