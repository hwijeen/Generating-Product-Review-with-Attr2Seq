import io 
from collections import Counter
import json
import random
import pickle
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm_notebook

params = {'PAD_WORD' : 'PAD', 'PAD_IDX' : 0, 'UNK_WORD' : 'UNK', 'UNK_IDX' : 1}

class Vocab:
	def __init__(self, which):
		self.which = which
		self.words = Counter()
		self.truncwords = []
		if self.which == "review":
			self.word2idx = {'UNK': 0, 'PAD' : 1, 'SOS' : 2, 'EOS' : 3}
			self.idx2word = {0 : 'UNK', 1 : 'PAD', 2 : 'SOS', 3 : 'EOS'}
		elif self.which == "tag":
			self.word2idx = {'UNK' : 0, 'PAD' : 1}
			self.idx2word = {0 : 'UNK', 1 : 'PAD'}
			
	def build_vocab(self, data):
		if self.which == "review":
			print("Building vocab for reviews ...")
			for p in data:
				tokens =  p[-1]
				self.words.update(tokens)
		elif self.which == "tag":
			print("Building vocab for tags ...")
			for p in data:
				tokens = p[-2]
				self.words.update(tokens)
		self.trunc_words = [tok for tok, count in self.words.items()]
		
	def init_vocab(self):
		if self.which == "review":
			self.word2idx = {'UNK': 0, 'PAD' : 1, 'SOS' : 2, 'EOS' : 3}
			self.idx2word = {0 : 'UNK', 1 : 'PAD', 2 : 'SOS', 3 : 'EOS'}
		elif self.which == "tag":
			self.word2idx = {'UNK' : 0, 'PAD' : 1}
			self.idx2word = {0 : 'UNK', 1 : 'PAD'}
			
	def filter_by_freq(self, min_count):
		trunc_words = [tok for tok, count in self.words.items() if count >= min_count]
		print(len(trunc_words), "out of" , len(self.words), "words left, which is",
			  len(trunc_words)/len(self.words)*100.0, "%")
		self.trunc_words = trunc_words
		self.init_vocab()
		
	def build_idx_mapping(self, min_count = 0):
		if min_count > 0:
			self.filter_by_freq(min_count)
		for t in self.trunc_words:
			if t not in self.word2idx:
				self.idx2word[len(self.word2idx)] = t
				self.word2idx[t] = len(self.word2idx)
			else:
				pass

class Products:
	def __init__(self):
		# rating 
		self.rating2idx = {}
		# category 
		self.upcat2idx = {}
		self.lowcat2idx = {}
	
	# Rating
	def addRating(self, rating_list):
		self.rating2idx = {}
		for rate in set(rating_list):
			self.rating2idx[rate] = len(self.rating2idx)
	
	# Category
	def addCategory(self, cat_list, which):
		if which == 'upper':
			self.upcat2idx = {}
			for cat in set(cat_list):
				self.upcat2idx[cat] = len(self.upcat2idx)
		elif which == 'lower':
			self.lowcat2idx = {}
			for cat in set(cat_list):
				self.lowcat2idx[cat] = len(self.lowcat2idx)

def filter_by_cnt(min_tag, min_rv, filepath = 'review_full_0624.pkl'):
	with open(filepath, 'rb') as data:
		all_prod = pickle.load(data)
	tagVocab = Vocab(which = 'tag')
	rvVocab = Vocab(which = 'review')

	tagVocab.build_vocab(all_prod)
	rvVocab.build_vocab(all_prod)
	
	tagVocab.build_idx_mapping(min_count = min_tag)
	rvVocab.build_idx_mapping(min_count = min_rv)

	#tag2idx, idx2tag = tagVocab.word2idx, tagVocab.idx2word
	#rv2idx, idx2rv = rvVocab.word2idx, rvVocab.idx2word

	return tagVocab, rvVocab

def build_meta():
	upper = ['top', 'outer', 'bottom', 'shoes', 'bags']
	top = ['12015003002','12015003003','12015003004',
	   '12015003005','12015003006','12015003007']
	outer = ['12015001001', '12015004001', '12015004002', '12015004003', '12015004004']
	bottom = ['12015009001', '12015009002', '12015009003', '12015009005', '12015009004']
	shoes = ['12016013001001', '12016013003001', '12016013007001', '12016013001002', '12016013002001',
			 '12016013004004', '12016013003002', '12016013002002', '12016013004005', '12016013001003',
			 '12016013004002', '12016013003003', '12016013001004', '12016013005', '12016013004003',
			 '12016013003004', '12016013001005', '12016013006', '12016013003005', '12016013007003',
			 '12016013002003', '12016013004001', '12016013008', '12016013009', '12016013001006',
			 '12016013003006', '12016013001007', '12016013003007', '12016013007004', '12016013007002',
			 '12016013010', '12016013001008', '12016013001009','12016013004006']
	bags = ['12016021001', '12016021002', '12016021003', '12016001001',
			'12016001004001', '12016001002', '12016001003', '12016001004002',
			'12016021004', '12016001004003', '12016001004004', '12016021005',
			'12016021006', '12016021007', '12016021008', '12016001004006',
			'12016001005', '12016001006', '12016001007', '12016001008', '12016001009']

	lower = top + outer + bottom + shoes + bags

	Rating = ['추천', '적극추천', '만족', '보통', '불만', '추천안함']

	meta = Products()
	meta.addCategory(cat_list = lower, which = 'lower')
	meta.addCategory(cat_list = upper, which = 'upper')
	meta.addRating(Rating)

	return meta

def review_to_num(review, metadict, vocab_tag, vocab_rv, cat = 'lower'):
	# 리뷰의 정보를 분리
	rating = review[0]
	if cat == 'both':
		uppercat = review[1]
	lowercat = str(review[2])
	tags = review[3]
	text = review[4]
		
	rating_num = torch.tensor([metadict.rating2idx.get(rating)]).type(torch.long)
	#cat_num = torch.tensor([metadict.upcat2idx.get(uppercat)]) # upper category
	cat_num = torch.tensor([metadict.lowcat2idx.get(lowercat)]).type(torch.long) # lower category
	tag_num = torch.tensor([vocab_tag.word2idx.get(t, params['UNK_IDX']) for t in tags]).type(torch.long)
	rv_num = torch.tensor([vocab_rv.word2idx.get(w, params['UNK_IDX']) for w in text]).type(torch.long)
	
	return [rating_num, cat_num, tag_num, rv_num]

def prepareData(metadict, vocab_tag, vocab_rv, filepath = 'review_full_0624.pkl'):
	"""
	all_reviews : 모든 리뷰에 대한 리스트
	p : Product 클라스의 객체 (인코딩할때 참조!)
	"""
	with open(filepath, 'rb') as data:
		all_reviews = pickle.load(data)

	encode_prod = [] # 숫자의 리스트
	for review in tqdm_notebook(all_reviews): # 일단 열개만 해봅시다
		encode_prod.append(review_to_num(review, metadict, vocab_tag, vocab_rv)) 
	return encode_prod

def pad(batch, which):
	if which == 'tag':
		idx = 2
	elif which == 'review':
		idx = 3
	max_len = np.max([len(sample[idx]) for sample in batch])
	tag_padding = params['PAD_IDX']
	#batch
	batch_data = tag_padding*np.ones((len(batch), max_len))
	for j in range(len(batch)):
		cur_len = len(batch[j][idx])
		if cur_len > 0:
			batch_data[j][:cur_len] = np.array(batch[j][idx])
	batch_data = batch_data.astype(np.int64)
	batch_data = torch.tensor(batch_data)
	return batch_data

def data_iterator(data, batch_size):
	batch = random.sample(data, batch_size)
	
	rating = torch.cat([sample[0] for sample in batch], dim=-1).view(-1,1)
	category = torch.cat([sample[1] for sample in batch], dim=-1).view(-1,1)
	
	tag = pad(batch = batch, which = 'tag')
	review = pad(batch = batch, which = 'review')
	

	return rating, category, tag, review

		
"""
if __name__ == '__main__':
	params = {'PAD_WORD' : 'PAD', 'PAD_IDX' : 0, 'UNK_WORD' : 'UNK', 'UNK_IDX' : 1}
	
	with open('review_full_0624.pkl', 'rb') as data:
		all_prod = pickle.load(data)
	
"""