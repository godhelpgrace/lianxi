# -*- coding: utf-8 -*-

import json
import csv, json
import re
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import BertTokenizer
"""
数据加载
"""

def csv2json(csvfile, json_file):
    csvfile = open(csvfile, 'r', encoding='utf8', errors='ignore')
    reader = csv.DictReader(csvfile)
    with open(json_file, 'w') as f:
        for row in reader:
            json.dump(row, f)
            f.write('\n')
    return json_file

class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.index_to_label = {0: '差评', 1: '好评'}
        self.label_to_index = dict((y, x) for x, y in self.index_to_label.items())
        self.config["class_num"] = len(self.index_to_label)
        if self.config["model_type"] == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.load()


    def load(self):
        self.data = []
        with open(self.path, encoding="utf8") as f:
            for line in f:
                line = json.loads(line)
                tag = line["label"].replace('0', '差评').replace('1', '好评')
                label = self.label_to_index[tag]
                review = line["review"]
                if self.config["model_type"] == "bert":
                    input_id = self.tokenizer.encode(review, max_length=self.config["max_length"], pad_to_max_length=True)
                else:
                    input_id = self.encode_sentence(review)
                input_id = torch.LongTensor(input_id)
                label_index = torch.LongTensor([label])
                self.data.append([input_id, label_index])
        return

    def encode_sentence(self, text):
        input_id = []
        for char in text:
            input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        input_id = self.padding(input_id)
        return input_id

    #补齐或截断输入的序列，使其可以在一个batch内运算
    def padding(self, input_id):
        input_id = input_id[:self.config["max_length"]]
        input_id += [0] * (self.config["max_length"] - len(input_id))
        return input_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1  #0留给padding位置，所以从1开始
    return token_dict


#用torch自带的DataLoader类封装数据
def load_data(data_path, config, k, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)   # batch_size:每个批要加载多少个样例, shuffle:是否重新整理数据,bool值
    return dl

def load_data_k_fold(data_path, config, k, shuffle=True):
    dg = DataGenerator(data_path, config)
    train_data, test_data = random_split(dataset=dg, lengths=[len(dg) - int(len(dg) / k), int(len(dg) / k)],
                                         generator=torch.Generator().manual_seed(config["seed"]))
    train_data = DataLoader(train_data, batch_size=config["batch_size"], shuffle=shuffle)
    test_data = DataLoader(test_data, batch_size=config["batch_size"], shuffle=shuffle)
    return train_data, test_data

if __name__ == "__main__":
    from config import Config

    csvfile = "../文本分类练习.csv"
    csv2json(csvfile)
    dg = DataGenerator("../test.json", Config)
    print(dg[1])
