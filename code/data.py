'''
Author: GitHub@Tom-Liii
'''

from datasets import Dataset
from transformers import BertTokenizer
import os
import dill

# 载入原始数据
def load_data(base_path):
    paths = os.listdir(base_path)
    result = []
    for path in paths:
        with open(os.path.join(base_path, path), 'r', encoding='utf-8') as f:
            result.append(f.readline())
    return result

# 读入数据并转化为datasets.Dataset
def get_dataset(base_path):
		# 为了展示方便，这里只取前3个数据，真实使用需要删掉切片操作
    pos_data = load_data(os.path.join(base_path, 'pos'))
    neg_data = load_data(os.path.join(base_path, 'neg'))
    
		# 列表合并
    texts = pos_data + neg_data
		# 生成标签，其中使用 '1.' 和 '0.' 是因为需要转化为浮点数，要不然模型训练时会报错
    labels = [[1., 0.]]*len(pos_data) + [[0., 1.]] * len(neg_data)
    dataset = Dataset.from_dict({'texts':texts, 'labels':labels})
    return dataset

# 加载数据
train_dataset = get_dataset('../data/aclImdb/train/')
test_dataset = get_dataset('../data/aclImdb/test/')

# cache_dir是预训练模型的地址
cache_dir="../transformersModels/bert-base-uncased/"
tokenizer = BertTokenizer.from_pretrained(cache_dir)

# 设置最大长度
MAX_LENGTH = 512

# 使用文本标记器对texts进行编码
# print(train_dataset['texts'])
# print(train_dataset['labels'])
# print(MAX_LENGTH)


train_dataset = train_dataset.map(lambda examples: tokenizer(examples['texts'], truncation=True, padding='max_length', max_length=MAX_LENGTH), batched=True)
test_dataset = test_dataset.map(lambda examples: tokenizer(examples['texts'], truncation=True, padding='max_length', max_length=MAX_LENGTH), batched=True)

train_dataset.save_to_disk('./data_new/train_dataset')
test_dataset.save_to_disk('./data_new/test_dataset')

# print(train_dataset['input_ids'])
# print(train_dataset)
# print(train_dataset['texts'])
# print(train_dataset['labels'])
# print(train_dataset.features)
