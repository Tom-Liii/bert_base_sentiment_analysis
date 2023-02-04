'''
Author: GitHub@Tom-Liii
The code is adapted from CSDN@HMTT

版权声明：本文为CSDN博主「HMTT」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/qq_42464569/article/details/123898549
'''
from transformers import BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments, BertConfig
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import json
import os
# 设定使用的GPU编号，也可以不设置，但trainer会默认使用多GPU
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# 使用cpu进行训练

os.environ["WANDB_DISABLED"] = "true"

def compute_metrics(pred):
    labels = pred.label_ids.argmax(-1)
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    print('---Evaluation Completed---')
    print('Accuracy = %.2f%%'%(acc*100))
    print('Precision = %.2f%%'%(precision*100))
    print('f1 = %.2f'%(f1))
    return {
        'accuracy': acc,
        # 'f1': f1,
        # 'precision': precision,
        # 'recall': recall
    }


# 将num_labels设置为2，因为我们训练的任务为2分类
# model = BertForSequenceClassification.from_pretrained('../transformersModels/bert-base-uncased/', num_labels=2)
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

train_dataset = Dataset.load_from_disk('./data_new/train_dataset/')
test_dataset = Dataset.load_from_disk('./data_new/test_dataset/')

# for param in model.base_model.parameters():
#     param.requires_grad = False

# 训练超参配置
training_args = TrainingArguments(
    output_dir='./my_results',          # output directory 结果输出地址
    num_train_epochs=10,              # total # of training epochs 训练总批次
    per_device_train_batch_size=4,  # batch size per device during training 训练批大小
    per_device_eval_batch_size=128,   # batch size for evaluation 评估批大小
    logging_dir='./my_logs',            # directory for storing logs 日志存储位置
    learning_rate=1e-6,			# learning rate
)



# 创建Trainer
trainer = Trainer(
    model=model.cuda(),              # the instantiated 🤗 Transformers model to be trained 需要训练的模型
    args=training_args,                  # training arguments, defined above 训练参数
    train_dataset=train_dataset,         # training dataset 训练集
    eval_dataset=test_dataset,           # evaluation dataset 测试集
    compute_metrics=compute_metrics
)


# 开始训练
trainer.train()

# 开始评估模型
trainer.evaluate()

# 保存模型 会保存到配置的output_dir处
trainer.save_model()

'''
# 模型配置文件
config.json

# 模型数据文件
pytorch-model.bin

# 训练配置文件
training_args.bin
'''

# load the saved model
output_config_file = './my_results/config.json'
output_model_file = './my_results/pytorch_model.bin'

config = BertConfig.from_json_file(output_config_file)
model = BertForSequenceClassification(config)
state_dict = torch.load(output_model_file)
model.load_state_dict(state_dict)

# use nlp to test the model
# cache_dir="../transformersModels/bert-base-uncased/"
cache_dir="bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(cache_dir)
data = tokenizer(['This is a good movie', 'This is a bad movie'], max_length=512, truncation=True, padding='max_length', return_tensors="pt")
print(model(**data))

