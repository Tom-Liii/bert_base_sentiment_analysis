'''
Author: GitHub@Tom-Liii
'''
from transformers import BertConfig, BertForSequenceClassification, BertTokenizer
import torch

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

review_f = open('../test_set/review.txt', 'r') 
     
test_set = review_f.readlines()

review_f.close()
# neg_review_f.close()
data = tokenizer(test_set, max_length=512, truncation=True, padding='max_length', return_tensors="pt")
# print(model(**data)['logits'])
t = model(**data)['logits']
i = 0
with open('../test_set/prediction.txt', 'w', encoding='utf-8') as prediction_file:
    while i < len(t):
        print('review: ', file=prediction_file)
        print(test_set[i], file=prediction_file) 
        print('Tensor for review # ' + str(i) + ' is: ', file=prediction_file)
        print(t[i], file=prediction_file)
        print('The prediction made by this model for this review is: ', file=prediction_file)
        if t[i][0] > t[i][1]: 
            print('pos', file=prediction_file)
        else:
            print('neg', file=prediction_file)
        print('-----------------', file=prediction_file)
        i+=1
print('test completed')
