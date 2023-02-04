# A simple library tester made by @Tom-Liii
import torch
import transformers
import datasets

print("torch version: ", torch.__version__)
print("CUDA availability: ", torch.cuda.is_available())
print("transformers version: ", transformers.__version__)
print("datasets version: ", datasets.__version__)