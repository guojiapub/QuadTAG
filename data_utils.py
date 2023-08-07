import random
import torch
from torch.utils.data import Dataset
from nltk.tokenize import TweetTokenizer
import json
from collections import defaultdict
import numpy as np


def get_transformed_io(data_path, data_dir):
    inputs, targets = [], []
    tk = TweetTokenizer()
    with open(data_path[:-4]+".json", 'r') as f:
        data = json.load(f)

    sent_nums = []
    stance = {"1": "supports the topic", "-1": "is against the topic"}

    for example in data:
        sent_nums.append(len(example['sents']) + 1)

        sents = [['<SS>', f"#{idx+1}:"] + tk.tokenize(s) + [f"<SE>"] for idx, s in enumerate(example['sents'])]

        label_dict = defaultdict(list)
        for c, e, s, t in example['labels']:
            label_dict[(c, s)].append((e, t))
        tmp = []
        for k, v in label_dict.items():
            assert len(v) == len(set(v))
            v = sorted(v, key=lambda x:x[0])
            tmp.append(f"#{k[0]+1} {stance[k[1]]} : " + " | ".join([f'#{evi+1} {evi_ty}' for evi, evi_ty in v]))   
        targets.append(' [SEP] '.join(tmp)) 
    
        topic = f'Topic: {example["topic"][:-1] if example["topic"][-1] == "?" else example["topic"]} ? <SE>'
        inputs.append([['<SS>'] + tk.tokenize(topic)]+ sents)

    return inputs, targets, sent_nums



def get_table_tags(data_path, max_sent_num):
    with open(data_path[:-4]+".json", 'r') as f:
        data = json.load(f)
    label_map = {'Research': 1, 'Expert': 2, 'Case': 3, 'Explanation': 4, 'Others':5, '1': 6, "-1": 7} 
    tags = torch.full((len(data), max_sent_num, max_sent_num), -1)
    for example_idx, example in enumerate(data):
        tags[example_idx, :len(example['sents'])+1, :len(example['sents'])+1] = 0
        for cid, eid, st, evi_type in example['labels']:
            tags[example_idx, cid+1, eid+1] = label_map[evi_type]
            tags[example_idx, cid+1, 0] = label_map[st]
    return tags


class MyDataset(Dataset):
    def __init__(self, tokenizer, data_dir, data_type, max_len=128):
        self.data_path = f'data/{data_dir}/{data_type}.txt'
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.data_dir = data_dir
        self.max_sent_num = 32

        self.inputs = []
        self.targets = []
        self.tags = get_table_tags(self.data_path, self.max_sent_num)
        self._build_examples()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        source_ids = torch.LongTensor(self.inputs[index]["input_ids"])
        target_ids = self.targets[index]["input_ids"].squeeze()

        src_mask = torch.LongTensor(self.inputs[index]["attention_mask"])  
        target_mask = self.targets[index]["attention_mask"].squeeze()  

        sent_mask = torch.LongTensor(self.inputs[index]["sent_mask"])
        sent_first_mask = torch.LongTensor(self.inputs[index]["sent_first_mask"])

        return {"source_ids": source_ids, "source_mask": src_mask, "sent_mask": sent_mask, "tags": self.tags[index],
                "target_ids": target_ids, "target_mask": target_mask, 'target_sent': self.targets[index]["target_sent"], 'sent_num': self.sent_nums[index], "sent_first_mask": sent_first_mask}



    def _build_examples(self):

        inputs, targets, self.sent_nums = get_transformed_io(self.data_path, self.data_dir)

        for i in range(len(inputs)):
            tokenized_input = {'input_ids': [], 'attention_mask': [], 'sent_mask': [], 'sent_first_mask': []}
            for j in range(len(inputs[i])):
                sent = ' '.join(inputs[i][j])
                if j == len(inputs[i]) - 1:
                    sent_input_ids = self.tokenizer.encode(sent)
                else:
                    sent_input_ids = self.tokenizer.encode(sent)[:-1]
                tokenized_input['input_ids'].extend(sent_input_ids)
                tokenized_input['sent_mask'].extend([0] * len(sent_input_ids))
                tokenized_input['sent_mask'][-1] = 1

                tokenized_input['sent_first_mask'].append(1)
                tokenized_input['sent_first_mask'].extend([0] * (len(sent_input_ids) -1))



            tokenized_input['attention_mask'].extend([1] * len(tokenized_input['input_ids']))   
            data_len = len(tokenized_input['input_ids'])
            assert len(tokenized_input['input_ids']) == len(tokenized_input['sent_mask']) == len(tokenized_input['attention_mask'])
            if data_len >= self.max_len:
                tokenized_input['input_ids'] = tokenized_input['input_ids'][:self.max_len] 
                tokenized_input['attention_mask'] = tokenized_input['attention_mask'][:self.max_len] 
                tokenized_input['sent_mask'] = tokenized_input['sent_mask'][:self.max_len] 
                tokenized_input['sent_first_mask'] = tokenized_input['sent_first_mask'][:self.max_len]                 

            else:
                tokenized_input['input_ids'] += [self.tokenizer.pad_token_id] * (self.max_len - data_len)
                tokenized_input['attention_mask'] += [0] * (self.max_len - data_len)
                tokenized_input['sent_mask'] += [-1] * (self.max_len - data_len)
                tokenized_input['sent_first_mask'] += [0] * (self.max_len - data_len)
          

            target = targets[i]
            tokenized_target = self.tokenizer.batch_encode_plus(
            [target], max_length=self.max_len, padding="max_length",
            truncation=True, return_tensors="pt"
            )
            tokenized_target['target_sent'] = targets[i]
            self.inputs.append(tokenized_input)
            self.targets.append(tokenized_target)


