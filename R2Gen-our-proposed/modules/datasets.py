import os
import json
import pdb

import torch
from PIL import Image
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, args, tokenizer, split, transform=None):
        self.image_dir = args.image_dir
        self.ann_path = args.ann_path
        self.max_seq_length = args.max_seq_length
        self.split = split
        self.tokenizer = tokenizer
        self.transform = transform
        self.ann = json.loads(open(self.ann_path, 'r').read())

        self.examples = self.ann[self.split]
        for i in range(len(self.examples)):
            self.examples[i]['ids'] = tokenizer(self.examples[i]['report'])[:self.max_seq_length]
            self.examples[i]['mask'] = [1] * len(self.examples[i]['ids'])

            candidates = self.examples[i]['candidates']
            tokenize_candidates = [tokenizer(each)[:self.max_seq_length] for each in candidates]
            candidate_mask = [[1] * len(ids) for ids in tokenize_candidates]
            self.examples[i]['ids'] = [self.examples[i]['ids']]
            self.examples[i]['ids'].extend(tokenize_candidates)  # 3, various_length
            self.examples[i]['mask'] = [self.examples[i]['mask']]
            self.examples[i]['mask'].extend(candidate_mask)  # 3, various_length

    def __len__(self):
        return len(self.examples)


class IuxrayMultiImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        image_1 = Image.open(os.path.join(self.image_dir, image_path[0])).convert('RGB')
        image_2 = Image.open(os.path.join(self.image_dir, image_path[1])).convert('RGB')
        if self.transform is not None:
            image_1 = self.transform(image_1)
            image_2 = self.transform(image_2)
        image = torch.stack((image_1, image_2), 0)
        report_ids = example['ids']
        report_masks = example['mask']
        seq_length = len(report_ids)
        sample = (image_id, image, report_ids, report_masks, seq_length)
        return sample


class MimiccxrSingleImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['study_id']
        image_path = example['image_path']
        image = Image.open(image_path[0]).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        report_ids = example['ids']
        report_masks = example['mask']
        # seq_length = len(report_ids)
        seq_length = max([len(each) for each in report_ids])
        sample = (image_id, image, report_ids, report_masks, seq_length)
        return sample
