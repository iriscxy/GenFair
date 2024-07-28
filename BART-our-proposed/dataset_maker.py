import logging
import pdb
import torch
from typing import Dict
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from args import DataTrainingArguments
from transformers.training_args_seq2seq import Seq2SeqTrainingArguments

from datasets import load_dataset, DownloadConfig

logger = logging.getLogger(__name__)


class DatasetMaker:
    def __init__(self, dataset_saved_path: str, data_args: DataTrainingArguments,
                 training_args: Seq2SeqTrainingArguments, tokenizer: PreTrainedTokenizerBase):
        self.data_args = data_args
        self.training_args = training_args
        self.tokenizer = tokenizer
        self.dataset_saved_path = dataset_saved_path

    def make_dataset(self):
        logger.info('******* Making Dataset **********')
        data_files = {}
        if self.data_args.train_file is not None:
            data_files["train"] = self.data_args.train_file
            extension = self.data_args.train_file.split(".")[-1]
        if self.data_args.validation_file is not None:
            data_files["validation"] = self.data_args.validation_file
            extension = self.data_args.validation_file.split(".")[-1]
        if self.data_args.test_file is not None:
            data_files["test"] = self.data_args.test_file
            extension = self.data_args.test_file.split(".")[-1]
        if extension == 'txt': extension = 'text'
        datasets = load_dataset(extension, data_files=data_files, download_config=DownloadConfig(use_etag=False))
        # Temporarily set max_target_length for training.
        max_target_length = self.data_args.max_target_length
        padding = "max_length" if self.data_args.pad_to_max_length else False

        if self.training_args.label_smoothing_factor > 0:
            logger.warn(
                "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for model. This will lead to loss being calculated twice and will take up more memory"
            )

        def preprocess_function(examples: Dict):
            inputs = [ex for ex in examples["findings"]]

            targets = [ex + self.tokenizer.eos_token for ex in examples["impression"]]#batch,length
            model_inputs = self.tokenizer(inputs, max_length=self.data_args.max_source_length, padding=padding,
                                          truncation=True)

            candidates = [[ex + self.tokenizer.eos_token for ex in candidate_list] for candidate_list in
                          examples["candidates"]]#batch, candidate_num,length
            combined = [targets[i:i + 1] + candidates[i] for i in range(len(targets))]  #batch, candidate_num+1,length


            flattened_combined = [item for sublist in combined for item in sublist]  #batch*(candidate_num+1),length

            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(flattened_combined, max_length=max_target_length, padding=padding,
                                        truncation=True)
            if padding == "max_length" and self.data_args.ignore_pad_token_for_loss:
                labels["input_ids"] = [
                    [(l if l != self.tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
                ]
            candidate_num = len(candidates[0])
            reshaped_labels = [labels["input_ids"][i * (1 + candidate_num):(i + 1) * (1 + candidate_num)] for i in
                               range(len(targets))]   #batch,candidate_num+1,length
            model_inputs["labels"] = reshaped_labels
            return model_inputs

        datasets = datasets.map(
            preprocess_function,
            batched=True,
            num_proc=self.data_args.preprocessing_num_workers,
            load_from_cache_file=not self.data_args.overwrite_cache,
        )

        logger.info('saving dataset')
        dataset_saved_path = self.dataset_saved_path
        datasets.save_to_disk(dataset_saved_path)
        logger.info(f'******* Dataset Finish {dataset_saved_path} **********')
        return datasets
