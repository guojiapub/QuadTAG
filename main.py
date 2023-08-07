# -*- coding: utf-8 -*-

import argparse
import os
import logging
import time
import pickle
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import seed_everything

from transformers import AdamW, T5Tokenizer, T5ForConditionalGeneration
from transformers import get_linear_schedule_with_warmup

from data_utils import MyDataset
from modules import *
from utils import * 
import numpy as np
import json
logger = logging.getLogger(__name__)


def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='QAM', type=str, required=True,
                        help="The name of the dataset, the default is [QAM]")
    parser.add_argument("--model_name_or_path", default='t5-base', type=str,
                        help="Path to pre-trained model or shortcut name")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_test", action='store_true', 
                        help="Whether to run eval on the test set.")
    parser.add_argument("--do_dev", action='store_true',
                        help="Whether to run eval on the dev set.")  
    parser.add_argument("--n_gpu", default=0, type=int,
                        help="GPU device")
    parser.add_argument("--train_batch_size", default=1, type=int,
                        help="Batch size per GPU for training.")
    parser.add_argument("--eval_batch_size", default=1, type=int,
                        help="Batch size per GPU for evaluation.")
    parser.add_argument("--max_seq_length", default=2496, type=int)
    parser.add_argument("--learning_rate", default=1e-4, type=float)
    parser.add_argument("--num_train_epochs", default=10, type=int, 
                        help="Total number of training epochs to perform.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--output_dir", default=f"./checkpoints/", type=str)
    parser.add_argument("--result_dir", default=f"./results/", type=str)
    parser.add_argument("--num_workers", default=8, type=int)  
    parser.add_argument("--negative_ratio", default=5, type=int)  
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--warmup_steps", default=0.0, type=float)
    parser.add_argument("--max_generate_len", default=512, type=int)
    parser.add_argument("--max_sent_num", default=32, type=int)
    parser.add_argument("--max_sent_len", default=400, type=int)
    parser.add_argument("--embed_dim", default=768, type=int)
    parser.add_argument("--hidden_embed_dim", default=256, type=int)
    parser.add_argument("--table_label_num", default=8, type=int)       
    parser.add_argument('--seed', type=int, default=68,
                        help="random seed for initialization")


    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    if not os.path.exists(args.result_dir):
        os.mkdir(args.result_dir)

    return args


def get_dataset(tokenizer, type_path, args):
    return MyDataset(tokenizer=tokenizer, data_dir=args.dataset, 
                       data_type=type_path, max_len=args.max_seq_length)


class T5FineTuner(pl.LightningModule):
    """
    Fine tune a pre-trained T5 model
    """
    def __init__(self, hparams, tfm_model, tokenizer):
        super(T5FineTuner, self).__init__()
        self.hparams = hparams
        self.model = tfm_model
        self.tokenizer = tokenizer
        self.biaf_layer = Biaffine(self.hparams.embed_dim, self.hparams.hidden_embed_dim, self.hparams.table_label_num)

    def is_logger(self):
        return True

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None,
                decoder_attention_mask=None, labels=None, output_hidden_states=False, sent_mask=None, sent_first_mask=None):

        return self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            output_hidden_states=output_hidden_states,
        )


    def _step(self, batch):
        lm_labels = batch["target_ids"]
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs = self(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            labels=lm_labels,
            decoder_attention_mask=batch['target_mask'],
            output_hidden_states=True,
        )
        last_hidden_states = outputs.encoder_hidden_states[-1]  

        sent_embeds, sent_token_masks = transform_doc_tokens_to_sent_tokens(last_hidden_states, batch['sent_mask'], self.hparams.max_sent_num, self.hparams.max_sent_len)
        updated_sent_embeds = sent_embeds[:, :, 0] 
        sent_num = (sent_token_masks.sum(-1) != 0).sum(-1) 
 

        truncated_dim = sent_num[0]
        table_logits = self.biaf_layer(updated_sent_embeds, updated_sent_embeds, sent_num[0])

        logits_flatten = table_logits.reshape(-1, table_logits.size()[-1])
        tags_flatten = batch['tags'][:, :truncated_dim, :truncated_dim].reshape(-1, 1)

        assert (tags_flatten == -1).sum() == 0

        logits_mask = (tags_flatten != 0).bool().to(table_logits.device)
        neg_idx = (logits_mask == 0).nonzero(as_tuple=False)
        negative_samples_num = (tags_flatten > 0).sum() * args.negative_ratio
        if negative_samples_num > neg_idx.size(0):
            negative_samples_num = torch.tensor(neg_idx.size(0))

        choice = torch.LongTensor(np.random.choice(neg_idx.size(0), negative_samples_num.item(), replace=False)).to(table_logits.device)
        logits_mask[neg_idx[choice][:, 0], neg_idx[choice][:, 1]] = True

        logits_flatten = torch.masked_select(logits_flatten, logits_mask).reshape(-1, self.hparams.table_label_num)
        tags_flatten =  torch.masked_select(tags_flatten, logits_mask).reshape(-1)

        table_loss = F.cross_entropy(logits_flatten, tags_flatten, ignore_index=-1, reduction='mean')

        loss = outputs[0] + table_loss  

        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)

        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def training_epoch_end(self, outputs):
        avg_train_loss = torch.stack([x["loss"] for x in outputs]).mean()
        tensorboard_logs = {"avg_train_loss": avg_train_loss}
        return {"avg_train_loss": avg_train_loss, "log": tensorboard_logs, 'progress_bar': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": tensorboard_logs, 'progress_bar': tensorboard_logs}

    def configure_optimizers(self):
        """ Prepare optimizer and schedule (linear warmup and decay) """
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)
        self.opt = optimizer
        return [optimizer]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, second_order_closure=None):
        if self.trainer.use_tpu:
            xm.optimizer_step(optimizer)
        else:
            optimizer.step()
        optimizer.zero_grad()
        self.lr_scheduler.step()

    def get_tqdm_dict(self):
        tqdm_dict = {"loss": "{:.4f}".format(self.trainer.avg_loss), "lr": self.lr_scheduler.get_last_lr()[-1]}
        return tqdm_dict

    def train_dataloader(self):
        train_dataset = get_dataset(tokenizer=self.tokenizer, type_path="train", args=self.hparams)
        dataloader = DataLoader(train_dataset, batch_size=self.hparams.train_batch_size,
                                drop_last=True, shuffle=True, num_workers=self.hparams.num_workers)
        t_total = (
            (len(dataloader.dataset) // (self.hparams.train_batch_size * max(1, len(self.hparams.n_gpu))))
            // self.hparams.gradient_accumulation_steps
            * float(self.hparams.num_train_epochs)
        )
        scheduler = get_linear_schedule_with_warmup(
            self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=t_total
        )
        self.lr_scheduler = scheduler
        return dataloader

    def val_dataloader(self):
        val_dataset = get_dataset(tokenizer=self.tokenizer, type_path="dev", args=self.hparams)
        return DataLoader(val_dataset, batch_size=self.hparams.eval_batch_size, num_workers=self.hparams.num_workers)


class LoggingCallback(pl.Callback):
    def on_validation_end(self, trainer, pl_module):
        logger.info("***** Validation results *****")
        if pl_module.is_logger():
            metrics = trainer.callback_metrics
        # Log results
        for key in sorted(metrics):
            if key not in ["log", "progress_bar"]:
                logger.info("{} = {}\n".format(key, str(metrics[key])))

    def on_test_end(self, trainer, pl_module):
        logger.info("***** Test results *****")

        if pl_module.is_logger():
            metrics = trainer.callback_metrics

        # Log and save results to file
        output_test_results_file = os.path.join(pl_module.hparams.output_dir, "test_results.txt")
        with open(output_test_results_file, "w") as writer:
            for key in sorted(metrics):
                if key not in ["log", "progress_bar"]:
                    logger.info("{} = {}\n".format(key, str(metrics[key])))
                    writer.write("{} = {}\n".format(key, str(metrics[key])))


def evaluate(data_loader, model, mode=''):
    """
    Compute scores given the predictions and gold labels
    """
    device = torch.device(f'cuda:{args.n_gpu}')
    model.to(device)
    model.model.eval()


    outputs, targets = [], []
    for batch in tqdm(data_loader):

        outs = model.model.generate(input_ids=batch['source_ids'].to(device), 
                                    attention_mask=batch['source_mask'].to(device), 
                                    max_length=args.max_generate_len,                                   
                                    )  

        dec = [tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]
        target = [tokenizer.decode(ids, skip_special_tokens=True) for ids in batch["target_ids"]]
        outputs.extend(dec)
        targets.extend(target)

    # return scores
    print("saving generation results...")
    with open(f"./results/{mode}[pred]{args.output_dir.split('/')[-1]}.txt", 'w') as fw:
        json.dump(outputs, fw)
    with open(f"./results/{mode}[gold]{args.output_dir.split('/')[-1]}.txt", 'w') as fw:
        json.dump(targets, fw)


# initialization
args = init_args()
print("\n", "="*30, f"NEW EXP on {args.dataset}", "="*30, "\n")


tokenizer = T5Tokenizer.from_pretrained(args.model_name_or_path)
special_tokens = ['<SS>', '<SE>', '[SEP]'] 
tokenizer.add_tokens(special_tokens)      

print(f"Here is an example (from the dev set):")
dataset = MyDataset(tokenizer=tokenizer, data_dir=args.dataset, 
                      data_type='dev', max_len=args.max_seq_length)
data_sample = dataset[7]  

example_input = tokenizer.decode(data_sample['source_ids'], skip_special_tokens=True)
example_output = tokenizer.decode(data_sample['target_ids'], skip_special_tokens=True)
print('Input :', example_input)
print('Output:', example_output)
print('\n')
print('dataset:  ', args.dataset)
if args.do_train:
    with open(f"./io_example/[example]{args.output_dir.split('/')[-1]}.txt", 'w') as fw:
        fw.write(example_input + '\n')
        fw.write(example_output + '\n')
        fw.write("\t".join(special_tokens))

print('\n')
print("output model dir", args.output_dir)
# training process
if args.do_train:

    seed_everything(args.seed)
    print("\n****** Conduct Training ******")
    # initialize the T5 model
    tfm_model = T5ForConditionalGeneration.from_pretrained(args.model_name_or_path)
    model = T5FineTuner(args, tfm_model, tokenizer)
    model.model.resize_token_embeddings(len(tokenizer))

    train_params = dict(
        default_root_dir=args.output_dir,
        accumulate_grad_batches=args.gradient_accumulation_steps,
        gpus=args.n_gpu,
        gradient_clip_val=1.0,
        max_epochs=args.num_train_epochs,
        callbacks=[LoggingCallback()],
    )

    trainer = pl.Trainer(**train_params)
    trainer.fit(model)

    # save the final model
    model.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Finish training and saving the model!", args.output_dir)
    torch.save(model.state_dict(), f'{args.output_dir}/model.pt')


# evaluation
if args.do_test:
    print("\n****** Conduct Evaluating with the last state ******")

    print(f"Load trained model from {args.output_dir}")
    print('Note that a pretrained model is required and `do_true` should be False')

    tokenizer = T5Tokenizer.from_pretrained(args.output_dir)
    tfm_model = T5ForConditionalGeneration.from_pretrained(args.output_dir)

    model = T5FineTuner(args, tfm_model, tokenizer)
    print("Reload other model paramters")
    model.load_state_dict(torch.load(f'{args.output_dir}/model.pt'))

    print("Reload pretrained model")
    model.model.from_pretrained(args.output_dir)
    model.tokenizer.from_pretrained(args.output_dir)


    test_dataset = MyDataset(tokenizer, data_dir=args.dataset, 
                               data_type='test', max_len=args.max_seq_length)
    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, num_workers=args.num_workers, shuffle=False)

    scores = evaluate(test_loader, model)



# evaluation
if args.do_dev:
    print("\n****** Conduct Evaluating with the last state ******")

    print(f"Load trained model from {args.output_dir}")
    print('Note that a pretrained model is required and `do_true` should be False')

    tokenizer = T5Tokenizer.from_pretrained(args.output_dir)
    tfm_model = T5ForConditionalGeneration.from_pretrained(args.output_dir)

    model = T5FineTuner(args, tfm_model, tokenizer)
    print("Reload other model paramters")
    model.load_state_dict(torch.load(f'{args.output_dir}/model.pt'))

    print("Reload pretrained model")
    model.model.from_pretrained(args.output_dir)
    model.tokenizer.from_pretrained(args.output_dir)


    test_dataset = MyDataset(tokenizer, data_dir=args.dataset, 
                               data_type='dev', max_len=args.max_seq_length)
    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, num_workers=args.num_workers, shuffle=False)

    scores = evaluate(test_loader, model, 'dev')