from __future__ import absolute_import, division, print_function
import argparse
import glob
import logging
import os
import random
import shutil
import math
from collections import OrderedDict

import json
import numpy as np
import torch
from torch import nn
from tensorboardX import SummaryWriter
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss, BCELoss, BCEWithLogitsLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torchsummary import summary
from tqdm import tqdm, trange
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    BertConfig,
    BertTokenizer,
    BertForTokenClassification,
    RobertaConfig,
    RobertaForTokenClassification,
    RobertaTokenizer,
    get_linear_schedule_with_warmup,
)

from ocr_dataset import OcrDataset
from models import LayoutlmConfig, LayoutlmForTokenClassification, DocumentClassifier
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

logger = logging.getLogger(__name__)


ALL_MODELS = sum(
    (
        tuple(conf.pretrained_config_archive_map.keys())
        for conf in (BertConfig, RobertaConfig, LayoutlmConfig)
    ),
    (),
)
    
    
MODEL_CLASSES = {
    "bert": (BertConfig, BertForTokenClassification, BertTokenizer),
    "roberta": (RobertaConfig, RobertaForTokenClassification, RobertaTokenizer),
    "layoutlm": (LayoutlmConfig, LayoutlmForTokenClassification, BertTokenizer),
}


def evaluate(args, model, tokenizer, labels, mode, prefix=""):
    eval_dataset = OcrDataset(args.data_dir, tokenizer, labels, "test_v2")
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(
        eval_dataset,
        sampler=eval_sampler,
        batch_size=args.eval_batch_size,
        collate_fn=None,
    )
    model.eval()

    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    
    if mode == "test_v2":
        dev_loss = 0
        dev_steps = 0
        logit_all, label_all, file_names_all = [], [], []
        logit_soft_all = []
        correct_files, error_files = [], []
        files = []
        entropy_all = []
        for batch in tqdm(eval_dataloader):
            with torch.no_grad():
                inputs = {
                    "input_ids": batch['input_ids'].to(args.device),
                    "attention_mask": batch['attention_mask'].to(args.device),
                    "image": batch['image'].to(args.device),
                    "label": batch['label'].to(args.device),
                }
                
                if args.model_type in ["layoutlm"]:
                    inputs["bbox"] = batch['bbox'].to(args.device)
                inputs["token_type_ids"] = (
                    batch['token_type_ids'].to(args.device)
                    if args.model_type in ["bert", "layoutlm"]
                    else None
                )  # RoBERTa don"t use segment_ids

                logit, loss = model(**inputs)

                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu parallel training   

                file_names = batch['file_names']                    
                
                logit_soft = F.softmax(logit, dim=-1)         
                entropy = (-torch.sum(logit_soft * torch.log(logit_soft), -1)).detach().cpu().numpy().tolist()
                logit_soft = logit_soft.detach().cpu().numpy().tolist()
                #import ipdb;ipdb.set_trace()
                dev_loss += loss.item()
                
                logit = logit.argmax(-1).detach().cpu().numpy().tolist()
                label = inputs['label'].detach().cpu().numpy().tolist()

                logit_all.extend(logit)
                logit_soft_all.extend(logit_soft)
                label_all.extend(label)
                file_names_all.extend(file_names)
                entropy_all.extend(entropy)
                
                
             
            dev_steps += 1
            
        acc = accuracy_score(logit_all, label_all)
        f1 = f1_score(logit_all, label_all, average=None)   # 'micro'/'macro'/'weighted'
        macro_f1 = f1_score(logit_all, label_all, average='macro') 
        micro_f1 = f1_score(logit_all, label_all, average='micro') 
        
        cm = confusion_matrix(logit_all, label_all)
        
        labels = open(args.labels).read().split('\n')
        label_map = {i: label for i, label in enumerate(labels)}
        
        for i, (logit, label) in enumerate(zip(logit_all, label_all)):
 
            files.append((file_names_all[i], label_map[logit], label_map[label], entropy_all[i]))
            #if logit != label:
            #import ipdb;ipdb.set_trace()
            if logit != label:
#                 print(logit_soft_all[i], max(logit_soft_all[i]))
#                 print('====='*20)
                error_files.append((file_names_all[i], label_map[logit], label_map[label]))
                
#             if (label == 10 and np.argmax(logit_soft_all[i]) != 10 and max(logit_soft_all[i]) > 0.5) or \
#             (label != 10 and np.argmax(logit_soft_all[i]) == 10 and max(logit_soft_all[i]) <= 0.3):
#                 error_files.append((file_names_all[i], label_map[logit], label_map[label]))
            else:
                correct_files.append((file_names_all[i], label_map[logit], label_map[label]))
        #print("Test: acc {:.5f}, f1 {}, cm {}".format(acc, f1, cm))  
        print("Test: acc {:.5f}, macro_f1 {}, micro_f1 {}".format(acc, macro_f1, micro_f1))  
        print("Test: f1 {}".format(f1))
        print("Test: cm {}".format(cm)) 
        return dev_loss / dev_steps, error_files, correct_files, files
        


def main():  
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the training files for the CoNLL-2003 NER task.",
    )
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--img_model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )    
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pre-trained model or shortcut name selected in the list: "
        + ", ".join(ALL_MODELS),
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )

    ## Other parameters
    parser.add_argument(
        "--labels",
        default="./CORD/labels.txt",
        type=str,
        help="Path to a file containing all labels. If not specified, CoNLL-2003 labels are used.",
    )
    parser.add_argument(
        "--config_name",
        default="",
        type=str,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from s3",
    )
    parser.add_argument(
        "--max_seq_length",
        default=512,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--do_train", action="store_true", help="Whether to run training."
    )
    parser.add_argument(
        "--do_eval", action="store_true", help="Whether to run eval on the dev set."
    )
    parser.add_argument(
        "--do_predict",
        action="store_true",
        help="Whether to run predictions on the test set.",
    )
    parser.add_argument(
        "--evaluate_during_training",
        action="store_true",
        help="Whether to run evaluation during training at each logging step.",
    )
    parser.add_argument(
        "--do_lower_case",
        action="store_true",
        help="Set this flag if you are using an uncased model.",
    )

    parser.add_argument(
        "--per_gpu_train_batch_size",
        default=8,
        type=int,
        help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size",
        default=16,
        type=int,
        help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--learning_rate",
        default=5e-5,   # 5e-5
        type=float,
        help="The initial learning rate for Adam.",
    )
    parser.add_argument(
        "--weight_decay", default=0.0, type=float, help="Weight decay if we apply some."
    )
    parser.add_argument(
        "--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer."
    )
    parser.add_argument(
        "--max_grad_norm", default=1.0, type=float, help="Max gradient norm."
    )
    parser.add_argument(
        "--num_train_epochs",
        default=3.0,
        type=float,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument(
        "--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps."
    )

    parser.add_argument(
        "--logging_steps", type=int, default=500, help="Log every X updates steps."
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=50,
        help="Save checkpoint every X updates steps.",
    )
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument(
        "--no_cuda", action="store_true", help="Avoid using CUDA when available"
    )
    parser.add_argument(
        "--overwrite_output_dir",
        action="store_true",
        help="Overwrite the content of the output directory",
    )
    parser.add_argument(
        "--overwrite_cache",
        action="store_true",
        help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="random seed for initialization"
    )

    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--fp16_opt_level",
        type=str,
        default="O1",
        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
        "See details at https://nvidia.github.io/apex/amp.html",
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="For distributed training: local_rank",
    )
    parser.add_argument(
        "--server_ip", type=str, default="", help="For distant debugging."
    )
    parser.add_argument(
        "--server_port", type=str, default="", help="For distant debugging."
    )
    args = parser.parse_args()

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
    ):
        if not args.overwrite_output_dir:
            raise ValueError(
                "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                    args.output_dir
                )
            )
        else:
            if args.local_rank in [-1, 0]:
                shutil.rmtree(args.output_dir)

    if not os.path.exists(args.output_dir) and (args.do_eval or args.do_predict):
        raise ValueError(
            "Output directory ({}) does not exist. Please train and save the model before inference stage.".format(
                args.output_dir
            )
        )

    if (
        not os.path.exists(args.output_dir)
        and args.do_train
        and args.local_rank in [-1, 0]
    ):
        os.makedirs(args.output_dir)

    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd

        print("Waiting for debugger attach")
        ptvsd.enable_attach(
            address=(args.server_ip, args.server_port), redirect_output=True
        )
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
        
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        )
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(
        filename=os.path.join(args.output_dir, "train.log")
        if args.local_rank in [-1, 0]
        else None,
        format="%(asctime)s - %(levelname)s - %(name)s -  %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        args.local_rank,
        device,
        args.n_gpu,
        bool(args.local_rank != -1),
        args.fp16,
    )

    labels = open(args.labels).read().split('\n')
    num_labels = len(labels)
    

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

                
    if args.do_predict and args.local_rank in [-1, 0]:
        
        model = DocumentClassifier(config, args.img_model_type, args.device, 2816, num_labels)   # 1280/3072

        pretrained_weights = (torch.load(os.path.join(args.model_name_or_path, 'pytorch_model.bin')))
        model.load_state_dict(pretrained_weights)
     
        model.to(args.device)
        dev_loss, error_files, correct_files, files = evaluate(
            args, model, tokenizer, labels, mode="test_v2"
        )
#         files.sort(key=lambda x: x[-1], reverse=True)        
#         json.dump(files, open('outputs/test_entropy_base.json', 'w'), indent=2, ensure_ascii=False)
        
        #print("dev loss: %f" % dev_loss)


if __name__ == "__main__":
    main()

