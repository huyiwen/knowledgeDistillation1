import accelerator
import datasets
import torch
import torch.nn as nn
from distillation.hintonDistiller import HintonDistiller
from prefetch_generator import BackgroundGenerator
from torch.nn import GRU, Embedding
from torch.utils.data import TensorDataset, DataLoader
from transformers import (AutoModelForSequenceClassification,
                          BertForSequenceClassification, BertModel,
                          BertTokenizer, DataCollatorWithPadding,
                          PreTrainedTokenizer, default_data_collator)
import re
from nltk.tokenize import SpaceTokenizer

from models.lstm import LSTMForSequenceClassification, LSTMConfig

class DataLoaderX(torch.utils.data.DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

def get_loader(raw_datasets, text_keys, label_key, tokenizer: PreTrainedTokenizer, max_seq_length=128, per_device_train_batch_size=64, per_device_eval_batch_size=64, pad_to_max_length=True, padding='max_length', truncation='longest_first'):

    # max_len = max([len(tokenizer.encode(ex[text_keys[0]])) for ex in raw_datasets["train"]])
    # print(f"Max length: {max_len}")

    def preprocess_function(tokenizer, padding, max_seq_length, truncation, label_key, text_keys):
        def callback(examples):
            texts = []
            for key in text_keys:
                if key in examples:
                    texts.append([
                        re.sub(r" ([\,\.\!\?\"\'])", r"\1", ex) for ex in examples[key]
                    ])
            result = tokenizer(*texts, padding=padding, max_length=max_seq_length, truncation=truncation)
            result["labels"] = examples[label_key]
            return result
        return callback

    # for ex in raw_datasets["train"]:
    #     print(ex["text"], ex["label"])

    processed_datasets = raw_datasets.map(
        preprocess_function(tokenizer, padding, max_seq_length, truncation, label_key, text_keys),
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
        desc="Running tokenizer on dataset",
    )

    if pad_to_max_length:
        # If padding was already done ot max length, we use the default data collator that will just convert everything
        # to tensors.
        data_collator = default_data_collator
    else:
        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
        data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=(8 if accelerator.use_fp16 else None))

    print(processed_datasets)

    train_dataloader = DataLoaderX(
        processed_datasets["train"], shuffle=True, collate_fn=data_collator, batch_size=per_device_train_batch_size
    )
    eval_dataloader = DataLoaderX(
        processed_datasets["validation"], collate_fn=data_collator, batch_size=per_device_eval_batch_size
    )
    return train_dataloader, eval_dataloader

TASK = "sst2"

if TASK == "sst2":
    MODEL = '/home/huyiwen/pretrained/bert-base-uncased-SST-2'
    DATA = '/home/huyiwen/datasets/sst2'
    TEXT = ('text',)
    S = 80
    B = 50
    epochs = 20
elif TASK == "qqp":
    MODEL = '/home/huyiwen/pretrained/bert-base-uncased-QQP'
    DATA = '/home/huyiwen/datasets/qqp'
    TEXT = ('text1', 'text2')
    S = 256
    B = 256
    epochs = 10

print("Loading models...")
teacher = BertForSequenceClassification.from_pretrained(MODEL).cuda()
# student = BertClassifier(MODEL).cuda()
student_config = LSTMConfig()
student = LSTMForSequenceClassification(student_config).cuda()
tokenizer = BertTokenizer.from_pretrained(MODEL)


# Initialize objectives and optimizer
objective = nn.CrossEntropyLoss()
distillObjective = nn.KLDivLoss(reduction='batchmean')
optimizer = torch.optim.AdamW(student.parameters())
scaler = torch.cuda.amp.GradScaler()


# Pseudo dataset and dataloader
print(f"Loading dataset {TASK}...")
data = datasets.load_from_disk(DATA)
trainloader, testloader = get_loader(data, TEXT, "label", tokenizer, max_seq_length=S, per_device_train_batch_size=B, per_device_eval_batch_size=B, truncation=False)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(trainloader) * epochs)
print(next(iter(trainloader)))


# Load state if checkpoint is provided
checkpoint = None
distiller = HintonDistiller(alpha=1, teacher=teacher, dataloader=trainloader, task=TASK, studentLayer=-1, teacherLayer=-1)
startEpoch = distiller.load_state(checkpoint, student, teacher, optimizer)


# Construct tensorboard logger
distiller.init_tensorboard_logger()

for epoch in range(startEpoch, epochs+1):
    print(f"Epoch {epoch}:")
    # Training step for one full epoch
    trainMetrics = distiller.train_step(
        student=student,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        objective=objective,
        distillObjective=distillObjective
    )

    # Validation step for one full epoch
    validMetrics = distiller.validate(
        student=student,
        dataloader=testloader,
        objective=objective
    )
    metrics = {**trainMetrics, **validMetrics}

    # Log to tensorbard
    distiller.log(epoch, metrics)

    # Save model
    distiller.save(epoch, student, teacher, optimizer)

    # Print epoch performance
    distiller.print_epoch(epoch, epochs, metrics)
