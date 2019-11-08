# -*- coding: utf-8 -*-
"""
PyTorch Implementation of NER with pretrained Bert

Code from https://github.com/Kyubyong/bert_ner
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from .model import Net
from .data_load import NerDataset, pad, tokenizer
import os
import numpy as np

def train(model, iterator, optimizer, criterion):
    model.train()
    for i, batch in enumerate(iterator):
        words, x, is_heads, tags, y, seqlens = batch
        _y = y # for monitoring
        optimizer.zero_grad()
        logits, y, _ = model(x, y) # logits: (N, T, VOCAB), y: (N, T)

        logits = logits.view(-1, logits.shape[-1]) # (N*T, VOCAB)
        y = y.view(-1)  # (N*T,)

        loss = criterion(logits, y)
        loss.backward()

        optimizer.step()

        if i==0:
            print("=====sanity check======")
            print("words:", words[0])
            print("x:", x.cpu().numpy()[0][:seqlens[0]])
            print("tokens:", tokenizer.convert_ids_to_tokens(x.cpu().numpy()[0])[:seqlens[0]])
            print("is_heads:", is_heads[0])
            print("y:", _y.cpu().numpy()[0][:seqlens[0]])
            print("tags:", tags[0])
            print("seqlen:", seqlens[0])
            print("=======================")

        if i%10==0: # monitoring
            print(f"step: {i}, loss: {loss.item()}")

def eval(model, iterator, f,tag2idx, idx2tag):
    model.eval()

    Words, Is_heads, Tags, Y, Y_hat = [], [], [], [], []
    with torch.no_grad():
        for i, batch in enumerate(iterator):
            words, x, is_heads, tags, y, seqlens = batch

            _, _, y_hat = model(x, y)  # y_hat: (N, T)

            Words.extend(words)
            Is_heads.extend(is_heads)
            Tags.extend(tags)
            Y.extend(y.numpy().tolist())
            Y_hat.extend(y_hat.cpu().numpy().tolist())

    ## gets results and save
    with open("temp", 'w') as fout:
        for words, is_heads, tags, y_hat in zip(Words, Is_heads, Tags, Y_hat):
            y_hat = [hat for head, hat in zip(is_heads, y_hat) if head == 1]
            preds = [idx2tag[hat] for hat in y_hat]
            assert len(preds)==len(words.split())==len(tags.split())
            for w, t, p in zip(words.split()[1:-1], tags.split()[1:-1], preds[1:-1]):
                fout.write(f"{w} {t} {p}\n")
            fout.write("\n")

    ## calc metric
    y_true =  np.array([tag2idx[line.split()[1]] for line in open("temp", 'r').read().splitlines() if len(line) > 0])
    y_pred =  np.array([tag2idx[line.split()[2]] for line in open("temp", 'r').read().splitlines() if len(line) > 0])

    num_proposed = len(y_pred[y_pred>1])
    num_correct = (np.logical_and(y_true==y_pred, y_true>1)).astype(np.int).sum()
    num_gold = len(y_true[y_true>1])

    print(f"num_proposed:{num_proposed}")
    print(f"num_correct:{num_correct}")
    print(f"num_gold:{num_gold}")
    try:
        precision = num_correct / num_proposed
    except ZeroDivisionError:
        precision = 1.0

    try:
        recall = num_correct / num_gold
    except ZeroDivisionError:
        recall = 1.0

    try:
        f1 = 2*precision*recall / (precision + recall)
    except ZeroDivisionError:
        if precision*recall==0:
            f1=1.0
        else:
            f1=0

    final = f + ".P%.2f_R%.2f_F%.2f" %(precision, recall, f1)
    with open(final, 'w') as fout:
        result = open("temp", "r").read()
        fout.write(f"{result}\n")

        fout.write(f"precision={precision}\n")
        fout.write(f"recall={recall}\n")
        fout.write(f"f1={f1}\n")

    os.remove("temp")

    print("precision=%.2f"%precision)
    print("recall=%.2f"%recall)
    print("f1=%.2f"%f1)
    return precision, recall, f1

def train_go(train_dataset,tag_all,logdir,eval_dataset=None,num_workers=1,ep=30,lr=0.0001,batch_size=128,finetuning=False,top_rnns=False):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_dataset = NerDataset(train_dataset,tag_all)
    if device == 'cuda':
        model = Net(top_rnns, len(train_dataset.VOCAB), device, finetuning).cuda()
    else:
        model = Net(top_rnns, len(train_dataset.VOCAB), device, finetuning)
    model = nn.DataParallel(model)

    train_iter = data.DataLoader(dataset=train_dataset,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=num_workers,
                                 collate_fn=pad)
    if eval_dataset!= None:
        eval_dataset = NerDataset(eval_dataset,tag_all)
        eval_iter = data.DataLoader(dataset=eval_dataset,batch_size=batch_size,shuffle=False,num_workers=num_workers,collate_fn=pad)

    optimizer = optim.Adam(model.parameters(), lr = lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    for epoch in range(1, ep+1):
        train(model, train_iter, optimizer, criterion)
        fname = os.path.join(logdir, str(epoch))
        if eval_dataset!= None:
            print(f"=========eval at epoch={epoch}=========")
            if not os.path.exists(logdir): os.makedirs(logdir)
            precision, recall, f1 = eval(model, eval_iter, fname,eval_dataset.tag2idx,eval_dataset.idx2tag)

        torch.save(model.state_dict(), f"{fname}.pt")
        print(f"weights were saved to {fname}.pt")
