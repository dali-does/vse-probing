import os
import nltk
import torch
import pickle
import numpy as np

from transformers import BertModel, BertTokenizer
from transformers import GPT2Model, GPT2Tokenizer

def compute_embeddings(embedder, train_set_annotation,
                       test_set_annotation, merge_funcs=[]):
    all_embs = []
    embs = embedder(train_set_annotation)
    embs_val = embedder(test_set_annotation)

    all_embs.append((embs[0],embs_val[0]))
    if (len(embs) > 1):
        all_embs.append((embs[1],embs_val[1]))
        for f in merge_funcs:
            all_embs.append((f(embs),f(embs_val)))

    return all_embs

def average_embeddings(embs):
    return np.mean(embs,axis=0)

def concatenate_embeddings(embs):
    return np.concatenate(embs, axis=1)

def compute_gpt_embeddings(annotation_data):
    pretrained_weights = 'gpt2'
    tokenizer = GPT2Tokenizer.from_pretrained(pretrained_weights)
    model = GPT2Model.from_pretrained(pretrained_weights)
    return compute_transformer_embeddings(model, tokenizer, annotation_data)

def compute_bert_embeddings(annotation_data):
    pretrained_weights = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(pretrained_weights)
    model = BertModel.from_pretrained(pretrained_weights)

    return compute_transformer_embeddings(model, tokenizer, annotation_data)

def compute_transformer_embeddings(model, tokenizer, annotation_data):
    model = model.to(torch.device('cuda'))

    vocab = lambda x: tokenizer.encode(x, add_special_tokens=True)
    dset = PrecompDataset(annotation_data, vocab)
    data_loader = get_precomp_loader(dset)
    
    embs = transformer_embed_data(model, tokenizer, data_loader)
    return (embs,)

def compute_vse_embeddings(model_constructor, model_path, annotation_data):
    checkpoint = torch.load(model_path)
    opt = checkpoint['opt']

    # load vocabulary used by the model
    with open(os.path.join(opt.vocab_path,
                           '%s_vocab.pkl' % opt.data_name), 'rb') as f:
        voc = pickle.load(f)
    opt.vocab_size = len(voc)

    vocab = lambda x: vse_vocab_transformer(voc, x)

    dset = PrecompDataset(annotation_data, vocab)
    data_loader = get_precomp_loader(dset)

    # construct model
    model = model_constructor(opt)
    model.load_state_dict(checkpoint['model'])

    return vse_embed_data(model, data_loader)

def vse_vocab_transformer(vocab, caption):
    tokens = nltk.tokenize.word_tokenize(str(caption).lower())
    caption_tokenized = []

    caption_tokenized.append(vocab('<start>'))
    caption_tokenized.extend([vocab(token) for token in tokens])
    caption_tokenized.append(vocab('<end>'))

    return caption_tokenized

def transformer_embed_data(model, tokenizer, data_loader):
    # numpy array to keep all the embeddings
    cap_embs = None

    model.eval()
    with torch.no_grad():
        for i, (images, captions, lengths, ids, img_ids, indices) in enumerate(data_loader):
            # compute the embeddings
            captions = captions.to('cuda')
            cap_emb = model(captions)
            # TODO describe
            cap_emb = cap_emb[0]

            # initialize the numpy arrays given the size of the embeddings
            if cap_embs is None:
                cap_embs = np.zeros((len(data_loader.dataset), cap_emb.size(2)))

            # preserve the embeddings by copying from gpu and converting to numpy
            cap_embs[ids] = np.sum(cap_emb.data.cpu().numpy().copy(), axis=1)

            del captions
    return cap_embs


def vse_embed_data(model, data_loader):
    # numpy array to keep all the embeddings
    img_embs = None
    cap_embs = None

    model.val_start()
    with torch.no_grad():
        for i, (images, captions, lengths, ids, img_ids, indices) in enumerate(data_loader):
            # compute the embeddings
            img_emb, cap_emb = model.forward_emb(images, captions, lengths,
                                                 volatile=True)
            # initialize the numpy arrays given the size of the embeddings
            if img_embs is None:
                img_embs = np.zeros((len(data_loader.dataset), img_emb.size(1)))
                cap_embs = np.zeros((len(data_loader.dataset), cap_emb.size(1)))
    
            # preserve the embeddings by copying from gpu and converting to numpy
            img_embs[ids] = img_emb.data.cpu().numpy().copy()
            cap_embs[ids] = cap_emb.data.cpu().numpy().copy()
    
            del images, captions

    return img_embs, cap_embs


def get_precomp_loader(dset, batch_size=1024,
                       shuffle=False, num_workers=3):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    return torch.utils.data.DataLoader(dataset=dset,
                                       batch_size=batch_size,
                                       shuffle=shuffle,
                                       pin_memory=True,
                                       num_workers=num_workers,
                                       collate_fn=collate_fn)
def collate_fn(data):
    """Build mini-batch tensors from a list of (image, caption) tuples.
    Args:
        data: list of (image, caption) tuple.
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.


    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions, ids, img_ids, coco_ids, indices = zip(*data)

    # Merge images (convert tuple of 3D tensor to 4D tensor)
    images = torch.stack(images, 0)

    # Merget captions (convert tuple of 1D tensor to 2D tensor)
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    return images, targets, lengths, ids, coco_ids, indices

class PrecompDataset(torch.utils.data.Dataset):
    """
    Load precomputed captions and image features
    Possible options: f8k, f30k, coco, 10crop
    """

    def __init__(self, annotation_data, vocab, caption_encoder=None):
        self.vocab = vocab
        self.caption_encoder = caption_encoder
        self.annotation_data = annotation_data

    def __len__(self):
        return len(self.annotation_data)

    def __getitem__(self, index):
        # handle the image redundancy
        image, caption, index, img_id, coco_id = self.annotation_data[index]

        ## Convert caption (string) to word ids.
        caption = self.vocab(caption)
        target = torch.Tensor(caption)

        return image, target, index, img_id, coco_id, index

