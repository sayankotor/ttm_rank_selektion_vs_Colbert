import os
import ujson
import torch
import random

from collections import defaultdict, OrderedDict

from colbert.parameters import DEVICE
from colbert.modeling.colbert import ColBERT
from colbert.utils.utils import print_message, load_checkpoint

from colbert.utils.utils import factorize_to_svd
import tntorch as tn

from src.ttm_linear.ttm_linear.ttm_linear import FactorizationTTMLinear
import tensorly
tensorly.set_backend("pytorch")

from .tensor_net import TTLayer


TT_SHAPES = (32, 48, 48, 32)
TT_RANKS = [1, 380, 390, 380, 1] # comp rate 0.5
SVD_RANKS = 350

# Initializing a BERT bert-base-uncased style configuration
#configuration = BertConfig()

# Initializing a model from the bert-base-uncased style configuration
#model = BertModel(configuration)

TT_SHAPES = [32, 48, 48, 32]
TT_RANKS = [1, 380, 390, 390, 1]


def load_model(args, do_print=True):
    colbert = ColBERT.from_pretrained('bert-base-uncased',
                                      query_maxlen=args.query_maxlen,
                                      doc_maxlen=args.doc_maxlen,
                                      dim=args.dim,
                                      similarity_metric=args.similarity,
                                      mask_punctuation=args.mask_punctuation)
    colbert = colbert.to(DEVICE)

    print_message("#> Loading model checkpoint.", condition=do_print)

    checkpoint = load_checkpoint(args.checkpoint, colbert, do_print=do_print)

    colbert.eval()

    return colbert, checkpoint

def load_compressed_model(args, do_print=True):
    colbert = ColBERT.from_pretrained('bert-base-uncased',
                                      query_maxlen=args.query_maxlen,
                                      doc_maxlen=args.doc_maxlen,
                                      dim=args.dim,
                                      similarity_metric=args.similarity,
                                      mask_punctuation=args.mask_punctuation)
    #dictt1 = torch.load(args.checkpoint + "model_tt.pth", map_location='cpu')
    #colbert.bert.load_state_dict(dictt1)   
    

    print_message("#> Loading model checkpoint.", condition=do_print)

    checkpoint = load_checkpoint(args.checkpoint, colbert, do_print=do_print)
    
    print_message("#> Compressed BERT.", condition=do_print)
    if (args.compressed_type == 'svd'):
        for i in [0, 2, 4, 6, 8, 10]:
            # fc part
            fc_w = colbert.bert.encoder.layer[i].intermediate.dense.weight.data.cpu().data.numpy()
            fc_b = colbert.bert.encoder.layer[i].intermediate.dense.bias.data.cpu().data.numpy()
            factorized_layer = factorize_to_svd(fc_w, fc_b, rank = args.c_rank)
            colbert.bert.encoder.layer[i].intermediate.dense = factorized_layer
            
            fc_w = colbert.bert.encoder.layer[i].output.dense.weight.data.cpu().data.numpy()
            fc_b = colbert.bert.encoder.layer[i].output.dense.bias.data.cpu().data.numpy()
            factorized_layer = factorize_to_svd(fc_w, fc_b, rank = args.c_rank)
            colbert.bert.encoder.layer[i].output.dense = factorized_layer
        
    if (args.compressed_type == 'ttm'): # only intermediate, to do add output
        for i in [0, 2, 4, 6, 8, 10]:
            # fc part
            fc_w = colbert.bert.encoder.layer[i].intermediate.dense.weight.data.cpu()
            fc_b = colbert.bert.encoder.layer[i].intermediate.dense.bias.data.cpu()
            (out_, in_) = fc_w.shape
            factorized_layer = FactorizationTTMLinear(in_, out_, rank=args.c_rank, max_core_dim_product =args.c_rank)
            factorized_layer.fill_with_pretrained_matrix(fc_w, reshape_sizes = (8, 12, 8, 16, 12, 16))
            print (factorized_layer.ttm.tt.cores)
            for elem in factorized_layer.ttm.tt.cores:
                print (elem.shape)
            colbert.bert.encoder.layer[i].intermediate.dense = factorized_layer
            
    if (args.compressed_type == 'tt'): 
        for i in [0, 2, 4, 6, 8, 10]:
            # fc part
            fc_w = colbert.bert.encoder.layer[i].intermediate.dense
            fc_b = colbert.bert.encoder.layer[i].intermediate.dense.bias
            (out_, in_) = fc_w.weight.shape
            factorized_layer = TTLayer(fc_w, shapes = TT_SHAPES, in_dims = [32, 24], ranks = TT_RANKS)
            for elem in factorized_layer.cores:
                print (elem.shape)

            colbert.bert.encoder.layer[i].intermediate.dense = factorized_layer

            fc_w = colbert.bert.encoder.layer[i].output.dense
            factorized_layer = TTLayer(fc_w, shapes = TT_SHAPES, in_dims = [32, 48, 2], ranks = TT_RANKS)
            colbert.bert.encoder.layer[i].output.dense = factorized_layer
            print (factorized_layer.cores)
            for elem in factorized_layer.cores:
                print (elem.shape)
    else:
        print_message("#> Compressing type is not supported.", condition=do_print)

    print ("parameter number in model", colbert.num_parameters())    
    colbert = colbert.to(DEVICE)

    colbert.eval()

    return colbert, checkpoint
