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

# inport different ttm compression realization

from exps.ttm_v2.modules import TTCompressedLinear
from exps.ttm.TTLinear import TTLinear
from src.ttm_linear.ttm_linear.ttm_linear import FactorizationTTMLinear
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
    
    print ("parameter number in model", colbert.num_parameters())  
    
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
            
            
    if (args.compressed_type == 'svd_select'):
        for i in [3, 4, 5]:
            # fc part
            rank = 1
            
            print ("fc shape", sum(p.numel() for p in colbert.bert.encoder.layer[i].intermediate.dense.parameters()))
            fc_w = colbert.bert.encoder.layer[i].intermediate.dense.weight.data.cpu().data.numpy()
            fc_b = colbert.bert.encoder.layer[i].intermediate.dense.bias.data.cpu().data.numpy()
            factorized_layer = factorize_to_svd(fc_w, fc_b, rank = rank)
            colbert.bert.encoder.layer[i].intermediate.dense = factorized_layer
            
            fc_w = colbert.bert.encoder.layer[i].output.dense.weight.data.cpu().data.numpy()
            fc_b = colbert.bert.encoder.layer[i].output.dense.bias.data.cpu().data.numpy()
            factorized_layer = factorize_to_svd(fc_w, fc_b, rank = rank)
            colbert.bert.encoder.layer[i].output.dense = factorized_layer
            
            
            print ("factorized_layer shape", sum(p.numel() for p in factorized_layer.parameters()))
        
    elif (args.compressed_type == 'ttm'): # only intermediate, to do add output
        for i in [0, 2, 4, 6, 8, 10]:
            # fc part
            rank = 50
            fc = colbert.bert.encoder.layer[i].intermediate.dense
            (out_, in_) = fc.weight.shape
            factorized_layer = TTLinear(in_features = 768, out_features = 3072, ranks =[rank, rank, rank] , input_dims = [12, 2, 2 ,16], output_dims= [32, 2, 3, 16])
            factorized_layer.set_weight(fc.weight, need_singular_values = False)
            factorized_layer.set_bias(fc.bias)
            print ("cores number = ", len(factorized_layer.cores))
            for elem in factorized_layer.cores:
                print (elem.shape)
            print ("\n")
            print ("fc shape", sum(p.numel() for p in fc.parameters()))
            print ("factorized_layer shape", sum(p.numel() for p in factorized_layer.parameters()))
            colbert.bert.encoder.layer[i].intermediate.dense = factorized_layer
            
            
            # output part
            fc = colbert.bert.encoder.layer[i].output.dense

            rank = 50  # Uniform TT-rank.
            factorized_layer = TTLinear(in_features = 3072, out_features = 768, ranks =[rank, rank, rank] , input_dims = [32, 2, 3, 16], output_dims= [12, 2, 2 ,16])
            factorized_layer.set_weight(fc.weight, need_singular_values = False)
            factorized_layer.set_bias(fc.bias)
            for elem in factorized_layer.cores:
                print (elem.shape)
            print ("\n")
            
            print ("fc shape", sum(p.numel() for p in fc.parameters()))
            print ("factorized_layer shape", sum(p.numel() for p in factorized_layer.parameters()))
            
            colbert.bert.encoder.layer[i].output.dense = factorized_layer
            
    elif (args.compressed_type == 'ttm_tntorch'): # only intermediate, to do add output
        for i in [3, 4, 5]:
            # fc part
            rank = 20
            fc = colbert.bert.encoder.layer[i].intermediate.dense
            (out_, in_) = fc.weight.shape
            factorized_layer = TTLinear(in_features = 768, out_features = 3072, ranks =[rank, rank, rank] , input_dims = [12, 2, 2 ,16], output_dims= [32, 2, 3, 16])
            factorized_layer.set_weight(fc.weight, need_singular_values = False)
            factorized_layer.set_bias(fc.bias)
            print ("cores number = ", len(factorized_layer.cores))
            for elem in factorized_layer.cores:
                print (elem.shape)
            print ("\n")
            print ("fc shape", sum(p.numel() for p in fc.parameters()))
            print ("factorized_layer shape", sum(p.numel() for p in factorized_layer.parameters()))
            colbert.bert.encoder.layer[i].intermediate.dense = factorized_layer
            
            
            # output part
            fc = colbert.bert.encoder.layer[i].output.dense

            rank = 5  # Uniform TT-rank.
            factorized_layer = TTLinear(in_features = 3072, out_features = 768, ranks =[rank, rank, rank] , input_dims = [32, 2, 3, 16], output_dims= [12, 2, 2 ,16])
            factorized_layer.set_weight(fc.weight, need_singular_values = False)
            factorized_layer.set_bias(fc.bias)
            for elem in factorized_layer.cores:
                print (elem.shape)
            print ("\n")
            
            print ("fc shape", sum(p.numel() for p in fc.parameters()))
            print ("factorized_layer shape", sum(p.numel() for p in factorized_layer.parameters()))
            
            colbert.bert.encoder.layer[i].output.dense = factorized_layer
            
    elif (args.compressed_type == 'ttm_custom'): # only intermediate, to do add output    
        for i in [3, 4, 5]:
            # fc part
            shape = (
            (12, 2, 2, 16),  # Row dimention.
            (32, 3, 2, 16),  # Column dimention.
            )

            rank = 20
            fc_w = colbert.bert.encoder.layer[i].intermediate.dense
            factorized_layer = TTCompressedLinear.from_linear(fc_w, shape=shape, rank=rank)
            colbert.bert.encoder.layer[i].intermediate.dense = factorized_layer
            
            # output part
            
            rank = 5  # Uniform TT-rank.
            shape = (
            (12, 2, 2, 16), # Column dimention.
            (32, 3, 2, 16), # Row dimention.    
            )
            
            fc_w = colbert.bert.encoder.layer[i].output.dense
            factorized_layer = TTCompressedLinear.from_linear(fc_w, shape=shape, rank=rank)
            colbert.bert.encoder.layer[i].output.dense = factorized_layer         
            
            
    elif (args.compressed_type == 'tt'): 
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

    print ("parameter number in model after compression", colbert.num_parameters())  
    

    colbert = colbert.to(DEVICE)

    colbert.eval()

    return colbert, checkpoint
