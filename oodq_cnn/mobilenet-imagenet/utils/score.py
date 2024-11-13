import torch
from torch.optim.optimizer import Optimizer, required
from utils.options import args
import numpy as np
from numpy.linalg import norm, pinv
from scipy.special import logsumexp
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

import json

from sklearn.covariance import EmpiricalCovariance

from utils.mahalanobis_lib import get_Mahalanobis_score
import math

device = torch.device(f"cuda:{args.gpus[0]}")


def get_msp_score(inputs, model, logits=None):
    if logits is None:
        with torch.no_grad():
            logits = model(inputs)
    scores = np.max(F.softmax(logits, dim=1).detach().cpu().numpy(), axis=1)
    return scores

def get_energy_score(inputs, model, logits=None):
    if logits is None:
        with torch.no_grad():
            logits = model(inputs)

    scores = torch.logsumexp(logits.data.cpu(), dim=1).numpy()
    return scores

def get_odin_score(inputs, model, temp = 1000.0, magnitude = 0.005):
    temper = temp
    noiseMagnitude1 = magnitude

    criterion = nn.CrossEntropyLoss()
    inputs = torch.autograd.Variable(inputs, requires_grad = True)
    # outputs = model(inputs)
    outputs = model(inputs)

    maxIndexTemp = np.argmax(outputs.data.cpu().numpy(), axis=1)

    # Using temperature scaling
    outputs = outputs / temper

    labels = torch.autograd.Variable(torch.LongTensor(maxIndexTemp).cuda())
    loss = criterion(outputs, labels)
    loss.backward()

    # Normalizing the gradient to binary in {0, 1}
    gradient =  torch.ge(inputs.grad.data, 0)
    gradient = (gradient.float() - 0.5) * 2

    # Adding small perturbations to images
    tempInputs = torch.add(inputs.data,  -noiseMagnitude1, gradient)
    # outputs = model(Variable(tempInputs))
    with torch.no_grad():
        outputs = model(tempInputs)
    outputs = outputs / temper
    # Calculating the confidence after adding perturbations
    nnOutputs = outputs.data.cpu()
    nnOutputs = nnOutputs.numpy()
    nnOutputs = nnOutputs - np.max(nnOutputs, axis=1, keepdims=True)
    nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs), axis=1, keepdims=True)
    scores = np.max(nnOutputs, axis=1)

    return scores


def get_mahalanobis_score(inputs, model, method_args):
    num_classes = method_args['num_classes']
    sample_mean = method_args['sample_mean']
    precision = method_args['precision']
    magnitude = method_args['magnitude']
    regressor = method_args['regressor']
    num_output = method_args['num_output']

    Mahalanobis_scores = get_Mahalanobis_score(inputs, model, num_classes, sample_mean, precision, num_output, magnitude)
    scores = -regressor.predict_proba(Mahalanobis_scores)[:, 1]

    return scores

def get_react_score(inputs, model, clip=None, logits=None):
    if logits is None:
        with torch.no_grad():
            _ = model(inputs)
        
        feat = model.feature
        
    if clip is None:
        clip = torch.quantile(feat.detach().cpu(), 0.9)
    
    feat_clip = torch.clip(feat.detach().cpu(), None, clip).to(device)
    
    logits = model.classifier(feat_clip)

    scores = torch.logsumexp(logits.data.cpu(), dim=1).numpy()
    
    return scores, clip

def get_vim_score(inputs, model, alpha=None, logits=None):
    with torch.no_grad():
        logits = model(inputs).cpu().numpy()
        
        feat = model.feature.cpu().numpy()
    
    w = model.classifier.weight.data.cpu().numpy()
    b = model.classifier.bias.data.cpu().numpy()
    
    u = -np.matmul(pinv(w), b)
    
    DIM = 1000 if feat.shape[-1] >= 2048 else 512
    
    ec = EmpiricalCovariance(assume_centered=True)
    ec.fit(feat - u)
    eig_vals, eigen_vectors = np.linalg.eig(ec.covariance_)
    NS = np.ascontiguousarray((eigen_vectors.T[np.argsort(eig_vals * -1)[DIM:]]).T)
    
    vim_scores = norm(np.matmul(feat - u, NS), axis=-1)
    
    if alpha is None:
        alpha = logits.max(axis=-1).mean() / vim_scores.mean()
 
    vim_scores *= alpha

    energy_scores = logsumexp(logits, axis=-1)
    
    scores = -vim_scores + energy_scores
    
    return scores, alpha


def get_quant_score(inputs, model, clip_l=None, clip=None, m=None, s=None, logits=None):
    if logits is None:
        with torch.no_grad():
            _ = model(inputs)
        
        feat = model.feature
    
    if clip_l is None:
        clip_l = torch.quantile(feat.detach().cpu(), args.act_clip_p_l)
    
    if clip is None:
        clip = torch.quantile(feat.detach().cpu(), 1 - args.act_clip_p)
        #m = torch.mean(feat.detach().cpu())
        #s = torch.std(feat.detach().cpu())
    
    feat_q, (m, s) = activation_quantize_fn(args.abitW, feat, clip_l, clip, m, s)
    
    feat_q = feat_q.to(device)
    
    logits = model.classifier(feat_q)
    
    scores = torch.logsumexp(logits.data.cpu(), dim=1).numpy()
    
    return scores, clip_l, clip, m, s

def get_quant_mixed_score(inputs, model, 
                          clip_l=None, clip=None, 
                          m=None, s=None, 
                          ood_m=None, ood_s=None, ood_n=None,
                          id_m=None, id_s=None,
                          thres=None, logits=None):
    if logits is None:
        with torch.no_grad():
            _ = model(inputs)
        
        feat = model.feature.detach().cpu()
    
    # quantize
    if clip is None:
        clip = torch.quantile(feat, 1 - args.act_clip_p)
        
    if clip_l is None:
        clip_l = torch.quantile(feat, args.act_clip_p_l)
    
    feat_q, (m, s) = activation_quantize_fn(args.abitW, feat, clip_l, clip, m, s)
    
    feat_q = feat_q.to(device)
    

    # calculate scores
    logits = model.classifier(feat_q)
    
    scores = torch.logsumexp(logits.data.cpu(), dim=1)
    
    # validation stage or not
    is_test = ('test' in args.csv_dir)
    
    # get pseudo labels 
    is_ood = (scores < thres).unsqueeze(-1)
    is_id = (scores >= thres).unsqueeze(-1)
        
    is_exist_pred_ood = sum(is_ood) > 0
    
    conf_scores = torch.zeros([1])
    scores2 = scores
    
    #print(is_test)
            
    if is_test and is_exist_pred_ood: 
        if ood_n is None:
            # update ood distribution (feature mean & std)
            pred_ood_features = feat[torch.where(is_ood)].detach().cpu().numpy()
        
            ood_n = np.prod(pred_ood_features.shape)
            ood_m = np.mean(pred_ood_features)
            ood_s = np.std(pred_ood_features)
            
            pred_id_features = feat[torch.where(is_id)].detach().cpu().numpy()
        
            id_m = np.mean(pred_id_features)
            id_s = np.std(pred_id_features)
    
        else:
            # calculate confidence from scores (pseudo labels)
            conf_scores = 1 - torch.exp(-torch.abs(scores - thres).unsqueeze(-1))
            id_conf = conf_scores*is_id
            ood_conf = conf_scores*is_ood
            
            # calculate post-processing features
            ood_indicator = torch.ones(feat.shape)*is_ood
            
            id_feat_q = feat_q.detach().cpu() * (1 - ood_indicator)
            
       
            id_m = id_feat_q.mean(-1).unsqueeze(-1) * (1-id_conf) + id_m * id_conf
            id_s = id_feat_q.std(-1).unsqueeze(-1) * (1-id_conf) + id_s * id_conf
            
            id_feat_q = (id_feat_q - m) / s * id_s + id_m
            #id_feat_q = (id_feat_q - m) / s * id_s + id_m
            
            ood_feat_q = feat_q.detach().cpu() * (ood_indicator)
            
            ood_m = ood_feat_q.mean(-1).unsqueeze(-1) * (1-ood_conf) + ood_m * ood_conf
            ood_s = ood_feat_q.std(-1).unsqueeze(-1) * (1-ood_conf) + ood_s * ood_conf
            
            ood_feat_q = (ood_feat_q - m) / s * ood_s + ood_m
            #ood_feat_q = (ood_feat_q - m) / s * ood_s + ood_m
            
            feat_q_post = id_feat_q + ood_feat_q
            
            # calculate scores
            logits = model.classifier(feat_q_post.to(device))
        
            scores2 = torch.logsumexp(logits.data.cpu(), dim=1)
            
            # update ood distribution (feature mean & std)
            pred_ood_features = feat[torch.where(is_ood)].detach().cpu().numpy()
            
            ood_n = np.prod(pred_ood_features.shape)
            ood_m = np.mean(pred_ood_features)
            ood_s = np.std(pred_ood_features)
            
            pred_id_features = feat[torch.where(is_id)].detach().cpu().numpy()

            id_m = np.mean(pred_id_features)
            id_s = np.std(pred_id_features)
            
    if is_test:
        total_conf = torch.mean(conf_scores) #args.lam
        scores = scores*(1-total_conf) + scores2*total_conf
        return scores.numpy(), ood_m, ood_s, ood_n, id_m, id_s
    else:
        return scores.numpy(), clip_l, clip, m, s 


def uniform_quantize(k, input):

    if k == 32:
      out = input
    elif k == 1:
      out = torch.sign(input)
    else:
      n = 2 ** k - 1
      out = torch.round(input * n) / n
    return out

def activation_quantize_fn(a_bit, x, clip_l, clip, m=None, s=None):
    x_ = x.detach().cpu()
    
    if m is None:
        m = torch.mean(x_)
        s = torch.std(x_)
        
    # clipping outliers (redundant info)
    x = torch.clip(x_, None, clip)
    #x = torch.clip(x_, clip_l, clip)
    
    # range (L, U) shifted/scaled to (0, 1)
    L = torch.min(x) #clip_l
    U = clip
    
    x = (x - L)/(U - L)
    
    # quantize
    activation_q = uniform_quantize(a_bit, x)
    
    # range (0, 1) shifted/scaled back to (L, U)
    activation_q = activation_q*(U - L) + L
    
    return activation_q, (m, s)


def activation_quantize_fn0(a_bit, x, clip, m=None, s=None):
    x_ = x.detach().cpu()
    
    
    # clipping outliers (redundant info)
    x = torch.clip(x_, None, clip)
    
    if m is None:
        q_err = torch.abs(x_ - x)
        m = torch.mean(q_err)
        s = torch.std(q_err)
            
    
    # range (L, U) shifted/scaled to (0, 1)
    L = torch.min(x)
    U = torch.max(x)
    
    x = (x - L)/(U - L)
    
    # quantize
    activation_q = uniform_quantize(a_bit, x)
    
    # range (0, 1) shifted/scaled back to (L, U)
    activation_q = activation_q*(U - L) + L
    
    return activation_q, (m, s)
