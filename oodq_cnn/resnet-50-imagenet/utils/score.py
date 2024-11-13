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
import warnings

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
    
    logits = model.fc(feat_clip)

    scores = torch.logsumexp(logits.data.cpu(), dim=1).numpy()
    
    return scores, clip

def get_vim_score(inputs, model, alpha=None, logits=None):
    with torch.no_grad():
        logits = model(inputs).cpu().numpy()
        
        feat = model.feature.cpu().numpy()
    
    w = model.fc.weight.data.cpu().numpy()
    b = model.fc.bias.data.cpu().numpy()
    
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
    
    logits = model.fc(feat_q)
    
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
    logits = model.fc(feat_q)
    
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
            logits = model.fc(feat_q_post.to(device))
        
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
        total_conf = args.lam # torch.mean(conf_scores)
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
    # x = torch.clip(x_, None, clip)
    x = torch.clip(x_, clip_l, clip)
    
    # range (L, U) shifted/scaled to (0, 1)
    L = clip_l # torch.min(x)
    U = clip
    
    x = (x - L)/(U - L)
    
    # quantize
    activation_q = uniform_quantize(a_bit, x)
    
    # range (0, 1) shifted/scaled back to (L, U)
    activation_q = activation_q*(U - L) + L
    
    return activation_q, (m, s)

def activation_quantize_fn2(a_bit, x, clip, m=None, s=None):
    x_ = x.detach().cpu()
    
    if m is None:
        m = torch.mean(x_)
        s = torch.std(x_)
        
    # clipping outliers (redundant info)
    x = torch.clip(x_, None, clip)

    
    # range (L, U) shifted/scaled to (0, 1)
    L = torch.min(x)
    U = clip
    
    x = (x - L)/(U - L)
    
    # quantize
    if args.method == 'uniform':
        activation_q = uniform_quantize(a_bit, x)
    elif args.method == 'brecq':
        activation_q = UniformAffineQuantizer(a_bit)(x)
    elif args.method == 'pdquant':
        activation_q = UniformAffineQuantizer2(a_bit)(x)
    elif args.method == 'adaround':
        u = UniformAffineQuantizer(a_bit)
        _ = u(x)
        activation_q = AdaRoundQuantizer(a_bit, u)(x)
    elif args.method == 'lifequant':
        activation_q = lifequant(a_bit)(x)
        
    # range (0, 1) shifted/scaled back to (L, U)
    activation_q = activation_q*(U - L) + L
    
    return activation_q, (m, s)


class StraightThrough(nn.Module):
    def __init__(self, channel_num: int = 1):
        super().__init__()

    def forward(self, input):
        return input


def round_ste(x: torch.Tensor):
    """
    Implement Straight-Through Estimator for rounding operation.
    """
    return (x.round() - x).detach() + x


def lp_loss(pred, tgt, p=2.0, reduction='none'):
    """
    loss function measured in L_p Norm
    """
    if reduction == 'none':
        return (pred-tgt).abs().pow(p).sum(1).mean()
    else:
        return (pred-tgt).abs().pow(p).mean()

class UniformAffineQuantizer2(nn.Module):
    """
    PyTorch Function that can be used for asymmetric quantization (also called uniform affine
    quantization). Quantizes its argument in the forward pass, passes the gradient 'straight
    through' on the backward pass, ignoring the quantization that occurred.
    Based on https://arxiv.org/abs/1806.08342.

    :param n_bits: number of bit for quantization
    :param symmetric: if True, the zero_point should always be 0
    :param channel_wise: if True, compute scale and zero_point in each channel
    :param scale_method: determines the quantization scale and zero point
    :param prob: for qdrop;
    """

    def __init__(self, n_bits: int = 8, symmetric: bool = False, channel_wise: bool = False,
                 scale_method: str = 'minmax',
                 leaf_param: bool = False, prob: float = 1.0):
        super(UniformAffineQuantizer2, self).__init__()
        self.sym = symmetric
        if self.sym:
            raise NotImplementedError
        #assert 2 <= n_bits <= 8, 'bitwidth not supported'
        self.n_bits = n_bits
        self.n_levels = 2 ** self.n_bits
        self.delta = 1.0
        self.zero_point = 0.0
        self.inited = True

        '''if leaf_param, use EMA to set scale'''
        self.leaf_param = leaf_param
        self.channel_wise = channel_wise
        self.eps = torch.tensor(1e-8, dtype=torch.float32)

        '''mse params'''
        self.scale_method = 'mse'
        self.one_side_dist = None
        self.num = 100

        '''for activation quantization'''
        self.running_min = None
        self.running_max = None

        '''do like dropout'''
        self.prob = prob
        self.is_training = False

    def set_inited(self, inited: bool = True):  # inited manually
        self.inited = inited

    def update_quantize_range(self, x_min, x_max):
        if self.running_min is None:
            self.running_min = x_min
            self.running_max = x_max
        self.running_min = 0.1 * x_min + 0.9 * self.running_min
        self.running_max = 0.1 * x_max + 0.9 * self.running_max
        return self.running_min, self.running_max

    def forward(self, x: torch.Tensor):
        if self.inited is False:
            if self.leaf_param:
                self.delta, self.zero_point = self.init_quantization_scale(x.clone().detach(), self.channel_wise)
            else:
                self.delta, self.zero_point = self.init_quantization_scale(x.clone().detach(), self.channel_wise)

        # start quantization
        x_int = round_ste(x / self.delta) + self.zero_point
        x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
        x_dequant = (x_quant - self.zero_point) * self.delta
        if self.is_training and self.prob < 1.0:
            x_ans = torch.where(torch.rand_like(x) < self.prob, x_dequant, x)
        else:
            x_ans = x_dequant
        return x_ans

    def lp_loss(self, pred, tgt, p=2.0):
        x = (pred - tgt).abs().pow(p)
        if not self.channel_wise:
            return x.mean()
        else:
            y = torch.flatten(x, 1)
            return y.mean(1)

    def calculate_qparams(self, min_val, max_val):
        # one_dim or one element
        quant_min, quant_max = 0, self.n_levels - 1
        min_val_neg = torch.min(min_val, torch.zeros_like(min_val))
        max_val_pos = torch.max(max_val, torch.zeros_like(max_val))

        scale = (max_val_pos - min_val_neg) / float(quant_max - quant_min)
        scale = torch.max(scale, self.eps)
        zero_point = quant_min - torch.round(min_val_neg / scale)
        zero_point = torch.clamp(zero_point, quant_min, quant_max)
        return scale, zero_point

    def quantize(self, x: torch.Tensor, x_max, x_min):
        delta, zero_point = self.calculate_qparams(x_min, x_max)
        if self.channel_wise:
            new_shape = [1] * len(x.shape)
            new_shape[0] = x.shape[0]
            delta = delta.reshape(new_shape)
            zero_point = zero_point.reshape(new_shape)
        x_int = torch.round(x / delta)
        x_quant = torch.clamp(x_int + zero_point, 0, self.n_levels - 1)
        x_float_q = (x_quant - zero_point) * delta
        return x_float_q

    def perform_2D_search(self, x):
        if self.channel_wise:
            y = torch.flatten(x, 1)
            x_min, x_max = torch._aminmax(y, 1)
            # may also have the one side distribution in some channels
            x_max = torch.max(x_max, torch.zeros_like(x_max))
            x_min = torch.min(x_min, torch.zeros_like(x_min))
        else:
            x_min, x_max = torch._aminmax(x)
        xrange = x_max - x_min
        best_score = torch.zeros_like(x_min) + (1e+10)
        best_min = x_min.clone()
        best_max = x_max.clone()
        # enumerate xrange
        for i in range(1, self.num + 1):
            tmp_min = torch.zeros_like(x_min)
            tmp_max = xrange / self.num * i
            tmp_delta = (tmp_max - tmp_min) / (2 ** self.n_bits - 1)
            # enumerate zp
            for zp in range(0, self.n_levels):
                new_min = tmp_min - zp * tmp_delta
                new_max = tmp_max - zp * tmp_delta
                x_q = self.quantize(x, new_max, new_min)
                score = self.lp_loss(x, x_q, 2.4)
                best_min = torch.where(score < best_score, new_min, best_min)
                best_max = torch.where(score < best_score, new_max, best_max)
                best_score = torch.min(best_score, score)
        return best_min, best_max

    def perform_1D_search(self, x):
        if self.channel_wise:
            y = torch.flatten(x, 1)
            x_min, x_max = torch._aminmax(y, 1)
        else:
            x_min, x_max = torch._aminmax(x)
        xrange = torch.max(x_min.abs(), x_max)
        best_score = torch.zeros_like(x_min) + (1e+10)
        best_min = x_min.clone()
        best_max = x_max.clone()
        # enumerate xrange
        for i in range(1, self.num + 1):
            thres = xrange / self.num * i
            new_min = torch.zeros_like(x_min) if self.one_side_dist == 'pos' else -thres
            new_max = torch.zeros_like(x_max) if self.one_side_dist == 'neg' else thres
            x_q = self.quantize(x, new_max, new_min)
            score = self.lp_loss(x, x_q, 2.4)
            best_min = torch.where(score < best_score, new_min, best_min)
            best_max = torch.where(score < best_score, new_max, best_max)
            best_score = torch.min(score, best_score)
        return best_min, best_max

    def get_x_min_x_max(self, x):
        if self.scale_method != 'mse':
            raise NotImplementedError
        if self.one_side_dist is None:
            self.one_side_dist = 'pos' if x.min() >= 0.0 else 'neg' if x.max() <= 0.0 else 'no'
        if self.one_side_dist != 'no' or self.sym:  # one-side distribution or symmetric value for 1-d search
            best_min, best_max = self.perform_1D_search(x)
        else:  # 2-d search
            best_min, best_max = self.perform_2D_search(x)
        if self.leaf_param:
            return self.update_quantize_range(best_min, best_max)
        return best_min, best_max

    def init_quantization_scale_channel(self, x: torch.Tensor):
        x_min, x_max = self.get_x_min_x_max(x)
        return self.calculate_qparams(x_min, x_max)

    def init_quantization_scale(self, x_clone: torch.Tensor, channel_wise: bool = False):
        if channel_wise:
            # determine the scale and zero point channel-by-channel
            delta, zero_point = self.init_quantization_scale_channel(x_clone)
            new_shape = [1] * len(x_clone.shape)
            new_shape[0] = x_clone.shape[0]
            delta = delta.reshape(new_shape)
            zero_point = zero_point.reshape(new_shape)
        else:
            delta, zero_point = self.init_quantization_scale_channel(x_clone)
        return delta, zero_point

    def bitwidth_refactor(self, refactored_bit: int):
        #assert 2 <= refactored_bit <= 8, 'bitwidth not supported'
        self.n_bits = refactored_bit
        self.n_levels = 2 ** self.n_bits

    @torch.jit.export
    def extra_repr(self):
        return 'bit={}, is_training={}, inited={}'.format(
            self.n_bits, self.is_training, self.inited
        )

class UniformAffineQuantizer(nn.Module):
    """
    PyTorch Function that can be used for asymmetric quantization (also called uniform affine
    quantization). Quantizes its argument in the forward pass, passes the gradient 'straight
    through' on the backward pass, ignoring the quantization that occurred.
    Based on https://arxiv.org/abs/1806.08342.

    :param n_bits: number of bit for quantization
    :param symmetric: if True, the zero_point should always be 0
    :param channel_wise: if True, compute scale and zero_point in each channel
    :param scale_method: determines the quantization scale and zero point
    """
    def __init__(self, n_bits: int = 8, symmetric: bool = False, channel_wise: bool = True, scale_method: str = 'max',
                 leaf_param: bool = False):
        super(UniformAffineQuantizer, self).__init__()
        self.sym = symmetric
        #assert 2 <= n_bits <= 8, 'bitwidth not supported'
        self.n_bits = n_bits
        self.n_levels = 2 ** self.n_bits
        self.delta = None
        self.zero_point = None
        self.inited = False
        self.leaf_param = leaf_param
        self.channel_wise = channel_wise
        self.scale_method = scale_method

    def forward(self, x: torch.Tensor):

        if self.inited is False:
            if self.leaf_param:
                delta, self.zero_point = self.init_quantization_scale(x, self.channel_wise)
                self.delta = torch.nn.Parameter(delta)
                # self.zero_point = torch.nn.Parameter(self.zero_point)
            else:
                self.delta, self.zero_point = self.init_quantization_scale(x, self.channel_wise)
            self.inited = True

        # start quantization
        x_int = round_ste(x / self.delta) + self.zero_point
        x_quant = torch.clamp(x_int, 0, self.n_levels - 1)
        x_dequant = (x_quant - self.zero_point) * self.delta
        return x_dequant

    def init_quantization_scale(self, x: torch.Tensor, channel_wise: bool = False):
        delta, zero_point = None, None
        if channel_wise:
            x_clone = x.clone().detach()
            n_channels = x_clone.shape[0]
            if len(x.shape) == 4:
                x_max = x_clone.abs().max(dim=-1)[0].max(dim=-1)[0].max(dim=-1)[0]
            else:
                x_max = x_clone.abs().max(dim=-1)[0]
            delta = x_max.clone()
            zero_point = x_max.clone()
            # determine the scale and zero point channel-by-channel
            for c in range(n_channels):
                delta[c], zero_point[c] = self.init_quantization_scale(x_clone[c], channel_wise=False)
            if len(x.shape) == 4:
                delta = delta.view(-1, 1, 1, 1)
                zero_point = zero_point.view(-1, 1, 1, 1)
            else:
                delta = delta.view(-1, 1)
                zero_point = zero_point.view(-1, 1)
        else:
            if 'max' in self.scale_method:
                x_min = min(x.min().item(), 0)
                x_max = max(x.max().item(), 0)
                if 'scale' in self.scale_method:
                    x_min = x_min * (self.n_bits + 2) / 8
                    x_max = x_max * (self.n_bits + 2) / 8

                x_absmax = max(abs(x_min), x_max)
                if self.sym:
                    x_min, x_max = -x_absmax if x_min < 0 else 0, x_absmax

                delta = float(x_max - x_min) / (self.n_levels - 1)
                if delta < 1e-8:
                    warnings.warn('Quantization range close to zero: [{}, {}]'.format(x_min, x_max))
                    delta = 1e-8

                zero_point = round(-x_min / delta)
                delta = torch.tensor(delta).type_as(x)

            elif self.scale_method == 'mse':
                x_max = x.max()
                x_min = x.min()
                best_score = 1e+10
                for i in range(80):
                    new_max = x_max * (1.0 - (i * 0.01))
                    new_min = x_min * (1.0 - (i * 0.01))
                    x_q = self.quantize(x, new_max, new_min)
                    # L_p norm minimization as described in LAPQ
                    # https://arxiv.org/abs/1911.07190
                    score = lp_loss(x, x_q, p=2.4, reduction='all')
                    if score < best_score:
                        best_score = score
                        delta = (new_max - new_min) / (2 ** self.n_bits - 1)
                        zero_point = (- new_min / delta).round()
            else:
                raise NotImplementedError
            
        #self.delta = delta
        return delta, zero_point

    def quantize(self, x, max, min):
        delta = (max - min) / (2 ** self.n_bits - 1)
        zero_point = (- min / delta).round()
        # we assume weight quantization is always signed
        x_int = torch.round(x / delta)
        x_quant = torch.clamp(x_int + zero_point, 0, self.n_levels - 1)
        x_float_q = (x_quant - zero_point) * delta
        return x_float_q

    def bitwidth_refactor(self, refactored_bit: int):
        #assert 2 <= refactored_bit <= 8, 'bitwidth not supported'
        self.n_bits = refactored_bit
        self.n_levels = 2 ** self.n_bits

    def extra_repr(self):
        s = 'bit={n_bits}, scale_method={scale_method}, symmetric={sym}, channel_wise={channel_wise},' \
            ' leaf_param={leaf_param}'
        return s.format(**self.__dict__)

class AdaRoundQuantizer(nn.Module):
    """
    Adaptive Rounding Quantizer, used to optimize the rounding policy
    by reconstructing the intermediate output.
    Based on
     Up or Down? Adaptive Rounding for Post-Training Quantization: https://arxiv.org/abs/2004.10568

    :param uaq: UniformAffineQuantizer, used to initialize quantization parameters in this quantizer
    :param round_mode: controls the forward pass in this quantizer
    :param weight_tensor: initialize alpha
    """

    def __init__(self, n_bits, uaq, round_mode='stochastic'):
        super(AdaRoundQuantizer, self).__init__()
        # copying all attributes from UniformAffineQuantizer
        self.n_bits = n_bits
        self.sym = uaq.sym
        self.delta = uaq.delta
        self.zero_point = uaq.zero_point
        self.n_levels = uaq.n_levels

        self.round_mode = round_mode
        self.alpha = None
        self.soft_targets = False

        # params for sigmoid function
        self.gamma, self.zeta = -0.1, 1.1
        self.beta = 2/3
        #self.init_alpha(x=weight_tensor.clone())

    def forward(self, x):
        if self.round_mode == 'nearest':
            x_int = torch.round(x / self.delta)
        elif self.round_mode == 'nearest_ste':
            x_int = round_ste(x / self.delta)
        elif self.round_mode == 'stochastic':
            x_floor = torch.floor(x / self.delta)
            rest = (x / self.delta) - x_floor  # rest of rounding
            x_int = x_floor + torch.bernoulli(rest)
            #print('Draw stochastic sample')
        elif self.round_mode == 'learned_hard_sigmoid':
            x_floor = torch.floor(x / self.delta)
            if self.soft_targets:
                x_int = x_floor + self.get_soft_targets()
            else:
                x_int = x_floor + (self.alpha >= 0).float()
        else:
            raise ValueError('Wrong rounding mode')

        x_quant = torch.clamp(x_int + self.zero_point, 0, self.n_levels - 1)
        x_float_q = (x_quant - self.zero_point) * self.delta

        return x_float_q

    def get_soft_targets(self):
        return torch.clamp(torch.sigmoid(self.alpha) * (self.zeta - self.gamma) + self.gamma, 0, 1)

    def init_alpha(self, x: torch.Tensor):
        x_floor = torch.floor(x / self.delta)
        if self.round_mode == 'learned_hard_sigmoid':
            print('Init alpha to be FP32')
            rest = (x / self.delta) - x_floor  # rest of rounding [0, 1)
            alpha = -torch.log((self.zeta - self.gamma) / (rest - self.gamma) - 1)  # => sigmoid(alpha) = rest
            self.alpha = nn.Parameter(alpha)
        else:
            raise NotImplementedError

class cdf(nn.Module):
    def __init__(self, m, s, quant_src):
        super(cdf, self).__init__()
    
        self.m = m
        self.s = s
        self.quant_src = quant_src

    def forward(self, tensor):
        normal = torch.distributions.Normal(self.m, self.s)
        cdf = normal.cdf(tensor)
    
        weight_cdf = cdf 

        weight_pdf = torch.exp(normal.log_prob(tensor)) * 2
        return weight_cdf, weight_pdf

def uniform_quantize_life(k):
  class qfn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
      if k == 32:
        out = input
      elif k == 1:
        out = torch.sign(input)
      else:
        n = 2 ** k - 1
        out = torch.round(input * n) / n
      return out

    @staticmethod
    def backward(ctx, grad_output):
      grad_input = grad_output.clone()
      return grad_input

  return qfn().apply

class lifequant(nn.Module):
  def __init__(self, a_bit):
    super(lifequant, self).__init__()
    #assert a_bit <= 8 or a_bit == 32
    self.a_bit = a_bit
    self.uniform_q = uniform_quantize_life(k=a_bit)
  
  def normalize(self, x, range_ = 2):
    new_s = range_ / 4
    m = torch.mean(x)
    s = torch.std(x)
    return (x - m) / s * new_s

  def forward(self, x):
    if self.a_bit == 32:
      activation_q = x
    else:
      x = self.normalize(x)
      activation_q = self.uniform_q(torch.clamp(x, 0, 1))
      # print(np.unique(activation_q.detach().numpy()))
    return activation_q
            


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
