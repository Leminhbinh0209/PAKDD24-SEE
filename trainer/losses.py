import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import sklearn.preprocessing
from pytorch_metric_learning import  losses
from pytorch_metric_learning.utils import loss_and_miner_utils as lmu

from sphericalexpansion import SphericalExpansion


def pairwise_distance(a, squared=False):
    """Computes the pairwise distance matrix with numerical stability."""
    pairwise_distances_squared = torch.add(
        a.pow(2).sum(dim=1, keepdim=True).expand(a.size(0), -1),
        torch.t(a).pow(2).sum(dim=0, keepdim=True).expand(a.size(0), -1)
    ) - 2 * (
        torch.mm(a, torch.t(a))
    )

    # Deal with numerical inaccuracies. Set small negatives to zero.
    pairwise_distances_squared = torch.clamp(
        pairwise_distances_squared, min=0.0
    )

    # Get the mask where the zero distances are at.
    error_mask = torch.le(pairwise_distances_squared, 0.0)
    #print(error_mask.sum())
    # Optionally take the sqrt.
    if squared:
        pairwise_distances = pairwise_distances_squared
    else:
        pairwise_distances = torch.sqrt(
            pairwise_distances_squared + error_mask.float() * 1e-16
        )

    # Undo conditionally adding 1e-16.
    pairwise_distances = torch.mul(
        pairwise_distances,
        (error_mask == False).float()
    )

    # Explicitly set diagonals to zero.
    mask_offdiagonals = 1 - torch.eye(
        *pairwise_distances.size(),
        device=pairwise_distances.device
    )
    pairwise_distances = torch.mul(pairwise_distances, mask_offdiagonals)

    return pairwise_distances

def binarize_and_smooth_labels(T, nb_classes, smoothing_const = 0):
    import sklearn.preprocessing
    T = T.cpu().numpy()
    T = sklearn.preprocessing.label_binarize(
        T, classes = range(0, nb_classes)
    )
    T = T * (1 - smoothing_const)
    T[T == 0] = smoothing_const / (nb_classes - 1)
    T = torch.FloatTensor(T).cuda()

    return T

class MSLoss(nn.Module):
    def __init__(self, tau=0.2, ):
        super().__init__()
        self.tau = tau
        self.mrg = 0.5
        self.alpha, self.beta = 1, 5
        

        self.sim_f = lambda x, y: x @ y.t()
        
            
    def forward(self, X, y):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """        
        batch_size = X.shape[0]
        device = X.device
                
        labels = y.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        class_eq_mask = torch.eq(labels, labels.T).float().to(device)
        
        # mask-out self-contrast cases
        self_mask = torch.scatter(torch.ones_like(class_eq_mask), 1, torch.arange(batch_size).view(-1, 1).to(device), 0)
        
        pos_mask = class_eq_mask * self_mask
        neg_mask = (1-class_eq_mask)
        
        # compute logits
        logits =  self.sim_f(X, X)
        
        mean_logit = logits[~torch.eye(logits.shape[0], dtype=torch.bool, device=logits.device)].mean()        
        pos_exp = torch.exp(-self.alpha * (logits - mean_logit)) * pos_mask
        neg_exp = torch.exp(self.beta * (logits - mean_logit)) * neg_mask
        
        pos_loss = 1.0 / self.alpha * torch.log(1 + torch.sum(pos_exp, dim=1))
        neg_loss = 1.0 / self.beta * torch.log(1 + torch.sum(neg_exp, dim=1))
        
        # loss
        loss = (pos_loss + neg_loss).mean()
                
        return loss

    
class MSLoss_Angle(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = 0.5
        self.alpha, self.beta = 2, 50
            
    def forward(self, X, y):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """        
        batch_size = X.shape[0]
        device = X.device
                
        labels = y.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        class_eq_mask = torch.eq(labels, labels.T).float().to(device)
        
        # mask-out self-contrast cases
        self_mask = torch.scatter(torch.ones_like(class_eq_mask), 1, torch.arange(batch_size).view(-1, 1).to(device), 0)
        
        pos_mask = class_eq_mask * self_mask
        neg_mask = (1-class_eq_mask)
        
        # compute logits
        X = F.normalize(X)
        logits =  F.linear(X, X) 
               
        pos_exp = torch.exp(-self.alpha * (logits - self.base)) * pos_mask
        neg_exp = torch.exp(self.beta * (logits - self.base)) * neg_mask
        
        pos_loss = 1.0 / self.alpha * torch.log(1 + torch.sum(pos_exp, dim=1))
        neg_loss = 1.0 / self.beta * torch.log(1 + torch.sum(neg_exp, dim=1))
        
        # loss
        loss = (pos_loss + neg_loss).mean()
                
        return loss
    
class PALoss_Angle(torch.nn.Module):
    def __init__(self, nb_classes, sz_embed, mrg = 0.1, alpha = 32):
        torch.nn.Module.__init__(self)
        # Proxy Anchor Initialization
        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.mrg = mrg
        self.alpha = alpha
        
        self.proxies = torch.nn.Parameter(torch.randn(self.nb_classes, self.sz_embed).cuda())
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')
        
    def forward(self, X, T, P=None):
        if P is None:
            P = self.proxies
        else:
            P = P[:self.nb_classes]
                
        cos = F.linear(F.normalize(X), F.normalize(P))  # Calcluate cosine similarity
        P_one_hot = F.one_hot(T, num_classes = self.nb_classes).float()        
        N_one_hot = 1 - P_one_hot
    
        pos_exp = torch.exp(-self.alpha * (cos - self.mrg))
        neg_exp = torch.exp(self.alpha * (cos + self.mrg))

        with_pos_proxies = torch.nonzero(P_one_hot.sum(dim = 0) != 0).squeeze(dim = 1)   # The set of positive proxies of data in the batch
        num_valid_proxies = len(with_pos_proxies)   # The number of positive proxies
        
        P_sim_sum = torch.where(P_one_hot == 1, pos_exp, torch.zeros_like(pos_exp)).sum(dim=0) 
        N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).sum(dim=0)
                
        pos_term = torch.log(1 + P_sim_sum).sum() / num_valid_proxies
        neg_term = torch.log(1 + N_sim_sum).sum() / self.nb_classes
        
        loss = (pos_term + neg_term)
        return loss
    
class PNCALoss_Angle(torch.nn.Module):
    def __init__(self, nb_classes, sz_embed, alpha = 32, normalize=True):
        torch.nn.Module.__init__(self)
        # Proxy Anchor Initialization
        self.nb_classes = nb_classes
        self.sz_embed = sz_embed

        self.alpha = alpha
        self.normalize = normalize
        
        self.proxies = torch.nn.Parameter(torch.randn(self.nb_classes, self.sz_embed).cuda())
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')
        
    def forward(self, X, T, P=None):
        P = self.alpha * F.normalize(P, p = 2, dim = -1)
        X = self.alpha * F.normalize(X, p = 2, dim = -1)
        
        D = pairwise_distance(
            torch.cat(
                [X, P]
            ),
            squared = True
        )[:X.size()[0], X.size()[0]:]

        T = binarize_and_smooth_labels(
            T = T, nb_classes = len(P), smoothing_const = 0
        )
        loss1 = torch.sum(T * torch.exp(-D), -1)
        loss2 = torch.sum((1-T) * torch.exp(-D), -1)
        loss = -torch.log(loss1/loss2)
        loss = loss.mean()
        return loss


class Norm_SoftMax(nn.Module):
    def __init__(self,  nb_classes, sz_embed, alpha=32.0):
        super(Norm_SoftMax, self).__init__()
        self.alpha = alpha
        self.nb_classes = nb_classes

        self.proxy = torch.nn.Parameter(torch.Tensor(nb_classes, sz_embed))
        
        torch.nn.init.kaiming_uniform_(self.proxy, a=math.sqrt(5))
        

    def forward(self, input, target):
        input_l2 = F.normalize(input, p=2, dim=1)
        proxy_l2 = F.normalize(self.proxy, p=2, dim=1)
        sim_mat = input_l2.matmul(proxy_l2.t())
        logits = self.alpha * sim_mat
        loss = F.cross_entropy(logits, target)
        return loss
      
class SoftTripleLoss_Angle(torch.nn.Module):
    def __init__(self, nb_classes, sz_embed, centers_per_class=10, la=20, gamma=0.1, margin=0.01):
        torch.nn.Module.__init__(self)
        self.loss_func = losses.SoftTripleLoss(nb_classes, sz_embed, centers_per_class, la, gamma, margin)
    
    def forward(self, X, T):
        X = F.normalize(X)
        loss = self.loss_func(X, T)
        return loss
    
class SupCon(torch.nn.Module):
    def __init__(self, tau=0.2,  IPC=1):
        torch.nn.Module.__init__(self)
        self.tau = tau

        self.IPC = IPC
        

        self.dist_f = lambda x, y: x @ y.t()
        
    def compute_loss(self, x0, x1):
        bsize = x0.shape[0]
        target = torch.arange(bsize).cuda()
        eye_mask = torch.eye(bsize).cuda() * 1e9
        logits00 = self.dist_f(x0, x0) / self.tau - eye_mask
        logits01 = self.dist_f(x0, x1) / self.tau
        logits = torch.cat([logits01, logits00], dim=1)
        logits -= logits.max(1, keepdim=True)[0].detach()
        loss = F.cross_entropy(logits, target)
        return loss
    
    def forward(self, X, T):
        # x0 and x1 - positive pair
        # tau - temperature
        loss = 0
        step = 0
        for i in range(self.IPC):
            for j in range(self.IPC):
                if i != j:
                    loss += self.compute_loss(X[:, i], X[:, j])
                step += 1
        loss /= step
        return loss
    
def binarize(T, nb_classes):
    T = T.cpu().numpy()
    
    T = sklearn.preprocessing.label_binarize(
        T, classes = range(0, nb_classes)
    )
    T = torch.FloatTensor(T).cuda()
    return T

def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)
    return output

    
class SEEProxyAnchor(SphericalExpansion):
    def __init__(self, nb_classes, 
                        sz_embed, 
                        n_expansion=16, 
                        mrg=0.1, 
                        alpha=32, 
                        keep_grad=True, 
                        lower_bound=0.0, 
                        max_thresh=0.5, 
                        min_thresh=0.1,
                        _type=1,
                        _lambda=0.1,
                        random=False):
        # Proxy Anchor Initialization
        SphericalExpansion.__init__(self, nb_classes=nb_classes, 
                                    sz_embed=sz_embed, 
                                    n_expansion=n_expansion,
                                    keep_grad=keep_grad, 
                                    lower_bound=lower_bound,
                                    max_thresh=max_thresh, 
                                    min_thresh=min_thresh,
                                    _type=_type,
                                    random=random)
        self.mrg = mrg
        self.alpha = alpha
        self._lambda = _lambda
    def proxy_loss(self, x, y, w, only_neg=False):
        cos = F.linear(l2_norm(x), l2_norm(w))  # Calcluate cosine similarity
        P_one_hot = binarize(T = y, nb_classes = self.nb_classes)
        N_one_hot = 1 - P_one_hot
    
        pos_exp = torch.exp(-self.alpha * (cos - self.mrg))
        neg_exp = torch.exp(self.alpha * (cos + self.mrg))

        with_pos_proxies = torch.nonzero(P_one_hot.sum(dim = 0) != 0).squeeze(dim = 1)   # The set of positive proxies of data in the batch
        num_valid_proxies = len(with_pos_proxies)   # The number of positive proxies
        
        P_sim_sum = torch.where(P_one_hot == 1, pos_exp, torch.zeros_like(pos_exp)).sum(dim=0) 
        N_sim_sum = torch.where(N_one_hot == 1, neg_exp, torch.zeros_like(neg_exp)).sum(dim=0)
        
        pos_term = torch.log(1 + P_sim_sum).sum() / num_valid_proxies
        neg_term = torch.log(1 + N_sim_sum).sum() / self.nb_classes
        loss = pos_term + neg_term 
        if only_neg:
            return neg_term
        return loss

    def forward(self, x_batch, y_batch, callback=None):
        self.proxies.requires_grad = True
        X, T, _ = self.spherical_aug(x_batch.clone().detach(), y_batch.clone().detach())
        batch_loss = self.proxy_loss(x_batch, y_batch, self.proxies)
        self.proxies.requires_grad = True
        if X.size(0) <= x_batch.size(0):
            aug_loss=0.0
        else:
            aug_loss = self.proxy_loss(X.detach()[x_batch.size(0):], T[x_batch.size(0):], self.proxies, only_neg=False)
        
        # Stat cosine similarity 
        cos = F.linear(l2_norm(x_batch), l2_norm(self.proxies.data))
        one_hot = torch.zeros_like(cos)
        one_hot.scatter_(1, y_batch.view(-1, 1), 1.0)
        pos_cos = (one_hot * cos).sum(dim=0).detach().cpu().numpy()
        if callback is not None:
            callback["pos_cos"] = np.hstack((callback["pos_cos"], pos_cos)) if len(callback["pos_cos"]) else pos_cos
            callback["rate"]  = len(X)
            callback["batch_loss"] += batch_loss.item()
            callback["aug_loss"] += aug_loss.item() if aug_loss >0 else 0.0
            # End stat
        
        return batch_loss + self._lambda*aug_loss
    