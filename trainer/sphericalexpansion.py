
import torch
import torch.nn as nn
import warnings
import numpy as np
import sklearn.preprocessing

def random_sum_zero(K):
    # Generate (K-1) random rows
    M = torch.randn(K-1, K)
    # Compute the Kth row
    last_row = -M.sum(dim=0)
    # Append the Kth row to the matrix
    M = torch.cat((M, last_row.unsqueeze(0)))
    return M

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


def gen_unit(dim: int=10):
    """
    Generate a unit vector with dimention `dim`
    """
    v = torch.randn(dim, dtype=torch.float32)
    return v / torch.norm(v)


def gram_schmidt(bases) :
    """
    Gram-Schmidt process to generate one more independent vector from a sec of bases
    args:
        bases:  dim X n_vector
    output:
        new_base: dim
    """
    dim, n_vector = bases.shape
    assert dim >= n_vector, "Number of bases already greater than dimension"
    assert torch.allclose(torch.norm(bases, dim=0), torch.tensor(1.0), rtol=1e-4, atol=1e-4), "Base vectors is not normailized"
    rand_vec = gen_unit(dim).cuda()
    new_base = rand_vec - torch.sum(rand_vec@bases * bases, dim=1)
    new_base /= new_base.norm()
    return new_base

class SphericalExpansion(torch.nn.Module):
    def __init__(self, nb_classes, 
                 sz_embed, 
                 n_expansion=16,
                 keep_grad=True, 
                 lower_bound=0.0, 
                 max_thresh=0.3, 
                 min_thresh=0.1,
                 _type=1,
                 random=False):
        torch.nn.Module.__init__(self)
        # Proxy Anchor Initialization
        self.proxies = torch.nn.Parameter(torch.randn(nb_classes, sz_embed).cuda())
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')

        self.n_expansion = n_expansion
        self.nb_classes = nb_classes
        self.sz_embed = sz_embed
        self.max_thresh = max_thresh
        self.min_thresh = min_thresh
        
        self.bweights = self.init_bweights().cuda()
        self.keep_grad = keep_grad
        self.lower_bound = lower_bound
        self.plan = _type
        if self.plan == 1:
            warnings.warn("Using the percentage for updating threshold")
        self.random = random
        if random:
            warnings.warn("Apply random generation")
    def update_threshold(self, epoch, max_epoch):
        """
            Update the threshold along the current epoch from max_threshold to min_threshold

        """
        max_ = self.max_thresh
        min_ = self.min_thresh
        delta = max_ - min_
        self.pos_thresh =  max(self.lower_bound, min_ + delta * ((np.cos(np.pi * epoch / max_epoch) + 1.0) / 2.0))
        if self.plan == 1:
            self.pos_thresh =    max_ - delta * ((np.cos(np.pi * min(epoch, 200) / min(max_epoch, 200)) + 1.0) / 2.0)
           
    def init_bweights(self):
        """
        Init weights of bases of expansion
        """
        bweights = torch.zeros((self.n_expansion+1, self.n_expansion+1), dtype=torch.float32) 
        bweights[:,0] = -1 / self.n_expansion
        bweights[0,0] = 1.0
        for i in range(1, self.n_expansion+1):
            for j in range(1, i): 
                bweights[i,j] = (-1 / self.n_expansion - bweights[i,:] @ bweights[j,:]) / bweights[j,j]
            bweights[i,i] = torch.sqrt(1-torch.sum(bweights[i,:i+1].pow(2)))
        bweights[-1,-1] = 0
        return  bweights

    def one2many_expansion(self, z, lb, w_c):
        """
            Expand one sample around its center class
        """
     
        M = z.dot(w_c)*w_c
        r_1 = z - M
        u_1 = r_1 / torch.norm(r_1)
        # Base set
        v = torch.zeros((self.sz_embed, self.n_expansion+2), dtype=torch.float32).cuda()
        # Gen set
        u = torch.zeros((self.n_expansion+2, self.sz_embed), dtype=torch.float32).cuda()
        v[:,0] = w_c
        v[:,1] = u_1
        u[0, :] = w_c
        u[1, :] = u_1

        for i in range(2, self.n_expansion+2):
            v[:,i] = gram_schmidt(v[:, :i])

        
        # Multiply coordinate with bases
        v = v.clone().detach()
        v[:,1] = u_1 # Keep gradient for the first base, i.e., u1
        u[1:, :] = self.bweights @ v[:,1:].T 
        if self.random:
           u[1:, :] = random_sum_zero(self.n_expansion+1).cuda() @ v[:,1:].T 


        r = u[1:] * torch.norm(r_1)
        z_expans = M + r
       
        lb_expans = torch.repeat_interleave(lb, z_expans.shape[0]).cuda()
        return z_expans[-self.n_expansion:], lb_expans[-self.n_expansion:]

    def spherical_aug(self, x_batch, y_batch):
        X  = l2_norm(x_batch) if self.keep_grad else l2_norm(x_batch).clone().detach()
        P = l2_norm(self.proxies).clone().detach()
        aug_x = x_batch
        aug_t = y_batch
        is_expan = []
        if self.plan == 0:
            for idx, (z, lb) in enumerate(zip(X, y_batch)):
                if z @ P[lb] < self.pos_thresh:
                    z_expans, lb_expans = None, None
                else:
                    z_expans, lb_expans = self.one2many_expansion(z, lb, P[lb])
                if z_expans is not None:
                    is_expan.append(True)
                    aug_x = torch.cat((aug_x, z_expans), dim=0) 
                    aug_t = torch.cat((aug_t, lb_expans), dim=0)
                else:
                    is_expan.append(False)
                    continue
        else:
            scores = torch.tensor([z @ P[lb] for idx, (z, lb) in enumerate(zip(X, y_batch))])
            top_k = int(np.round(scores.numel() * self.pos_thresh))
            if top_k <=0 :
                return aug_x.contiguous(), aug_t.contiguous(), [False]*y_batch.size(0)
            _, top_indices = torch.topk(scores.float(), k=top_k, largest=True)
            top_indices_list = top_indices.tolist()
            for idx, (z, lb) in enumerate(zip(X, y_batch)):
                if idx not in top_indices_list:
                    z_expans, lb_expans = None, None
                    is_expan.append(False)
                else:     
                    z_expans, lb_expans = self.one2many_expansion(z, lb, P[lb])
                    if z_expans is not None:
                        is_expan.append(True)
                        aug_x = torch.cat((aug_x, z_expans), dim=0) 
                        aug_t = torch.cat((aug_t, lb_expans), dim=0)
                    else:
                        is_expan.append(False)
                        continue
        return aug_x.contiguous(), aug_t.contiguous(), is_expan

    def forward(self, x_batch, y_batch):
        raise NotImplementedError("Forward function is not implemented")


