import os.path as osp

import argparse
import torch
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GAE, VGAE,GCNConv
#from GCNLayer import *
from torch_geometric.utils import train_test_split_edges
from torch_geometric.data import InMemoryDataset,Data
import numpy as np
import torch.nn as nn
from torch_sparse import coalesce
from torch.nn import Sequential as Seq, Linear, ReLU
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops
import torch.nn.functional as F
from torch_geometric.nn import TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import torch.nn.functional as F
from torch_geometric.data import DataLoader
import scipy
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils

import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
import torch.optim as optim
from torch_geometric.utils import negative_sampling
import copy

parser = argparse.ArgumentParser()
parser.add_argument('--variational', action='store_true')
parser.add_argument('--linear', action='store_true')
parser.add_argument('--threelayer', action='store_true')
parser.add_argument('--dataset', type=str, default='Cora',
                    choices=['Cora', 'CiteSeer', 'PubMed'])
parser.add_argument('--perturbStyle', type=str, default='None',
                    choices=['features', 'None', 'structure','alternative','together'])
parser.add_argument('--epochs', type=int, default=400)
parser.add_argument('--RepeatRuns', type=int, default=10)
parser.add_argument('--n_layers', type=int, default=0)
parser.add_argument('--epsSt', type=float, default=1e-1)
parser.add_argument('--lam', type=float, default=5.0)
parser.add_argument('--epsFeat', type=float, default=5e-2)
parser.add_argument('--steps', type=int, default=1)
parser.add_argument('--norm',type=str,default='linf',choices=['linf','l2'])

args = parser.parse_args()

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Planetoid')
dataset=Planetoid(path,args.dataset,transform=T.NormalizeFeatures())
data = dataset[0]
data.train_mask = data.val_mask = data.test_mask = data.y = None
data = train_test_split_edges(data,val_ratio=0.05,test_ratio=0.1)


class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels,n=0):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=False)
        self.conv_n=nn.ModuleList()
        self.n=n
        if n!=0:
            for i in range(n):
                self.conv_n.append(GCNConv(2 * out_channels, 2 * out_channels, cached=False))
        self.conv2 = GCNConv(2 * out_channels, out_channels, cached=False)

    def forward(self, x, edge_index,edge_weight=None):
        x = self.conv1(x, edge_index,edge_weight=edge_weight).relu()
        if self.n!=0:
            for i in range(self.n):
                x=self.conv_n[i](x, edge_index,edge_weight=edge_weight).relu()
        return self.conv2(x, edge_index,edge_weight=edge_weight)


class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels,n=0):
        super(VariationalGCNEncoder, self).__init__()
        self.n=n
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=False)
        self.conv_n=nn.ModuleList()
        if n!=0:
            for i in range(self.n):
                self.conv_n.append(GCNConv(2*out_channels, 2 * out_channels, cached=False))
        self.conv_mu = GCNConv(2 * out_channels, out_channels, cached=False)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels, cached=False)

    def forward(self, x, edge_index,edge_weight=None):
        x = self.conv1(x, edge_index,edge_weight=edge_weight).relu()
        if self.n!=0:
            for i in range(self.n):
                x=self.conv_n[i](x, edge_index,edge_weight=edge_weight).relu()
        return self.conv_mu(x, edge_index,edge_weight=edge_weight), self.conv_logstd(x, edge_index,edge_weight=edge_weight)

class DeepGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DeepGCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=False)
        self.conv2 = GCNConv(2 * out_channels, 2 * out_channels, cached=False)
        self.conv3 = GCNConv(2 * out_channels, out_channels, cached=False)

    def forward(self, x, edge_index,edge_weight=None):
        x = self.conv1(x, edge_index,edge_weight=edge_weight).relu()
        x = self.conv2(x, edge_index,edge_weight=edge_weight).relu()
        return self.conv3(x, edge_index,edge_weight=edge_weight)


class DeepVariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DeepVariationalGCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=False)
        self.conv2=GCNConv(2 * out_channels, 2 * out_channels, cached=False)
        self.conv_mu = GCNConv(2 * out_channels, out_channels, cached=False)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels, cached=False)

    def forward(self, x, edge_index,edge_weight=None):
        x = self.conv1(x, edge_index,edge_weight=edge_weight).relu()
        x=self.conv2(x, edge_index,edge_weight=edge_weight).relu()
        return self.conv_mu(x, edge_index,edge_weight=edge_weight), self.conv_logstd(x, edge_index,edge_weight=edge_weight)


class LinearEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LinearEncoder, self).__init__()
        self.conv = GCNConv(in_channels, out_channels)

    def forward(self, x, edge_index,edge_weight=None):
        return self.conv(x, edge_index,edge_weight=edge_weight)


class VariationalLinearEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VariationalLinearEncoder, self).__init__()
        self.conv_mu = GCNConv(in_channels, out_channels, cached=False)
        self.conv_logstd = GCNConv(in_channels, out_channels, cached=False)

    def forward(self, x, edge_index,edge_weight=None):
        return self.conv_mu(x, edge_index,edge_weight=edge_weight), self.conv_logstd(x, edge_index,edge_weight=edge_weight)


out_channels = 16
num_features = dataset.num_features

if not args.variational:
    if not args.linear:
        if not args.threelayer:
            model = GAE(GCNEncoder(num_features, out_channels))
        else:
            model = GAE(DeepGCNEncoder(num_features, out_channels,n=args.n_layers))
    else:
        model = GAE(LinearEncoder(num_features, out_channels))
else:
    if args.linear:
        model = VGAE(VariationalLinearEncoder(num_features, out_channels))
    else:
        if not args.threelayer:
            model = VGAE(VariationalGCNEncoder(num_features, out_channels))
        else:
            model = VGAE(DeepVariationalGCNEncoder(num_features, out_channels,n=args.n_layers))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
x = data.x.to(device)
train_pos_edge_index = data.train_pos_edge_index.to(device)
edge_weights=torch.ones(train_pos_edge_index.shape[1]).to(device)
num_nodes=x.shape[0]

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

def attack_features(x,model,n_iters,epsilon,train_pos_edge_index,norm='linf',variational=False):
    delta=torch.zeros_like(x).requires_grad_()
    length=torch.norm(x,dim=1,p=2,keepdim=True)
    if norm =="linf":
        alpha=epsilon/n_iters  #make sure the total perturbations after n_iters steps is less than epsilon
    elif norm=="l2":
        max_norm=length*epsilon  #to make sure the norm of  total perturbations after n_iters steps is less than epsilon*||X||_2
        alpha=max_norm/n_iters
    for i in range(n_iters):
        z = model.encode(x+delta, train_pos_edge_index)
        loss = model.recon_loss(z, train_pos_edge_index)
        if variational:
            loss=loss+(1.0/z.shape[0])*model.kl_loss()
        if norm=='linf':
            grad=torch.autograd.grad(loss,delta)[0].sign()
        elif norm=='l2':
            grad=torch.autograd.grad(loss,delta)[0]
            length_grad=torch.norm(grad.detach(),dim=1,p=2,keepdim=True)
            grad=grad*length/(length_grad+1e-20)
        else:
            print("Wrong in Norm!")
        delta.data=delta.data+grad.data*alpha
    return delta
    
def attack_edges(x,model,n_iters,epsilon,train_pos_edge_index,edge_weights,norm='linf',variational=False):
    delta=torch.zeros_like(edge_weights).requires_grad_()
    length=torch.norm(edge_weights,dim=0,p=2,keepdim=True)
    if norm =="linf":
        alpha=epsilon/n_iters
    elif norm=="l2":
        max_norm=length*epsilon
        alpha=max_norm/n_iters
        
    for i in range(n_iters):
        z=model.encode(x,train_pos_edge_index,edge_weight=(edge_weights+delta))
        loss=model.recon_loss(z,train_pos_edge_index)
        if variational:
            loss=loss+(1.0/z.shape[0])*model.kl_loss()
        if norm=='linf':
            grad=torch.autograd.grad(loss,delta)[0].sign()
        elif norm=='l2':
            grad=torch.autograd.grad(loss,delta)[0]
            grad=grad*length/torch.norm(grad.data,dim=0,p=2,keepdim=True)
            
        delta.data=delta.data+grad.data*alpha  #make sure weights>0
        temp_weights=edge_weights+delta.data
        temp_weights[temp_weights<0]=0
        delta.data=(temp_weights-edge_weights).data
    return delta.detach()
kl = nn.KLDivLoss(reduction="sum")        
        
def train():
    global x
    model.train()
    if args.perturbStyle =="None":
        z = model.encode(x, train_pos_edge_index)
        optimizer.zero_grad()
        loss = model.recon_loss(z, train_pos_edge_index)
        if args.variational:
            loss=loss+(1.0/z.shape[0])*model.kl_loss()
        loss.backward()
        optimizer.step()
    elif args.perturbStyle=="structure":
        
        delta=attack_edges(x,model,args.steps,args.epsSt,train_pos_edge_index,edge_weights,args.norm)
        edge_weights1=edge_weights.data+delta
        z = model.encode(x, train_pos_edge_index,edge_weight=edge_weights1)
        optimizer.zero_grad()
        loss = model.recon_loss(z, train_pos_edge_index)
        if args.variational:
            loss=loss+(1.0/z.shape[0])*model.kl_loss()
        loss.backward()
        optimizer.step()
    elif args.perturbStyle=="features":
 
        delta=attack_features(x,model,args.steps,args.epsFeat,train_pos_edge_index,args.norm)
        x1=x+delta
        z1 = model.encode(x1, train_pos_edge_index,edge_weight=edge_weights)
        optimizer.zero_grad()
        loss = model.recon_loss(z1, train_pos_edge_index)
        if args.variational:
            loss=loss+(1.0/z1.shape[0])*model.kl_loss()
        loss.backward()
        optimizer.step()
        
    elif args.perturbStyle=="alternative":
        delta=attack_edges(x,model,args.steps,args.epsSt,train_pos_edge_index,edge_weights,args.norm)
        edge_weights1=edge_weights.data+delta
        z_origin=model.encode(x,train_pos_edge_index)
        
        z = model.encode(x, train_pos_edge_index,edge_weight=edge_weights1)
        robust_loss=(1.0 / z.shape[0]) * kl(F.log_softmax(z, dim=1),F.softmax(z_origin, dim=1))*args.lam
        
        optimizer.zero_grad()
        loss = model.recon_loss(z_origin, train_pos_edge_index)+robust_loss
        
        if args.variational:
            loss=loss+(1.0/z.shape[0])*model.kl_loss()
        loss.backward()
        optimizer.step()
        
        delta=attack_features(x,model,args.steps,args.epsFeat,train_pos_edge_index,args.norm)
        x1=x+delta
        
        z1 = model.encode(x1, train_pos_edge_index,edge_weight=edge_weights)
        z_origin=model.encode(x,train_pos_edge_index)
        robust_loss=(1.0 / z.shape[0]) * kl(F.log_softmax(z1, dim=1),F.softmax(z_origin, dim=1))*args.lam
        optimizer.zero_grad()

        loss = model.recon_loss(z_origin, train_pos_edge_index)+robust_loss
        if args.variational:
            loss=loss+(1.0/z1.shape[0])*model.kl_loss()
        loss.backward()
        optimizer.step()
    elif args.perturbStyle=='together':
        delta=attack_edges(x,model,args.steps,args.epsSt,train_pos_edge_index,edge_weights,args.norm)
        delta1=attack_features(x,model,args.steps,args.epsFeat,train_pos_edge_index,args.norm)
        edge_weights1=edge_weights.data+delta
        x1=x+delta1
        z_origin=model.encode(x,train_pos_edge_index)
        z = model.encode(x1, train_pos_edge_index,edge_weight=edge_weights1)
        robust_loss=(1.0 / z.shape[0]) * kl(F.log_softmax(z, dim=1),F.softmax(z_origin, dim=1))*args.lam
        optimizer.zero_grad()
        loss = model.recon_loss(z_origin, train_pos_edge_index)+robust_loss
        if args.variational:
            loss=loss+(1.0/z.shape[0])*model.kl_loss()
        loss.backward()
        optimizer.step()        
    else:
        print("perturbStyple is wrong!")
    return float(loss)


def test(pos_edge_index, neg_edge_index):
    model.eval()
    with torch.no_grad():
        z = model.encode(x, train_pos_edge_index)
    return model.test(z, pos_edge_index, neg_edge_index)

repeat_auc=[]
repeat_ap=[]
print("perturbation Style:",args.perturbStyle)
for t  in range(args.RepeatRuns):
    print("Repeats number:",t)
    data = dataset[0]
    data.train_mask = data.val_mask = data.test_mask = data.y = None
    data = train_test_split_edges(data,val_ratio=0.05,test_ratio=0.1)
    x = data.x.to(device)
    train_pos_edge_index = data.train_pos_edge_index.to(device)
    edge_weights=torch.ones(train_pos_edge_index.shape[1]).to(device)
    if not args.variational:
        if not args.linear:
            if not args.threelayer:
                model = GAE(GCNEncoder(num_features, out_channels,n=args.n_layers))
            else:
                model = GAE(DeepGCNEncoder(num_features, out_channels))
        else:
            model = GAE(LinearEncoder(num_features, out_channels))
    else:
        if args.linear:
            model = VGAE(VariationalLinearEncoder(num_features, out_channels))
        else:
            if not args.threelayer:
                model = VGAE(VariationalGCNEncoder(num_features, out_channels,n=args.n_layers))
            else:
                model = VGAE(DeepVariationalGCNEncoder(num_features, out_channels))
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
    for epoch in range(1, args.epochs + 1):
        loss = train()
        val_auc,val_ap=test(data.val_pos_edge_index,data.val_neg_edge_index)
        auc, ap = test(data.test_pos_edge_index, data.test_neg_edge_index)
        if epoch%100==0:
            print('Epoch: {:03d}, Val AUC: {:.4f}, Val AP: {:.4f}, Test AUC: {:.4f}, Test AP: {:.4f}'.format(epoch, val_auc,val_ap,auc, ap))
    repeat_auc.append(auc)
    repeat_ap.append(ap)
    
average_auc=np.average(repeat_auc)
std_auc=np.std(repeat_auc)
average_ap=np.average(repeat_ap)
std_ap=np.std(repeat_ap)
print("auc average:",average_auc, "auc std:",std_auc, "ap average",average_ap,"ap std", std_ap)