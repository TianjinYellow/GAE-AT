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
from sklearn.metrics.cluster import (v_measure_score, homogeneity_score,
                                     completeness_score)
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn import metrics
from munkres import Munkres, print_matrix
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
import torch.optim as optim
from torch_geometric.utils import negative_sampling
import copy


class clustering_metrics():
    def __init__(self, true_label, predict_label):
        self.true_label = true_label
        self.pred_label = predict_label


    def clusteringAcc(self):
        # best mapping between true_label and predict label
        l1 = list(set(self.true_label))
        numclass1 = len(l1)

        l2 = list(set(self.pred_label))
        numclass2 = len(l2)
        if numclass1 != numclass2:
            print('Class Not equal, Error!!!!')
            return 0

        cost = np.zeros((numclass1, numclass2), dtype=int)
        for i, c1 in enumerate(l1):
            mps = [i1 for i1, e1 in enumerate(self.true_label) if e1 == c1]
            for j, c2 in enumerate(l2):
                mps_d = [i1 for i1 in mps if self.pred_label[i1] == c2]

                cost[i][j] = len(mps_d)

        # match two clustering results by Munkres algorithm
        m = Munkres()
        cost = cost.__neg__().tolist()

        indexes = m.compute(cost)

        # get the match results
        new_predict = np.zeros(len(self.pred_label))
        for i, c in enumerate(l1):
            # correponding label in l2:
            c2 = l2[indexes[i][1]]

            # ai is the index with label==c2 in the pred_label list
            ai = [ind for ind, elm in enumerate(self.pred_label) if elm == c2]
            new_predict[ai] = c

        acc = metrics.accuracy_score(self.true_label, new_predict)
        f1_macro = metrics.f1_score(self.true_label, new_predict, average='macro')
        precision_macro = metrics.precision_score(self.true_label, new_predict, average='macro')
        recall_macro = metrics.recall_score(self.true_label, new_predict, average='macro')
        f1_micro = metrics.f1_score(self.true_label, new_predict, average='micro')
        precision_micro = metrics.precision_score(self.true_label, new_predict, average='micro')
        recall_micro = metrics.recall_score(self.true_label, new_predict, average='micro')
        return acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro
        
    def evaluationClusterModelFromLabel(self):
        nmi = metrics.normalized_mutual_info_score(self.true_label, self.pred_label)
        adjscore = metrics.adjusted_rand_score(self.true_label, self.pred_label)
        acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro = self.clusteringAcc()
   
        print('ACC=%f, f1_macro=%f, precision_macro=%f, recall_macro=%f, f1_micro=%f, precision_micro=%f, recall_micro=%f, NMI=%f, ADJ_RAND_SCORE=%f' % (acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro, nmi, adjscore),flush=True)

        fh = open('recoder.txt', 'a')

        fh.write('ACC=%f, f1_macro=%f, precision_macro=%f, recall_macro=%f, f1_micro=%f, precision_micro=%f, recall_micro=%f, NMI=%f, ADJ_RAND_SCORE=%f' % (acc, f1_macro, precision_macro, recall_macro, f1_micro, precision_micro, recall_micro, nmi,adjscore) )
        fh.write('\r\n')
        fh.flush()
        fh.close()
        return acc, nmi,f1_macro,precision_macro,adjscore

        return acc, nmi, adjscore        
parser = argparse.ArgumentParser()
parser.add_argument('--variational', action='store_true')
parser.add_argument('--linear', action='store_true')
parser.add_argument('--dataset', type=str, default='Cora',
                    choices=['Cora', 'CiteSeer', 'PubMed'])
parser.add_argument('--perturbStyle', type=str, default='None',
                    choices=['features', 'None', 'structure','alternative','together'])
parser.add_argument('--epochs', type=int, default=400)
parser.add_argument('--RepeatRuns', type=int, default=10)
parser.add_argument('--n_layers', type=int, default=0, help='total layers minus 2')
parser.add_argument('--epsSt', type=float, default=1e-1)
parser.add_argument('--lam', type=float, default=5.0)
parser.add_argument('--epsFeat', type=float, default=5e-2)
parser.add_argument('--steps', type=int, default=1)
parser.add_argument('--norm',type=str,default='linf',choices=['linf','l2'])

args = parser.parse_args()

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Planetoid')
dataset=Planetoid(path,args.dataset,transform=T.NormalizeFeatures())
data = dataset[0]
data.train_mask = data.val_mask = data.test_mask= None
data = train_test_split_edges(data)

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
        model = GAE(GCNEncoder(num_features, out_channels,n=args.n_layers))
    else:
        model = GAE(LinearEncoder(num_features, out_channels))
else:
    if args.linear:
        model = VGAE(VariationalLinearEncoder(num_features, out_channels))
    else:
        model = VGAE(VariationalGCNEncoder(num_features, out_channels,n=args.n_layers))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
x = data.x.to(device)
train_pos_edge_index = data.train_pos_edge_index.to(device)
edge_weights=torch.ones(train_pos_edge_index.shape[1]).to(device)
num_nodes=x.shape[0]

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
@torch.no_grad()
def plot_points(colors):
    model.eval()
    z = model.encode(data.x.to(device), data.train_pos_edge_index.to(device))
    z = TSNE(n_components=2).fit_transform(z.cpu().numpy())
    y = data.y.cpu().numpy()

    plt.figure(figsize=(8, 8))
    for i in range(dataset.num_classes):
        plt.scatter(z[y == i, 0], z[y == i, 1], s=20, color=colors[i])
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("./visulization.png",dpi=600)
    plt.show()
@torch.no_grad()
def test_cluster():
    model.eval()
    z = model.encode(x, train_pos_edge_index)
    kmeans_input = z.cpu().numpy()
    kmeans = KMeans(n_clusters=data.y.max()+1, random_state=0).fit(kmeans_input)
    predict_labels = kmeans.predict(kmeans_input)
    labels = data.y.cpu().numpy()
    cm = clustering_metrics(labels, predict_labels)
    auc, ap = model.test(z, data.test_pos_edge_index, data.test_neg_edge_index)
    acc,nmi,f1,precision,ari=cm.evaluationClusterModelFromLabel()
    return acc,nmi,f1,precision,ari,auc,ap

def attack_features(x,model,n_iters,epsilon,train_pos_edge_index,norm='linf',variational=False):
    delta=torch.zeros_like(x).requires_grad_()
    length=torch.norm(x,dim=1,p=2,keepdim=True)
    if norm =="linf":
        alpha=epsilon/n_iters  #it controls the total perturbations after n_iters steps is less than epsilon
    elif norm=="l2":
        max_norm=length*epsilon
        alpha=max_norm/n_iters  #it controls the norm of  total perturbations after n_iters steps is less than epsilon*||X||_2
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
            
        delta.data=delta.data+grad.data*alpha
        temp_weights=torch.clamp(edge_weights+delta.data,0,1)
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

repeat_acc=[]
repeat_nmi=[]
repeat_f1=[]
repeat_precision=[]
repeat_adj=[]
print("perturbation Style:",args.perturbStyle)
for t  in range(args.RepeatRuns):
    print("Repeats number:",t)
    data = dataset[0]
    data.train_mask = data.val_mask = data.test_mask  = None
    data = train_test_split_edges(data,val_ratio=0.05,test_ratio=0.1)
    x = data.x.to(device)
    train_pos_edge_index = data.train_pos_edge_index.to(device)
    edge_weights=torch.ones(train_pos_edge_index.shape[1]).to(device)
    if not args.variational:
        if not args.linear:
            model = GAE(GCNEncoder(num_features, out_channels,n=args.n_layers))
        else:
            model = GAE(LinearEncoder(num_features, out_channels))
    else:
        if args.linear:
            model = VGAE(VariationalLinearEncoder(num_features, out_channels))
        else:
            model = VGAE(VariationalGCNEncoder(num_features, out_channels,n=args.n_layers))
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
    for epoch in range(1, args.epochs + 1):
        loss = train()
        if epoch%100==0:
            acc, nmi,f1_micro,precision_micro,adjscore,auc,ap=test_cluster()
            print('Epoch: {:03d}, Test AUC: {:.4f}, Test AP: {:.4f}'.format(epoch,auc, ap))
    repeat_acc.append(acc)
    repeat_nmi.append(nmi)
    repeat_f1.append(f1_micro)
    repeat_precision.append(precision_micro)
    repeat_adj.append(adjscore)

#colors = [
#    '#ffc0cb', '#bada55', '#008080', '#420420', '#7fe5f0', '#065535', '#ffd700'
#]
#plot_points(colors)
acc_mean=np.average(repeat_acc)
nmi_mean=np.average(repeat_nmi)
f1_mean=np.average(repeat_f1)
precision_mean=np.average(repeat_precision)
adj_mean=np.average(repeat_adj)

acc_std=np.std(repeat_acc)
nmi_std=np.std(repeat_nmi)
f1_std=np.std(repeat_f1)
precision_std=np.std(repeat_precision)
adj_std=np.std(repeat_adj)
print("acc mean:",acc_mean,"nmi_mean:",nmi_mean,"f1_mean:",f1_mean,"precision mean:",precision_mean,"adj_mean",adj_mean)
print("acc std:",acc_std,"nmi_std:",nmi_std,"f1_std:",f1_std,"precision_std:",precision_std,"adj std:",adj_std)
    