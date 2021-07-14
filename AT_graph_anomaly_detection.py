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
from sklearn.metrics import roc_auc_score,average_precision_score
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch
import torch.optim as optim
from torch_geometric.utils import negative_sampling
import copy
import scipy.sparse as sp
import scipy.io
import numpy as np
from sklearn.decomposition import PCA
from random import shuffle


def largest_distance(target, neighbors, X,binary=True):
    
    distance = []
    if binary:
        X1=sp.csc_matrix(X)
        X1.eliminate_zeros()
        simi=jaccard_similarity
    else:
        X1=X
        simi=eulicdean

    for item in neighbors:
        distance.append(simi(X1[target], X1[item]))

    largest_distance_index = np.argmax(distance)
    return neighbors[largest_distance_index]
def jaccard_similarity( a, b):
    intersection = a.multiply(b).count_nonzero()
    J = intersection * 1.0 / (a.count_nonzero() + b.count_nonzero() - intersection)
    return 1-J
def eulicdean(a,b):
    return np.linalg.norm(a - b, ord=2, keepdims=True)


def gen_cliques(m, indices):
    edges = []
    for i in range(m-1):
        for j in range(i+1, m):
            edges.append((indices[i], indices[j]))
    return edges


def add_structure_anomalies(m, n, num_of_nodes, A):
    indices = [i for i in range(num_of_nodes)]
    shuffle(indices)

    edges = []
    for i in range(n):
        edges += gen_cliques(m, indices[m*i: m*(i+1)])
    #print(A.shape)
    for item in edges:
        (x, y) = item
        A[x,y] = 1
        A[y,x] = 1

    return indices[: m*n]


def add_attribute_anomalies(m, n, k, num_of_nodes, X,binary=False):
    indices = [i for i in range(num_of_nodes)]
    shuffle(indices)

    anomalies = indices[: m*n]
    for a in anomalies:
        # select k random elements
        shuffle(indices)
        neighbors = indices[: k]
        select_id = largest_distance(a, neighbors, X,binary=binary)
        X[a] = X[select_id]

    return anomalies


def load_data_p(path,filename):
    data = scipy.io.loadmat(path+"{}.mat".format(filename))
    A = data["Network"]
    labels_c=data["Label"]
    PCA_dim = 100
    m = 25
    
    k =50
    if filename=="ACM":
        num_of_nodes = 16484
        n = 20
    elif filename=="BlogCatalog":
        num_of_nodes=5196
        n=10
    elif filename=="Flickr":
        num_of_nodes=7575
        n=15
    else:
        print(filename)
        print("wrong")

    A = data["Network"].toarray()
    X = PCA(n_components=PCA_dim).fit_transform(data["Attributes"].todense())
    if filename=="ACM":
        A=A-np.eye(A.shape[0])

    anomalies = add_structure_anomalies(m, n, num_of_nodes, A)
    anomalies += add_attribute_anomalies(m, n, k, num_of_nodes,X)
    
    labels = [[0] for i in range(num_of_nodes)]
    for i in anomalies:
        labels[i] = [1]
    labels=np.array(labels)
    A=sp.csc_matrix(A)
    A.eliminate_zeros()
    A=A.todense()
    data={"A":A,"X":X,"gnd":labels}
    return data



class MyOwnDataset(InMemoryDataset):
    def __init__(self, root,filename,transform=None, pre_transform=None):
        self.filename=filename
        super(MyOwnDataset, self).__init__(root, transform, pre_transform)
        
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    @property
    def raw_dir(self):
        return osp.join(self.root, self.filename, 'raw')

    @property
    def processed_dir(self):
        return osp.join(self.root, self.filename, 'processed')
    @property
    def processed_file_names(self):
        return ['data.pt']
    def index_to_mask(self,index, size):
        mask = torch.zeros((size, ), dtype=torch.bool)
        mask[index] = 1
        return mask

    def process(self):
        # Read data into huge `Data` list.
        path="./data/"
        if self.filename in ["ACM","BlogCatalog","Flickr"]:
            data=load_data_p(path,self.filename)
        else:
            data=scipy.io.loadmat(path+"{}.mat".format(self.filename))
        labels=data['gnd']
        x=data['X']
        adj=sp.lil_matrix(data['A'])
        
        num_nodes=adj.shape[0]
        adj_coo=adj.tocoo()
        row=adj_coo.row
        col=adj_coo.col
        edge_index=torch.stack([torch.tensor(row).long(),torch.tensor(col).long()],dim=0)
        edge_index,_=coalesce(edge_index,None,num_nodes,num_nodes)

        x=torch.tensor(x).float()
        labels=torch.tensor(labels).long()
        train_index=torch.arange(labels.size(0))
        val_index=train_index
        test_index=train_index
        train_mask=self.index_to_mask(train_index,num_nodes)
        val_mask=self.index_to_mask(val_index,num_nodes)
        test_mask=self.index_to_mask(test_index,num_nodes)
        data=Data(x=x,edge_index=edge_index,y=labels)
        data.train_mask = train_mask
        data1, slices = self.collate([data])
        torch.save((data1, slices), self.processed_paths[0])



parser = argparse.ArgumentParser()
parser.add_argument('--variational', action='store_true')
parser.add_argument('--linear', action='store_true')
parser.add_argument('--dataset', type=str, default='Cora',
                    choices=['Disney', 'Amazon', 'Enron','BlogCatalog','ACM','Flickr'])
parser.add_argument('--perturbStyle', type=str, default='None',
                    choices=['features', 'None', 'structure','alternative','together'])
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--RepeatRuns', type=int, default=10)
parser.add_argument('--epsSt', type=float, default=1e-1)
parser.add_argument('--lam', type=float, default=5.0)
parser.add_argument('--epsFeat', type=float, default=5e-2)
parser.add_argument('--steps', type=int, default=1)
parser.add_argument('--norm',type=str,default='linf',choices=['linf','l2'])

args = parser.parse_args()

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Planetoid')
dataset = MyOwnDataset(path, args.dataset,transform=None)
data = dataset[0]
data.train_mask = data.val_mask = data.test_mask = data.y = None
data = train_test_split_edges(data,val_ratio=0.0,test_ratio=0.0)




class GCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=False)
        self.conv2 = GCNConv(2 * out_channels, 2*out_channels, cached=False)
        #self.conv3 = GCNConv(2 * out_channels, out_channels, cached=False)
    def forward(self, x, edge_index,edge_weight=None):
        x = self.conv1(x, edge_index,edge_weight=edge_weight).relu()
        x=self.conv2(x, edge_index,edge_weight=edge_weight)
        return x


class InnerProductDecoder(torch.nn.Module):
    r"""The inner product decoder from the `"Variational Graph Auto-Encoders"
    <https://arxiv.org/abs/1611.07308>`_ paper

    .. math::
        \sigma(\mathbf{Z}\mathbf{Z}^{\top})

    where :math:`\mathbf{Z} \in \mathbb{R}^{N \times d}` denotes the latent
    space produced by the encoder."""
    def forward(self, z, edge_index, sigmoid=True):
        r"""Decodes the latent variables :obj:`z` into edge probabilities for
        the given node-pairs :obj:`edge_index`.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            sigmoid (bool, optional): If set to :obj:`False`, does not apply
                the logistic sigmoid function to the output.
                (default: :obj:`True`)
        """
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value


    def forward_all(self, z, sigmoid=True):
        r"""Decodes the latent variables :obj:`z` into a probabilistic dense
        adjacency matrix.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            sigmoid (bool, optional): If set to :obj:`False`, does not apply
                the logistic sigmoid function to the output.
                (default: :obj:`True`)
        """
        adj = torch.matmul(z, z.t())
        return torch.sigmoid(adj) if sigmoid else adj
EPS = 1e-15
MAX_LOGSTD = 10


class GCNDecoder(torch.nn.Module):
    def __init__(self, input_channels, out_channels):
        super(GCNDecoder, self).__init__()
        #decoder1
        self.conv1_att = GCNConv( 2 * out_channels,2*out_channels, cached=False)
        self.conv2_att = GCNConv( 2 * out_channels,input_channels, cached=False)
         
        #decoder2
        self.conv1_st=GCNConv( 2 * out_channels,input_channels, cached=False)
        self.innerproduct=InnerProductDecoder()
        
    def forward_st(self, x,train_pos_index, edge_index,sigmoid=True):
        x_st=self.conv1_st(x,train_pos_index).relu()
        x_st=self.innerproduct.forward(x_st,edge_index,sigmoid=sigmoid)
        return x_st
    def forward_att(self, x,train_pos_index):
        x_att = self.conv1_att(x, train_pos_index).relu()
        x_att=self.conv2_att(x_att,train_pos_index)
        return x_att
    def forward_all(self,z,train_pos_index,sigmoid=True):
        z_att=self.conv1_att(z,train_pos_index).relu()
        z_att=self.conv2_att(z_att,train_pos_index)
        
        z_st = self.conv1_st(z, train_pos_index).relu()
        z_st=self.innerproduct.forward_all(z_st,sigmoid=sigmoid)
        return z_st,z_att

class MyGAE(GAE):
    def __init__(self,encoder,decoder=None):
        super(MyGAE,self).__init__(encoder,decoder)
        
    
    def recon_loss(self,x, z, pos_edge_index, neg_edge_index=None):
        r"""Given latent variables :obj:`z`, computes the binary cross
        entropy loss for positive edges :obj:`pos_edge_index` and negative
        sampled edges.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (LongTensor): The positive edges to train against.
            neg_edge_index (LongTensor, optional): The negative edges to train
                against. If not given, uses negative sampling to calculate
                negative edges. (default: :obj:`None`)
        """
        pos_st=self.decoder.forward_st(z,pos_edge_index,pos_edge_index, sigmoid=True)
        x_att=self.decoder.forward_att(z,pos_edge_index)
        pos_loss = -torch.log(pos_st+ EPS).mean()
        att_loss=torch.nn.MSELoss()(x_att,x)
        
        # Do not include self-loops in negative samples
        pos_edge_index, _ = remove_self_loops(pos_edge_index)
        pos_edge_index, _ = add_self_loops(pos_edge_index)
        if neg_edge_index is None:
            neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
        neg_st=self.decoder.forward_st(z,pos_edge_index, neg_edge_index, sigmoid=True)
        neg_loss = -torch.log(1 -neg_st+EPS).mean()

        return pos_loss + neg_loss+att_loss


    def test(self, z, train_pos_index,pos_edge_index, neg_edge_index):
        r"""Given latent variables :obj:`z`, positive edges
        :obj:`pos_edge_index` and negative edges :obj:`neg_edge_index`,
        computes area under the ROC curve (AUC) and average precision (AP)
        scores.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (LongTensor): The positive edges to evaluate
                against.
            neg_edge_index (LongTensor): The negative edges to evaluate
                against.
        """
        pos_y = z.new_ones(pos_edge_index.size(1))
        neg_y = z.new_zeros(neg_edge_index.size(1))
        y = torch.cat([pos_y, neg_y], dim=0)

        pos_pred = self.decoder.forward_st(z,train_pos_index, pos_edge_index, sigmoid=True)
        neg_pred = self.decoder.forward_st(z,train_pos_index, neg_edge_index, sigmoid=True)
        pred = torch.cat([pos_pred, neg_pred], dim=0)

        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()

        return roc_auc_score(y, pred), average_precision_score(y, pred)

class VariationalGCNEncoder(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VariationalGCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=False)
        self.conv_mu = GCNConv(2 * out_channels, out_channels, cached=False)
        self.conv_logstd = GCNConv(2 * out_channels, out_channels, cached=False)

    def forward(self, x, edge_index,edge_weight=None):
        x = self.conv1(x, edge_index,edge_weight=edge_weight).relu()
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
        model = MyGAE(GCNEncoder(num_features, out_channels))
    else:
        model = GAE(LinearEncoder(num_features, out_channels))
else:
    if args.linear:
        model = VGAE(VariationalLinearEncoder(num_features, out_channels))
    else:
        model = VGAE(VariationalGCNEncoder(num_features, out_channels))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
x = data.x.to(device)
train_pos_edge_index = data.train_pos_edge_index.to(device)
edge_weights=torch.ones(train_pos_edge_index.shape[1]).to(device)
num_nodes=x.shape[0]

temp_adj=torch.ones(num_nodes,num_nodes).to(device)
temp_adj[train_pos_edge_index[0],train_pos_edge_index[1]]=0
train_neg_edge_row,train_neg_edge_col=temp_adj.nonzero(as_tuple=False).t()
train_neg_edge_index=torch.stack([train_neg_edge_row,train_neg_edge_col])
ADJ=torch.zeros(num_nodes,num_nodes).to(device)
ADJ[train_pos_edge_index[0],train_pos_edge_index[1]]=1

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def attack_features(x,model,n_iters,epsilon,train_pos_edge_index,norm='linf',variational=False):
    delta=torch.zeros_like(x).requires_grad_()
    length=torch.norm(x,dim=1,p=2,keepdim=True)
    if norm =="linf":
        alpha=epsilon/n_iters      #to make sure the total perturbations after n_iters steps is less than epsilon
    elif norm=="l2":
        max_norm=length*epsilon
        alpha=max_norm/n_iters     #to make sure the norm of  total perturbations after n_iters steps is less than epsilon*||X||_2
    for i in range(n_iters):
        z = model.encode(x+delta, train_pos_edge_index)
        loss = model.recon_loss(x,z, train_pos_edge_index)
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
def Anomaly_detection(x,model,adj):
    model.eval()
    with torch.no_grad():
        z = model.encode(x, train_pos_edge_index)
    adj_predict,att_predict=model.decoder.forward_all(z,train_pos_edge_index,sigmoid=True)
    reconstruct_att_errors=torch.norm(att_predict-x,p=2,dim=1)
    reconstruct_errors=torch.norm(adj_predict-adj,p=2,dim=1)
    return reconstruct_errors.detach()*0.5+reconstruct_att_errors.detach()*0.5
    
    

    
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
        loss=model.recon_loss(x,z,train_pos_edge_index)
        if variational:
            loss=loss+(1.0/z.shape[0])*model.kl_loss()
        if norm=='linf':
            grad=torch.autograd.grad(loss,delta)[0].sign()
        elif norm=='l2':
            grad=torch.autograd.grad(loss,delta)[0]
            grad=grad*length/torch.norm(grad.data,dim=0,p=2,keepdim=True)
            
        delta.data=delta.data+grad.data*alpha
        temp_weights=edge_weights+delta.data
        temp_weights[temp_weights<0]=0
    return delta.detach()
kl = nn.KLDivLoss(reduction="sum")        
        
def train():
    global x
    model.train()
    if args.perturbStyle =="None":
        z = model.encode(x, train_pos_edge_index)
        optimizer.zero_grad()
        loss = model.recon_loss(x,z, train_pos_edge_index)
        if args.variational:
            loss=loss+(1.0/z.shape[0])*model.kl_loss()
        loss.backward()
        optimizer.step()
    elif args.perturbStyle=="structure":
        
        delta=attack_edges(x,model,args.steps,args.epsSt,train_pos_edge_index,edge_weights,args.norm)
        edge_weights1=edge_weights.data+delta
        z = model.encode(x, train_pos_edge_index,edge_weight=edge_weights1)
        optimizer.zero_grad()
        loss = model.recon_loss(x,z, train_pos_edge_index)
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
        loss = model.recon_loss(x,z_origin, train_pos_edge_index)+robust_loss
        
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

        loss = model.recon_loss(x,z_origin, train_pos_edge_index)+robust_loss
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
        loss = model.recon_loss(x,z_origin, train_pos_edge_index)+robust_loss
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
    return model.test(z, train_pos_edge_index,pos_edge_index, neg_edge_index)

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
            model = MyGAE(GCNEncoder(num_features, out_channels),decoder=GCNDecoder(num_features,out_channels))
        else:
            model = GAE(LinearEncoder(num_features, out_channels))
    else:
        if args.linear:
            model = VGAE(VariationalLinearEncoder(num_features, out_channels))
        else:
            model = VGAE(VariationalGCNEncoder(num_features, out_channels))
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        
    for epoch in range(1, args.epochs + 1):
        loss = train()
    reconstructed_errors=Anomaly_detection(x,model,ADJ)
    auc_anomaly=roc_auc_score(dataset.data.y.squeeze().cpu().numpy(),reconstructed_errors.cpu().numpy())
    print("anomaly auc",auc_anomaly)
    repeat_auc.append(auc_anomaly)
    
average_auc=np.average(repeat_auc)
std_auc=np.std(repeat_auc)
print("auc average:",average_auc, "auc std:",std_auc)