import torch
import dgl
import torch.nn as nn
import dgl.nn as dglnn
import numpy as np
import scipy.sparse as sp
import dgl.function as fn
import matplotlib.pyplot as plt
from dgl.data.utils import load_graphs,save_graphs

def split(g,test_len=0.1):
    np.random.seed(10)
    sub_graph = dgl.edge_type_subgraph(g,[('drug', 'treat', 'micor')])
    u, v = sub_graph.edges()

    eids = np.arange( sub_graph.number_of_edges())
    eids = np.random.permutation(eids)
    test_size = int(len(eids) * test_len)
    train_size = sub_graph.number_of_edges() - test_size
    test_pos_u, test_pos_v = u[eids[:test_size]], v[eids[:test_size]]
    train_pos_u, train_pos_v = u[eids[test_size:]], v[eids[test_size:]]

    # Find all negative edges and split them for training and testing
    adj = sp.coo_matrix((np.ones(len(u)), (u.numpy(), v.numpy())))

    adj_neg = 1 - adj.todense() 
    neg_u, neg_v = np.where(adj_neg != 0)

    neg_eids = np.random.choice(len(neg_u),  sub_graph.number_of_edges())
    test_neg_u, test_neg_v = neg_u[neg_eids[:test_size]], neg_v[neg_eids[:test_size]]
    train_neg_u, train_neg_v = neg_u[neg_eids[test_size:]], neg_v[neg_eids[test_size:]]
    #return train_pos_u,train_pos_v,test_pos_u,test_pos_v,train_neg_u,train_neg_v,test_neg_u,test_neg_v
    return eids[:test_size],eids[test_size:]

def figure(n,data,path,title):
    fig = plt.figure(figsize=(10,6))
    color = ['red','blue','green']
    label = ['negtive sample 4','negtive sample 5',"negtive sample 6"]
    for  i in range(n):
        plt.plot(np.arange(data[i].shape[0]),data[i],c=color[i],label = label[i])
    plt.xlabel("Epoch")
    plt.ylabel(path)
    plt.legend(label)
    plt.title(title)
    plt.savefig(path+".png")


    
class HeteroDotProductPredictor(nn.Module):
    def forward(self, graph, h, etype):
        # h contains the node representations for each node type computed from
        # the GNN defined in the previous section (Section 5.1).
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(fn.u_dot_v('h', 'h', 'score'), etype=etype)
            return graph.edges[etype].data['score']


def construct_negative_graph(graph, k, etype,device):
    utype, _, vtype = etype
    src, dst = graph.edges(etype=etype)
    neg_src = src.repeat_interleave(k).to(device)

    neg_dst = torch.randint(0, graph.num_nodes(vtype), (len(src) * k,)).to(device)

    return dgl.heterograph(
        {etype: (neg_src, neg_dst)},
        num_nodes_dict={ntype: graph.num_nodes(ntype) for ntype in graph.ntypes})


def construct_negative_graph_(train_g, etype):
    u,_,v = etype
    etype = _
    train_g_ = train_g.edge_type_subgraph([etype])
    train_g_ = dgl.to_homogeneous(train_g_)
    train_adj =dgl.khop_adj(train_g_,1)
    train_neg_adj = 1-train_adj
    train_neg_u,train_neg_v= np.where(train_neg_adj != 0)
    u_len = train_g.number_of_nodes(u)
    idx = np.where(train_neg_u<u_len)
    train_neg_u=train_neg_u[idx]
    train_neg_v = train_neg_v[idx]
    idx = np.where(train_neg_v>=u_len)
    train_neg_u=train_neg_u[idx]
    train_neg_v = train_neg_v[idx]-u_len
    _ = train_g_.edges()
    len_ = train_g.edges(etype=etype)[0].shape[0]
    train_g = dgl.remove_edges(train_g,torch.tensor([i for i in range(len_)]),etype = etype)
    #train_g.edges(etype="treat") = (train_neg_u,train_neg_v)
    train_neg_g = dgl.add_edges(train_g,train_neg_u,train_neg_v,etype = etype)
    return train_neg_g

if __name__ == '__main__':
    g = load_graphs("Data/graph")[0][0]
 
    
    
    