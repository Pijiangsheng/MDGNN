import torch
import dgl
import torch.nn as nn
import dgl.nn as dglnn
import dgl.function as fn
import torch.nn.functional as F
from dgl.nn import GraphConv,GATConv
from utils import HeteroDotProductPredictor





"""
Relation Graph Convolution Network
"""
class RGCN(nn.Module):
    def __init__(self, node_feats, out_feats,etype):
        super().__init__()

        self.conv1 = dglnn.HeteroGraphConv({
            'interaction': GraphConv(node_feats,out_feats),
            'treat':GraphConv(node_feats,out_feats,),
            'similar':GraphConv(node_feats,out_feats)
            }, aggregate='sum')
        self.conv2 = dglnn.HeteroGraphConv({
            'interaction': GraphConv(out_feats,out_feats),
            'treat':GraphConv(out_feats,out_feats),
            'similar':GraphConv(out_feats,out_feats)
            }, aggregate='sum')

    def forward(self, graph, inputs):
        # inputs are features of nodes
        h = self.conv1(graph, inputs)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv2(graph, h)
        return h


        

"""
        Node Attention and Feature Attention
    """
class Block(nn.Module):
    def __init__(self,  node_features,out_features, rel_names):
        super(Block,self).__init__()
        self.sage1 = RGCN(node_features, out_features, rel_names)
        self.sage2 = RGCN(out_features, out_features, rel_names)
        self.sage3 = RGCN(out_features,out_features, rel_names)
        self.sage4 = RGCN(out_features, out_features, rel_names)
        self.sage5 = RGCN(out_features,out_features, rel_names)
        self.sage6 = RGCN(out_features,out_features, rel_names)
        #self.linear = nn.Sequential(nn.Linear(out_features,out_features),nn.ReLU(),nn.Dropout(),nn.Linear(out_features,out_features))
        self.linear = nn.Linear(out_features,out_features)
        self.linear1 = nn.Linear(out_features,out_features,bias=False)
        self.linear2 = nn.Linear(out_features,out_features,bias=False)
        self.linear3 = nn.Linear(out_features,out_features,bias=False)
        self.linear4 = nn.Linear(out_features,out_features,bias=False)
        self.linear5 = nn.Linear(out_features,out_features)
        self.linear6 = nn.Linear(out_features,out_features)

        self.activate =nn.ReLU()
        self.mlp = nn.Sequential(nn.Linear(out_features,out_features),nn.Dropout(),nn.ReLU(),nn.Linear(out_features,out_features))
        self.pred = HeteroDotProductPredictor()
        self.dropout = nn.Dropout()
    def forward(self, g,  x, etype):
        """
            Node Attention
        """
        h1 = self.sage1(g, x)
        h2 = self.sage2(g, x)
        h3 = self.sage3(g, x)
        u,_,v = etype
        u_len = h1[u].shape[0]
        h1 = torch.cat([h1[u],h1[v]],dim=0)
        h2 = torch.cat([h2[u],h2[v]],dim=0)
        h3 = torch.cat([h3[u],h3[v]],dim=0)
        
        #node_score = torch.matmul(self.activate(self.linear1(h1)),self.activate(self.linear2(h2)).t())
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            node_score = torch.matmul(h1,h2.t())
        node_score = F.softmax(node_score,dim=1)
        h = h3+self.activate(self.linear5(torch.matmul(node_score,h3)))
        
        """
            Feature Attention
        """
        h =  {u:h[:u_len],v:h[u_len:]}
        h4 = self.sage4(g, h)
        h5 = self.sage5(g, h)
        h6 = self.sage6(g,h)
        h4 = torch.cat([h4[u],h4[v]],dim=0)
        h5 = torch.cat([h5[u],h5[v]],dim=0)
        h6 = torch.cat([h6[u],h6[v]],dim=0)
        #feature_score = torch.matmul(self.activate(self.linear3(h4)).t(),self.activate(self.linear4(h5)))
        feature_score = torch.matmul(h4.t(),h5)
    
        feature_score = F.softmax(feature_score,dim=0)
        h = h6+self.activate(self.linear6(torch.matmul(h6,feature_score)))
        
        return h



class Transformer(nn.Module):
    def __init__(self,in_dim,out_dim):
        super(Transformer,self).__init__()
        self.q = nn.Linear(in_dim,out_dim)
        self.k = nn.Linear(in_dim,out_dim)
        self.v = nn.Linear(in_dim,out_dim)
        self.activate = nn.ReLU()
    
    def forward(self,x):
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)
        score = F.softmax(torch.matmul(q,k.t()),dim=0)
        result = x+self.activate(torch.matmul(score,v))

        return result

class Model(nn.Module):
    def __init__(self,layer,node_features, out_features, rel_names):
        super(Model,self).__init__()
        self.layer = layer
        self.dim = out_features
        
        
        self.block1 = Block(node_features, out_features, rel_names)
        self.block2 = Block(out_features, out_features, rel_names)
        self.block3 = Block(out_features, out_features, rel_names)
        self.block4 = Block(out_features, out_features, rel_names)
        self.block5 = Block(out_features, out_features, rel_names)
        self.block6 = Block(out_features, out_features, rel_names)
        self.pred = HeteroDotProductPredictor()
        self.linear1 = nn.Linear(out_features,out_features)
        self.linear2 = nn.Linear(out_features,out_features)
        self.linear3 = nn.Linear(out_features,out_features)
        self.linear4 = nn.Linear(out_features,out_features)
        self.linear5 = nn.Linear(out_features,out_features)
        self.linear6 = nn.Linear(out_features,out_features)
        self.activate =nn.ReLU()
        self.dropout = nn.Dropout()
        self.trans1_1 = Transformer(out_features,out_features)
        self.trans2_1 = Transformer(out_features,out_features)
        self.trans3_1 = Transformer(out_features,out_features)
        self.trans4_1 = Transformer(out_features,out_features)
        self.trans5_1 = Transformer(out_features,out_features)
        self.trans1_2 = Transformer(out_features,out_features)
        self.trans2_2 = Transformer(out_features,out_features)
        self.trans3_2 = Transformer(out_features,out_features)
        self.trans4_2 = Transformer(out_features,out_features)
        self.trans5_2 = Transformer(out_features,out_features)
        self.mlp = nn.Sequential(nn.Linear(5*out_features,2*out_features),nn.ReLU(),nn.Dropout(),nn.Linear(2*out_features,out_features))
    def forward(self, g, neg_g, x, etype):
        h_list =[x]
        u,_,v = etype
        u_len = x[u].shape[0]
        h1 = self.dropout(self.linear1(self.block1(g,x,etype)))
        
        x = torch.cat([x[u],x[v]],dim=0)
        h = x + h1
        x1=h
        h =  {u:h[:u_len],v:h[u_len:]}
        h2 = self.dropout(self.linear2(self.block2(g,h,etype)))
        h = torch.cat([h[u],h[v]],dim=0)
        h = h + h2
        x2= h 
        h =  {u:h[:u_len],v:h[u_len:]}
        h3 = self.dropout(self.linear3(self.block3(g,h,etype)))
        h = torch.cat([h[u],h[v]],dim=0)
        h = h + h3
        x3= h 
        h =  {u:h[:u_len],v:h[u_len:]}
        h4 = self.dropout(self.linear4(self.block4(g,h,etype)))
        h = torch.cat([h[u],h[v]],dim=0)
        h = h + h4
        x4= h 
        
        h =  {u:h[:u_len],v:h[u_len:]}
        h5 = self.dropout(self.linear5(self.block5(g,h,etype)))
        h = torch.cat([h[u],h[v]],dim=0)
        h = h + h5
        x5=h
        h =  {u:h[:u_len],v:h[u_len:]}
        """
        h6 = self.dropout(self.linear6(self.block6(g,h,etype)))
        h = torch.cat([h[u],h[v]],dim=0)
        h = h + h6
        """


        x1=self.trans1_2(self.trans1_1(x1))
        x2=self.trans2_2(self.trans2_1(x2))
        x3=self.trans3_2(self.trans3_1(x3))
        x4=self.trans4_2(self.trans4_1(x4))
        x5=self.trans5_2(self.trans5_1(x5))
        h= torch.cat([x1,x2,x3,x4,x5],dim=1)
        h=self.mlp(h)
        h =  {u:h[:u_len],v:h[u_len:]}
        
        return  torch.sigmoid(self.pred(g, h, etype)),torch.sigmoid(self.pred(neg_g, h, etype))
