import torch
import torch.nn as nn
from dgl.data.utils import load_graphs
from utils import *
from model import Model
from sklearn import metrics
import argparse
import random

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

setup_seed(10)

parser = argparse.ArgumentParser('Set Model', add_help=False)
parser.add_argument('--device', default="cuda:1", type=str)
parser.add_argument('--lr', default=1e-3, type=float)
parser.add_argument('--layer', default=4, type=int)
parser.add_argument('--attention', default="BiAttention", type=str)
parser.add_argument('--drop', default=0.5, type=float)
parser.add_argument('--negative_num', default=4, type=int)

args = parser.parse_args()

g = load_graphs("Data/graph")[0][0]
g.node_dict = {}
g.edge_dict = {}
for ntype in g.ntypes:
    g.node_dict[ntype] = len(g.node_dict)
for etype in g.etypes:
    g.edge_dict[etype] = len(g.edge_dict)
    g.edges[etype].data['id'] = torch.ones(g.number_of_edges(etype), dtype=torch.long) * g.edge_dict[etype] 



model =Model(5,100, 100,('drug', 'treat', 'micor')).to(args.device)
opt = torch.optim.Adam(model.parameters(),lr =args.lr)
BCEloss = nn.BCELoss()

etype = ('drug', 'treat', 'micor')


def eval(test_g):
    test_g.node_dict = {}
    test_g.edge_dict = {}
    for ntype in test_g.ntypes:
        test_g.node_dict[ntype] = len(test_g.node_dict)
    for etype in test_g.etypes:
        test_g.edge_dict[etype] = len(test_g.edge_dict)
        test_g.edges[etype].data['id'] = torch.ones(test_g.number_of_edges(etype), dtype=torch.long,device=args.device) * test_g.edge_dict[etype] 
    node_features = {"drug":test_g.nodes['drug'].data['h'],"micor":test_g.nodes['micor'].data['h']}
    negative_graph = construct_negative_graph(test_g,args.negative_num, ('drug', 'treat', 'micor'),args.device)
    model.eval()
    pos_score,neg_score =model(test_g,negative_graph,node_features,('drug', 'treat', 'micor'))
    pos_score = torch.sigmoid(pos_score).squeeze(1)
    neg_score = torch.sigmoid(neg_score).squeeze(1)
    pred = torch.cat([pos_score,neg_score]).cpu().detach().numpy()
    label = torch.cat([torch.ones(pos_score.shape[0]),torch.zeros(neg_score.shape[0])]).detach().numpy()
    return metrics.roc_auc_score(label,pred),metrics.average_precision_score(label,pred)


def train(model):
    auc =[0.]
    aupr =[0.]
    for epoch in range(1,5001):
        train_g = load_graphs("Data/train_g")[0][0].to(args.device)
        test_g = load_graphs("Data/test_g")[0][0].to(args.device)
        model.train()
        node_features = {"drug":train_g.nodes['drug'].data['h'],"micor":train_g.nodes['micor'].data['h']}
        negative_graph = construct_negative_graph(train_g,args.negative_num, ('drug', 'treat', 'micor'),args.device)
        

        train_g.node_dict = {}
        train_g.edge_dict = {}
        for ntype in train_g.ntypes:
            train_g.node_dict[ntype] = len(train_g.node_dict)
        for etype in train_g.etypes:
            train_g.edge_dict[etype] = len(train_g.edge_dict)
            train_g.edges[etype].data['id'] = torch.ones(train_g.number_of_edges(etype), dtype=torch.long,device=args.device) * train_g.edge_dict[etype]
        
       
        pos_score,neg_score =model(train_g, negative_graph,node_features, ('drug', 'treat', 'micor'))
        pred = torch.cat([pos_score,neg_score])
        label = torch.cat([torch.ones(pos_score.shape[0]),torch.zeros(neg_score.shape[0])]).reshape(-1,1).to(args.device)
        
        loss = BCEloss(pred, label)
        opt.zero_grad()
        loss.backward()
        opt.step()
        model_auc,model_aupr= eval(test_g)
        print("Epoch:{} finishing! Loss:{:.5f} AUC:{:.5f}  AUPR:{:.5f} ".format(epoch,loss,model_auc,model_aupr))

        auc.append(model_auc.item())
        aupr.append(model_aupr.item())
        
        if epoch%1000==0:
            
            for param_group in opt.param_groups:
                param_group['lr'] = param_group['lr']*0.5
          
        
        
    return torch.tensor(auc).max().item(),torch.tensor(aupr).max().item()


def main():
    print("*"*10,"train","*"*10)
    auc,aupr = train(model)
    print("auc:",auc)
    print("aupr:",aupr)

if __name__ == '__main__':
    main()

        
    
