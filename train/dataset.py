from config import boardH,boardW

from torch.utils.data import Dataset
import numpy as np


def apply_sym(d,sym,dimY,dimX):
    x_sym=sym%2
    sym=sym//2

    y_sym=sym%2
    sym=sym//2

    transp=sym%2

    if(x_sym):
        d=np.flip(d,axis=dimX)
    if(y_sym):
        d=np.flip(d,axis=dimY)
    if(transp):
        d=np.swapaxes(d,dimX,dimY)

    return d.copy()

def apply_sym_policyTarget(pt,sym):
    assert(pt.ndim==1)
    assert(pt.shape[0]==boardH*boardW)
    pt=pt.reshape(boardH,boardW)
    pt=apply_sym(pt,sym,0,1)
    pt=pt.reshape(-1)
    return pt.copy()

class trainset(Dataset):
    #randomsym: Randomly symmetric each sample. Does not change total sample num
    def __init__(self, npz_path,randomsym=True):

        #bf: board features. shape=[N,C,H,W] eg. where are black stones and white stones
        #gf: global features. shape=[N,C] eg. which color is the next player
        #vt: value target. shape=[N,3]  probability of win/loss/draw for the next player
        #pt: policy target. shape=[N,H*W]  probability of the next move

        data = np.load(npz_path)
        self.bf=data["bf"]
        self.gf=data["gf"]
        self.vt=data["vt"]
        self.pt=data["pt"]

        if randomsym:
            self.syms=np.random.randint(0,8,self.vt.shape[0])
        else:
            self.syms=np.zeros(self.vt.shape[0],dtype=np.int)

    def __getitem__(self, index):

        sym=self.syms[index]

        bf1=self.bf[index].astype(np.float32)
        pt1=self.pt[index].astype(np.float32)

        #apply symmetry
        bf1=apply_sym(bf1,sym,1,2)
        pt1=apply_sym_policyTarget(pt1,sym=sym)

        #concat bf and gf
        gf1=self.gf[index].astype(np.float32)
        gf1 = gf1.reshape((gf1.shape[0], 1, 1)).repeat(bf1.shape[1], axis=1).repeat(bf1.shape[2], axis=2)
        bf1 = np.concatenate((bf1, gf1), axis=0)

        vt1=self.vt[index].astype(np.float32)

        return bf1,vt1,pt1
    def __len__(self):
        return self.vt.shape[0]



