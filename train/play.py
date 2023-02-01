from config import boardH,boardW
from model import input_c,ModelDic

import torch
import numpy as np
from copy import deepcopy

C_EMPTY=0
C_BLACK=1
C_WHITE=2

botColor = C_WHITE # what color does bot play

class Board:
    #add your game logic in this class
    def __init__(self):
        self.board = np.zeros(shape=(boardH, boardW),dtype=np.int8)

    def play(self,x,y,player):
        self.board[y,x]=player

    def isLegal(self,x,y,nextPlayer):
        if self.board[y,x] != C_EMPTY:
            return False
        return True

    def nnInput(self,nextPlayer):
        input=np.zeros([input_c,boardH,boardW],dtype=np.float32)
        if nextPlayer==C_BLACK:
            input[0] = self.board==C_BLACK
            input[1] = self.board==C_WHITE
        else:
            input[0] = self.board==C_WHITE
            input[1] = self.board==C_BLACK

        input[2] = self.board==C_BLACK

        return input

    def print(self,markLocX=-1,markLocY=-1):
        print(" "*3,end="")
        for x in range(boardW):
            print(chr(x+ord("A")),end="")
            print(" ",end="")
        print("")
        for y in range(boardH):
            yOrd=boardH-y
            print(yOrd//10 if yOrd>=10 else " ",end="")
            print(yOrd%10,end="")
            print(" ",end="")
            for x in range(boardW):
                color=self.board[y,x]
                c="#"
                if x==markLocX and y==markLocY:
                    c="@"
                elif color==C_EMPTY:
                    c="."
                elif color==C_BLACK:
                    c="x"
                elif color==C_WHITE:
                    c="o"
                print(c,end="")
                print(" ",end="")
            print("")
        print(" "*3,end="")
        for x in range(boardW):
            print(chr(x+ord("A")),end="")
            print(" ",end="")
        print("")
        print("")


#Load model
def loadModel(modelpath):
    modeldata = torch.load(modelpath, map_location="cpu")
    model_type = modeldata['model_type']
    model_param = modeldata['model_param']
    model = ModelDic[model_type](*model_param)
    model.load_state_dict(modeldata['state_dict'])
    model.eval()
    print(f"Loaded model: type={model_type}, size={model_param}, totalstep={modeldata['totalstep']}")
    return model

# choose a move based on policy, with some randomness
# temperture means randomness
# temperture=0.0 means choose the move with the largest policy
# temperture=infinity means choose the move completely randomly
def getRandomMoveUsingPolicy(board,policy,player,temperture=0.0):
    policy=policy-np.max(policy)
    for i in range(boardW*boardH):
        if not board.isLegal(i%boardW,i//boardW,player):
            policy[i]=-10000
    policy=policy-np.max(policy)
    for i in range(boardW*boardH):
        if(policy[i]<-1):
            policy[i]=-10000
    policy=policy/temperture
    probs=np.exp(policy)
    probs=probs/sum(probs)
    for i in range(boardW*boardH):
        if(probs[i]<1e-3):
            probs[i]=0
    move = int(np.random.choice([i for i in range(boardW*boardW)],p=probs))
    x = move % boardW
    y = move // boardW
    return x,y


def genmove(board,model,player): #generate a move using model
    nninput=board.nnInput(player)
    nninput=torch.tensor(nninput).unsqueeze(0)
    v,p = model(nninput)

    value=torch.softmax(v,dim=1)
    value=value.detach().numpy().reshape((-1))
    winrate=0.5*(value[0]-value[1])+0.5

    policy=p.detach().numpy().reshape((-1))
    policytemp=0.3
    x,y=getRandomMoveUsingPolicy(board,policy,player,policytemp)

    return x,y,winrate

def loc2str(x,y):
    return chr(x + ord("A"))+str(boardH-y)

str2loc={}
for y in range(boardH):
    for x in range(boardW):
        str2loc[chr(x + ord("A"))+str(boardH-y)]=(x,y)
        str2loc[chr(x + ord("a"))+str(boardH-y)]=(x,y)
        str2loc[chr(x + ord("A"))+" "+str(boardH-y)]=(x,y)
        str2loc[chr(x + ord("a"))+" "+str(boardH-y)]=(x,y)

if __name__ == '__main__':
    modelpath="../saved_models/model1/model.pth"
    model=loadModel(modelpath)

    board = Board()
    boardHistory=[] #for undo
    boardHistory.append(deepcopy(board))
    nextPlayer=C_BLACK

    board.print()


    while True:
        if nextPlayer==botColor:
            x,y,winrate=genmove(board,model,nextPlayer)

            board.play(x,y,nextPlayer)
            boardHistory.append(deepcopy(board))
            nextPlayer=3-nextPlayer

            board.print(x,y)
            print("Bot played " + ("Black" if nextPlayer==C_WHITE else "WHITE") + " at "+loc2str(x,y))
            print("Bot's winrate = {:.1f} %".format(winrate*100))
        else:
            s=input("Your play "+ ("Black" if nextPlayer==C_BLACK else "WHITE") + "(\"undo\" if you want to undo, \"switch\" if you want to switch color) :")
            if(s=="switch" or s=="\"switch\""):
                botColor=3-botColor
            elif(s=="undo" or s=="\"undo\""):
                if(len(boardHistory)<3):
                    print("Cannot undo")
                else:
                    board=deepcopy(boardHistory[-3])
                    boardHistory.pop()
                    boardHistory.pop()
                    board.print()
            elif s not in str2loc:
                print("Wrong location")
            else:
                x,y=str2loc[s]
                if not board.isLegal(x,y,nextPlayer):
                    print("Illegal move")
                else:
                    board.play(x,y,nextPlayer)
                    boardHistory.append(deepcopy(board))
                    nextPlayer=3-nextPlayer

                    board.print(x,y)
                    print("You played " + ("Black" if nextPlayer==C_WHITE else "WHITE") + " at "+loc2str(x,y))


