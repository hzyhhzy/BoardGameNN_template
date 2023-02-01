from dataset import trainset
from model import ModelDic

import argparse
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch
import os
import time


if not os.path.exists("../saved_models"):
    os.mkdir("../saved_models")

def cross_entropy_loss(output, target):
    t = torch.log_softmax(output,dim=1)
    losses = torch.sum(-t*target, dim=1)+torch.sum(torch.log(target+1e-10)*target, dim=1)
    return torch.mean(losses,dim=0)

def calculatePolicyLoss(output,pt):
    output=torch.flatten(output,start_dim=1)
    pt = pt+1e-10
    wsum = torch.sum(pt, dim=1, keepdims=True)
    pt = pt/wsum
    return cross_entropy_loss(output,pt)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    #data settings
    parser.add_argument('--tdata', type=str, default='../alldata.npz', help='npz file of training data')
    parser.add_argument('--maxstep', type=int, default=100000, help='Max step to train')
    parser.add_argument('--infostep', type=int, default=500, help='Print loss every # steps')
    parser.add_argument('--savestep', type=int, default=500, help='Save model every # steps')


    #model parameters
    parser.add_argument('--modeltype', type=str, default='resnet',help='Model type, defined in model.py. Default resnet')
    parser.add_argument('--modelsize', nargs='+',type=int,
                        default=(10,128), help='Model size. "--modelsize blocks channels" if using ResNet')
    parser.add_argument('--savename', type=str ,default='model1', help='Model save path. If already existing, continue training it and ignore --modeltype and --modelsize settings')


    #training parameters
    parser.add_argument('--gpu', type=int,
                        default=0, help='Which gpu, -1 means cpu')
    parser.add_argument('--batchsize', type=int,
                        default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-3, help='Learning rate')
    parser.add_argument('--wd', type=float, default=0, help='Weight decay')

    args = parser.parse_args()

    if(args.gpu==-1):
        device=torch.device('cpu')
    else:
        device = torch.device(f"cuda:{args.gpu}")


    basepath = f'../saved_models/{args.savename}/'
    if not os.path.exists(basepath):
        os.mkdir(basepath)
    tensorboardpath=os.path.join(basepath,"tensorboardData")

    #tensorboard writer
    if not os.path.exists(tensorboardpath):
        os.mkdir(tensorboardpath)
    train_writer=SummaryWriter(os.path.join(tensorboardpath,"train"))
    val_writer=SummaryWriter(os.path.join(tensorboardpath,"val"))

    print("Building model..............................................................................................")
    modelpath=os.path.join(basepath,"model.pth")
    if os.path.exists(modelpath):
        modeldata = torch.load(modelpath,map_location="cpu")
        model_type=modeldata['model_type']
        model_param=modeldata['model_param']
        model = ModelDic[model_type](*model_param).to(device)

        model.load_state_dict(modeldata['state_dict'])
        totalstep = modeldata['totalstep']
        print(f"Loaded model: type={model_type}, size={model_param}, totalstep={totalstep}")
    else:
        totalstep = 0
        model_type=args.modeltype
        model_param=args.modelsize
        model = ModelDic[model_type](*model_param).to(device)

    startstep=totalstep


    optimizer = optim.Adam(model.parameters(),lr=args.lr,weight_decay=args.wd)
    model.train()




    time0=time.time()
    loss_record_init=[0,0,0,1e-30]
    loss_record=loss_record_init.copy()


    print("Start training..............................................................................................")
    for epoch in range(100000000):

        # dataset should be reloaded every epoch to get a different symmetry
        print("Loading dataset..............................................................................................")
        tDataset = trainset(args.tdata, randomsym=True)
        print(f"Data file has {tDataset.__len__()} rows")
        tDataloader = DataLoader(tDataset, shuffle=True, batch_size=args.batchsize)

        for _ , (board, valueTarget, policyTarget) in enumerate(tDataloader):
            if(board.shape[0]!=args.batchsize): #只要完整的batch
                continue

            # data
            board = board.to(device)
            valueTarget = valueTarget.to(device)
            policyTarget = policyTarget.to(device)

            # calculate loss
            optimizer.zero_grad()
            value, policy = model(board)
            vloss = cross_entropy_loss(value, valueTarget)
            ploss = calculatePolicyLoss(policy, policyTarget)
            loss = ploss+vloss

            # optimize
            loss.backward()
            optimizer.step()

            # log
            loss_record[0]+=(vloss.detach().item()+ploss.detach().item())
            loss_record[1]+=vloss.detach().item()
            loss_record[2]+=ploss.detach().item()
            loss_record[3]+=1

            #maybe save model
            totalstep += 1
            if(totalstep % args.infostep == 0):
                time1=time.time()
                time_used=time1-time0
                time0=time1
                totalloss_train=loss_record[0]/loss_record[3]
                vloss_train=loss_record[1]/loss_record[3]
                ploss_train=loss_record[2]/loss_record[3]
                print("name: {}, time: {:.2f} s, step: {}, totalloss: {:.4f}, vloss: {:.4f}, ploss: {:.4f}"
                      .format(args.savename,time_used,totalstep,totalloss_train,vloss_train,ploss_train))
                train_writer.add_scalar("steps_per_second",loss_record[3]/time_used,global_step=totalstep)
                train_writer.add_scalar("totalloss",totalloss_train,global_step=totalstep)
                train_writer.add_scalar("vloss",vloss_train,global_step=totalstep)
                train_writer.add_scalar("ploss",ploss_train,global_step=totalstep)

                loss_record = loss_record_init.copy()

            if((totalstep % args.savestep == 0) or (totalstep-startstep==args.maxstep)):

                print(f"Finished training {totalstep} steps")
                torch.save(
                    {'totalstep': totalstep,
                     'state_dict': model.state_dict(),
                     'model_type': model.model_type,
                     'model_param':model.model_param},
                    modelpath)
                print('Model saved in {}\n'.format(modelpath))

            #maybe exit
            if(totalstep - startstep >= args.maxstep):
                print("Reached max step, exiting")
                exit(0)