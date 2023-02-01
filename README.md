# BoardGameNN_template
A simple neural network training template for board games   
   
# Data format
Put your dataset at the root dir as alldata.npz   
alldata.npz should contain 4 matrices: bf,gf,vt,pt   
**bf**: Board Features, Location-specific features. shape=[N,C1,H,W] eg. where are black stones and white stones   
**gf**: Global Features. shape=[N,C2] eg. which color is the next player   
**vt**: Value Target. shape=[N,3]  probability of win/loss/draw for the next player   
**pt**: Policy Target. shape=[N,H*W]  probability of the next move   
   
input_c in model.py should be C1+C2   
C2 should not be zero. If you don't need global features, you can let C2=1 and fill it with zero   
You should modify Board.nninput() in play.py based on the format of your dataset   
   
# Training scripts
## train.py
Run `train.py -h` to see the usage
## play.py   
You can play with your trained model here.   
You should implement your game rules and NNinput(bf and gf) in "Board" class   
## model.py
A very simple ResNet.    
You should change input_c here based on your dataset    
You can design other models in this file and add it to ModelDic
## dataset.py
This script will concat bf and gf, and apply random symmetry.   
## config.py
Global variables

