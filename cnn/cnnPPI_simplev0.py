import torch.optim as optim
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import csv
import numpy as np
import pickle
from torch.nn import BCELoss

from torch import device
from torch import cuda
from torch.utils.data import Dataset,DataLoader
#import GPUtil as GPU
import copy
from sklearn import metrics
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import accuracy_score
import math
import sys
import matplotlib.pyplot as plt
import gc
torch.manual_seed(7)   

class SiameseNetworkDataset():

    def __init__(self, csv_file, directory, folder,transform=None):
        # used to prepare the labels and images path

        #folderLoc = '/Users/sabby/Documents/TPD/files/'
        folderLoc = '/data/data/profiles_cropped/'
        
        loc = folderLoc + 'proteins_number_crop_512.p'
        number = pickle.load(open(loc, "rb"),encoding='latin1')

        loc = folderLoc + 'proteins_profile_crop_512.p'
        #loc = folderLoc + 'save.p'
        crop = pickle.load(open(loc, "rb"),encoding='latin1')
        fLog = open('/data/logs/log.txt','w')
        #fLog.write('keys')
        #fLog.write('\n')
        #fLog.write(str(crop.keys()))
        fLog.close()
        position_trg = []
        labels_trg = []
        output_trg = []

        fileName = csv_file
        num_rec = getSizeofFile(fileName, directory, number)

        print('total lines myTrain', num_rec)

        input_a = np.zeros((num_rec, 20, 1, 512), dtype=np.float32)
        input_b = np.zeros((num_rec, 20, 1, 512), dtype=np.float32)

        #def populateData(input_x, input_y, fileName, folderLoc, crop, number):

        outList = populateData(input_a,input_b,fileName,folderLoc,crop,number)
        self.input_x = outList[0]
        self.input_y = outList[1]
        self.position = outList[2]
        self.output = outList[3]
        self.labels = outList[4]

        #print('out dataset', len(self.output))
        #print('label dataset', len(self.labels))


        self.num_rec = num_rec

    def __getitem__(self, index):

        input1 = self.input_x[index]
        input2 = self.input_y[index]
        label = self.output[index]

        return input1,input2,label

    def __len__(self):
        return self.num_rec

class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        # Setting up the Sequential of CNN Layers
        self.cnn1 = nn.Conv2d(20, 64, kernel_size=(1,5), stride=(1,1),padding=(0,2))
        self.bn1 =  nn.BatchNorm2d(64)
        self.act1 = nn.ReLU(inplace=True)
        self.pool1 = nn.AvgPool2d(kernel_size=(1,4), stride=(1,4),padding=(0,0))

        #torch.Size([1, 64, 1, 128])
        #input_dim, hidden_dim, layer_dim, output_dim
        #self.lstm1 = LSTMModel(64, 64, 2, 64)
        
        self.cnn2 = nn.Conv2d(64,128, kernel_size=(1,5), stride=(1,1),padding=(0,2))
        self.bn2 =  nn.BatchNorm2d(128)
        self.act2 = nn.ReLU(inplace=True)
        self.pool2 = nn.AvgPool2d(kernel_size=(1,4), stride=(1,4),padding=(0,0))

        self.cnn3 = nn.Conv2d(128, 256, kernel_size=(1,5), stride=(1,1),padding=(0,2))
        self.bn3 =  nn.BatchNorm2d(256)
        self.act3 = nn.ReLU(inplace=True)
        self.pool3 = nn.AvgPool2d(kernel_size=(1,4), stride=(1,4),padding=(0,0))

        #torch.Size([1, 128, 1, 32])
        #self.lstm2 = LSTMModel(128, 64, 2, 128)
            
        self.cnn4 = nn.Conv2d(256, 512,kernel_size=(1,5), stride=(1,1),padding=(0,2))
        self.bn4 =  nn.BatchNorm2d(512)
        self.act4 =  nn.ReLU(inplace=True)
        self.pool4= nn.AvgPool2d(kernel_size=(1,8), stride=(1,8),padding=(0,0))

        #torch.Size([1, 256, 1, 8])
        #self.lstm3 = LSTMModel(256, 64, 2, 256)

        #self.cnn4 = nn.Conv2d(512, 1024, kernel_size=(1,5), stride=(1,1),padding=(0,2))
        #self.bn4 =  nn.BatchNorm2d(1024)
        #self.act4 = nn.ReLU(inplace=True)
        #self.pool4 = nn.AvgPool2d(kernel_size=(1,8), stride=(1,8),padding=(0,0))

        # Defining the fully connected layers
        self.fc1 = nn.Linear(512,1)
        self.sigmoidOut = nn.Sigmoid()

        lin1 = nn.Linear(512,512)
        lin2 = nn.Linear(512,512)
        bn1K = nn.BatchNorm1d(512,eps=1e-05, momentum=0.1,affine=False)
        bn2K = nn.BatchNorm1d(512,eps=1e-05, momentum=0.1,affine=False)
        act1K = nn.ReLU(inplace=True)
        act2K = nn.ReLU(inplace=True)

        lin1c = copy.deepcopy(lin1)
        lin2c = copy.deepcopy(lin2)
        bn1Kc = copy.deepcopy(bn1K)
        bn2Kc = copy.deepcopy(bn2K)
        act1Kc = copy.deepcopy(act1K)
        act2Kc = copy.deepcopy(act2K)

        self.lin1 = lin1
        self.lin2 = lin2
        self.bn1K = bn1K
        self.bn2K = bn2K
        self.act1K = act1K
        self.act2K = act2K
        self.lin1c = lin1c
        self.lin2c = lin2c
        self.bn1Kc = bn1Kc
        self.bn2Kc = bn2Kc
        self.act1Kc = act1Kc
        self.act2Kc = act2Kc

    def forward_once(self, x):
        # Forward pass
        #print('init ',x)
        output = self.cnn1(x)
        output = self.bn1(output)
        output = self.act1(output)
        output = self.pool1(output)
        #print('out 1',output.size())

        #print('LSTM in array',output.size()[0],output.size()[1],output.size()[2],output.size()[3])
        #outputLSTM =output.view(output.size()[3],output.size()[0],output.size()[1])
        #outputLSTM = self.lstm1(outputLSTM)
        #print('LSTM out array',outputLSTM.size())
        #outputLSTM = outputLSTM.view(output.size())
        
        #print('outputLSTM size',outputLSTM.size())
        #interOutput = torch.cat((output,outputLSTM),1)
        #interOutput = torch.add(output,outputLSTM)
        #print('interOutput size',interOutput.size())
        #print('out 1',output.size())
        #torch.Size([1, 64, 1, 128i])
        #torch.Size([100, 64, 1, 128])

        output = self.cnn2(output)
        output = self.bn2(output)
        output = self.act2(output)
        output = self.pool2(output)
        #print('out 2',output.size())

        #print('LSTM in array', output.size()[0], output.size()[1], output.size()[2], output.size()[3])
        #output = output.view(output.size()[3], output.size()[0], output.size()[1])
        #output = self.lstm2(output)
        #print('LSTM out array', output.size()[0], output.size()[1], output.size()[2])
        #output = output.view(output.size()[1], output.size()[2], 1, output.size()[0])
        #print('out 2',output.size())


        output = self.cnn3(output)
        output = self.bn3(output)
        output = self.act3(output)
        output = self.pool3(output)
        #print('out 3',output.size())

        #print('LSTM in array', output.size()[0], output.size()[1], output.size()[2], output.size()[3])
        #output = output.view(output.size()[3], output.size()[0], output.size()[1])
        #output = self.lstm3(output)
        #print('LSTM out array', output.size()[0], output.size()[1], output.size()[2])
        #output = output.view(output.size()[1], output.size()[2], 1, output.size()[0])

        output = self.cnn4(output)
        output = self.bn4(output)
        output = self.act4(output)
        output = self.pool4(output)
        #print('out 4',output.size())

        output = output.view(output.size()[0], -1)
        return output

    def forward(self, input1, input2):
        # forward pass of input 1
        #output1 = torch.cat((input1,input2),1)

        output1 = self.forward_once(input1)
        
        #output1a = self.lin1(output1)
        #print('output1a',output1a.size())
        #output1a = self.bn1K(output1a)
        #output1a = self.act1K(output1a)

        #output1b = self.lin2(output1)
        #output1b = self.bn2K(output1b)
        #output1b = self.act2K(output1b)
        
        #output1c = torch.cat((output1a,output1b),1)
        # forward pass of input 2
        
        output2 = self.forward_once(input2)
        
        #output2a = self.lin2c(output2)
        #output2a = self.bn2Kc(output2a)
        #output2a = self.act2Kc(output2a)
        
        #output2b = self.lin1c(output2)
        #output2b = self.bn1Kc(output2b)
        #output2b = self.act1Kc(output2b)

        #output2c = torch.cat((output2a,output2b),1)

        hadamard_product = output1*output2
        #print('output1',output1c.shape)

        out = self.fc1(hadamard_product)
        out = self.sigmoidOut(out)
        out = out.view(out.size(0), -1)

        # returning the feature vectors of two inputs
        return out

def main():
    #folder = sys.argv[1]
    folder=''
    folderLoc = '/data/data/csv/profppi/'

    print('In main ...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('In main ...1')
    siamese_dataset_trg = SiameseNetworkDataset('0.train.csv', folderLoc,folder)
    print('In main ...2')
    siamese_dataset_tst = SiameseNetworkDataset('0.test.csv', folderLoc,folder)
    print('In main ...3',siamese_dataset_trg)
    batchsize = 100
    trg_loader = DataLoader(siamese_dataset_trg,batch_size=batchsize,shuffle=True,drop_last=True)
    print('In main ...4')
    tst_loader = DataLoader(siamese_dataset_tst,batch_size=1)
    print('In main ...5')
    #print('labels in test',siamese_dataset_tst.labels)
    fLog = open('/data/logs/log.txt','a')

    fLog.write('Number of Training')
    fLog.write("\n")
    fLog.write(str(len(siamese_dataset_trg.output)))
    fLog.write('Number of Testing')
    fLog.write("\n")
    fLog.write(str(len(siamese_dataset_tst.output)))
    #fLog.write(str(epoch))
    fLog.write("\n")
    fLog.write('Number of Training labels')
    fLog.write("\n")
    fLog.write(str(len(siamese_dataset_trg.output)))
    fLog.write('Number of Testing labels')
    fLog.write("\n")
    #print('labels in trg',siamese_dataset_trg.labels)
    #print('labels in trg loader',trg_loader)


    tst_labels = siamese_dataset_tst.labels
    #print('tst_labels',tst_labels)
    #print('len(tst_labels)',len(tst_labels))
    criterion = BCELoss()
    
    model = SiameseNetwork().cuda()

    print('In main ...6')
    #optimizer = optim.SGD(model.parameters(),lr=0.01, momentum=0)
    print('Training size',len(trg_loader))
    print('Testing size', tst_loader)
    model.lin1.weight.requires_grad = False
    model.lin1.bias.requires_grad = False

    model.lin2.weight.requires_grad = False
    model.lin2.bias.requires_grad = False

    model.lin1c.weight.requires_grad = False
    model.lin1c.bias.requires_grad = False

    model.lin2c.weight.requires_grad = False
    model.lin2c.bias.requires_grad = False
    
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            #m.weight.data.normal_(0, math.sqrt(2./n))
            torch.nn.init.orthogonal_(m.weight.data)
            m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            #torch.nn.init.orthogonal(m.weight.data)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            stdv = 1./math.sqrt(m.weight.size(1))
            #m.weight.data.uniform_(-stdv, stdv) 
            torch.nn.init.orthogonal_(m.weight.data)
            m.bias.data.zero_()
    
    #fLog = open('/data/work/pyUbi/logs/log.txt','w')
    fLog.write('Begining')
    fLog.close()

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01,momentum=0, weight_decay=0)
    trg_loss_hist =[]
    val_loss_hist=[]
    positions = siamese_dataset_tst.position

    for epoch in range(1, 100):
        #torch.cuda.empty_cache()
        fLog = open('/data/logs/log.txt','a')
        print('Epoch',epoch)
        fLog.write('Epoch = ')
        fLog.write(str(epoch))
        fLog.write("\n")
        model.train()
        fLog.close()
        trg_loss = train(criterion, optimizer, model, trg_loader, epoch, batchsize, device)
        trg_loss_hist.append(trg_loss)
        
        #with torch.no_grad():
            #val_loss = calculatetest(model, tst_loader, device, siamese_dataset_tst)
            #print('results',results)
            #finalTensor= torch.cat(results)
            #finalTensor = results
            #import numpy as np
            #finalTensor = np.concatenate(finalTensor).ravel()
            #finalTensor = finalTensor.detach().numpy()
            #finalTensor = np.vstack(finalTensor)
            #print('final Tensor',finalTensor)
            #preds =[]
            
            #for x in positions:
                #preds.append(max(finalTensor[x[0]:x[1]])[0])

            #tst_labels = np.array(tst_labels,dtype='float64')
            #print('tst_labels',tst_labels)
            #print('preds',preds)
            #fpr, tpr, thresholds = metrics.roc_curve(tst_labels, preds)
            #roc_auc = metrics.auc(fpr, tpr)
            #print ('ROC-AUC Testing in epochs ',roc_auc)
            
            #val_loss_hist.append(val_loss)
        
    
   
    # fig = plt.figure()
    # epoch_hist = np.arange(1,100,step=1)
    # plt.plot(epoch_hist,trg_loss_hist, label = 'Training loss')
    # plt.plot(epoch_hist,val_loss_hist, label = 'Validation loss')
    # plt.legend(loc="best")
    # plt.ylabel('BCE', fontsize = 14)
    # plt.xlabel('Epoch number', fontsize = 14)
    # x=1
    # if (x==1):
    #     #fig.savefig('/home/saby2k13/projects/def-ilie/saby2k13/pyDPPI/scripts/classic_learning_curves.png')
    #     #fig.savefig('/home/saby2k13/projects/def-ilie/saby2k13/pyDPPI/organised/deepPSSM/optimise.png')
    #     fig.savefig( '/home/ubuntu/Work/optimise.png')
    # else:
    #     fig.savefig('/Users/sabby/Documents/classic_learning_curves.png')

    print(torch.cuda.memory_allocated() / 1024**2) 
    print(torch.cuda.memory_reserved()/ 1024**2)
    fLog = open('/data/logs/log.txt','a')

    results = test(model.cpu().eval(),tst_loader)
    print('Length results',len(results))
    positions = siamese_dataset_tst.position
    #print('positions[1]',positions[1],'positions[5]',positions[5])
    finalTensor= torch.cat(results)
    finalTensor = finalTensor.cpu()
    finalTensor = finalTensor.detach().numpy()

    preds =[]
    
    for x in positions:
        preds.append(max(finalTensor[x[0]:x[1]])[0])

    #print('preds',preds)
    #print('preds',len(preds))
    #print('Here')
    #print('tst_labels', tst_labels)
    #print('len(tst_labels)', len(tst_labels))
    tst_labels = np.array(tst_labels,dtype='float64')
    #print('tst_labels',tst_labels)
    print('preds',preds)
    fpr, tpr, thresholds = metrics.roc_curve(tst_labels, preds)
    roc_auc = metrics.auc(fpr, tpr)
    print ('ROC-AUC Testing',roc_auc)
    fLog.write('ROC = ')
    fLog.write(str(roc_auc))
    fLog.write("\n")
    precision, recall, thresholds = precision_recall_curve(tst_labels, preds)
    aupr = metrics.auc(recall, precision)
    print('AUPR Testing', aupr)
    fLog.write('AUPR = ')
    fLog.write(str(aupr))
    fLog.write("\n")    
    binResults =  [0 if a_ < 0.5 else 1 for a_ in preds]
    print('binResults',binResults)
    tr_acc_tst = accuracy_score(tst_labels,binResults )
    
    #fig = plt.figure()
    #epoch_hist = np.arange(200)
    #plt.plot(epoch_hist,trg_loss_hist, label = 'Training loss')
    #plt.plot(epoch_hist,val_loss_hist, label = 'Validation loss')
    #plt.legend(loc="best")
    #plt.ylabel('BCE', fontsize = 14)
    #plt.xlabel('Epoch number', fontsize = 14)
    #fig.savefig('/data/classic_learning_curves.png')
    # x=1
    # if (x==1):
    #     #fig.savefig('/home/saby2k13/projects/def-ilie/saby2k13/pyDPPI/scripts/classic_learning_curves.png')
    #     #fig.savefig('/home/saby2k13/projects/def-ilie/saby2k13/pyDPPI/organised/deepPSSM/optimise.png')
    #     fig.savefig( '/home/ubuntu/Work/optimise.png')
    # else:
    #     fig.savefig('/Users/sabby/Documents/classic_learning_curves.png')


    print('Test Accuracy',tr_acc_tst)
    fLog.write('Accuracy = ')
    fLog.write(str(tr_acc_tst))
    fLog.write('Loss = ')
    fLog.write('Epoch = ')
    fLog.close()

def test(net,tst_loader) :
    accuracy = 0
    counter = 0
    correct = 0
    results = []
    for i, data in enumerate(tst_loader):
        #crop0, crop1, label = data[0].to(device),data[1].to(device),data[2].to(device)
        crop0, crop1, label = data[0], data[1], data[2]
        #crop0 = np.nan_to_num(crop0.cpu(), copy=True, nan=0.0, posinf=None, neginf=None)
        #crop1 = np.nan_to_num(crop1.cpu(), copy=True, nan=0.0, posinf=None, neginf=None)
        #crop0 = torch.from_numpy(crop0).cuda()
        #crop1 = torch.from_numpy(crop1).cuda()
        x=0
        if (x==0):
            output = net(crop0, crop1)
        else:
            output = net(crop0, crop1).cuda()

        #crop0.detach().cpu()
        #crop1.detach().cpu()

        #del crop0
        #del crop1
        #gc.collect()

        #torch.cuda.empty_cache()
        results.append(output)

    return results


def calculatetest(model, tst_loader, device, siamese_dataset_tst):
    # for idx, (data, targets) in enumerate(tst_loader):
    #tst_labels = siamese_dataset_tst.ind
    # results = test(model.eval(),tst_loader, device)
    net = model.eval()
    results = []
    test_loss = 0
    for i, data in enumerate(tst_loader):
        # print('Enumerate i ',i)
        crop0, crop1, label = data[0].to(device), data[1].to(device), data[2].to(device)
        #crop0 = np.nan_to_num(crop0.cpu(), copy=True, nan=0.0, posinf=None, neginf=None)
        #crop1 = np.nan_to_num(crop1.cpu(), copy=True, nan=0.0, posinf=None, neginf=None)
        #crop0 = torch.from_numpy(crop0).cuda()
        #crop1 = torch.from_numpy(crop1).cuda()

        x=1
        if (x==0):
            output = net(crop0, crop1)
        else:
            output = net(crop0, crop1).cuda()
         
        crop0.detach().cpu()
        crop1.detach().cpu()

        #del crop0
        #del crop1
        gc.collect()

        torch.cuda.empty_cache()

        labels = label.float()
        labels = labels.view(labels.size(0), -1)
        #print('labels.size', labels.size())
        #print('output.size',output.size())
        output = output.squeeze()
        #output = labels
        labels = labels.squeeze()
        bce = BCELoss()
        loss = bce(output, labels)
        test_loss = test_loss + loss.item()
        results.append(output.cpu().numpy())
    test_loss = test_loss / len(tst_loader)
    print('test_loss', test_loss)
    #print('results in',results)
    return test_loss


def train(criterion,optimizer,net,train_loader,epoch,batchsize,device):
    counter = []
    loss_history = []
    iteration_number = 0
    trg_loss=0
    print('In train ...')        
    for i,data in enumerate(train_loader):
            #print('In train ...',i)
            #print(data)
            crop0, crop1, labels = data[0].to(device), data[1].to(device),data[2].to(device)
            #crop0 = np.nan_to_num(crop0.cpu(), copy=True, nan=0.0, posinf=None, neginf=None)
            #crop1 = np.nan_to_num(crop1.cpu(), copy=True, nan=0.0, posinf=None, neginf=None)
            #crop0 = torch.from_numpy(crop0).cuda()
            #crop1 = torch.from_numpy(crop1).cuda()

            #crop0 = crop0.cuda()
            #crop1 = crop1.cuda()
            #print('crop0',crop0.size())
            #print('crop1',crop1.size())
            #print('labels',labels)

            optimizer.zero_grad()
            x=0
            if (x==0):
                output = net(crop0, crop1)
            else:
                output = net(crop0, crop1).cuda()
            
            #crop0.detach().cpu()
            #crop1.detach().cpu()
            
            #del crop0
            #del crop1
            #torch.cuda.empty_cache()
            #gc.collect()
            #print('crop0',crop0)
            #print('crop1',crop1)

            #print('output',output)
            criterion = BCELoss()
            labels = labels.float()
            #print('labels',labels)

            weights = [10 if a_ == 1 else 1 for a_ in labels]
            weights = torch.FloatTensor(weights).to(device)
            #print('weights ...',weights)
            #print('labels ...',labels)
            #print('output ...',output)
            bce = BCELoss(weight=weights)
            #labels =labels.unsqueeze(1)

            #output = output.squeeze()
            #print('output',output.size())
            #print('weights',weights.size())

            #output = torch.squeeze(output)
            #labels =labels.unsqueeze(1)
            #output = output.squeeze()

            labels = labels.view(labels.size(0), -1)
            #print('labels.size', labels.size())
            #print('output.size',output.size())
            output = output.squeeze()
            #output = labels
            labels = labels.squeeze()
            #output = labels
            #print('labels ',labels)
            #print('output',output)
            #print('labels.size', labels.size())
            #print('output.size',output.size())
            loss = bce(output, labels)
            trg_loss = trg_loss + float(loss.item())
            loss.backward()
            optimizer.step()                            
            # Now we can do an optimizer step
            if i % 50 == 0:
                print("Epoch number {}\n Current loss {}\n".format(epoch, loss.item()))
                iteration_number += 10
                counter.append(iteration_number)
                loss_history.append(loss.item())

    trg_loss = trg_loss / len(train_loader)
    print('train_loss', trg_loss)
    
    return trg_loss


def getSizeofFile(fileName, folderLoc, number):
    #print('number keys',number.keys())
    numRecords = 0
    fileCalc = folderLoc + fileName
    print('fileCalc',fileCalc)
    with open(fileCalc) as csv_file:
        print('num recors')
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            key1 = row[0]
            key2 = row[1]
            #print('key1',key1)
            #key1 = bytes(row[0],'utf-8')
            #key2 = bytes(row[1],'utf-8')
            if ((key1 in number) and (key2 in number)):  
                num1 = number[key1]
                num2 = number[key2]
                numRecords += num1 * num2
    return numRecords


def populateData(input_x,input_y,fileName,folderLoc,crop,number):
    #print('crop keys',crop.keys())
    index=0
    count=0
    output=[]
    position=[]
    labels=[]
    print(folderLoc,fileName)
    folderLoc = '/data/data/csv/profppi/'
    fileInput =  folderLoc + fileName
    with open(fileInput) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            key1 = bytes(row[0] , 'utf-8')
            key1 = row[0]
            #num1 = number[key1]
            key2 = bytes(row[1] , 'utf-8')
            key2 = row[1]
            #num2 = number[key2]
            #print('crop keys',crop.keys())
            if ((key1 in number) and (key2 in number)):
                num1 = number[key1]
                num2 = number[key2]
                #print('labels',row[2])
                labels.append(row[2])
                tempList = []
                tempList.append(count)
                tempList.append(count + num1 * num2)
                tempList.append(num1 * num2)
                count = count + num1 * num2
                position.append(tempList)
                for i in range(0, num1):
                    for j in range(0, num2):
                        #val1 = bytes(row[0] + '-sub' + str(i),'utf-8')
                        #val2 = bytes(row[1] + '-sub' + str(j),'utf-8')
                        val1 = row[0]+'-sub'+str(i)
                        val2 = row[1]+'-sub'+str(j)
                        mat1 = crop[val1]
                        mat2 = crop[val2]
                        print('Values found')
                        #mat1 = crop[row[0] + '-sub' + str(i)]
                        #mat2 = crop[row[1] + '-sub' + str(j)]
                        mat1 = mat1.reshape(1, 20, 1, 512)
                        mat2 = mat2.reshape(1, 20, 1, 512)
                        mat1 = np.array(mat1)
                        mat2 = np.array(mat2)
                        #print('mat1')
                        #print(np.sum(mat1))
                        ran = random.randint(1, 10)
                        #print('ran',ran)
                        #print('input_x[index]',input_x[index])
                        all_zeros = not np.any(input_x[index])
                        #print('all_zeros',all_zeros)
                        all_zeros1 = not np.any(input_y[index])
                        #print('all_zeros1',all_zeros)
                        if ( not(all_zeros) or not(all_zeros1)):
                            print('Sane ..............................')
                        if (ran > 5) :
                            input_x[index] = mat1
                            input_y[index] = mat2
                        else:
                            input_x[index] = mat2
                            input_y[index] = mat1
                        index +=1
                        if (row[2]=='0'):
                            output.append(-1)
                        else:
                            output.append(1)

    #returnList =[]
    #print('input_x',input_x)
    #returnList.append(input_x)
    #returnList.append(input_y)
    #returnList.append(position)
    #returnList.append(output)
    #returnList.append(labels)
    return [input_x,input_y,position,output,labels]

if __name__ == '__main__':
    main()
