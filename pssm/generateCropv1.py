import numpy as np
import math
import os
import pickle

background = {0: 0.0799912015849807, 1: 0.0484482507611578, 2: 0.044293531582512, 3: 0.0578891399707563,
              4: 0.0171846021407367, 5: 0.0380578923048682, 6: 0.0638169929675978, 7: 0.0760659374742852,
              8: 0.0223465499452473, 9: 0.0550905793661343, 10: 0.0866897071203864, 11: 0.060458245507428,
              12: 0.0215379186368154, 13: 0.0396348024787477, 14: 0.0465746314476874, 15: 0.0630028230885602,
              16: 0.0580394726014824, 17: 0.0144991866213453, 18: 0.03635438623143, 19: 0.0700241481678408}


def normalize(value, k, c):
    #print(background[k],value,k,c)
    ret = math.log((c*value)/100 + (1 - c) * (background[k]))
    #print('ret',ret)
    return ret


def process_msa(file_name, dict_of_matrices, dict_num_feat):

    window = 512
    pssm_file = open(file_name, 'r')
    file_name = os.path.basename(pssm_file.name)
    print('file_name',file_name)
    lines = list(pssm_file.readlines())
    size = lines.__len__()
    # print('File size ', size)
    feature_array = np.zeros((size, 42), dtype=float)
    
    if (size <= 512):
        steps=1
    else:
        steps = math.ceil(float(size/256) - 1)
    
    print('steps',steps)
    if (steps > 10):
        print('file_name', file_name,'size',size,'window',window,'steps',steps)
    

    pssmMat = np.zeros((size,42))
    cnt=0
    print('lines',lines)
    for line in lines:
        print('line',line.strip().split())
        pssmMat[cnt,]=line.strip().split()
        #print('cnt',pssmMat[cnt,])
        cnt+=1
    
    print('pssmMat',pssmMat)
    #pssmMatVal=pssmMat[:,20:40]
    currentIndex=0
    c=0.8
    #print('pssmMat',pssmMatVal)
    
    #for i in range(0,20):
    #    print('pssmMatVal',pssmMatVal[:,i])
    
    #print('pssmMatVal',pssmMatVal)

    dict_num_feat[file_name]=steps
    #print('pssmMat',pssmMat)
    print('name',file_name ,' size',size,'steps',steps);
    #print('steps',steps)
    
    pssmMatVal = np.zeros((size, 20))
    pssmMatVal = pssmMat[:,20:40]
    print('pssmMatVal',pssmMatVal)
    
    for step in range(0, (steps)):
        #pssmMatVal = np.zeros((size, 20))
        #pssmMatVal = pssmMat[:,20:40]
        
        #for a in range(0,20):
        #    print('pssmMatVal','col',a ,'step',i,pssmMatVal[:,a])

        #val = np.zeros((512, 20))
        computed_feature = np.zeros((512, 20))
    
        if ((currentIndex + window) < size):
            #print('start',start,'start+window',start+window)
            #print(' pssm mat val if ',pssmMatVal[start:start+window,:])
            val=pssmMatVal[currentIndex:currentIndex+window,:]
            print('val.shape init ',val.shape)
            for aa in range(0, 20):
                pos=0
                for p in val[:, aa]:
                    computed_feature[pos,aa] = normalize(p, aa, c)
                    pos+=1
        else:
            print('pssmMatVal',pssmMatVal.shape)
            #print('start',start,'size',size+100,pssmMatVal[start:111780,:])
            print(' pssm mat val else ',pssmMatVal[currentIndex:size,:])   

            #val[0:size-start,:]=pssmMatVal[start:size,:]
            val = pssmMatVal[currentIndex:size,:]
            print('val.shape last',val.shape)
            filler = np.zeros((512-size+currentIndex, 20))
            print('filler.shape',filler.shape)
            #import numpy as np
            final = np.vstack((val, filler))

            #print('val.shape',val.shape)
            for aa in range(0, 20):
                pos=0
                for p in final[:, aa]:
                    computed_feature[pos,aa] = normalize(p, aa, c)
                    pos+=1
            
            print(' pssm mat val else ',pssmMatVal[currentIndex:size,:])
 
        #for i in range(0,20):
        #    print('Val',val[:,i])

        #print('val',val)
        dict_of_matrices[file_name+"-sub"+str(step)]=computed_feature
        currentIndex = currentIndex + 256
        
        #print('dict_of_matrices',dict_of_matrices)
        #print('dict_num_feat',dict_num_feat)

    return dict_of_matrices,dict_num_feat
    

if __name__ == '__main__':

    directory = "/data/data/test1"
    # directory = "/Users/sabyasachi/PycharmProjects/pythonTPD/files"
    dict_of_matrices={}
    dict_num_feat={}
    num_files=0
    for filename in os.listdir(directory):
        with open(os.path.join(directory, filename), 'r') as f:
            #print(f.__getattribute__('name'))
            dict_of_matrices, dict_num_feat = process_msa(f.__getattribute__('name'), dict_of_matrices, dict_num_feat )
            #process_msa(f.__getattribute__('name'))
        num_files+=1
        #print('number of files ',num_files)

    # print('dictOfMatrices keys', dict_of_matrices.keys())
    pickle.dump(dict_of_matrices, open("proteins_profile_crop_512.p", "wb"))
    pickle.dump(dict_num_feat, open("proteins_number_crop_512.p", "wb"))

    crop_dict = open("proteins_profile_crop_512.p",'rb')
    num_dict = open("proteins_number_crop_512.p",'rb')

    #crop_file = pickle.load(crop_dict,encoding='latin1')
    #num_crop = pickle.load(num_dict,encoding='latin1'

    crop_file = pickle.load(crop_dict)
    num_crop = pickle.load(num_dict)

    print(' crop_file ', crop_file)
    print(' num_crop ', num_crop)


