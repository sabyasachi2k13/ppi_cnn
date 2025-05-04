

if __name__ == '__main__':
    path="/Users/saby/Documents/ppi-generic-data/dataset_challenges_2.0/redundancyTest/nonRed/C3/humanCV/split_0"
    fTestOutput = open(path+"/0.test.csv","w")
    fTestNeg = open(path+"/0.test.neg","r")
    fTestPos = open(path+"/0.test.pos","r")

    fTrainOutput = open(path+"/0.train.csv","w")
    fTrainNeg = open(path + "/0.train.neg", "r")
    fTrainPos = open(path + "/0.train.pos", "r")

    for l in fTestNeg:
        fTestOutput.write( l.strip().split()[0]+","+ l.strip().split()[1]+","+"0"+"\n")

    fTestNeg.close()

    for l in fTestPos:
        fTestOutput.write(l.strip().split()[0] + "," + l.strip().split()[1] +","+"1" + "\n")

    fTestPos.close()

    fTestOutput.close()

    for l in fTrainNeg:
        fTrainOutput.write( l.strip().split()[0]+","+ l.strip().split()[1]+","+"0"+"\n")

    fTestNeg.close()

    for l in fTrainPos:
        fTrainOutput.write(l.strip().split()[0] + "," + l.strip().split()[1] +","+"1" + "\n")

    fTestPos.close()
    fTrainOutput.close()

