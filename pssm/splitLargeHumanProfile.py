import pandas as pd
def main():
  print('kkkk')

if __name__ == '__main__':
    main()
# protUbi =[]
# #/Users/sabby/Documents/TPD/Ubi/UbiBrowser/dataset/golden standard
# f = open("/Users/sabby/Documents/TPD/Ubi/UbiBrowser/dataset/golden standard/train_positive.txt", "r")
# for x in f:
#     txt = x.split()
#     protUbi.append(txt[0])
#     protUbi.append(txt[1])
#     #print(txt)
#
# f = open("/Users/sabby/Documents/TPD/Ubi/UbiBrowser/dataset/golden standard/test_positive.txt", "r")
# for x in f:
#     txt = x.split()
#     protUbi.append(txt[0])
#     protUbi.append(txt[1])
#     #print(txt)
#
# f = open("/Users/sabby/Documents/TPD/Ubi/UbiBrowser/dataset/golden standard/train_negative.txt", "r")
# for x in f:
#     txt = x.split()
#     protUbi.append(txt[0])
#     protUbi.append(txt[1])
#     #print(txt)
#
# f = open("/Users/sabby/Documents/TPD/Ubi/UbiBrowser/dataset/golden standard/test_negative.txt", "r")
# for x in f:
#     txt = x.split()
#     protUbi.append(txt[0])
#     protUbi.append(txt[1])
#     #print(txt)
#
# print('prot ubi::',len(protUbi),len(set(protUbi)))
#
# prot =[]
#
# f = open("/Users/sabby/Documents/TPD/Ubi/ppi/myTrain.csv", "r")
# for x in f:
#     txt = x.split(',')
#     prot.append(txt[0])
#     prot.append(txt[1])
#     #print(txt)
#
# f = open("/Users/sabby/Documents/TPD/Ubi/ppi/myTrain_valid.csv", "r")
# for x in f:
#     txt = x.split(',')
#     prot.append(txt[0])
#     prot.append(txt[1])
#     #print(txt)
#
# print('prot ::',len(prot),len(set(prot)))
#
# diff = []
# for x in protUbi:
#  if (x not in prot):
#      diff.append(x)
#      print('in list ',x)
#  else:
#      print('not in list ',x)
#
# print('prot diff ::',len(diff),len(set(diff)))
#
# #print(set(protUbi).issubset(set(prot)))
cnt  =0
f = open("/data/work/work-pssm/largefiles/human.profiles", "r")
for x in f:
     if x.startswith('>'):
        fname = str('/data/work/profiles/')+str(x[1:]).rstrip()
        f = open(fname, "a")
        #print('hello ', x)
        cnt += 1
     else:
        x = x.lstrip()
        if (not ((x.startswith('Last') or x.startswith('Standard') or x.startswith('PSI') or x.startswith('K') or x.startswith('A')))):
            arr = x.split(" ")
            arr = list(filter(None, arr))
            print('Count ',cnt)
            #print(arr)
            line=''
            arr=arr[2:]
            for x in arr:
                line = line+'\t'+x
            f.write(line)
            # arr = x.split(" ")
            #
            # if (len(arr)>1 and arr[0]!='A'):
            #     arr = list(filter(None, arr))
            #     arr = arr[2:22]
            #     print(x)
            #     print(arr)
            #     print(len(arr))
            #     line = ''
            #     for x in arr:
            #         line = line+' '+x

f.close()
#txt = x.split()
print(cnt)
