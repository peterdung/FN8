import h5py
import numpy as np
from tqdm import tqdm
f=h5py.File('../getmax/embedding.h5','r')
class_arr=f['class_name'][:]
class_arr=[k.decode() for k in class_arr]
emb_arr=f['embeddings'][:]
classcount = 0
total_count = len(class_arr)
accuracy_count = 0
THRED = 0.000000001
tmp = class_arr[0]
target_count = 0
check = 0
file_total = 0
class_total = 1
pbar = tqdm(total=87562761)
for i in class_arr:
    pbar.update(1)
    if i == tmp:
        
        if check ==1:
            diff = np.mean(np.square(emb_arr[target_count]-emb_arr[classcount]))
#            print(diff)
            if diff<=THRED:
                accuracy_count = accuracy_count + 1
#            print(tmp,i)
            file_total = file_total + 1
        check = 1
    elif i!=tmp: 
        target_count = classcount
#        print('no match',i,tmp)
        tmp = i
        check = 0
        class_total = class_total + 1
    classcount = classcount+1
#    print('file accuracy = ',accuracy_count)
#    print('target_count = ', target_count)
#    print('class_count = ', classcount)
emb_count = 0
total_count = 0
t_accuracy = 0
E_count = 0
em_diff_array = []
Tp = 0
Fp = 0
Tn = 0
Fn = 0
for emb in emb_arr:
    j_count = emb_count
    for j in emb_arr[j_count:]:
        pbar.update(1)
        #if j_count != emb_count:
        em_diff = np.mean(np.square(emb-j))
        em_diff_array.append(em_diff)
        if em_diff <= THRED:
            if class_arr[emb_count] == class_arr[j_count]:
#                print(class_arr[emb_count],'=',class_arr[j_count])
                t_accuracy = t_accuracy + 1
                Tp = Tp + 1
            else:
                E_count = E_count + 1
                Fp = Fp + 1
                
        else:
            if class_arr[emb_count] != class_arr[j_count]:
#                print(class_arr[emb_count],'!=',class_arr[j_count])
                t_accuracy = t_accuracy + 1
                Tn = Tn + 1
            else:
                Fn = Fn + 1
        j_count = j_count + 1
        total_count = total_count + 1
    emb_count = emb_count + 1
#    total_count = total_count + j_count
#    print('process',emb_count/j_count)
pbar.close()
f.close()
Precicion = Tp/(Tp+Fp)
Recall = Tp/(Tp + Fn)
F1_score = (2*Precicion*Recall)/(Precicion+Recall)
print('total count = ',total_count)
#print("Feature Distance Mean =", np.mean(em_diff_array))
print("Precicion = ",Precicion)
print("Recall = ",Recall)
print("F1-Score = ",F1_score)
print('FRR = ',E_count/total_count)
print('total folder accuracy = ',t_accuracy/total_count)
print(accuracy_count/file_total)
print(class_total)