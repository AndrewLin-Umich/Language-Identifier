# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 16:18:47 2017

@author: Andrew Jun Lin
@unique name: ljumsi
"""
from __future__ import division
import sys
import numpy as np
from scipy.special import expit
from sklearn.preprocessing import LabelBinarizer

train_file, dev_file, test_file = sys.argv[1], sys.argv[2], sys.argv[3]

with open(train_file) as f:
	f_train = f.readlines()

with open(dev_file) as f:
	f_dev = f.readlines()

with open(test_file) as f:
	f_test = f.readlines()
    
with open('languageIdentification.data/test_solutions') as f:
	f_test_solutions = f.readlines()

train = [(i.split(' ')[0],' '.join(i.lower().split(' ')[1:-1])) for i in f_train]
dev = [(i.split(' ')[0],' '.join(i.lower().split(' ')[1:-1])) for i in f_dev]
test = [i[:-1].lower() for i in f_test]


train_char = []
for file in [train, dev]:
    for x in file:
        s = x[-1]
        for j in s:
            train_char.append(j)
for x in test:
    for s in x:
        train_char.append(s)
alphabet = set(train_char)

ec = LabelBinarizer()
encoded = ec.fit_transform(list(alphabet))
encode_dict = {a:b for (a,b) in zip(alphabet, encoded)}

train_5 = []
for i in train:
    for j in range(len(i[1])-4):
        train_5.append((i[0],i[1][j:j+5]))

train_5c = []
for i in train_5:
    s = encode_dict[i[1][0]]
    for j in i[1][1:]:
        s=np.append(s,encode_dict[j])
    train_5c.append((i[0],s))

np.random.shuffle(train_5c)
np.random.shuffle(train_5c)

# train_x = np.array([i[1] for i in train_5c])
encoder_y = LabelBinarizer()
train_y = encoder_y.fit_transform(np.array([i[0] for i in train_5c]))

feature_size = len(alphabet)*5

def sigmoid(x):
	return expit(x)

def softmax(x):
	s = np.exp(x - np.max(x))
	return s / s.sum()


dev = [(i.split(' ')[0],' '.join(i.lower().split(' ')[1:-1])) for i in f_dev]

dev_5 = []
for i in dev:
    for j in range(len(i[1])-4):
        dev_5.append((i[0],i[1][j:j+5]))

dev_5c = []
for i in dev_5:
    s = encode_dict[i[1][0]]
    for j in i[1][1:]:
        s=np.append(s,encode_dict[j])
    dev_5c.append(s)

dev_x = np.array(dev_5c)
dev_y = encoder_y.fit_transform(np.array([i[0] for i in dev]))
#%%


def encode_str(x):
    r = []
    for i in x:
        if i in encode_dict:
            r.append(encode_dict[i])
        else:
            r.append(np.zeros(81))
    r = np.array(r).reshape(1,-1)
    return r

def pred(x,w1,b1,w2,b2):
#    w1,b1,w2,b2 = model
    x1 = [x[i:i+5] for i in range(len(x)-4)]
    x2 = [encode_str(i) for i in x1]
    d = {0:0, 1:0, 2:0}
    for i in x2:
        
        layer_hp_p = np.matmul(i,w1) + b1
        layer_h_p = sigmoid(layer_hp_p)
        layer_yp_p = np.matmul(layer_h_p,w2) + b2
        output_y_p = softmax(layer_yp_p)
        r = np.argmax(output_y_p)
        d[r] += 1
    return np.argmax(np.array([d[0],d[1],d[2]]))

#%%
#def train_network(x, y, eta, num_units_in_layer_d, epochs, feature_size_c, num_classes):
# x ,y = np.array([i[1] for i in train_5c]), train_y
sample_size = len(train_5)
feature_size_c = feature_size
num_units_in_layer_d = 100
num_classes =3
max_epoch = 3
eta=0.1
	# np.random.seed(36)
w1 = np.random.rand(feature_size_c, num_units_in_layer_d)
b1 = np.random.rand(1, num_units_in_layer_d)
w2 = np.random.rand(num_units_in_layer_d,num_classes)
b2 = np.random.rand(1,num_classes)


train_accuracy = []
dev_accuracy = []

for _ in range(max_epoch):
    losses = []
    l = {'ENGLISH':0, 'FRENCH':1, 'ITALIAN':2}

    c1 = c2 = 0
    for i in range(sample_size):
        x1 = np.array([i[1] for i in train_5c])[i].reshape(1,-1)
        y1 = train_y[i].reshape(1,-1)
        
        layer_hp = np.matmul(x1,w1) + b1
        layer_h = sigmoid(layer_hp)
        layer_yp = np.matmul(layer_h,w2) + b2
        output_y = softmax(layer_yp)
        loss = sum([i**2 for i in (y1[0] - output_y[0])])/2
        losses.append(loss)
        grad_L_y = output_y - y1
        # grad_L_y = y1 - output_y
           
        grad_L_yp = np.zeros((1,3))
        for n in range(3):
        	r = 0
        	for m in range(3):
        		if n == m:
        			r += grad_L_y[0,m]*output_y[0,m]*(1-output_y[0,m])
        		else:
        			r += grad_L_y[0,m]*output_y[0,m]*(-output_y[0,n])
        	grad_L_yp[0,n] = r
        
        grad_L_w2 = np.matmul(grad_L_y.T,layer_h).T
        grad_L_b2 = grad_L_y
        grad_L_h = np.matmul(grad_L_yp, w2.T)
        grad_L_hp = grad_L_h*layer_h*(1-layer_h)
        grad_L_w1 = np.matmul(grad_L_hp.T, x1).T
        grad_L_b1 = grad_L_hp
        
        w1 -= eta * grad_L_w1
        b1 -= eta * grad_L_b1
        w2 -= eta * grad_L_w2
        b2 -= eta * grad_L_b2
        
    print('epoch', _+1, 'finished. loss =', sum(losses))
    for q1 in train:
        if pred(q1[1], w1,b1,w2,b2) == l[q1[0]]:
            c1 += 1
    print('accuracy on training set =', c1/len(train))
    for q2 in dev:
        if pred(q2[1], w1,b1,w2,b2) == l[q2[0]]:
            c2 += 1
    print('accuracy on dev set =', c2/len(dev))  
    

test = [i[:-1].lower() for i in f_test]
dl = {0:'English', 1:'French', 2:'Italian'}
res = []
with open('languageIdentificationPart1.output','w') as w:
    for i in test:
        x = str(i) +' '+ dl[pred(i, w1,b1,w2,b2)] + '\n'
        w.write(x)
        res.append(dl[pred(i, w1,b1,w2,b2)])
    
#%%

    
ans = [i[2:-1] for i in f_test_solutions]
#%%
c3 = 0
for i in range(300):
    print(ans[i][2:5], res[i][:3], ans[i][2]==res[i][0])
    if ans[i][2] == res[i][0]:
        c3+=1
print('test set accuracy= '. c3/300)
