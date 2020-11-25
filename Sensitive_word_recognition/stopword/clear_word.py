# -*- coding: utf-8 -*-

data=[]
with open('all_stopward.txt','r') as f:
    for i in f.readlines():
        data.extend(i.split(","))

data_set=list(set(data))

with open('clear_stopward.txt','w') as f:
        for i in data_set:
            f.write(i+"\n")




