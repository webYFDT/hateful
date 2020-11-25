# -*- coding: utf-8 -*-

import json

from tqdm import tqdm


import spacy
from spacy import displacy
#import neuralcoref
#nlp =  spacy.load('en_core_web_sm')
nlp=spacy.load('en_core_web_md')

def ner(sent):
    doc=nlp(sent)

    #print("名词短语")
    return [nc for nc in doc.noun_chunks]

##读取stopward
with open("../stopword/clear_stopward.txt") as o:
    stop_ward=[i.strip() for i in o.readlines()]

rm_sig='.'               #删除的符号
rm_space=['\"','\'','-',',']   #替换为空格

def  get_noun(sent):
        all_noun=''
        noun=ner(sent)
        ##先拆分短语，去除停用词后再组合短语
        for noun_one in noun:
            noun_one=noun_one.string
            noun_one_list=noun_one.split()
            noun_one_rm_stop_word=''
            for word_one in noun_one_list:
                if rm_sig in word_one:
                    word_one=word_one.replace(".","")
                for sig in rm_space:
                    if sig in word_one:
                      word_one=word_one.replace(sig," ")  
                word_one=word_one.split()
                for word_one_one in word_one:
                    if word_one_one not in stop_ward:
                        noun_one_rm_stop_word+=word_one_one+" "
            
            if noun_one_rm_stop_word!='':
                all_noun+=noun_one_rm_stop_word.strip()+' '
            
                
        #all_noun+='end'
        return all_noun
            
        
    #nouns=[i.string for i in noun]
def main(path,save_path):
    #path="/Users/liuguihua/Downloads/竞赛/fb_code/data/trainmore.jsonl"
    f=open(path)
    

    
    
        
    dataf=open(save_path,'w')
    
    for line in tqdm(f.readlines()):
        data=json.loads(line)
        text=data['text']
        one_noun=get_noun(text)
        noun_len=len(one_noun.split(" "))
        noun_len=noun_len-1
        data['noun']=one_noun
        data['noun_len']=noun_len
        
        data_noun=json.dumps(data)
        dataf.write(data_noun+'\n')
        
        
    f.close()    

#trainmore 
path="../../data/trainmore.jsonl"    
save_path='trainmore_plus_noun_Smbed40.jsonl'    
main(path,save_path)





