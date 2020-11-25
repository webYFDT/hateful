import pandas as pd
import numpy as np
import os
roc_dic={}

#search CSV
def walkFile(p,file):
    file=os.path.join(p,file)
    csv_list=[]
    for root, dirs, files in os.walk(file):
        for f in files:
            if 'csv' in f:
                csv_path=os.path.join(root, f)
                if pd.read_csv(csv_path).shape[0]==2000:
                    csv_list.append(csv_path)
            #print(os.path.join(root, f))

        for d in dirs:
            walkFile(root, d)
    return csv_list
#The absolute path of the project
path=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#file="6.2.1_41版本vilNO7291_50x1版本visNO7190_cat_lessFC_YesBalance_shuffleData_918版本_没有预训练模型"
#walkFile(path,file)

p6_=0.7
p6=walkFile(path,'6vil_vis_cat_Pre_lessFC_YesBalance_shuffleData')
for i in p6:
    roc_dic[i]=p6_

p6x2x1_=0.7
p6x2x1=walkFile(path,'6.2.1_41版本vilNO7291_50x1版本visNO7190_cat_lessFC_YesBalance_shuffleData_918版本_没有预训练模型')
for i in p6x2x1:
    roc_dic[i]=p6x2x1_

p6x2x2p1_=0.7502
p6x2x2p1=walkFile(path,"6.2.2_41版本vil7291_官网版本vis72_cat_lessFC_NoBalance_train上shuffleData_val上不shuffleData_918版本")
for i in p6x2x2p1:
    roc_dic[i]=p6x2x2p1_


p7_=0.7
p7=walkFile(path,'7vil_vis_cat_Pre_lessFC_YesBalance_shuffleData_train+dev')
for i in p7:
    roc_dic[i]=p7_


p8_=0.7
p8=walkFile(path,'8vis_vis_cat_Pre_lessFC_YesBalance_shuffleData')
for i in p8:
    roc_dic[i]=p8_


p21_=0.7
p21=walkFile(path,'21vil_vis_cat_Pre_lessFC_YesBalance_shuffleData7307_ImgGobalFeatureAddObj')
for i in p21:
    roc_dic[i]=p21_

ps27_=0.7
ps27=walkFile(path,'27vil_noPre_trainmorePlusNoun711_V4x图像全局特征_加_文本bert特征2_正确版本')
for i in ps27:
    roc_dic[i]=ps27_

ps28_=0.7
ps28=walkFile(path,'28vil_x预训练7191_trainmore_27V4x7230_19图片add到obj上7245_合成模型叫V5模型')
for i in ps28:
    roc_dic[i]=ps28_

ps35_=0.7
ps35=walkFile(path,'35_ps19模型和p6模型融合_双模型')
for i in ps35:
    roc_dic[i]=ps35_


ps41_=0.72
ps41=walkFile(path,"41_vil_Sembed/0.raw_vil_sembed")
for i in ps41:
    roc_dic[i]=ps41_


ps41p1_=0.75
ps41p1=walkFile(path,"41_vil_Sembed/1.7291模型基础上在newFeature即train_pluse_dev_seen上fintune_dev_unseen上选模型")
for i in ps41p1:
    roc_dic[i]=ps41p1_

ps41p2_=0.75
ps41p2=walkFile(path,"41_vil_Sembed/2.与1相比_不加载预训练模型_从头开始在newFeature即train_plus_dev_unseen训练模型")
for i in ps41p2:
    roc_dic[i]=ps41p2_


ps41p3_=0.75
ps41p3=walkFile(path,"41_vil_Sembed/3.在2的基础上加单膜态loss")
for i in ps41p3:
    roc_dic[i]=ps41p3_

ps41x4s21000_=0.72
ps41x4s21000=walkFile(path,"41.4_vil_Sembed_FineTune_Tain+val")
for i in ps41x4s21000:
    roc_dic[i]=ps41x4s21000_

ps41x4s22000_=0.72
ps41x4s22000=walkFile(path,"41.4_vil_Sembed_FineTune_Tain+val")
for i in ps41x4s22000:
    roc_dic[i]=ps41x4s22000_

ps41x4s23000_=0.72
ps41x4s23000=walkFile(path,"41.4_vil_Sembed_FineTune_Tain+val")
for i in ps41x4s23000:
    roc_dic[i]=ps41x4s23000_

ps41x4s24000_=0.72
ps41x4s24000=walkFile(path,"41.4_vil_Sembed_FineTune_Tain+val")
for i in ps41x4s24000:
    roc_dic[i]=ps41x4s24000_

ps41p4_=0.75
ps41p4=walkFile(path,"41_vil_Sembed/4.7330模型即_41.4x22000对应模型_基础上在newFeature即train_pluse_dev_seen上finetune_dev_unseen上选模型")
for i in ps41p4:
    roc_dic[i]=ps41p4_

ps41p5_=0.75
ps41p5=walkFile(path,"41_vil_Sembed/5.7291模型基础上在newFeature即train_pluse_dev_same上fintune_dev_different上选模型")
for i in ps41p5:
    roc_dic[i]=ps41p5_

ps41p6_=0.75
ps41p6=walkFile(path,"41_vil_Sembed/6.与1相同处理_但是使用随机擦除来随机抹掉一些图像obj")
for i in ps41p6:
    roc_dic[i]=ps41p6_

ps41p10_=0.75
ps41p10=walkFile(path,"41_vil_Sembed/10.与1相同_但为1的数据翻倍")
for i in ps41p10:
    roc_dic[i]=ps41p10_

ps41p13_=0.75
ps41p13=walkFile(path,"41_vil_Sembed/13.与1相同_但是abcd数据集上分别训练/dcba_fusion/")
for i in ps41p13:
    roc_dic[i]=ps41p13_


ps50x1p1_=0.72
ps50x1p1=walkFile(path,"50.1_vis_Sembed_Pre/1.vis_Sembed_加载官网模型_加载sembed和globalimg继续fintune_918")
for i in ps50x1p1:
    roc_dic[i]=ps50x1p1_

################################

threshold=0
roc_dic_1={}
for k,v in roc_dic.items():
    if v>=threshold:
        roc_dic_1[k]=v

csv_path=roc_dic_1.keys()



csv_num=len(csv_path)


data_all={}
for path_sub in csv_path:
    
    data_sub=pd.read_csv(path_sub)
    
    data_all[path_sub]=data_sub
    
     
def save_csv(data_all,proba_num,result_path):
    
    id_num=np.array(data_all[i].id)
    
    label=np.array(proba_num>0.5,int)
    
    data={'id':id_num,'proba':proba_num,'label':label}
    
    data=pd.DataFrame(data)
    
    data.to_csv(result_path,index=None)
    

roc_sum=0
for roc in roc_dic_1.values():
    roc_sum+=roc
    
proba_num2=np.zeros(2000)

for k,v in  data_all.items():
    np_proba=np.array(v.proba)
    proba_num2+=np_proba*(roc_dic_1[k]/roc_sum)
    

save_csv(data_all,proba_num2,'lucky_lucky_lukcy_model_result.csv')  









