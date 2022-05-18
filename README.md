## Environment and strategy description


	This experiment is all conducted under the mmf framework. The VisualBert and ViLBert models have been improved in many ways to obtain models with different designation. Finally, the prediction results of all models are fused at the decision level to obtain the final result.

## Model

![Image text](https://github.com/webYFDT/hateful/blob/main/model.png)

## Effective models:

	1. Resplit the train data set and dev data set
		a. Motivation:
		We can ovserver that the samples with label 0 and label 1 in the train dataset are not balanced. While through the inference score, we find that the distribution of the official dev data set and the test data set is relatively consistent. So we can resplit the two set. 
		b. Methods:
		When training, we use two versions of the training set. The first one is to put a part of the samples with label 1 in dev_unseen.jsonl into the train.jsonl data set and repeat the samples with label 1 to balance the samples to get trainmore.jsonl training set. The second is to add dev_seen.jsonl to train.jsonl to form the training set train_plus_dev_seen.jsonl, and use dev_unseen.jsonl as the validation set.

	2. Add global features to the picture (main improvement module):
		a. Motivation:
		In the process of using ViLBert and VisualBert, we found that the input on the image side of the model is only the local features of the object extracted with faster-rcnn, while lacks the global features of the image which is usually helpful because the scene/background information of the picture is also very important in this task. For example, the background of picutre -- "look how many people love you" is an empty beach, so it is a hateful sample.
		b. Methods:
		The restnet152 model is used to extract the global feature of each sample, and then the feature is added to each object (the feature extracted by the faster-rcnn) at a time with point addition. 

	3. Adding Sembed word vector into text (main improvement module)
		a. Motivation:
		Through careful observation of this task, we can find that there are some key words (called S word) in the text side, which have a great impact on the final classification results of samples after interacting with images, and many of these words are "noun phrase". For example, the words "Christian" and "bad drivers" in "ever notice that Christians are bad drivers" will have a great impact on the classification results of samples.
		b. Methods:
		Spacy's "noun_chunks" function is utilized to extract the noun phrase from the text of each sample, and then removes some stop words through the stopword table to get the keywords of the text. After that, we construct a 3x768 look-up embedding (called Sembedding, which is similar to segment embedding). The token of "keyword"(S word) is 1, "other words" is 2, and "padding word" is 0. 

	4. Multi-fold model
		a. Motivation
		Models of the same version are trained with different sub-data, and then the results of these models can be fused to improve model performance.
		b. Methods
		Fusion train.jsonl and dev.jsonl data sets, and then re-divide four points, each model training uses 3/4 of them, testing uses the remaining 1/4.

	5.Feature level fusion:
		a. Motivation
		The two models of VisualBert and ViLBert have their own advantages, and the two models can be fused together for training. And we propose a data augmentation strategy in which the input of the two models is different samples with the same label, which enhances the generalization ability of the model.
		b. Methods
		The last layer of the VisualBert and ViLBert models are fused and then input to a classification layer for category prediction.

	6. Weighted average of model decision-making level (main improvement module)
		a. Motivation:
		The prediction ability of different models for a task is different. A stronger model can be obtained by fusing the results of several weak models.
		b. Methods:
		For all the effective modles, the weighted average of decision-making level shows that the result can be improved by 2%.

## Data configuration:
The data configuration includes three parts: 1. Re-division of training set and verification set. 2. Adding "noun" keyword(S word) to all jsonl data sets. 3. Extracting global features of images.

	1. Re-division of training set and validation set (./data/raw_data folder contains the officially released raw data):
		-./data/trainmore.jsonl-->/root/.cache/torch/mmf/data/datasets/hateful_memes/defaults/annotations/
			trainmore.jsonl is the re-divided training set. A part of dev.jsonl is added to the training set and put it into the "hateful_memes/defaults/annotations/" folder of the mmf framework
		-./data/train_plus_dev_seen.jsonl-->/root/.cache/torch/mmf/data/datasets/hateful_memes/defaults/annotations/
			Add dev_seen.jsonl to the training set and put it in the "hateful_memes/defaults/annotations/" folder of the mmf framework.
		- ./data/data_partition/*.jsonl-->/root/.cache/torch/mmf/data/datasets/hateful_memes/defaults/annotations/
			Fuse the train.jsonl and dev.jsonl data sets, and then re-divide the same number of quarters for the multi-fold model

	2. Add "noun" keywords to all jsonl data sets (./Sensitive_word_recognition/recognized_by_spacy/, The folder contains "keyword" processing programs and processing out Result file):
		-./Sensitive_word_recognition/recognized_by_spacy/get_SembedSent_nostopword.py
			It is used to add "keyword S information" to the text data set.
		-./Sensitive_word_recognition/recognized_by_spacy/*.jsonl-->/root/.cache/torch/mmf/data/datasets/hateful_memes/defaults/annotations/
			Put all "*.jsonl" files into the "hateful_memes/defaults/annotations/" folder.

	3. Extraction of image global features ("./extract_image_feature_restne/"" folder contains the program and result files for extracting global features of images):
		-./extract_image_feature_restne/extract_img_feature_restnet.py
			Use the restnet152 model to extract the global features of the picture and save it as img_feature.pkl file.
		-./extract_image_feature_restne/img_feature.pkl-->/root/img_feature.pkl
			Global feature files of all pictures, copy the file "img_feature.pkl" to the "/root" directory

## Training model configuration and commands:
Model is trained under the mmf framework, so different model training only needs to modify the training configuration file and the source code of the loaded data. Different models have different configurations. The configuration and running commands of each model are described in detail below:

	0. Document organization description
		The file organization format of all models is similar to the following structure:
			.
			├── dataset.py
			├── projecs.yaml
			├── run.txt
			├── save
			│   ├── config.yaml
			│   ├── hateful_memes_vil_v13_46405938
			│   │   ├── reports
			│   │   │   └── hateful_memes_run_test_2020-09-09T09:05:02.csv
			│   │   └── ?\210??\2172020-09-09\ ?\212?\215\2109.19.47.png
			│   ├── logs
			│   │   ├── train_2020_09_08T09_18_26.log
			│   │   ├── train_2020_09_09T08_59_47.log
			│   │   └── train_2020_09_09T09_02_46.log
			│   └── train.log
			├── vil_V13.py
			└── vil_V13_model.yaml
		a. dataset.py-->->mmf-master/mmf/datasets/builders/hateful_memes/dataset.py (The symbol "-->->" denotes replace)
			Use the "dataset.py" in the directory to replace "mmf-master/mmf/datasets/builders/hateful_memes/dataset.py" in the mmf framework source code.
		b. projecs.yaml
			Training configuration file of the model.
		c. run.txt
			Contains training, testing, and prediction commands for this version of the model.
		c. save
			This folder contains logs, models, prediction results and configuration files generated by the mmf framework.
		d. vil_V13.py-->mmf-master/mmf/models/ (the symbol "-->" denotes copy)
			Copy "vil_V13.py" in this directory to the "mmf-master/mmf/models/vil_V13.py" file in the mmf framework.
		e. vil_V13_model.yaml-->mmf-master/mmf/configs/models/vilbert/vil_V13_model.yaml
			The configuration file of this version of the model. "vil_V13_model.yaml" in this directory is copied to the "mmf-master/mmf/configs/models/vilbert/vil_V13_model.yaml" file in the mmf framework.

	1. ./6vil_vis_cat_Pre_lessFC_YesBalance_shuffleData/
		- Configuration
			vil_vis_cat_lessFC_shuffleData.py-->mmf-master/mmf/models/
			vvc_model_config_lessFC_shuffleData.yaml-->mmf-master/mmf/configs/models/vil_vis_cat_Pre_lessFC_noBalance_shuffleData/vvc_model_config_lessFC_shuffleData.yaml
		- Run
			Open the run.txt file, which contains the commands for model training, testing and prediction of this version

	2. ./6.2.1_41版本vilNO7291_50x1版本visNO7190_cat_lessFC_YesBalance_shuffleData_918版本_没有预训练模型
		- Configuration
			vil_vis_catV621.py-->mmf-master/mmf/models/
			vil_vis_catV621_model.yaml-->mmf-master/mmf/configs/models/vil_vis_cat_Pre_lessFC_noBalance_shuffleData/vil_vis_catV621_model.yaml
			dataset.py-->->mmf-master/mmf/datasets/builders/hateful_memes/dataset.py
		- Run
			Open the run.txt file, which contains the commands for model training, testing and prediction of this version

	3. ./6.2.2_41版本vil7291_官网版本vis72_cat_lessFC_NoBalance_train上shuffleData_val上不shuffleData_918版本
		- Configuration
			41_vil_Sembed/0.raw_vil_sembed/save/vil_v11_final.pth-->/root/.cache/torch/mmf/data/models/visual_bert.finetuned.hateful_memes.direct/vil_v11_final.pth
			dataset.py-->->mmf-master/mmf/datasets/builders/hateful_memes/dataset.py
			training_loop.py-->->mmf-master/mmf/trainers/core/training_loop.py
			evaluation_loop.py-->->mmf-master/mmf/trainers/core/evaluation_loop.py
			vil_vis_catV622.py-->mmf-master/mmf/models/
			vil_vis_catV622_model.yaml-->mmf-master/mmf/configs/models/vil_vis_cat_Pre_lessFC_noBalance_shuffleData/vil_vis_catV622_model.yaml
		- Run
			Open the run.txt file, which contains the commands for model training, testing and prediction of this version

	4. ./7vil_vis_cat_Pre_lessFC_YesBalance_shuffleData_train+dev
		- Run
			Open the run.txt file, which contains the commands for model training, testing and prediction of this version

	5. ./8vis_vis_cat_Pre_lessFC_YesBalance_shuffleData
		- Configuration
			vis_vis_cat_lessFC_shuffleData.py-->mmf-master/mmf/models/
			visvisc_model_config_lessFC_shuffleData.yaml-->mmf-master/mmf/configs/models/vis_vis_cat_Pre_lessFC_noBalance_shuffleData/visvisc_model_config_lessFC_shuffleData.yaml
		- Run
			Open the run.txt file, which contains the commands for model training, testing and prediction of this version

	6. ./21vil_vis_cat_Pre_lessFC_YesBalance_shuffleData7307_ImgGobalFeatureAddObj
		- Configuration
			dataset.py-->->mmf-master/mmf/datasets/builders/hateful_memes/dataset.py
		- Run
			Open the run.txt file, which contains the commands for model training, testing and prediction of this version

	7. ./27vil_noPre_trainmorePlusNoun711_V4x图像全局特征_加_文本bert特征2_正确版本
		- Configuration
			dataset.py-->->mmf-master/mmf/datasets/builders/hateful_memes/dataset.py
			vil_V4x.py-->mmf-master/mmf/models/
			vil_V4x_model.yaml-->mmf-master/mmf/configs/models/vilbert/vil_V4x_model.yaml
		- Run
			Open the run.txt file, which contains the commands for model training, testing and prediction of this version

	8. ./28vil_x预训练7191_trainmore_27V4x7230_19图片add到obj上7245_合成模型叫V5模型
		- Configuration
			dataset.py-->->mmf-master/mmf/datasets/builders/hateful_memes/dataset.py
			vil_V5.py-->mmf-master/mmf/models/
			vil_V5_model.yaml-->mmf-master/mmf/configs/models/vilbert/vil_V5_model.yaml
		- Run
			Open the run.txt file, which contains the commands for model training, testing and prediction of this version

	9. ./35_ps19模型和p6模型融合_双模型
		- Configuration
			dataset.py-->->mmf-master/mmf/datasets/builders/hateful_memes/dataset.py
			vil_vis_cat_lessFC_shuffleData.py-->mmf-master/mmf/models/
			vvc_model_config_lessFC_shuffleData.yaml-->mmf-master/mmf/configs/models/vil_vis_cat_Pre_lessFC_noBalance_shuffleData/vvc_model_config_lessFC_shuffleData.yaml
		- Run
			打开Open the run.txt file, which contains the commands for model training, testing and prediction of this version

	10. ./41_vil_Sembed/0.raw_vil_sembed
		- Configuration
			dataset.py-->->mmf-master/mmf/datasets/builders/hateful_memes/dataset.py
			vil_V11.py-->mmf-master/mmf/models/vil_V11.py
			vil_V11_model.yaml-->mmf-master/mmf/configs/models/vilbert/vil_V11_model.yaml
		- Run
			Open the run.txt file, which contains the commands for model training, testing and prediction of this version

	11. ./41_vil_Sembed/1.7291模型基础上在newFeature即train_pluse_dev_seen上fintune_dev_unseen上选模型
		- Configuration
			41_vil_Sembed/0.raw_vil_sembed/save/best.ckpt-->41_vil_Sembed/1.7291模型基础上在newFeature即train_pluse_dev_seen上fintune_dev_unseen上选模型/save/best.ckpt
		- Run
			Open the run.txt file, which contains the commands for model training, testing and prediction of this version

	12. ./41_vil_Sembed/2.与1相比_不加载预训练模型_从头开始在newFeature即train_plus_dev_unseen训练模型
		- Configuration
			dataset.py-->->mmf-master/mmf/datasets/builders/hateful_memes/dataset.py
			vil_V11.py-->mmf-master/mmf/models/vil_V11.py
			vil_V11_model.yaml-->mmf-master/mmf/configs/models/vilbert/vil_V11_model.yaml
		- Run
			Open the run.txt file, which contains the commands for model training, testing and prediction of this version

	13. ./41_vil_Sembed/3.在2的基础上加单膜态loss
		- Configuration
			dataset.py-->->mmf-master/mmf/datasets/builders/hateful_memes/dataset.py
			losses.py-->->mmf-master/mmf/modules/losses.py
			vil_V11x1.py-->mmf-master/mmf/models/vil_V11x1.py
			vil_V11x1_model.yaml-->mmf-master/mmf/configs/models/vilbert/vil_V11x1_model.yaml
		- Run
			Open the run.txt file, which contains the commands for model training, testing and prediction of this version

	14. ./41.4_vil_Sembed_FineTune_Tain+val
		- Configuration
		 	41_vil_Sembed/0.raw_vil_sembed/save/best.ckpt-->41.4_vil_Sembed_FineTune_Tain+val/save/best.ckpt
		 	dataset.py-->->mmf-master/mmf/datasets/builders/hateful_memes/dataset.py
		 	vil_V11.py-->mmf-master/mmf/models/vil_V11.py
			vil_V11_model.yaml-->mmf-master/mmf/configs/models/vilbert/vil_V11_model.yaml
		- Run
			Open the run.txt file, which contains the commands for model training, testing and prediction of this version

	15. ./41_vil_Sembed/4.7330模型即_41.4x22000对应模型_基础上在newFeature即train_pluse_dev_seen上finetune_dev_unseen上选模型
		- Configuration
			41.4_vil_Sembed_FineTune_Tain+val/save/models/model_22000.ckpt-->41_vil_Sembed/4.7330模型即_41.4x22000对应模型_基础上在newFeature即train_pluse_dev_seen上finetune_dev_unseen上选模型/save/best.ckpt
		- Run
			Open the run.txt file, which contains the commands for model training, testing and prediction of this version


	16. ./41_vil_Sembed/5.7291模型基础上在newFeature即train_pluse_dev_same上fintune_dev_different上选模型/
		- Configuration
			41_vil_Sembed/0.raw_vil_sembed/save/best.ckpt-->41_vil_Sembed/5.7291模型基础上在newFeature即train_pluse_dev_same上fintune_dev_different上选模型/save/
		- Run
			Open the run.txt file, which contains the commands for model training, testing and prediction of this version

	17. ./41_vil_Sembed/6.与1相同处理_但是使用随机擦除来随机抹掉一些图像obj
		- Configuration
			dataset.py-->->mmf-master/mmf/datasets/builders/hateful_memes/dataset.py
			41_vil_Sembed/0.raw_vil_sembed/save/best.ckpt-->41_vil_Sembed/6.与1相同处理_但是使用随机擦除来随机抹掉一些图像obj/save/best.ckpt
		- Run
			Open the run.txt file, which contains the commands for model training, testing and prediction of this version

	18. ./41_vil_Sembed/10.与1相同_但为1的数据翻倍
		- Configuration
			phase2_train_plus_dev_seen_doubleone_plus_noun_Smbed40.jsonl-->/root/.cache/torch/mmf/data/datasets/hateful_memes/defaults/annotations/
			41_vil_Sembed/0.raw_vil_sembed/save/best.ckpt-->41_vil_Sembed/10.与1相同_但为1的数据翻倍/save/best.ckpt
		- Run
			Open the run.txt file, which contains the commands for model training, testing and prediction of this version

	19. ./41_vil_Sembed/13.与1相同_但是abcd数据集上分别训练
		- Configuration
			41_vil_Sembed/0.raw_vil_sembed/save/best.ckpt-->41_vil_Sembed/13.与1相同_但是abcd数据集上分别训练/a/save
			41_vil_Sembed/0.raw_vil_sembed/save/best.ckpt-->41_vil_Sembed/13.与1相同_但是abcd数据集上分别训练/b/save
			41_vil_Sembed/0.raw_vil_sembed/save/best.ckpt-->41_vil_Sembed/13.与1相同_但是abcd数据集上分别训练/c/save
			41_vil_Sembed/0.raw_vil_sembed/save/best.ckpt-->41_vil_Sembed/13.与1相同_但是abcd数据集上分别训练/d/save
		- Run
			./a/ab.sh
			./c/cd.sh

	20. ./50.1_vis_Sembed_Pre/1.vis_Sembed_加载官网模型_加载sembed和globalimg继续fintune_918
		- Configuration
			dataset.py-->->mmf-master/mmf/datasets/builders/hateful_memes/dataset.py
			embeddings.py-->->mmf-master/mmf/modules/embeddings.py
			vis_V33.py-->mmf-master/mmf/configs/models/vilbert/
			vis_V33_model.yaml-->mmf-master/mmf/configs/models/visual_bert/vis_V33_model.yaml
		- Run
			Open the run.txt file, which contains the commands for model training, testing and prediction of this version
		

## Model Fusion(Weighted average of model decision-making level)
Configure the CSV file path of the prediction phase2 results of all the above models to the "./0decision-making_level/V.py" file

	- run
		python V.py

## Code and model download address
	
	link: https://pan.baidu.com/s/1IH7hMz7Yp1nXzVV_P5DXjQ  password: k740
	(链接: https://pan.baidu.com/s/1e2DqCN44r8Tq_up2bF3hOQ 提取码: hdwf)
