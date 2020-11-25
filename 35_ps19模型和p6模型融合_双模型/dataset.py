# Copyright (c) Facebook, Inc. and its affiliates.
import copy
import os
import math
import numpy as np
import omegaconf
import torch
from PIL import Image

from mmf.common.sample import Sample
from mmf.datasets.mmf_dataset import MMFDataset
from mmf.utils.general import get_mmf_root
from mmf.utils.visualize import visualize_images
import warnings
#import transforms
import torchvision.transforms as transforms
import random
from PIL import Image
import torch
import  warnings
import pickle
class RandomErasingT(object):
    def __init__(self, EPSILON = 0.5, sl = 0.02, sh = 0.4, r1 = 0.3, mean=[0.4914, 0.4822, 0.4465]):
        self.EPSILON = EPSILON
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1
       
    def __call__(self, img):

        if random.uniform(0, 1) > self.EPSILON:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]
       
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1/self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    #img[0, x1:x1+h, y1:y1+w] = random.uniform(0, 1)
                    #img[1, x1:x1+h, y1:y1+w] = random.uniform(0, 1)
                    #img[2, x1:x1+h, y1:y1+w] = random.uniform(0, 1)
                    img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                    img[1, x1:x1+h, y1:y1+w] = self.mean[1]
                    img[2, x1:x1+h, y1:y1+w] = self.mean[2]
                    #img[:, x1:x1+h, y1:y1+w] = torch.from_numpy(np.random.rand(3, h, w))
                else:
                    img[0, x1:x1+h, y1:y1+w] = self.mean[1]
                    # img[0, x1:x1+h, y1:y1+w] = torch.from_numpy(np.random.rand(1, h, w))
                return img

        return img


class HatefulMemesFeaturesDataset(MMFDataset):
    def __init__(self, config, *args, dataset_name="hateful_memes", **kwargs):
        super().__init__(dataset_name, config, *args, **kwargs)
        assert (
            self._use_features
        ), "config's 'use_images' must be true to use image dataset"
        #图片全局特征
        with open('/mnt/liu/extract_image_feature_restne/img_feature.pkl', 'rb') as file:
            self.img_info =pickle.load(file)

    def preprocess_sample_info(self, sample_info):
        image_path = sample_info["img"]
        # img/02345.png -> 02345
        feature_path = image_path.split("/")[-1].split(".")[0]
        # Add feature_path key for feature_database access
        sample_info["feature_path"] = f"{feature_path}.npy"
        return sample_info

    def RandomErasing(self,data_tensor,p=0.5,p_object=0.7,sl=0.02,sh=0.4):
        if random.uniform(0,1)>p:
            return data_tensor
        feature_len=data_tensor.shape[1]
        for id_object in range(data_tensor.shape[0]):

            if random.uniform(0,1)<p_object:
                #RE长度
                l=int(random.uniform(sl,sh)*feature_len)
                #起点
                x=random.randint(0,feature_len-l)
                
                
                #计算掩盖量
                mean_data=data_tensor[id_object,x:x+l].mean().item()
                
                data_tensor[id_object,x:x+l]=mean_data
                       
        return data_tensor

    #图片特征！！
    def transfor_img(self,img_id):
        img_id=str(img_id)
        if len(img_id)==4:
            img_id='0'+img_id
        img_path='/root/.cache/torch/mmf/data/datasets/hateful_memes/defaults/images/img/'+img_id+'.png'
        mode = Image.open(img_path).convert('RGB')
        transform1 = transforms.Compose([
            transforms.Resize((256,256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
            #RandomErasingT()
        ])
    
        torch_img = transform1(mode)
        mode.close()
        return torch_img

    #给每个object添加图片全局特征
    def add_img_global_feature(self,c_id,image_feature_0):
        c_id=c_id.item()
        c_id=str(c_id)
        if len(c_id)==4:
            c_id='0'+c_id
        k=c_id

        feature_global=self.img_info[k]
        warnings.warn(str(feature_global.shape))
        image_feature_0+=feature_global
        warnings.warn(str(image_feature_0.shape))

        return image_feature_0    

    def __getitem__(self, idx):
        sample_info = self.annotation_db[idx]
        sample_info = self.preprocess_sample_info(sample_info)

        current_sample = Sample()

        processed_text = self.text_processor({"text": sample_info["text"]})
        current_sample.text = processed_text["text"]
        if "input_ids" in processed_text:
            current_sample.update(processed_text)

        current_sample.id = torch.tensor(int(sample_info["id"]), dtype=torch.int)

        # Instead of using idx directly here, use sample_info to fetch
        # the features as feature_path has been dynamically added
        features = self.features_db.get(sample_info)
        current_sample.update(features)

        if "label" in sample_info:
            current_sample.targets = torch.tensor(
                sample_info["label"], dtype=torch.long
            )
        #擦除算法
        #att=copy.deepcopy(current_sample.image_feature_0)
        #current_sample.image_feature_0=self.RandomErasing(current_sample.image_feature_0)
        #warnings.warn(str((att==current_sample.image_feature_0).sum()))
        #warnings.warn(str(current_sample.image_feature_0.shape))


        #添加图片信息
        #current_sample.img_RGB=self.transfor_img(current_sample.id.item())
        #warnings.warn(str(current_sample.image_feature_0.shape))
        

        #添加全局信息
        current_sample.image_feature_0=self.add_img_global_feature(current_sample.id,current_sample.image_feature_0)

        return current_sample

    def format_for_prediction(self, report):
        return generate_prediction(report)


class HatefulMemesImageDataset(MMFDataset):
    def __init__(self, config, *args, dataset_name="hateful_memes", **kwargs):
        super().__init__(dataset_name, config, *args, **kwargs)
        assert (
            self._use_images
        ), "config's 'use_images' must be true to use image dataset"

    def init_processors(self):
        super().init_processors()
        # Assign transforms to the image_db
        self.image_db.transform = self.image_processor

    def __getitem__(self, idx):
        sample_info = self.annotation_db[idx]
        current_sample = Sample()

        processed_text = self.text_processor({"text": sample_info["text"]})
        current_sample.text = processed_text["text"]
        if "input_ids" in processed_text:
            current_sample.update(processed_text)

        current_sample.id = torch.tensor(int(sample_info["id"]), dtype=torch.int)

        # Get the first image from the set of images returned from the image_db
        current_sample.image = self.image_db[idx]["images"][0]

        if "label" in sample_info:
            current_sample.targets = torch.tensor(
                sample_info["label"], dtype=torch.long
            )

        return current_sample

    def format_for_prediction(self, report):
        return generate_prediction(report)

    def visualize(self, num_samples=1, use_transforms=False, *args, **kwargs):
        image_paths = []
        random_samples = np.random.randint(0, len(self), size=num_samples)

        for idx in random_samples:
            image_paths.append(self.annotation_db[idx]["img"])

        images = self.image_db.from_path(image_paths, use_transforms=use_transforms)
        visualize_images(images["images"], *args, **kwargs)


def generate_prediction(report):
    scores = torch.nn.functional.softmax(report.scores, dim=1)
    _, labels = torch.max(scores, 1)
    # Probability that the meme is hateful, (1)
    probabilities = scores[:, 1]

    predictions = []

    for idx, image_id in enumerate(report.id):
        proba = probabilities[idx].item()
        label = labels[idx].item()
        predictions.append(
            {"id": image_id.item(), "proba": proba, "label": label,}
        )
    return predictions
