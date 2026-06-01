
##### old multiclass dataset loader class
import random
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import json
from PIL import Image
import monai.transforms as mtf
from monai.data import set_track_meta
from datasets import load_dataset, concatenate_datasets
import spacy
import logging

logger = logging.getLogger(__name__)

class VQA_Slake_Dataset(Dataset):
    def __init__(self, args, tokenizer, close_ended=True, mode="train", knowledge_encoder=False):
        self.args = args
        self.data_root = args.data_root
        self.tokenizer = tokenizer
        self.mode = mode
        self.close_ended = close_ended
        self.knowledge_encoder = knowledge_encoder

        kg_path = 'medvqa_rssm_et/Datasets/Slake1.0/KG/kg_refined_SLAKE.json'
        
        self.kg = {}
        try:
            self.kg = json.load(open(kg_path, 'r'))
            logger.info(f"Loaded knowledge graph from: {kg_path}")
        except  FileNotFoundError:
            self.kg = {}

        try:
            self.term_parser = spacy.load("en_core_sci_sm")
        except OSError:
            self.term_parser = spacy.blank("en")

        self.image_tokens = "<im_patch>" * args.proj_out_num

        if knowledge_encoder:
            # Handle case when kg_proj_out_num might be None  
            kg_proj_out_num = getattr(args, 'kg_proj_out_num', 0) or 0
            self.knowledge_tokens = "<kg_token>" * (kg_proj_out_num * 2) #for 3D embeddings.

        self.data_list = [] 
        
        if mode == "train":
            possible_paths = [
                #os.path.join(self.data_root, "KG_Slake_Train.json"),
                "/home/cv/Blaise/RSSMET/Datasets/Slake1.0/KG_Slake_Train.json"
            ]
            path = "/home/cv/Blaise/RSSMET/Datasets/Slake1.0/KG_Slake_Train.json"
            try:
                with open(path, 'r') as f:
                    self.data_list = json.load(f)
            except FileNotFoundError:
                self.data_list = []

            self.image_to_head_entity = {}
            for item in self.data_list:
                if 'image' in item and 'head_entity' in item:
                    self.image_to_head_entity[item['image']] = item['head_entity']
               
            
        elif mode == "validation":
            # Try multiple possible paths for the validation data files
            possible_paths = [
                os.path.join(self.data_root, "KG_Slake_Val.json"),
                "../Datasets/Slake1.0/KG_Slake_Val.json",
                "medvqa_rssm_et/Datasets/Slake1.0/KG_Slake_Val.json"
            ]
            path = "/home/cv/Blaise/RSSMET/Datasets/Slake1.0/KG_Slake_Val.json"
            
            try:
                with open(path, 'r') as f:
                    self.data_list = json.load(f)
            except FileNotFoundError:
                self.data_list = []

            self.image_to_head_entity = {}
            for item in self.data_list:
                if 'image' in item and 'head_entity' in item:
                    self.image_to_head_entity[item['image']] = item['head_entity']

        elif "test" in mode:
            # Try multiple possible paths for the test data files
            possible_paths = [
                os.path.join(self.data_root, "KG_Slake_Test.json"),
                "../Datasets/Slake1.0/KG_Slake_Test.json",
                "medvqa_rssm_et/Datasets/Slake1.0/KG_Slake_Test.json"
            ]
            
            path = "/home/cv/Blaise/RSSMET/Datasets/Slake1.0/KG_Slake_Test.json"
            try:
                with open(path, 'r') as f:
                    self.data_list = json.load(f)
            except FileNotFoundError:
                self.data_list = []

            self.image_to_head_entity = {}
            for item in self.data_list:
                if 'image' in item and 'head_entity' in item:
                    self.image_to_head_entity[item['image']] = item['head_entity']
        else:
            print("The mode is not desired ! ")


        if(args.pre_processor_type == "BiomedCLIP"):
            train_transform = mtf.Compose(
                [
                    mtf.EnsureChannelFirst(channel_dim=-1),
                    mtf.Resize(spatial_size=(224, 224)),
                    mtf.NormalizeIntensity(subtrahend = ([0.48145466, 0.4578275, 0.40821073]), divisor = ([0.26862954, 0.26130258, 0.27577711]), channel_wise=True),
                    mtf.RandRotate90(prob=0.5),
                    mtf.RandFlip(prob=0.10),
                    mtf.RandScaleIntensity(factors=0.1, prob=0.5),
                    mtf.RandShiftIntensity(offsets=0.1, prob=0.5),
                    mtf.ToTensor(dtype=torch.float),
                ]
            )
            
            val_transform = mtf.Compose(
                [
                    mtf.EnsureChannelFirst(channel_dim=-1),
                    mtf.Resize(spatial_size=(224, 224)),
                    mtf.NormalizeIntensity(subtrahend = ([0.48145466, 0.4578275, 0.40821073]), divisor = ([0.26862954, 0.26130258, 0.27577711]), channel_wise=True),
                    mtf.ToTensor(dtype=torch.float),
                ]
            )

        else:
            train_transform = mtf.Compose(
                [
                    mtf.EnsureChannelFirst(channel_dim=-1),
                    mtf.Resize(spatial_size=(256, 256)),
                    mtf.NormalizeIntensity(nonzero=True, channel_wise=False),
                    mtf.RandRotate90(prob=0.5),
                    mtf.RandFlip(prob=0.10),
                    mtf.RandScaleIntensity(factors=0.1, prob=0.5),
                    mtf.RandShiftIntensity(offsets=0.1, prob=0.5),
                    mtf.ToTensor(dtype=torch.float),
                ]
            )
            val_transform = mtf.Compose(
                [
                    mtf.EnsureChannelFirst(channel_dim=-1),
                    mtf.Resize(spatial_size=(256, 256)),
                    mtf.ToTensor(dtype=torch.float),
                ]
            )
        set_track_meta(False)

        if mode == 'train':
            self.transform = train_transform
        elif mode == 'validation':
            self.transform = val_transform
        elif 'test' in mode:
            self.transform = val_transform

    def __len__(self):
        if isinstance(self.data_list, list):
            return len(self.data_list)
        elif isinstance(self.data_list, dict) and 'question' in self.data_list:
            return len(self.data_list['question'])
        else:
            return 0

    def __getitem__(self, idx):
        max_attempts = 100
        
        # Ensure idx is within bounds
        if idx >= len(self.data_list):
            idx = idx % len(self.data_list)
        

        try:
            data_item = self.data_list[idx]
            question = data_item["question"]
            answer = data_item["answer"]
            img_name = data_item["image"] 
            head_entity = data_item.get("head_entity", None)

            img_path = os.path.join('/home/cv/Blaise/RSSMET/Datasets/Slake1.0/', "imgs/", img_name)
            image = np.array(Image.open(img_path).convert('RGB'))
            image = self.transform(image)

            if(self.knowledge_encoder):
                term_list = head_entity if head_entity else question
                question = self.knowledge_tokens + self.image_tokens + ' ' + question
            else:
                term_list = None
                question = self.image_tokens + ' ' + question


            text_tensor = self.tokenizer(
                question + ' ' + answer, 
                max_length=self.args.max_length, 
                truncation=True, padding="max_length",
                return_tensors="pt",
            )

            input_id = text_tensor["input_ids"][0]
            attention_mask = text_tensor["attention_mask"][0]
            
            valid_len = torch.sum(attention_mask)
            if valid_len < len(input_id):
                input_id[valid_len] = self.tokenizer.eos_token_id

            question_tensor = self.tokenizer(
                question, 
                max_length=self.args.max_length, 
                truncation=True, 
                padding="max_length", 
                return_tensors="pt"
            )
            question_len = torch.sum(question_tensor["attention_mask"][0])

            label = input_id.clone()
            label[:question_len] = -100
            if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
                label[label == self.tokenizer.pad_token_id] = -100
                if valid_len < len(label):
                    label[valid_len] = self.tokenizer.eos_token_id
            else:
                label[label == self.tokenizer.pad_token_id] = -100

            # Get head entity for this image if available
            try:
                img_name_for_entity = data_item["image"]
                question_original = data_item["question"]
            except Exception as e:
                img_name_for_entity = img_name
                question_original = question

            ret = {
                'image': image,
                'input_id': input_id,
                'label': label,
                'attention_mask': attention_mask,
                'question': question,
                'answer': answer,
                "question_original": question_original,
                "term_list": term_list if self.knowledge_encoder else "None" if self.mode == "test" else None,
                "head_entity": head_entity,  # Add head entity
                "image_path": img_name  # Add image path
            }

            if self.args.seg_enable:
                ret.update({'seg': torch.zeros_like(image)})

            return ret

        except Exception as e:
            print(f"Error in __getitem__ at index {idx}: {e}")
            idx = random.randint(0, self.__len__() - 1)
        
