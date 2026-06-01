import random
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import json
from PIL import Image
import monai.transforms as mtf
from monai.data import set_track_meta
import spacy
import logging

logger = logging.getLogger(__name__)

class VQA_PathVQA_Dataset(Dataset):
    pass
class VQA_VQARad_Dataset(Dataset):
    pass

class VQA_Slake_Dataset(Dataset):
    def __init__(self, args, tokenizer, close_ended=True, mode="train", knowledge_encoder=False):
        self.args = args
        self.data_root = args.data_root
        self.tokenizer = tokenizer
        self.mode = mode
        self.close_ended = close_ended
        self.knowledge_encoder = knowledge_encoder

        kg_path = '/home/cv/Blaise/EvidenceT/KSpace/Datasets/Slake1.0/KG/kg_refined_SLAKE.json'
        self.kg = {}
        try:
            self.kg = json.load(open(kg_path, 'r'))
            logger.info(f"Loaded knowledge graph from: {kg_path}")
        except FileNotFoundError:
            self.kg = {}

        try:
            self.term_parser = spacy.load("en_core_sci_sm")
        except OSError:
            self.term_parser = spacy.blank("en")

        self.image_tokens = "<im_patch>" * args.proj_out_num

        if knowledge_encoder:
            kg_proj_out_num = getattr(args, 'kg_proj_out_num', 0) or 0
            self.knowledge_tokens = "<kg_token>" * (kg_proj_out_num * 2)

        self.data_list = []

        # ── SLAKE paths ───────────────────────────────────────────
        slake_paths = {
            "train":      "/home/cv/Blaise/EvidenceT/KSpace/Datasets/Slake1.0/train.json",
            "validation": "/home/cv/Blaise/EvidenceT/KSpace/Datasets/Slake1.0/val.json",
            "test":       "/home/cv/Blaise/EvidenceT/KSpace/Datasets/Slake1.0/test.json",
        }

        # ── VQARAD paths ──────────────────────────────────────────
        vqarad_paths = {
            "train":      "/home/cv/Blaise/EvidenceT/KSpace/Datasets/VQARAD/train.json",
        }

        # ── Image root dirs ───────────────────────────────────────
        self.slake_img_root  = "/home/cv/Blaise/EvidenceT/KSpace/Datasets/Slake1.0/imgs"
        self.vqarad_img_root = "/home/cv/Blaise/EvidenceT/KSpace/Datasets/VQARAD/VQA_RAD_Image_Folder"

        # ── Load data ─────────────────────────────────────────────
        if mode == "train":
            slake_data  = self._load_json(slake_paths["train"],  source_tag="slake")
            
            #vqarad_data = self._load_json(vqarad_paths["train"], source_tag="vqarad")
            vqarad_data = []
            self.data_list = slake_data + vqarad_data
            logger.info(f"Train set — Slake: {len(slake_data)}, VQARAD: {len(vqarad_data)}, "
                        f"Total: {len(self.data_list)}")

        elif mode == "validation":
            self.data_list = self._load_json(slake_paths["validation"], source_tag="slake")
            logger.info(f"Validation set — Slake: {len(self.data_list)}")

        elif "test" in mode:
            self.data_list = self._load_json(slake_paths["test"], source_tag="slake") 
            
            logger.info(f"Test set — Slake: {len(self.data_list)}")
            print(f"Test set — Slake: {len(self.data_list)}")

        else:
            print("The mode is not desired!")

        # ── Build image → head_entity lookup ─────────────────────
        self.image_to_head_entity = {
            item["image"]: item["head_entity"]
            for item in self.data_list
            if "image" in item and "head_entity" in item
        }

        # ── Transforms ───────────────────────────────────────────
        if args.pre_processor_type == "BiomedCLIP":
            train_transform = mtf.Compose([
                mtf.EnsureChannelFirst(channel_dim=-1),
                mtf.Resize(spatial_size=(224, 224)),
                mtf.NormalizeIntensity(
                    subtrahend=[0.48145466, 0.4578275, 0.40821073],
                    divisor=[0.26862954, 0.26130258, 0.27577711],
                    channel_wise=True
                ),
                mtf.RandRotate90(prob=0.5),
                mtf.RandFlip(prob=0.10),
                mtf.RandScaleIntensity(factors=0.1, prob=0.5),
                mtf.RandShiftIntensity(offsets=0.1, prob=0.5),
                mtf.ToTensor(dtype=torch.float),
            ])
            val_transform = mtf.Compose([
                mtf.EnsureChannelFirst(channel_dim=-1),
                mtf.Resize(spatial_size=(224, 224)),
                mtf.NormalizeIntensity(
                    subtrahend=[0.48145466, 0.4578275, 0.40821073],
                    divisor=[0.26862954, 0.26130258, 0.27577711],
                    channel_wise=True
                ),
                mtf.ToTensor(dtype=torch.float),
            ])
        else:
            train_transform = mtf.Compose([
                mtf.EnsureChannelFirst(channel_dim=-1),
                mtf.Resize(spatial_size=(256, 256)),
                mtf.NormalizeIntensity(nonzero=True, channel_wise=False),
                mtf.RandRotate90(prob=0.5),
                mtf.RandFlip(prob=0.10),
                mtf.RandScaleIntensity(factors=0.1, prob=0.5),
                mtf.RandShiftIntensity(offsets=0.1, prob=0.5),
                mtf.ToTensor(dtype=torch.float),
            ])
            val_transform = mtf.Compose([
                mtf.EnsureChannelFirst(channel_dim=-1),
                mtf.Resize(spatial_size=(256, 256)),
                mtf.ToTensor(dtype=torch.float),
            ])

        set_track_meta(False)

        self.transform = train_transform if mode == "train" else val_transform

    # ── Helpers ───────────────────────────────────────────────────
    def _load_json(self, path, source_tag):
        """Load a JSON file and stamp every item with a dataset source tag."""
        try:
            with open(path, 'r') as f:
                data = json.load(f)
            for item in data:
                item["_source"] = source_tag   # track origin for image path resolution
            logger.info(f"Loaded {len(data)} items from {path}")
            return data
        except FileNotFoundError:
            logger.warning(f"File not found: {path}")
            return []

    def _get_img_path(self, item):
        """Return the correct image path depending on the source dataset."""
        source = item.get("_source", "slake")
        if source == "vqarad":
            img_name = item["image_name"]
            return os.path.join(self.vqarad_img_root, img_name)
        img_name = item["img_name"]
        return os.path.join(self.slake_img_root, img_name)

    # ── Dataset protocol ──────────────────────────────────────────
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        max_attempt = 100
        for _ in range(max_attempt):
            try:
                data_item  = self.data_list[idx]
                question   = data_item["question"]
                answer     = data_item["answer"]
                head_entity = data_item.get("head_entity", None)

                # ── Load image ────────────────────────────────────────
                img_path = self._get_img_path(data_item)
                image = np.array(Image.open(img_path).convert('RGB'))
                image = self.transform(image)

                # ── Build question prompt ─────────────────────────────
                if self.knowledge_encoder:
                    term_list = head_entity if head_entity else question
                    question  = self.knowledge_tokens + self.image_tokens + ' ' + question
                else:
                    term_list = None
                    question  = self.image_tokens + ' ' + question

                # ── Tokenize ──────────────────────────────────────────
                text_tensor = self.tokenizer(
                    question + ' ' + answer,
                    max_length=self.args.max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )
                input_id       = text_tensor["input_ids"][0]
                attention_mask = text_tensor["attention_mask"][0]

                valid_len = torch.sum(attention_mask)
                if valid_len < len(input_id):
                    input_id[valid_len] = self.tokenizer.eos_token_id

                question_tensor = self.tokenizer(
                    question,
                    max_length=self.args.max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
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

                ret = {
                    'image':             image,
                    'input_id':          input_id,
                    'label':             label,
                    'attention_mask':    attention_mask,
                    'question':          question,
                    'answer':            answer,
                    'question_original': data_item["question"],
                    'term_list':         term_list if self.knowledge_encoder else ("None" if self.mode == "test" else None),  # useful for debugging
                }

                if self.args.seg_enable:
                    ret.update({'seg': torch.zeros_like(image)})

                return ret

            except Exception as e:
                print(f"Error in __getitem__ at index {idx}: {e}")
                return self.__getitem__(random.randint(0, self.__len__() - 1))
        
 