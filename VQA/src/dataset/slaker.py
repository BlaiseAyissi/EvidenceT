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

class VQA_Slake_Dataset(Dataset):
    def __init__(
        self, 
        args, 
        tokenizer, 
        close_ended=True, 
        mode="train", 
        knowledge_encoder=False,
        curriculum_learning=False, 
        current_epoch=0,
        lang_mode="all"   # ← NEW PARAMETER
    ):
        self.args = args
        self.data_root = args.data_root
        self.tokenizer = tokenizer
        self.mode = mode
        self.close_ended = close_ended
        self.knowledge_encoder = knowledge_encoder
        self.lang_mode = lang_mode  # ← store mode

        # curriculum
        self.curriculum_learning = curriculum_learning and (mode == "train")
        self.current_epoch = current_epoch
        self.difficulty_scores = {}

        self.kg = json.load(open('/home/cv/Blaise/BaMCo/KSpace/Datasets/Slake1.0/KG_Slake_Train.json', 'r'))

        self.image_tokens = "<im_patch>" * args.proj_out_num
        if knowledge_encoder:
            self.knowledge_tokens = "<kg_token>" * (args.kg_proj_out_num * 2)

        # Load dataset files
        base_path = "/home/cv/Blaise/BaMCo/KSpace/Datasets/Slake1.0"
        if mode == "train":
            file_name = "train.json"
        elif mode == "validation":
            file_name = "val.json"
        elif "test" in mode:
            file_name = mode + ".json"
        else:
            raise ValueError("Invalid mode")

        with open(os.path.join(base_path, file_name), 'r') as f:
            self.data_list = json.load(f)

        # 🆕 LANGUAGE FILTERING
        self._apply_language_filter()

        # curriculum
        if self.curriculum_learning:
            self._compute_all_difficulty_scores()
            self._apply_curriculum_ordering()

        # Define transforms
        if args.pre_processor_type == "BiomedCLIP":
            train_transform = mtf.Compose([
                mtf.EnsureChannelFirst(channel_dim=-1),
                mtf.Resize((224, 224)),
                mtf.NormalizeIntensity(
                    subtrahend=[0.48145466, 0.4578275, 0.40821073],
                    divisor=[0.26862954, 0.26130258, 0.27577711],
                    channel_wise=True,
                ),
                mtf.RandRotate90(prob=0.5),
                mtf.RandFlip(prob=0.10),
                mtf.RandScaleIntensity(factors=0.1, prob=0.5),
                mtf.RandShiftIntensity(offsets=0.1, prob=0.5),
                mtf.ToTensor(dtype=torch.float),
            ])

            val_transform = mtf.Compose([
                mtf.EnsureChannelFirst(channel_dim=-1),
                mtf.Resize((224, 224)),
                mtf.NormalizeIntensity(
                    subtrahend=[0.48145466, 0.4578275, 0.40821073],
                    divisor=[0.26862954, 0.26130258, 0.27577711],
                    channel_wise=True,
                ),
                mtf.ToTensor(dtype=torch.float),
            ])
        else:
            train_transform = mtf.Compose([
                mtf.EnsureChannelFirst(channel_dim=-1),
                mtf.Resize((256, 256)),
                mtf.NormalizeIntensity(nonzero=True, channel_wise=False),
                mtf.RandRotate90(prob=0.5),
                mtf.RandFlip(prob=0.10),
                mtf.RandScaleIntensity(factors=0.1, prob=0.5),
                mtf.RandShiftIntensity(offsets=0.1, prob=0.5),
                mtf.ToTensor(dtype=torch.float),
            ])

            val_transform = mtf.Compose([
                mtf.EnsureChannelFirst(channel_dim=-1),
                mtf.Resize((256, 256)),
                mtf.ToTensor(dtype=torch.float),
            ])

        set_track_meta(False)

        self.transform = train_transform if mode == "train" else val_transform

    # 🆕 LANGUAGE FILTER METHOD
    def _apply_language_filter(self):
        """
        Filters dataset based on the lang_mode parameter.
        lang_mode = "en" → keep only english
        lang_mode = "all" → keep everything
        """
        if self.lang_mode == "all":
            return

        filtered = []
        for s in self.data_list:
            lang = s.get("lang") or s.get("q_lang") or "en"
            if self.lang_mode == "en" and lang.lower().startswith("en"):
                filtered.append(s)

        print(f"[Language Filter] Keeping {len(filtered)}/{len(self.data_list)} English samples.")
        self.data_list = filtered

    def _compute_difficulty(self, sample):
        """
        Computes difficulty score for a sample based on multiple criteria.
        Lower score = easier, higher score = harder.
        """
        difficulty_score = 0.0
        
        # 1. Question length (longer questions are harder)
        question_length = len(sample["question"].split())
        difficulty_score += question_length * 0.5
        
        # 2. Answer length (longer answers are harder)
        answer_length = len(sample["answer"].split())
        difficulty_score += answer_length * 0.3
        
        # 3. Question type complexity
        question_lower = sample["question"].lower()
        if any(word in question_lower for word in ["why", "how", "explain"]):
            difficulty_score += 5.0  # Complex reasoning questions
        elif any(word in question_lower for word in ["what", "which", "where"]):
            difficulty_score += 2.0  # Medium difficulty
        else:
            difficulty_score += 1.0  # Simple yes/no or identification
        
        # 4. Medical terminology density (if knowledge encoder is used)
        if self.knowledge_encoder:
            words = sample["question"].split()
            medical_term_count = sum(1 for w in words if (w[0].isupper() and len(w) > 3) or len(w) > 8)
            difficulty_score += medical_term_count * 1.5
        
        return difficulty_score

    def _compute_all_difficulty_scores(self):
        """Pre-compute difficulty scores for all samples"""
        print("[Curriculum Learning] Computing difficulty scores for all samples...")
        for idx, sample in enumerate(self.data_list):
            self.difficulty_scores[idx] = self._compute_difficulty(sample)
        print(f"[Curriculum Learning] Difficulty scores computed. Range: {min(self.difficulty_scores.values()):.2f} - {max(self.difficulty_scores.values()):.2f}")

    def _apply_curriculum_ordering(self):
        """
        Reorder the dataset based on curriculum learning strategy.
        All samples are kept, but ordered by difficulty based on current epoch.
        """
        total_samples = len(self.data_list)
        
        # Create list of (index, sample, difficulty) tuples
        indexed_samples = [
            (idx, sample, self.difficulty_scores[idx]) 
            for idx, sample in enumerate(self.data_list)
        ]
        
        # Sort by difficulty (easy to hard)
        indexed_samples.sort(key=lambda x: x[2])
        
        # Calculate curriculum pacing based on current epoch
        # This determines what percentage of hard samples to mix in
        if hasattr(self.args, 'curriculum_total_epochs'):
            total_epochs = self.args.curriculum_total_epochs
        else:
            total_epochs = self.args.num_train_epochs if hasattr(self.args, 'num_train_epochs') else 20
        
        # Pacing function: starts at 0.3 (30% hardest samples) and grows to 1.0 (all samples shuffled)
        pacing_ratio = min(1.0, 0.3 + (self.current_epoch / total_epochs) * 0.7)
        
        # Determine how much of the difficulty ordering to maintain
        # Early epochs: strict easy-to-hard ordering
        # Later epochs: more shuffling/randomness
        if pacing_ratio < 0.5:
            # Early training: Pure easy-to-hard ordering
            self.data_list = [sample for _, sample, _ in indexed_samples]
            ordering_type = "Strict Easy-to-Hard"
        elif pacing_ratio < 0.8:
            # Mid training: Group by difficulty buckets, shuffle within buckets
            num_buckets = 5
            bucket_size = total_samples // num_buckets
            reordered_list = []
            
            for bucket_idx in range(num_buckets):
                start_idx = bucket_idx * bucket_size
                end_idx = start_idx + bucket_size if bucket_idx < num_buckets - 1 else total_samples
                bucket_samples = [sample for _, sample, _ in indexed_samples[start_idx:end_idx]]
                
                # Shuffle within bucket
                random.shuffle(bucket_samples)
                reordered_list.extend(bucket_samples)
            
            self.data_list = reordered_list
            ordering_type = "Difficulty Buckets (shuffled within)"
        else:
            # Late training: Weighted sampling favoring harder samples
            # But still maintaining some easy samples at the start
            easy_samples = [sample for _, sample, _ in indexed_samples[:int(total_samples * 0.2)]]
            hard_samples = [sample for _, sample, _ in indexed_samples[int(total_samples * 0.2):]]
            
            random.shuffle(easy_samples)
            random.shuffle(hard_samples)
            
            # Mix: 20% easy at start, then hard samples
            self.data_list = easy_samples + hard_samples
            ordering_type = "Mixed (Easy warmup + Hard focus)"
        
        print(f"[Curriculum Learning] Epoch {self.current_epoch + 1}: Pacing={pacing_ratio:.2f}, Strategy={ordering_type}")

    def update_epoch(self, new_epoch):
        """
        Update the current epoch and reorder the dataset accordingly.
        Call this at the beginning of each epoch.
        """
        if self.curriculum_learning and self.mode == "train":
            self.current_epoch = new_epoch
            self._apply_curriculum_ordering()
            print(f"[Curriculum Learning] Dataset reordered for epoch {self.current_epoch + 1}")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        max_attempts = 100
        for _ in range(max_attempts):
            try:
                img_path = os.path.join(
                    "/home/cv/Blaise/BaMCo/KSpace/Datasets/Slake1.0/imgs", 
                    self.data_list[idx]["img_name"]
                )
                image = np.array(Image.open(img_path).convert('RGB'))
                image = self.transform(image)
                
                question = self.data_list[idx]["question"]
                answer = self.data_list[idx]["answer"]

                if self.knowledge_encoder:
                    term_list = question
                else:
                    term_list = None

                # Add tokens based on configuration
                if self.knowledge_encoder:
                    question = self.knowledge_tokens + self.image_tokens + ' ' + question
                else:
                    question = self.image_tokens + ' ' + question

                # Tokenize question + answer
                text_tensor = self.tokenizer(
                    question + ' ' + answer, 
                    max_length=self.args.max_length, 
                    truncation=True, 
                    padding="max_length", 
                    return_tensors="pt",
                )

                input_id = text_tensor["input_ids"][0]
                attention_mask = text_tensor["attention_mask"][0]

                valid_len = torch.sum(attention_mask)
                if valid_len < len(input_id):
                    input_id[valid_len] = self.tokenizer.eos_token_id

                # Tokenize question only to get question length
                question_tensor = self.tokenizer(
                    question, 
                    max_length=self.args.max_length, 
                    truncation=True, 
                    padding="max_length", 
                    return_tensors="pt"
                )
                question_len = torch.sum(question_tensor["attention_mask"][0])

                # Create labels (mask question part)
                label = input_id.clone()
                label[:question_len] = -100
                if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
                    label[label == self.tokenizer.pad_token_id] = -100
                    if valid_len < len(label):
                        label[valid_len] = self.tokenizer.eos_token_id
                else:
                    label[label == self.tokenizer.pad_token_id] = -100

                ret = {
                    'image': image,
                    'input_id': input_id,
                    'label': label,
                    'attention_mask': attention_mask,
                    'question': question,
                    'answer': answer,
                    "question_original": self.data_list[idx]["question"],
                    "term_list": term_list if self.knowledge_encoder else "None" if self.mode == "test" else None,
                }

                if self.args.seg_enable:
                    ret.update({'seg': torch.zeros_like(image)})

                return ret

            except Exception as e:
                print(f"Error in __getitem__ at index {idx}: {e}")
                idx = random.randint(0, self.__len__() - 1)
                

class VQA_RAD_Dataset(Dataset):
    def __init__(
        self, 
        args, 
        tokenizer, 
        close_ended=True, 
        mode="train", 
        knowledge_encoder=False,
        curriculum_learning=False, 
        current_epoch=0,
        lang_mode="all"   # ← NEW PARAMETER
    ):
        self.args = args
        self.data_root = args.data_root
        self.tokenizer = tokenizer
        self.mode = mode
        self.close_ended = close_ended
        self.knowledge_encoder = knowledge_encoder
        self.lang_mode = lang_mode  # ← store mode

        # curriculum
        self.curriculum_learning = curriculum_learning and (mode == "train")
        self.current_epoch = current_epoch
        self.difficulty_scores = {}

        self.kg = json.load(open('/home/cv/Blaise/BaMCo/KSpace/Datasets/VQARAD/KG_VQARAD_Train.json', 'r'))

        self.image_tokens = "<im_patch>" * args.proj_out_num
        if knowledge_encoder:
            self.knowledge_tokens = "<kg_token>" * (args.kg_proj_out_num * 2)

        # Load dataset files
        base_path = "/home/cv/Blaise/BaMCo/KSpace/Datasets/VQARAD"
        if mode == "train":
            file_name = "train.json"
        elif mode == "validation":
            file_name = "val.json"
        elif "test" in mode:
            file_name = mode + ".json"
        else:
            raise ValueError("Invalid mode")

        with open(os.path.join(base_path, file_name), 'r') as f:
            self.data_list = json.load(f)

        # 🆕 LANGUAGE FILTERING
        # curriculum
        if self.curriculum_learning:
            self._compute_all_difficulty_scores()
            self._apply_curriculum_ordering()

        # Define transforms
        if args.pre_processor_type == "BiomedCLIP":
            train_transform = mtf.Compose([
                mtf.EnsureChannelFirst(channel_dim=-1),
                mtf.Resize((224, 224)),
                mtf.NormalizeIntensity(
                    subtrahend=[0.48145466, 0.4578275, 0.40821073],
                    divisor=[0.26862954, 0.26130258, 0.27577711],
                    channel_wise=True,
                ),
                mtf.RandRotate90(prob=0.5),
                mtf.RandFlip(prob=0.10),
                mtf.RandScaleIntensity(factors=0.1, prob=0.5),
                mtf.RandShiftIntensity(offsets=0.1, prob=0.5),
                mtf.ToTensor(dtype=torch.float),
            ])

            val_transform = mtf.Compose([
                mtf.EnsureChannelFirst(channel_dim=-1),
                mtf.Resize((224, 224)),
                mtf.NormalizeIntensity(
                    subtrahend=[0.48145466, 0.4578275, 0.40821073],
                    divisor=[0.26862954, 0.26130258, 0.27577711],
                    channel_wise=True,
                ),
                mtf.ToTensor(dtype=torch.float),
            ])
        else:
            train_transform = mtf.Compose([
                mtf.EnsureChannelFirst(channel_dim=-1),
                mtf.Resize((256, 256)),
                mtf.NormalizeIntensity(nonzero=True, channel_wise=False),
                mtf.RandRotate90(prob=0.5),
                mtf.RandFlip(prob=0.10),
                mtf.RandScaleIntensity(factors=0.1, prob=0.5),
                mtf.RandShiftIntensity(offsets=0.1, prob=0.5),
                mtf.ToTensor(dtype=torch.float),
            ])

            val_transform = mtf.Compose([
                mtf.EnsureChannelFirst(channel_dim=-1),
                mtf.Resize((256, 256)),
                mtf.ToTensor(dtype=torch.float),
            ])

        set_track_meta(False)

        self.transform = train_transform if mode == "train" else val_transform


    def _compute_difficulty(self, sample):
        """
        Computes difficulty score for a sample based on multiple criteria.
        Lower score = easier, higher score = harder.
        """
        difficulty_score = 0.0
        
        # 1. Question length (longer questions are harder)
        question_length = len(sample["question"].split())
        difficulty_score += question_length * 0.5
        
        # 2. Answer length (longer answers are harder)
        answer_length = len(sample["answer"].split())
        difficulty_score += answer_length * 0.3
        
        # 3. Question type complexity
        question_lower = sample["question"].lower()
        if any(word in question_lower for word in ["why", "how", "explain"]):
            difficulty_score += 5.0  # Complex reasoning questions
        elif any(word in question_lower for word in ["what", "which", "where"]):
            difficulty_score += 2.0  # Medium difficulty
        else:
            difficulty_score += 1.0  # Simple yes/no or identification
        
        # 4. Medical terminology density (if knowledge encoder is used)
        if self.knowledge_encoder:
            words = sample["question"].split()
            medical_term_count = sum(1 for w in words if (w[0].isupper() and len(w) > 3) or len(w) > 8)
            difficulty_score += medical_term_count * 1.5
        
        return difficulty_score

    def _compute_all_difficulty_scores(self):
        """Pre-compute difficulty scores for all samples"""
        print("[Curriculum Learning] Computing difficulty scores for all samples...")
        for idx, sample in enumerate(self.data_list):
            self.difficulty_scores[idx] = self._compute_difficulty(sample)
        print(f"[Curriculum Learning] Difficulty scores computed. Range: {min(self.difficulty_scores.values()):.2f} - {max(self.difficulty_scores.values()):.2f}")

    def _apply_curriculum_ordering(self):
        """
        Reorder the dataset based on curriculum learning strategy.
        All samples are kept, but ordered by difficulty based on current epoch.
        """
        total_samples = len(self.data_list)
        
        # Create list of (index, sample, difficulty) tuples
        indexed_samples = [
            (idx, sample, self.difficulty_scores[idx]) 
            for idx, sample in enumerate(self.data_list)
        ]
        
        # Sort by difficulty (easy to hard)
        indexed_samples.sort(key=lambda x: x[2])
        
        # Calculate curriculum pacing based on current epoch
        # This determines what percentage of hard samples to mix in
        if hasattr(self.args, 'curriculum_total_epochs'):
            total_epochs = self.args.curriculum_total_epochs
        else:
            total_epochs = self.args.num_train_epochs if hasattr(self.args, 'num_train_epochs') else 20
        
        # Pacing function: starts at 0.3 (30% hardest samples) and grows to 1.0 (all samples shuffled)
        pacing_ratio = min(1.0, 0.3 + (self.current_epoch / total_epochs) * 0.7)
        
        # Determine how much of the difficulty ordering to maintain
        # Early epochs: strict easy-to-hard ordering
        # Later epochs: more shuffling/randomness
        if pacing_ratio < 0.5:
            # Early training: Pure easy-to-hard ordering
            self.data_list = [sample for _, sample, _ in indexed_samples]
            ordering_type = "Strict Easy-to-Hard"
        elif pacing_ratio < 0.8:
            # Mid training: Group by difficulty buckets, shuffle within buckets
            num_buckets = 5
            bucket_size = total_samples // num_buckets
            reordered_list = []
            
            for bucket_idx in range(num_buckets):
                start_idx = bucket_idx * bucket_size
                end_idx = start_idx + bucket_size if bucket_idx < num_buckets - 1 else total_samples
                bucket_samples = [sample for _, sample, _ in indexed_samples[start_idx:end_idx]]
                
                # Shuffle within bucket
                random.shuffle(bucket_samples)
                reordered_list.extend(bucket_samples)
            
            self.data_list = reordered_list
            ordering_type = "Difficulty Buckets (shuffled within)"
        else:
            # Late training: Weighted sampling favoring harder samples
            # But still maintaining some easy samples at the start
            easy_samples = [sample for _, sample, _ in indexed_samples[:int(total_samples * 0.2)]]
            hard_samples = [sample for _, sample, _ in indexed_samples[int(total_samples * 0.2):]]
            
            random.shuffle(easy_samples)
            random.shuffle(hard_samples)
            
            # Mix: 20% easy at start, then hard samples
            self.data_list = easy_samples + hard_samples
            ordering_type = "Mixed (Easy warmup + Hard focus)"
        
        print(f"[Curriculum Learning] Epoch {self.current_epoch + 1}: Pacing={pacing_ratio:.2f}, Strategy={ordering_type}")

    def update_epoch(self, new_epoch):
        """
        Update the current epoch and reorder the dataset accordingly.
        Call this at the beginning of each epoch.
        """
        if self.curriculum_learning and self.mode == "train":
            self.current_epoch = new_epoch
            self._apply_curriculum_ordering()
            print(f"[Curriculum Learning] Dataset reordered for epoch {self.current_epoch + 1}")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        max_attempts = 100
        for _ in range(max_attempts):
            try:
                img_path = os.path.join(
                    "/home/cv/Blaise/BaMCo/KSpace/Datasets/VQARAD/VQA_RAD_Image_Folder", 
                    self.data_list[idx]["image_name"]
                )
                image = np.array(Image.open(img_path).convert('RGB'))
                image = self.transform(image)
                
                question = self.data_list[idx]["question"]
                answer = self.data_list[idx]["answer"]

                if self.knowledge_encoder:
                    term_list = question
                else:
                    term_list = None

                # Add tokens based on configuration
                if self.knowledge_encoder:
                    question = self.knowledge_tokens + self.image_tokens + ' ' + question
                else:
                    question = self.image_tokens + ' ' + question

                # Tokenize question + answer
                text_tensor = self.tokenizer(
                    question + ' ' + answer, 
                    max_length=self.args.max_length, 
                    truncation=True, 
                    padding="max_length", 
                    return_tensors="pt",
                )

                input_id = text_tensor["input_ids"][0]
                attention_mask = text_tensor["attention_mask"][0]

                valid_len = torch.sum(attention_mask)
                if valid_len < len(input_id):
                    input_id[valid_len] = self.tokenizer.eos_token_id

                # Tokenize question only to get question length
                question_tensor = self.tokenizer(
                    question, 
                    max_length=self.args.max_length, 
                    truncation=True, 
                    padding="max_length", 
                    return_tensors="pt"
                )
                question_len = torch.sum(question_tensor["attention_mask"][0])

                # Create labels (mask question part)
                label = input_id.clone()
                label[:question_len] = -100
                if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
                    label[label == self.tokenizer.pad_token_id] = -100
                    if valid_len < len(label):
                        label[valid_len] = self.tokenizer.eos_token_id
                else:
                    label[label == self.tokenizer.pad_token_id] = -100

                ret = {
                    'image': image,
                    'input_id': input_id,
                    'label': label,
                    'attention_mask': attention_mask,
                    'question': question,
                    'answer': answer,
                    "question_original": self.data_list[idx]["question"],
                    "term_list": term_list if self.knowledge_encoder else "None" if self.mode == "test" else None,
                }

                if self.args.seg_enable:
                    ret.update({'seg': torch.zeros_like(image)})

                return ret

            except Exception as e:
                print(f"Error in __getitem__ at index {idx}: {e}")
                idx = random.randint(0, self.__len__() - 1)