import os
import logging
from typing import Optional, List, Dict
import numpy as np
import torch
import transformers
from transformers import AutoTokenizer, LlamaForCausalLM, GPT2Tokenizer
from dataclasses import dataclass, field
from dataset.multi_dataset_copyy import VQA_PathVQA_Dataset, VQA_Slake_Dataset, VQA_VQARad_Dataset
from model.language_model import BaMCoLlamaForCausalLM, BaMCoGPT2ForCausalLM
from train.BaMCo_VQA_trainer import BaMCoVQATrainer
from huggingface_hub import login
from tqdm import tqdm
import evaluate
import json
import wandb

from accelerate import Accelerator
from transformers import BitsAndBytesConfig, Trainer
from huggingface_hub import hf_hub_download, HfApi
api = HfApi()

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

from reasoning.config import PRIMEDiffuConfig, RSSMConfig, ETMConfig
from train.two_stage_callback import TwoStageCallback


local_rank = -1
tokenizer = None
model_path = "/home/cv/Blaise/LLMs/Llama-3.2-3B"
eval_path = "/home/cv/Blaise/EvidenceT/VQA/src/checkpoints/"
knowledge_encoder_path = "/home/cv/Blaise/EvidenceT/KSpace/src/checkpoint/highest_val_acc_Slake.pt"
data_path = "/home/cv/Blaise/EvidenceT/KSpace/Datasets"
class_embedding = "/home/cv/Blaise/EvidenceT/VQA/src/dataset"

def rank0_print(*args):
    if local_rank == 0:
        print(*args)

@dataclass
class ModelArguments:
    version: Optional[str] = field(default="v0")
    model_name_or_path: Optional[str] = field(default=model_path, metadata={"help": "Path to the LLM or MLLM."}) #"meta-llama/Llama-3.2-1B", "openai-community/gpt2-xl"
    model_type: Optional[str] = field(default="llama3", metadata={"help": "llama3, gpt2"})

    freeze_backbone: bool = field(default=True)
    pretrain_mllm: Optional[str] = field(default=None)

    tune_mm_mlp_adapter: bool = field(default=False, metadata={"help": "Used in pretrain: tune mm_projector and embed_tokens"})
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None, metadata={"help": "Path to pretrained mm_projector and embed_tokens."})

    eval_model_path: Optional[str] = field(default=eval_path, metadata={"help": "Path to the model for evaluation."})

    # image
    image_channel: int = field(default=3)
    image_size: tuple = field(default=(224, 224))
    patch_size: tuple = field(default=(16, 16))

    #eval
    max_new_tokens: int = field(default=256)
    do_sample: bool = field(default=False)
    top_p: float = field(default=None)
    temperature: float = field(default=0.8)

    # vision
    vision_tower: Optional[str] = field(default="BiomedCLIP") # None, "vit", "BiomedCLIP"
    vision_select_layer: Optional[int] = field(default=-1)
    vision_select_feature: Optional[str] = field(default="patch")
    pretrain_vision_model: str = field(default=None, metadata={"help": "Path to pretrained model for ViT."})
    freeze_vision_tower: bool = field(default=True)

    #knowledge encoder
    knowledge_encoder: bool = field(default=True)
    knowledge_encoder_checkpoint: Optional[str] = field(default=knowledge_encoder_path)
    freeze_knowledge_encoder: bool = field(default=True)
    class_embedding = class_embedding
    data_path: str = field(default=data_path, metadata={"help": "Root directory for all data."})

    # projector
    mm_projector_type: Optional[str] = field(default='spp', metadata={"help": "spp"}) #image projection layer.
    proj_layer_type: str = field(default="mlp", metadata={"help": "Type of layer in projector. options: [linear, mlp]."})
    proj_layer_num: int = field(default=2, metadata={"help": "Number of layers in projector."})
    proj_pooling_type: str = field(default="spatial", metadata={"help": "Type of pooling in projector. options: [spatial, sequence]."})
    proj_pooling_size: int = field(default=2, metadata={"help": "Size of pooling in projector."})

    # segvol
    segmentation_module: str = field(default=None, metadata={"help": "segvol"})
    pretrain_seg_module: str = field(default=None, metadata={"help": "Pretrained segvol model."})


@dataclass
class DataArguments:
    data_root: str = field(default=data_path, metadata={"help": "Root directory for all data."})

@dataclass
class TrainingArguments(transformers.TrainingArguments):

    eval_only: bool = field(default=False)

    # LoRA
    lora_enable: bool = field(default=True)
    lora_r: int = field(default=64)
    lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0.1)

    cache_dir: Optional[str] = field(default="/cache")
    remove_unused_columns: bool = field(default=False)
    model_max_length: int = field(
        default=320,#256
        metadata={"help": "Maximum sequence length."},
    )
    seed: int = field(default=25)
    optim: str = field(default="adamw_torch")

    # Training hyperparameters — now proper fields, not bare assignments
    output_dir: str = field(default="./outputs")
    per_device_train_batch_size: int = field(default=10)
    per_device_eval_batch_size: int = field(default=100)
    num_train_epochs: float = field(default=20)
    learning_rate: float = field(default=1e-4)
    weight_decay: float = field(default=0.0)
    #warmup_ratio: float = field(default=0.1)
    lr_scheduler_type: str = field(default="cosine")
    gradient_accumulation_steps: int = field(default=1)

    # Precision / checkpointing
    bf16: bool = field(default=False)
    fp16: bool = field(default=True)          # bf16 and fp16 are mutually exclusive;
                                               # set one True via CLI, leave the other False
    #gradient_checkpointing: bool = field(default=True)

    # Eval / save
    eval_strategy: str = field(default="epoch")
    eval_accumulation_steps: int = field(default=1)
    eval_steps: float = field(default=0.1)
    save_strategy: str = field(default="epoch")
    #save_steps: float = field(default=0.5)
    #save_total_limit: int = field(default=2)
    load_best_model_at_end: bool = field(default=True)
    metric_for_best_model: str = field(default="eval_loss")
    greater_is_better: bool = field(default=False)

    # Logging
    logging_steps: int = field(default=1000)

    # Distributed
    local_rank: int = field(default=-1)
    save_on_each_node: bool = field(default=True)
    n_gpu: int = field(default=1)
    dataloader_pin_memory: bool = field(default=True)
    dataloader_num_workers: int = field(default=16)

    # Curriculum learning
    curriculum_learning: bool = field(default=False)
    curriculum_stages: int = field(default=3)

    checkpoint_dir: str = field(default="./checkpoints")
    run_name: str = field(default="Llama3.2_3B_Slake")

    auto_find_batch_size: bool = field(default=True)

def preprocess_logits_for_metrics(logits, labels):
    pred_ids = torch.argmax(logits, dim=-1)
    return pred_ids

def find_all_linear_names(model):
    if("gpt2" in model.name_or_path):
        cls = transformers.pytorch_utils.Conv1D
        model = model.base_model
    elif("Llama-3.2" in model.name_or_path):
        cls = torch.nn.Linear
    else: print("Unknown model type")

    lora_module_names = set()
    # Process of elimination: LoRA only targets on LLM backbone
    ignore_keywords = ['vision_tower', 'mm_projector', 'embed_tokens', 'lm_head', 'seg_projector', 'seg_module', 'rssm', 'etm']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in ignore_keywords):
            continue
        if isinstance(module, cls):
            lora_module_names.add(name)
    return list(lora_module_names)

@dataclass
class DataCollator:
    def __init__(self, seg_enable):
        self.seg_enable = seg_enable
    def __call__(self, batch: list) -> dict:
        if self.seg_enable:
            images, input_ids, labels, attention_mask, segs, term_list = tuple(
                [b[key] for b in batch] for key in ('image', 'input_id', 'label', 'attention_mask', 'seg', 'term_list'))

            images = torch.cat([_.unsqueeze(0) for _ in images], dim=0)
            input_ids = torch.cat([_.unsqueeze(0) for _ in input_ids], dim=0)
            labels = torch.cat([_.unsqueeze(0) for _ in labels], dim=0)
            attention_mask = torch.cat([_.unsqueeze(0) for _ in attention_mask], dim=0)

            for i, seg in enumerate(segs):
                if seg.sum() == 0:
                    segs[i] = torch.zeros((1, 1, 32, 256, 256))
                else:
                    segs[i] = seg.unsqueeze(0)
            segs = torch.cat(segs, dim=0)

            return_dict = dict(
                images=images,
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
                segs=segs,
                term_list=term_list,
            )
        else:
            images, input_ids, labels, attention_mask, term_list = tuple(
                [b[key] for b in batch] for key in ('image', 'input_id', 'label', 'attention_mask', 'term_list'))

            images = torch.cat([_.unsqueeze(0) for _ in images], dim=0)
            input_ids = torch.cat([_.unsqueeze(0) for _ in input_ids], dim=0)
            labels = torch.cat([_.unsqueeze(0) for _ in labels], dim=0)
            attention_mask = torch.cat([_.unsqueeze(0) for _ in attention_mask], dim=0)

            return_dict = dict(
                images=images,
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
                term_list=term_list
            )

        return return_dict

def main():
    global local_rank
    global tokenizer
    from accelerate import Accelerator
    accelerator = Accelerator()
    
    from accelerate.state import AcceleratorState
    print("AcceleratorState instance:", AcceleratorState().__dict__)

    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    local_rank = training_args.local_rank
    
    seed = training_args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    rank0_print("="*20 + " Tokenizer preparation " + "="*20)
    # Load tokenizer from the given path with specified configurations

    if 'gpt2' in model_args.model_type:
        tokenizer = GPT2Tokenizer.from_pretrained(model_args.model_name_or_path) 
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )

    # Define and add special tokens
    if model_args.knowledge_encoder:
        special_token = {"additional_special_tokens": ["<im_patch>", "<kg_token>", "<bx_start>", "<bx_end>"]}
        model_args.num_new_tokens = 5
    else: 
        special_token = {"additional_special_tokens": ["<im_patch>", "<bx_start>", "<bx_end>"]}
        model_args.num_new_tokens = 4

    tokenizer.add_special_tokens(
        special_token
    )
    tokenizer.add_tokens("[SEG]")

    if tokenizer.unk_token is not None and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.unk_token

    if("gpt2" in model_args.model_type):
        tokenizer.eos_token_id = 50256
        
    if 'llama3' in model_args.model_type:
        tokenizer.eos_token_id = 128001
        tokenizer.pad_token = tokenizer.eos_token

    # Convert special tokens to token IDs and set related arguments
    model_args.img_token_id = tokenizer.convert_tokens_to_ids("<im_patch>")
    model_args.seg_token_id = tokenizer.convert_tokens_to_ids("[SEG]")

    if model_args.knowledge_encoder: model_args.kg_token_id = tokenizer.convert_tokens_to_ids("<kg_token>")

    model_args.vocab_size = len(tokenizer)
    rank0_print("seg_token_id: ", model_args.seg_token_id)
    rank0_print("vocab_size: ", model_args.vocab_size)

    rank0_print("="*20 + " Model preparation " + "="*20)
    
    if model_args.vision_tower is not None:
        if 'llama' in model_args.model_type:
            model = BaMCoLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        elif 'gpt2' in model_args.model_type:
            model = BaMCoGPT2ForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        else:
            raise ValueError(f"Unknown Model Type {model_args.model_type}")
    else:
        model = LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir
        )

    if model_args.freeze_backbone:
        model.requires_grad_(False)

    model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
    if model_args.tune_mm_mlp_adapter:
        model.requires_grad_(False)
        for p in model.get_model().mm_projector.parameters():
            p.requires_grad = True

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            #bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )

        rank0_print("Adding LoRA adapters only on LLM.")
        model = get_peft_model(model, lora_config)

    
    model.config.seg_token_id = model_args.seg_token_id
    model.config.use_cache = False

    model.enable_input_require_grads()
    if training_args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # initialize vision and seg modules on LLM
    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(model_args=model_args)
    if model_args.segmentation_module is not None:
        model.get_model().initialize_seg_modules(model_args=model_args)
    if model_args.knowledge_encoder:
        model.get_model().initialize_knowledge_module(model_args=model_args)
    
    # Initialize RSSM + ETM reasoning modules
    reasoning_config = PRIMEDiffuConfig(
        rssm=RSSMConfig(num_steps=4),          # K: reasoning depth (2-8)
        etm=ETMConfig(num_evidence_tokens=16), # L: soft prefix length (8-64)
    )
    model.initialize_reasoning_modules(reasoning_config)
    rank0_print(f"Reasoning modules initialized: RSSM K={reasoning_config.rssm.num_steps}, "
                f"ETM L={reasoning_config.etm.num_evidence_tokens}")

    if model_args.pretrain_mllm:
        ckpt = torch.load(model_args.pretrain_mllm, map_location="cpu")
        model.load_state_dict(ckpt, strict=True)
        rank0_print("load pretrained MLLM weights.")

    model.initialize_vision_tokenizer(model_args, tokenizer) #output lm_head is disabled.

    model.print_trainable_parameters()

    # ckpt = torch.load("PATH/model_with_lora.bin", map_location="cpu")
    # model.load_state_dict(ckpt, strict=True)

    rank0_print("="*20 + " Dataset preparation " + "="*20)
    data_args.max_length = training_args.model_max_length
    data_args.proj_out_num = model.get_model().mm_projector.proj_out_num

    if model_args.knowledge_encoder:
        data_args.kg_proj_out_num = model.get_model().kg_projector.proj_out_num

    if(model_args.vision_tower == "BiomedCLIP"):
        data_args.pre_processor_type = "BiomedCLIP"
    else: data_args.pre_processor_type = "Default"

    rank0_print("vision tokens output from projector: ", data_args.proj_out_num)

    if model_args.knowledge_encoder:
        rank0_print("knowledge tokens output from projector: ", data_args.kg_proj_out_num)
    data_args.seg_enable = hasattr(model.get_model(), "seg_module")

    
    
    train_dataset = VQA_Slake_Dataset(data_args, tokenizer, True, "train", True)
    test_dataset = VQA_Slake_Dataset(data_args, tokenizer, False, "test", False)
    eval_dataset = VQA_Slake_Dataset(data_args, tokenizer, False, 'validation', False)
    

    data_collator = DataCollator(data_args.seg_enable)

    if not training_args.eval_only:
        rank0_print("="*20 + " Training " + "="*20)
        
        # Check if curriculum learning is enabled
        use_curriculum = hasattr(training_args, 'curriculum_learning') and training_args.curriculum_learning
        two_stage_cb = TwoStageCallback(
            model=model,
            verbose=True,
        )
        #use_curriculum = False
        
        if use_curriculum:
            rank0_print("="*20 + " Curriculum Learning Enabled " + "="*20)
            class CurriculumEpochCallback(transformers.TrainerCallback):
                def __init__(self, dataset):
                    self.dataset = dataset
                
                def on_epoch_begin(self, args, state, control, **kwargs):
                    """Update dataset ordering at the start of each epoch"""
                    current_epoch = int(state.epoch) if state.epoch is not None else 0
                    self.dataset.update_epoch(current_epoch)
                    rank0_print(f"\n{'='*20} Epoch {current_epoch + 1} - Dataset Reordered {'='*20}\n")
        
            curriculum_callback = CurriculumEpochCallback(train_dataset)
            
            trainer = BaMCoVQATrainer(
                model=model,
                args=training_args,
                data_collator=data_collator,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                processing_class=tokenizer,
                callbacks=[
                    transformers.EarlyStoppingCallback(4),
                    curriculum_callback  # Add curriculum callback
                ],
            )
            
            print(trainer.place_model_on_device)
            trainer.train()
            
            rank0_print("="*20 + " Saving Best Model (Curriculum) " + "="*20)
            torch.save(
                model.state_dict(), 
                os.path.join(training_args.checkpoint_dir, 'pytorch_model_best_curriculum.bin')
            )
            
        else:

            # Standard training without curriculum learning
            rank0_print("="*20 + " Standard Training (No Curriculum) " + "="*20)
            trainer = BaMCoVQATrainer(
                model=model,
                args=training_args,
                data_collator=data_collator,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                processing_class=tokenizer,
                callbacks=[transformers.EarlyStoppingCallback(4), two_stage_cb],
            )

            print(trainer.place_model_on_device)
            
            trainer.train()

            rank0_print("="*20 + " Saving the Best Model " + "="*20)
            torch.save(
                model.state_dict(), 
                os.path.join(training_args.checkpoint_dir, 'pytorch_model_best6.bin')
            )
        
    #===================== mod end ===============================
    else: 
        rank0_print("="*20 + " Loading and Pushing to the Hub " + "="*20)
        state_dict = torch.load(os.path.join(training_args.checkpoint_dir, 'pytorch_model_best6.bin'), map_location="cuda:0", weights_only=True)
        model.load_state_dict(state_dict, strict=False)

        # Push the model to the Hugging Face Hub before the evaluation phase
        """ try:
            api.create_repo("<your_repo_name>/" + training_args.run_name, private=True)
            api.upload_file( # Upload in the background (non-blocking action)
                repo_id="<your_repo_name>/" + training_args.run_name,
                path_or_fileobj=os.path.join(training_args.checkpoint_dir, 'pytorch_model_best.bin'),
                repo_type="model",
                path_in_repo="pytorch_model_best.bin",
            )
        except:
            print("Repo already exists. Skipping the upload step...") """
        #model = LamedLlamaForCausalLM.from_pretrained("<your_repo_name>/" + training_args.run_name)

    rank0_print("="*20 + " Evaluate on Test Set " + "="*20)
    evaluater(model, test_dataset, training_args, model_args)

def evaluater(model, dataset, training_args, model_args):
    # Evaluation metrics are initialized from the Hugging Face Evaluate library
    import evaluate
    
    #bleu = evaluate.load("evaluate_metric/sacrebleu")
    #bertscore = evaluate.load("evaluate_metric/bertscore")
    #meteor = evaluate.load("evaluate_metric/meteor")
    #rouge = evaluate.load("evaluate_metric/rouge")   
    
    try:
        bleu = None
    except:
        rank0_print("Warning: Could not load BLEU metric, skipping BLEU evaluation")
        bleu = None
    
    try:
        bertscore = None
    except (FileNotFoundError, ConnectionError):
        rank0_print("Warning: Could not load BERTScore metric, skipping BERTScore evaluation")
        bertscore = None
    
    try:
        meteor = None
    except (FileNotFoundError, ConnectionError):
        rank0_print("Warning: Could not load METEOR metric, skipping METEOR evaluation")
        meteor = None
    
    try:
        rouge = None
    except (FileNotFoundError, ConnectionError):
        rank0_print("Warning: Could not load ROUGE metric, skipping ROUGE evaluation")
        rouge = None
      

    model.to("cuda:0")
    model.eval()

    from torch.utils.data import DataLoader
    test_dataloader = DataLoader(
            dataset,
            batch_size=1,
            num_workers=2,
            pin_memory=True,
            shuffle=False,
            drop_last=False,
    )
    
    if not os.path.exists(training_args.checkpoint_dir):
        os.makedirs(training_args.checkpoint_dir)
    
    eval_json_path = os.path.join(training_args.checkpoint_dir, "eval.json")

    result_bleu = 0
    result_rouge = 0
    result_meteor = 0
    result_bert = 0
    result_accuracy_oe = 0
    result_accuracy_ce = 0

    c_ce = 1 # count of closed-ended questions
    c_oe = 1 # count of open-ended questions

    outputs = []
    
    # Iterate through the test dataset
    for sample in tqdm(test_dataloader):
        question = sample["question"]
        answer = sample['answer']

        image = sample["image"].to(device="cuda:0")
        input_id = tokenizer(question, return_tensors="pt")['input_ids'].to(device="cuda:0")
        attention_mask = torch.ones((input_id.shape[0], input_id.shape[1])).to(device="cuda:0")
        
        #in term_list, convert all "None" to None
        for i, term in enumerate(sample["term_list"]):
            if term == "None":
                sample["term_list"][i] = None

        with torch.inference_mode():
            #The knowledge encoder is used in the generate function of the model.
            generation = model.generate(image, input_id, max_new_tokens=model_args.max_new_tokens,
                                        do_sample=model_args.do_sample, top_p=model_args.top_p,
                                        temperature=model_args.temperature, pad_token_id=tokenizer.eos_token_id,
                                        attention_mask=attention_mask, term_list=sample["term_list"])
            
        generated_texts = tokenizer.batch_decode(generation, skip_special_tokens=True)

        result = dict()
        decoded_preds, decoded_labels = postprocess_text(generated_texts, answer)

        if("yes" in decoded_labels[0] or "no" in decoded_labels[0]):
            c_ce += 1

            if(decoded_labels[0] == decoded_preds):
                result_accuracy_ce += 1
        
            outputs.append({"Question": sample["question_original"], "Answer": answer[0], "Prediction": decoded_preds, "Result ACC": decoded_labels[0][0] == decoded_preds[0]})
            
        else:
            
            c_oe += 1

            doesItContain = doesContain(decoded_preds, decoded_labels[0])

            if(doesItContain):
                result_accuracy_oe += 1

            if bleu is not None:
                bleu_score = bleu.compute(predictions=decoded_preds, references=decoded_labels, max_order=1)
                result["bleu"] = bleu_score['bleu']
                result_bleu += bleu_score['bleu']
            else:
                result["bleu"] = 0

            if rouge is not None:
                rouge_score = rouge.compute(predictions=decoded_preds, references=decoded_labels, rouge_types=['rouge1'])
                result["rouge1"] = rouge_score['rouge1']
                result_rouge += rouge_score['rouge1']
            else:
                result["rouge1"] = 0

            if meteor is not None:
                meteor_score = meteor.compute(predictions=decoded_preds, references=decoded_labels)
                result["meteor"] = meteor_score['meteor']
                result_meteor += meteor_score['meteor']
            else:
                result["meteor"] = 0

            if bertscore is not None:
                bert_score = bertscore.compute(predictions=decoded_preds, references=decoded_labels, lang="en")
                result["bert_f1"] = sum(bert_score['f1']) / len(bert_score['f1'])
                result_bert += result["bert_f1"]
            else:
                result["bert_f1"] = 0

            outputs.append({"Question": sample["question_original"], "Answer": answer[0], "Prediction": decoded_preds[0], "Result ACC": doesItContain, "Bleu": result["bleu"], "Rouge": result["rouge1"]})


    result_bleu /= c_oe
    result_rouge /= c_oe
    #result_meteor /= c_oe
    #result_bert /= c_oe
    result_accuracy_oe /= c_oe
    result_accuracy_ce /= c_ce

    outputs.append({"Bleu": result_bleu, "Rouge": result_rouge, "Accuracy OE": result_accuracy_oe, "Accuracy CE": result_accuracy_ce})

    print(outputs)

    #save json
    with open(eval_json_path, 'w') as f:
        json.dump(outputs, f, indent=4)

def postprocess_text(preds, labels):
    preds = [pred.strip().lower() for pred in preds]
    labels = [[label.strip().lower()] for label in labels]
    return preds, labels

def doesContain(sub_answer_texts, sub_correct_texts):
    """
    Check if any of the sub_answer_texts is contained in any of the sub_correct_texts.
    Args:
        sub_answer_texts (list): List of sub-answer texts.
        sub_correct_texts (list): List of sub-correct texts.
    Returns:
        bool: True if any sub-answer text is contained in any sub-correct text, False otherwise.
    """
    sub_answer_texts = sub_answer_texts[0].lower().split(',')
    for sub_answer_text in sub_answer_texts:
        for sub_correct_text in sub_correct_texts:
            try:
              if sub_answer_text[0] == ' ': 
                  sub_answer_text = sub_answer_text[1:]
              if sub_correct_text[0] == ' ': 
                  sub_correct_text = sub_correct_text[1:]
            except:
              pass
            if sub_answer_text in sub_correct_text or sub_correct_text in sub_answer_text:
                return True
    return False

if __name__ == "__main__":
    main()
