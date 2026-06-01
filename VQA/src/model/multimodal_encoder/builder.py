from .vit import ViTTower
from .kg import KGEncoder

import torch.nn as nn
from torch import load
import os

import open_clip

"""This module provides functions to build the vision tower and knowledge encoder."""

def build_vision_tower(config, **kwargs):
    vision_tower = getattr(config, 'vision_tower', None)
    if 'vit' in vision_tower.lower():
        return ViTTower(config, **kwargs)
    elif 'BiomedCLIP':
        checkpoint = load(config.knowledge_encoder_checkpoint,  weights_only=False)

        config.num_classes = checkpoint['state_dict']['fc.weight'].shape[-1]
        model = KGEncoder(num_classes=config.num_classes, config=config.eval_model_path)

        model.load_state_dict(checkpoint["state_dict"], strict=False)

        for param in model.parameters():
            param.requires_grad = False

        # Get only the pretrained vision tower, ViT of BiomedCLIP.
        # We do not further fine-tune or apply a projection layer
        # to prioritize the alignment towards the reference image features.
        #print(model.model[0].visual.children())
        model = nn.Sequential(*list(model.model[0].visual.children()))
        return model
    else:
        raise ValueError(f'Unknown vision tower: {vision_tower}')
    
def build_knowledge_encoder(config, **kwargs):
    knowledge_encoder = getattr(config, 'knowledge_encoder', None)
    if knowledge_encoder == True:
        checkpoint = load(config.knowledge_encoder_checkpoint, weights_only=False)

        config.num_classes = checkpoint['state_dict']['fc.weight'].shape[-1]
        model = KGEncoder(num_classes=config.num_classes, config=config.eval_model_path)

        model.load_state_dict(checkpoint["state_dict"], strict=False)

        model.tokenizer = open_clip.get_tokenizer(os.path.join(config.eval_model_path, 'BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'))

        for param in model.parameters():
            param.requires_grad = False
            
        return model
    else:
        raise ValueError(f'Unknown knowledge encoder: {knowledge_encoder}')