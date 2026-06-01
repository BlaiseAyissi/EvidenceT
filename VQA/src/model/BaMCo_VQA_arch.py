from abc import ABC, abstractmethod
import torch
import torch.nn as nn
import sys
import random
from .multimodal_encoder.builder import build_vision_tower, build_knowledge_encoder
from .multimodal_projector.builder import build_mm_projector, build_kg_projector
from torchvision.transforms import transforms
from PIL import Image
import torch.nn.functional as F
import os
import numpy as np
from . import loss

from dataset.dataset_info import get_RAG_classes_dict

from reasoning.arch_mixin import PRIMEDiffuMixin
from reasoning.config import PRIMEDiffuConfig

#knowledge_encoder_checkpoint = "/home/cv/Blaise/BaMCo/KSpace/src/checkpoints/Slake_KnowledgeSpace.pt"

#The language models will inherit from this class
#This class will be used to build the multimodal visual language models, which will be used for VQA tasks
class BaMCoMetaModel:
    def __init__(self, config):
        super(BaMCoMetaModel, self).__init__(config)

        self.config = config
        self.seg_enable = False

        if hasattr(config, "vision_tower"):
            self.vision_tower = build_vision_tower(config)
            if(config.vision_tower == "BiomedCLIP"):
                self.vision_tower, self.vision_tower_preprocessor = self.vision_tower
            self.mm_projector = build_mm_projector(config)

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        return vision_tower

    def initialize_vision_modules(self, model_args):
        global knowledge_encoder_checkpoint

        self.config.image_channel = model_args.image_channel
        self.config.image_size = model_args.image_size
        self.config.patch_size = model_args.patch_size
        self.config.knowledge_encoder_checkpoint = model_args.knowledge_encoder_checkpoint
        knowledge_encoder_checkpoint = model_args.knowledge_encoder_checkpoint

        self.config.vision_tower = model_args.vision_tower
        self.config.vision_select_layer = model_args.vision_select_layer
        self.config.vision_select_feature = model_args.vision_select_feature

        self.config.mm_projector_type = model_args.mm_projector_type
        self.config.proj_layer_type = model_args.proj_layer_type
        self.config.proj_layer_num = model_args.proj_layer_num
        self.config.proj_pooling_type = model_args.proj_pooling_type
        self.config.proj_pooling_size = model_args.proj_pooling_size
        self.config.eval_model_path = model_args.eval_model_path

        # vision tower
        if self.get_vision_tower() is None:
            self.model = build_vision_tower(self.config)
            # If you have a more robust vision encoder, try freezing the vision tower by requires_grad_(False)

            if(self.config.vision_tower != "BiomedCLIP"):
                self.model.requires_grad_(not model_args.freeze_vision_tower)

        if(self.config.vision_tower == "BiomedCLIP"):
            #print(self.model)
            self.config.mm_hidden_size = 768
            self.vision_tower = self.model
        else: 
            self.config.mm_hidden_size = self.model.hidden_size
            self.vision_tower = self.model

        # mm_projector
        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_mm_projector(self.config)


    def initialize_knowledge_module(self, model_args):
        self.config.knowledge_encoder = model_args.knowledge_encoder
        self.config.knowledge_encoder_checkpoint = model_args.knowledge_encoder_checkpoint
        
        self.config.data_path = model_args.data_path  
        self.config.class_embedding = model_args.class_embedding

        self.knowledge_encoder = build_knowledge_encoder(self.config)
        self.knowledge_encoder.requires_grad_(not model_args.freeze_knowledge_encoder)

        self.kg_projector = build_kg_projector(self.config)
        self.kg_projector_intra = build_kg_projector(self.config)

        self.bce_loss = loss.BCELoss()

class BaMCoMetaForCausalLM(PRIMEDiffuMixin,ABC):

    @abstractmethod
    
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_images(self, images):
        """Encodes images using the vision tower and normalizes the features."""

        #RAG_query_image_features = self.get_model().get_vision_tower()(images)
        vision_outputs = self.get_model().get_vision_tower()(images).to("cuda:0")
        
        # Extract the tensor
        if hasattr(vision_outputs, "last_hidden_state"):
            RAG_query_image_features = vision_outputs.last_hidden_state
        elif hasattr(vision_outputs, "pooler_output"):
            RAG_query_image_features = vision_outputs.pooler_output
        elif isinstance(vision_outputs, (tuple, list)):
            RAG_query_image_features = vision_outputs[0]
        else:
            RAG_query_image_features = vision_outputs

        RAG_query_image_features = F.normalize(RAG_query_image_features, dim=1)

        # gets the part of the module until the last 3 layers for the image features
        image_features = nn.Sequential(*list(self.get_model().get_vision_tower()._modules['0']._modules.values())[:-3])(images)
        image_features = self.get_model().mm_projector(image_features)
        return image_features, RAG_query_image_features
    
    def encode_knowledge(self, term_list, intra_class_images = False):
        tokenizer = self.get_model().knowledge_encoder.tokenizer

        self.get_model().knowledge_encoder.eval()

        knowledge_features = []

        if(intra_class_images):
            #tokenized_terms = tokenizer(term_list).to(self.device)
            embeddings = self.get_model().knowledge_encoder.encode_text(term_list).to(self.device)
            return embeddings

        for _, term in enumerate(term_list):
            embedding = self.get_model().knowledge_encoder.encode_text(term).to(self.device)
            embedding = self.get_model().kg_projector(embedding)
            knowledge_features.append(embedding)

        knowledge_features = torch.stack(knowledge_features).squeeze(1)
        return knowledge_features

    def retrieve_intra_class_images(self, image_features):
        all_image_data = []

        aug = transforms.Compose([
            transforms.RandomResizedCrop(84, scale=(0.08, 1.)),
            transforms.ToTensor(),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)) 
        ])

        config = self.get_model().config
        if("Slake" in knowledge_encoder_checkpoint):
            RAG_classes_dict = get_RAG_classes_dict("Slake", config)
            file = os.path.join(config.class_embedding, "class_embeddings_Slake.npy")
            #file = "/home/cv/Blaise/BaMCo/VQA/src/dataset/class_embeddings_Slake.npy"
        elif("PathVQA" in knowledge_encoder_checkpoint):
            RAG_classes_dict = get_RAG_classes_dict("PathVQA", config)
            file = os.path.join(config.class_embedding, "class_embeddings_PathVQA.npy")
            #file = "/home/cv/Blaise/BaMCo/VQA/src/dataset/class_embeddings_PathVQA.npy"
        elif("VQARAD" in knowledge_encoder_checkpoint):
            RAG_classes_dict = get_RAG_classes_dict("VQARAD", config)
            file = os.path.join(config.class_embedding, "class_embeddings_VQARAD.npy")
            #file = "/home/cv/Blaise/BaMCo/VQA/src/dataset/class_embeddings_VQARAD.npy"

        RAG_classes_dict = RAG_classes_dict
        classes = list(RAG_classes_dict.keys())[:-1] #last one is the num of classes

        """Find the most similar classes to the image features and retrieve the intra-class images."""
        with torch.no_grad():
            if not os.path.exists(file):
                #for each text, get its embedding and save into a cpu tensor
                class_embedding = []
                for idx, class_ in enumerate(classes):
                    class_features = self.encode_knowledge(class_, intra_class_images=True).detach()
                    class_embedding.append(class_features)
                class_features = torch.stack(class_embedding).squeeze(1)

                #if the npy file of the embeddings does not exist in the cache folder, create it
                if not os.path.exists(file):
                    np.save(file, class_features.cpu().numpy())
            else: class_features = torch.tensor(np.load(file)).to(self.device)

        logit_scale = self.get_model().knowledge_encoder.logit_scale
        logits_image_head = (logit_scale * image_features @ class_features.t()).detach().softmax(dim=-1)
        sorted_indices_image_head = torch.argsort(logits_image_head, dim=-1, descending=True).cpu().numpy()[:, 0]
        intra_classes = [classes[i] for i in sorted_indices_image_head]

        # Retrieve the intra-class images for the top classes for each item in the batch
        for idx, class_ in enumerate(intra_classes):
            image_data = []

            all_intra_class_images = RAG_classes_dict[class_]
            
            # 36 images per class, randomly selected
            selected_intra_class_images = random.choices(all_intra_class_images, k=36)
            for image in selected_intra_class_images:
                image_data.append(aug(Image.open(image).convert('RGB')))
            image_data = torch.stack(image_data)
            
            all_image_data.append(image_data)
        
        all_image_data = torch.stack(all_image_data).to(self.device)
        all_image_data = all_image_data.view(all_image_data.shape[0], all_image_data.shape[2], all_image_data.shape[1], all_image_data.shape[3], all_image_data.shape[4])

        # Encode the intra-class images, GLIMS is used
        all_image_data = self.get_model().knowledge_encoder.encode_intra_class_images(all_image_data)
        all_image_data = self.get_model().kg_projector_intra(all_image_data) #Trainable projector for the intra-class images

        return all_image_data


    #comment this method
    """def prepare_inputs_for_multimodal(
        self, 
        input_ids, 
        position_ids, 
        attention_mask, 
        past_key_values, 
        labels,
        images, 
        term_list):

        #Prepares the inputs for multimodal processing by encoding images and knowledge terms, 
           # and concatenating them with input embeddings. 
        
        num_layers = 4  # Increased from 2 (deeper = more capacity)
        num_heads = 8   # Increased from 4 (more attention patterns)
        dropout = 0.15  # Slightly increased for regularization
        embed_dim = 3072
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads,
            dropout=dropout,
            dim_feedforward=embed_dim * 4,
            activation='gelu',
            batch_first=True,   # More intuitive batch ordering
            norm_first=True,
        )
        
        cross_modal_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(embed_dim)
        )    
        
        input_norm = nn.LayerNorm(embed_dim)
        input_norm = input_norm.to('cuda:0')
        cross_modal_transformer = cross_modal_transformer.to('cuda:0')
        
        
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels
        else:
            #Reference image features and RAG query image features (to be used with the intra-class image sampler)
            image_features, RAG_query_image_features = self.encode_images(images)
            
            if not None in term_list:
                intra_class_images = self.retrieve_intra_class_images(RAG_query_image_features).unsqueeze(1)
                knowledge_features = self.encode_knowledge(term_list).unsqueeze(1) #1, 64, 2048]

            if("llama" in self.get_model().name_or_path or "Llama" in self.get_model().name_or_path):
                inputs_embeds = self.get_model().embed_tokens(input_ids)
            else: 
                #In BaMCo, only use the Llama and GPT2 as the base LLM models.
                inputs_embeds = self.get_model().base_model.wte(input_ids)
            
            #If there is no term-list, only the image is used. For the ablation only.
            #print("knowledge_features shape:", knowledge_features.shape)
            #print("intra_class_images shape:", intra_class_images.shape)
            #print("image_features shape:", image_features.shape)
            if not None in term_list:
                inputs_embeds = torch.cat((inputs_embeds[:, :1, :], knowledge_features, intra_class_images, image_features, inputs_embeds[:, (knowledge_features.shape[1] + intra_class_images.shape[1] + image_features.shape[1] + 1):, :]), dim=1)
                    #(inputs_embeds[:, :1, :], image_features, inputs_embeds[:, (image_features.shape[1] + 1):, :]),  dim=1)
                #knowledge_features.to('cuda:0')
                #intra_class_images.to('cuda:0')
                #image_features.to('cuda:0')
                #multimodal_embeds = torch.cat((knowledge_features, intra_class_images, image_features), dim=1)
                #multimodal_embeds = multimodal_embeds.to('cuda:0')
                #x = input_norm(multimodal_embeds)
                #fused_multimodal = cross_modal_transformer(x).to('cuda:0')
                #inputs_embeds = torch.cat(
                #    (
                #        inputs_embeds[:, :1, :],  # First token, e.g., <bos>
                #        fused_multimodal,
                #        inputs_embeds[:, (fused_multimodal.shape[1] + 1):, :],  # Remaining text tokens
                #    ), dim=1
                #)
                    
            else: 
                inputs_embeds = torch.cat(
                    (inputs_embeds[:, :1, :], image_features, inputs_embeds[:, (image_features.shape[1] + 1):, :]),
                    dim=1
                )
                #inputs_embeds = torch.cat((inputs_embeds[:, :1, :], image_features, inputs_embeds[:, (image_features.shape[1] + 1):, :]), dim=1)
                

        return None, position_ids, attention_mask, past_key_values, inputs_embeds, labels
    """

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        num_new_tokens = model_args.num_new_tokens

        self.resize_token_embeddings(len(tokenizer))

        if num_new_tokens > 0:
            input_embeddings = self.get_input_embeddings().weight.data
            output_embeddings = self.get_output_embeddings().weight.data

            input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                dim=0, keepdim=True)
            output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                dim=0, keepdim=True)

            input_embeddings[-num_new_tokens:] = input_embeddings_avg
            output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
            else:
                # we add 4 new tokens
                # if new tokens need input, please train input_embeddings
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False #The defined new tokens are replaced before feeding to the model, no need to train them
                # if new tokens need predict, please train output_embeddings
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False #Will not be predicted!

        if model_args.pretrain_mm_mlp_adapter:
            mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
            embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']

            if input_embeddings.shape == embed_tokens_weight.shape:
                input_embeddings = embed_tokens_weight
            elif embed_tokens_weight.shape[0] == num_new_tokens:
                input_embeddings[-num_new_tokens:] = embed_tokens_weight
            else:
                raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")