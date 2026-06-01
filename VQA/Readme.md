# BaMCo VQA - Visual Question Answering Module

Welcome to the VQA component of BaMCo, a novel framework for multimodal, knowledge-driven biomedical Visual Question Answering. This module contains the end-to-end VQA pipeline, including data loading, model training, evaluation, and inference.

## Folder Structure

```
VQA/
├── src/
│   ├── main.py                       # Main training and evaluation script
│   ├── dataset/
│   │   ├── class_embeddings_*.npy    # Precomputed class embeddings
│   │   ├── dataset_info.py           # Dataset metadata and helpers
│   │   └── multi_dataset.py          # Dataset loaders for VQA tasks
│   ├── model/
│   │   ├── __init__.py
│   │   ├── BaMCo_VQA_arch.py         # Model architecture definitions
│   │   ├── loss.py                   # Loss functions (e.g., BCELoss)
│   │   ├── cache/                    # Model cache
│   │   ├── language_model/           # Language model wrappers (GPT2, Llama, etc.)
│   │   ├── multimodal_encoder/       # Vision and knowledge encoder modules
│   │   └── multimodal_projector/     # Projector modules for multimodal fusion
│   ├── outputs/                      # Model outputs and checkpoints
│   ├── train/
│   │   └── BaMCo_VQA_trainer.py      # Custom Trainer class
│   └── utils/
│       └── dist_utils.py             # Distributed training utilities
└── Readme.md
```

## Features

- **Multimodal VQA**: Integrates vision, language, and knowledge graph information
- **Flexible Model Architectures**: Supports Llama, GPT2, and custom adapters
- **Custom Loss Functions**: Includes BCELoss and others for VQA tasks
- **Dataset Support**: PathVQA, Slake, VQARAD, and more
- **Trainer Integration**: Uses Hugging Face Trainer with custom hooks
- **Evaluation Metrics**: BLEU, ROUGE, METEOR, BERTScore, and accuracy

## Getting Started

### 1. Install Requirements

Ensure you have installed the dependencies from the main repository:

```bash
# From the root directory (BaMCo/PRIME/)
conda env create -f environment.yml
conda activate bamco
```

### 2. Prepare Datasets

Place your datasets under `KSpace/Datasets/` or use the predefined datasets (Slake, PathVQA, and VQA-RAD).

### 3. Download Model Weights

**Knowledge Encoder**: Download `slake_knowledgeSpace.pt` from [Google Drive Knowledge Space Weights](https://drive.google.com/drive/folders/...) and place it in `KSpace/src/checkpoints/`.

**Llame 3.2 model**: Downlaod the Llame 3.2 model from [Huggingface Llama 3.2](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct/tree/main) and place the files in `VQA/src/checkpoints/Llama-3.2-3B`

**BiomedNLP-BiomedBERT-base-uncased-abstract**: Downlaod from [Huggingface BiomedNLP](https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext/tree/main) and place the files in `VQA/src/checkpoints/BiomedNLP-BiomedBERT-base-uncased-abstract,`

**BiomedCLIP-PubMedBERT_256-vit_base_patch16_224**: Downlaod from [Huggingface BiomedCLIP](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/tree/main) and place the files in `VQA/src/checkpoints/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224,`


### 4. Update Model Paths

Edit `main.py` in `VQA/src/` to point to the correct checkpoint files respective to your file path

```python
# Example configuration in main.py
model_path = "../VQA/src/checkpoints/Llama-3.2-3B"
eval_path = "../VQA/src/checkpoints"
knowledge_encoder_path = "../KSpace/src/checkpoints/Slake_KnowledgeSpace.pt"
data_path = "../KSpace/Datasets"
class_embedding = "../VQA/src/dataset/"
```

### 5. Run Training

```bash
cd VQA/src
python main.py
```

### 6. Run Evaluation

Evaluation is performed automatically after training, or you can set `eval_only=True` in the arguments:

```bash
python main.py --eval_only True --checkpoint_path outputs/checkpoint_best.pth
```

## Checkpoints & Outputs

- Model checkpoints and logs are saved in the `outputs/` directory
- Evaluation results are saved as JSON files in `outputs/`
- Training logs can be monitored via wandb (optional)


## Contact

For questions, issues, or contributions, please open an issue or pull request on GitHub.