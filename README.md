# PRIME-VQA: Progressive Reasoning through Integrated Multimodal Enhancement

Official implementation of "Progressive Reasoning through Integrated Multimodal Enhancement for Knowledge-Driven Medical Visual Question Answering"

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## Overview

PRIME-VQA is a knowledge-driven Medical Visual Question Answering framework that addresses training inefficiency and shallow multimodal integration through:
- **Adaptive Curriculum Learning**: Progressive difficulty ordering based on question complexity, answer depth, and medical terminology density
- **Cross-Modal Transformer Fusion**: Explicit integration of reference images, UMLS knowledge, and intra-class visual prototypes

### Key Results
- **Slake**: 87.95% accuracy (+2.15% over baselines)
- **VQA-RAD**: 77.4% accuracy (+0.7% over baselines)
- **Efficiency**: 3.1B parameters, 26.1 A100-hours training


## Installation

### Requirements
- Python >= 3.8
- PyTorch >= 2.0
- CUDA >= 11.8 (for GPU support)

### Setup

```bash
# Clone the repository
git clone https://github.com/BlaiseAyissi/-Progressive-Reasoning-through-Integrated-Multimodal-Enhancement.git
cd PRIME-VQA

# Create conda environment
conda create -n primevqa python=3.8
conda activate primevqa

# Install dependencies
pip install -r requirements.txt



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

### Prepare Datasets

Place your datasets under `KSpace/Datasets/` or use the predefined datasets (Slake, PathVQA, and VQA-RAD).

### Download Model Weights

**Knowledge Encoder**: Download `slake_knowledgeSpace.pt` from [Google Drive Knowledge Space Weights](https://drive.google.com/drive/folders/...) and place it in `KSpace/src/checkpoints/`.

**Llame 3.2 model**: Downlaod the Llame 3.2 model from [Huggingface Llama 3.2](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct/tree/main) and place the files in `VQA/src/checkpoints/Llama-3.2-3B`

**BiomedNLP-BiomedBERT-base-uncased-abstract**: Downlaod from [Huggingface BiomedNLP](https://huggingface.co/microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext/tree/main) and place the files in `VQA/src/checkpoints/BiomedNLP-BiomedBERT-base-uncased-abstract,`

**BiomedCLIP-PubMedBERT_256-vit_base_patch16_224**: Downlaod from [Huggingface BiomedCLIP](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/tree/main) and place the files in `VQA/src/checkpoints/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224,`


### Update Model Paths

Edit `main.py` in `VQA/src/` to point to the correct checkpoint files respective to your file path

```python
# Example configuration in main.py
model_path = "../VQA/src/checkpoints/Llama-3.2-3B"
eval_path = "../VQA/src/checkpoints"
knowledge_encoder_path = "../KSpace/src/checkpoints/Slake_KnowledgeSpace.pt"
data_path = "../KSpace/Datasets"
class_embedding = "../VQA/src/dataset/"
```

### Run Training

```bash
cd VQA/src
python main.py
```

### Run Evaluation

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


## Model Zoo

| Model | Dataset | Accuracy | Checkpoint |
|-------|---------|----------|------------|
| PRIME-VQA (Llama-3B) | Slake | 87.95% | 
| PRIME-VQA (Llama-3B) | VQA-RAD | 77.4% | 
| PRIME-VQA (GPT2-XL) | Slake | 85.4% |


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For questions or issues, please:
- Open an issue on GitHub

---

**Note**: This is a research project. Please ensure compliance with data usage agreements and ethical guidelines when using medical datasets.
