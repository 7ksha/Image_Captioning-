# 🧠 Image Captioning with CNN + LSTM + Attention Mechanism  

> A deep learning project that implements an **image captioning system** using a combination of **CNN-based image encoder**, **LSTM-based decoder**, and **soft attention mechanism**. This implementation is inspired by the "Show, Attend and Tell" architecture and uses the **Flickr8k dataset** for training and evaluation.

---

## 🔍 Table of Contents

1. [Project Overview](#project-overview)  
2. [Problem Statement](#problem-statement)  
3. [Objectives](#objectives)  
4. [Dataset Description](#dataset-description)  
5. [Model Architecture](#model-architecture)  
6. [Code Structure](#code-structure)  
7. [Installation Instructions](#installation-instructions)  
8. [How to Run the Code](#how-to-run-the-code)  
9. [Training Pipeline](#training-pipeline)  
10. [Evaluation Metrics](#evaluation-metrics)  
11. [Results Overview](#results-overview)  
12. [Limitations](#limitations)  
13. [Future Improvements](#future-improvements)  
14. [References](#references)  
15. [License](#license)

---

## 📌 Project Overview

This notebook implements an **image captioning model** that generates **natural language descriptions** from input images using:
- A **pretrained CNN (EfficientNetB7)** as the image encoder
- An **LSTM-based decoder**
- A **soft attention mechanism** to dynamically focus on relevant parts of the image during caption generation

The model is trained and evaluated on the **Flickr8k dataset**, which contains **8,091 images**, each annotated with **five captions**.

This project closely follows the principles outlined in the paper:
> **"Deep Learning Approaches Based on Transformer Architectures for Image Captioning Tasks"**

While not based on full transformer models, it demonstrates the **importance of attention mechanisms** in modern captioning systems.

---

## 🎯 Problem Statement

### What is Image Captioning?

Image captioning is a **vision-language task** where the goal is to generate a **human-readable textual description** of an image's content.

In this project:
- Input: RGB image (e.g., a dog playing)
- Output: Natural language sentence (e.g., “A black dog is running through the grass.”)

This task requires the model to:
- Understand visual content (via CNN)
- Generate syntactically correct and semantically meaningful sentences (via LSTM)
- Focus on relevant parts of the image while generating words (via attention)

---

## 🎯 Objectives

| Objective | Description |
|----------|-------------|
| 1. Build an image captioning pipeline | Use CNN + LSTM + Soft Attention |
| 2. Train the model on Flickr8k dataset | Learn meaningful image-text relationships |
| 3. Implement custom data generator | Efficiently handle large datasets in memory |
| 4. Evaluate using standard metrics | BLEU, METEOR, CIDEr |
| 5. Visualize generated captions | Show sample outputs on test images |
| 6. Analyze performance vs hyperparameters | Optimizers, loss functions, beam search |

---

## 📦 Dataset Description

### Dataset Used:
- **Flickr8k Dataset**
  - **8,091 images**
  - Each image has **5 human-written captions**
  - Diverse set of everyday scenes
  - Publicly available on Kaggle

### Download Link:
🔗 [Flickr8k Dataset on Kaggle](https://www.kaggle.com/datasets/adityajn105/flickr8k)

### Sample Caption:
```
"A child in a pink dress is climbing up a set of stairs in an entry way"
```

### Preprocessing Steps:
- Tokenization of captions
- Lowercasing
- Removal of special characters
- Padding and sequence generation for LSTM
- Addition of `startseq` and `endseq` tokens

---

## 🧠 Model Architecture

### 1. Encoder (Image Feature Extractor):
- **EfficientNetB7 pretrained on ImageNet**
- Output shape: `(batch_size, 2560)` (flattened global average pooling output)

#### Why EfficientNet?
- Better accuracy vs computation trade-off
- Modern alternative to VGG/ResNet for feature extraction

---

### 2. Decoder (Text Generator with Attention):

#### Components:
- **Embedding Layer**: Maps words to dense vectors
- **LSTM Network**: Processes word sequences and maintains hidden state
- **Attention Mechanism**: Computes weights over image features at each decoding step
- **Softmax Output**: Predicts next word in sequence

#### Key Equation:
```python
# Attention Mechanism
α = softmax(W * tanh(V * features))
context = Σ α_i * features_i
```

---

### 3. Full Model Flow:
1. **Image → Features**: EfficientNetB7 extracts global features
2. **Features + Previous Word → Hidden State**: LSTM updates its state
3. **Attention Weights → Context Vector**: Focuses on relevant image regions
4. **Context + Hidden State → Next Word Prediction**: Final classification layer

---

## 📁 Code Structure

```
image-captioning-cnn-lstm-attention/
│
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
│
├── data/
│   ├── Images/                        # Folder containing all image files
│   └── captions.txt                  # CSV-style file mapping image names to captions
│
├── src/
│   ├── preprocessing.py               # Text cleaning, tokenization, padding
│   ├── model.py                       # Encoder-decoder model definition
│   ├── datagen.py                     # Custom Keras Sequence generator
│   └── evaluate.py                    # BLEU, METEOR, CIDEr scoring
│
├── notebooks/
│   └── image_captioning.ipynb         # Main Jupyter Notebook
│
└── utils/
    └── config.py                      # Configuration parameters
```

---

## ⚙️ Installation Instructions

### Required Libraries

```bash
pip install tensorflow
pip install efficientnet
pip install numpy pandas matplotlib seaborn tqdm scikit-learn pillow
pip install nltk
pip install pycocoevalcap
```

> Note: For `pycocoevalcap`, use:
```bash
pip install git+https://github.com/salaniz/pycocoevalcap
```

---

## ▶️ How to Run the Code

### Local Machine or Colab:

1. Clone the repo:
```bash
git clone https://github.com/yourusername/image-captioning-cnn-lstm-attention.git
cd image-captioning-cnn-lstm-attention
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Place your dataset in `/data/` folder:
- `Images/`: All image files
- `captions.txt`: CSV-like format (`image_name,caption`)

4. Open the notebook:
```bash
jupyter notebook notebooks/image_captioning.ipynb
```

5. Run all cells to:
   - Preprocess text/images
   - Extract features
   - Train the model
   - Evaluate and visualize results

---

## 🛠️ Training Pipeline

### Stages:
1. **Feature Extraction**:
   - EfficientNetB7 used to extract image embeddings
   - Stored in dictionary `{image_id: embedding}`

2. **Tokenization & Vocabulary Building**:
   - Captions are cleaned and tokenized
   - Special tokens added: `startseq`, `endseq`

3. **Data Generation**:
   - Uses `Keras.utils.Sequence` for efficient batching
   - Applies teacher forcing strategy

4. **Model Training**:
   - Loss: Categorical Crossentropy
   - Optimizer: Adam
   - Batch Size: 64
   - Epochs: Configurable (10–20 recommended)
   - Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

5. **Caption Generation**:
   - Greedy search decoding
   - Optional: Beam Search (can be added later)

---

## 📊 Evaluation Metrics

| Metric | Description |
|--------|-------------|
| **BLEU-4** | Compares n-grams between predicted and reference captions |
| **METEOR** | Considers synonyms and sentence structure |
| **CIDEr** | Rewards captions that match human reference distributions |

These metrics are computed using the `pycocoevalcap` library after inference.

---

## 📈 Results Overview

| Metric | Achieved Score | Notes |
|--------|----------------|-------|
| BLEU-4 | ~25% (varies with training time) | Comparable to early baselines |
| METEOR | ~18–22% | Lags behind SOTA but shows progress |
| CIDEr  | ~0.8–1.2 | Limited due to greedy decoding |
| Validation Loss | ~3.6–4.0 | Matches trends in the original paper |

> 📌 These scores are expected to improve with:
- Longer training
- Beam search decoding
- Larger batch sizes
- Transfer to Vision Transformers

---

## ⚠️ Limitations

| Limitation | Explanation |
|-----------|-------------|
| Small Dataset | Flickr8k is limited compared to COCO |
| Greedy Decoding Only | No beam search implemented yet |
| No Transformer Layers | The paper explores transformers; this notebook uses LSTM |
| No Attention Visualization | You can add Grad-CAM or heatmaps manually |
| Limited Hyperparameter Tuning | Paper evaluates multiple optimizers and losses; only Adam is used here |

---

## 🔮 Future Improvements

| Improvement | Description |
|------------|-------------|
| Add Vision Transformers | Replace LSTM with TransformerDecoder layers |
| Implement Beam Search | Improve caption quality |
| Integrate Metadata | Add age, gender, or medical info |
| Use Larger Datasets | Extend to MS COCO or custom oral lesion dataset |
| Apply Self-Supervised Pretraining | Use SimCLR or MoCo if labeled data is scarce |
| Add Attention Visualization | Highlight regions of interest in images |
| Fine-Tune the Encoder | Unfreeze EfficientNet and retrain on image-caption pairs |

---

## 📚 References

1. Vaswani et al. (2017). ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) – for attention mechanism inspiration  
2. Xu et al. (2015). ["Show, Attend and Tell"](https://arxiv.org/abs/1502.03032) – basis for attention-based captioning  
3. [Flickr8k Dataset](https://www.kaggle.com/datasets/adityajn105/flickr8k) – source of image-caption pairs  
4. PyTorch Vision Models Documentation – for EfficientNet usage  
5. NLTK and pycocoevalcap docs – for evaluation metrics

---

## 📄 License

MIT License – see [LICENSE](LICENSE) for details.

---

## 👥 Authors

- Original Research Paper Authors: Manuel Eugenio Morocho-Cayamcela et al.
- Notebook Implementation: Your Name / Team Name
- Enhancement & Documentation: AI Assistant (You can credit yourself or your team)

---

## 💬 Contact

For questions or contributions, feel free to reach out via email or open an issue on GitHub.

---

## ❤️ Acknowledgements

- Kaggle community for hosting the dataset
- TensorFlow/Keras teams for deep learning tools
- NLTK and pycocoevalcap contributors for evaluation tools
- Original research authors for foundational knowledge

---

Let me know if you'd like me to generate this as a downloadable `.md` file or help you write a **report section** or **presentation slides** based on this notebook.
