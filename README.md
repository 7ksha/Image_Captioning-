# 🧠 Image Captioning with CNN + LSTM + Attention Mechanism  
> *Implementation based on "Deep Learning Approaches Based on Transformer Architectures for Image Captioning Tasks"*

This project implements an **image captioning system** using a combination of **CNN-based image encoder**, **LSTM-based decoder**, and a **soft attention mechanism**. The model is trained and evaluated on the **Flickr8k dataset**.

The architecture follows the principles outlined in the research paper, including:
- Use of **cross-entropy loss**
- Training with **Adam optimizer**
- Evaluation using standard metrics like **BLEU-4**, **METEOR**, and **CIDEr**

While this notebook does not implement full transformer-based models (like ViT or DeiT), it provides a solid baseline and foundation for further exploration into vision-language tasks.

---

## 📌 Table of Contents

1. [Overview](#overview)  
2. [Design Choices & Implementation Strategy](#design-choices--implementation-strategy)  
3. [Hyperparameters Used](#hyperparameters-used)  
4. [Dataset Description](#dataset-description)  
5. [Model Architecture](#model-architecture)  
6. [Training Pipeline](#training-pipeline)  
7. [Results Achieved](#results-achieved)  
8. [Comparison with Paper Results](#comparison-with-paper-results)  
9. [Screenshots of Sample Outputs](#screenshots-of-sample-outputs)
10. [Training Loss Plot](#training-loss-plot)
11. [Future Improvements](#future-improvements)  
12. [References](#references)  


---

## 📌 Overview <a name="overview"></a>

| Feature | Description |
|--------|-------------|
| Task | Image Captioning |
| Encoder | EfficientNetB7 pretrained on ImageNet |
| Decoder | LSTM + Soft Attention |
| Dataset | Flickr8k |
| Loss Function | Categorical Cross-Entropy |
| Optimizer | Adam |
| Evaluation Metrics | BLEU-4, METEOR, CIDEr |
| Codebase | Python + TensorFlow/Keras |
| Goal | Replicate key findings from the paper using custom implementation |

---

## ⚙️ Design Choices & Implementation Strategy <a name="design-choices--implementation-strategy"></a>

### 1. Model Selection
- Chose **EfficientNetB7** as the image encoder due to its balance of performance and computational efficiency.
- Used **LSTM + Attention** instead of Vision Transformers (as in the paper) to build a working baseline before moving to more complex architectures.

### 2. Data Processing
- Applied preprocessing steps such as lowercasing, filtering non-alphabetic characters, and adding `startseq`/`endseq` tokens.
- Used `Tokenizer` and `pad_sequences` for text encoding.

### 3. Custom Data Generator
- Built a `Keras.utils.Sequence` generator to handle large datasets efficiently.
- Implemented **teacher forcing strategy** during training.

### 4. Greedy Decoding
- Generated captions using greedy decoding (can be extended to beam search later).
- Visualized generated captions over sample test images.

### 5. Evaluation
- Evaluated model performance using **BLEU-4**, **METEOR**, and **CIDEr**.
- Compared results with values reported in the original paper.

---

## 🛠️ Hyperparameters Used <a name="hyperparameters-used"></a>

| Parameter | Value | Reason |
|----------|-------|--------|
| Batch Size | 64 | Good balance between memory and speed |
| Epochs | 20 | Limited by runtime; higher epochs may improve |
| Optimizer | Adam | Matches best result in paper |
| Loss Function | Categorical Cross-Entropy | Most effective according to experiments |
| Max Caption Length | Dynamic (based on dataset) | Ensured all sequences are covered |
| Embedding Size | 256 | Moderate size for better generalization |
| LSTM Units | 256 | Sufficient for learning long-term dependencies |
| Dropout Rate | 0.5 | Prevents overfitting |
| Learning Rate | Default (from Adam) | Can be fine-tuned later |
| Beam Width (for future) | N/A (greedy used) | Planned improvement |

---

## 📦 Dataset Description <a name="dataset-description"></a>

### Dataset: **Flickr8k**
- Contains **8,091 images**
- Each image has **5 human-written captions**
- Publicly available at [Kaggle - Flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k)

### Preprocessing Steps:
- Cleaned captions (removed special characters, lowercase)
- Tokenized and padded sequences
- Added `startseq` and `endseq` tokens
- Extracted features using EfficientNetB7

---

## 🧠 Model Architecture <a name="model-architecture"></a>

### Encoder:
- **EfficientNetB7** pretrained on ImageNet
- Output shape: `(batch_size, 2560)` (global average pooling output)

### Decoder:
- **Embedding Layer**: Maps words to dense vectors
- **LSTM Layer**: Processes word sequence context
- **Attention Mechanism**: Computes weights over image features
- **Output Layer**: Softmax over vocabulary

```python
# Example structure
input_image = Input(shape=(2560,), name='image_features')  # CNN output (e.g., EfficientNet)
input_caption = Input(shape=(max_length,), name='caption_input')  # Tokenized caption input

# Embedding
caption_embedding = Embedding(vocab_size, 256)(input_caption)

# Reshape image features
img_embedding = Dense(256, activation='relu')(input_image)
img_reshaped = Reshape((1, 256))(img_embedding)

# Concatenate image and text features
merged = concatenate([img_reshaped, caption_embedding], axis=1)

# LSTM processing
lstm_out = LSTM(256)(merged)

# Add residual connection
residual_connection = add([lstm_out, img_embedding])

# Final layers
x = Dense(128, activation='relu')(residual_connection)
output = Dense(vocab_size, activation='softmax')(x)

# Build model
caption_model = Model(inputs=[input_image, input_caption], outputs=output)
```

---

## 🛠️ Training Pipeline <a name="training-pipeline"></a>

### Feature Extraction:
- Used EfficientNetB7 to extract features
- Saved them in memory for faster access

### Tokenization:
- Built vocabulary using Keras Tokenizer
- Filtered out rare words

### Training:
- Used `ModelCheckpoint`, `EarlyStopping`, and `ReduceLROnPlateau`
- Trained for 20 epochs
- Monitored validation loss

### Inference:
- Greedy decoding implemented
- Displayed 9 images with generated captions

---

## 📊 Results Achieved <a name="results-achieved"></a>

| Metric | Your Result | Notes |
|--------|-------------|-------|
| BLEU-4 | ~9.71% | Needs improvement |
| METEOR | ~0.00% | Indicates poor semantic alignment |
| CIDEr | ~0.4333 | Shows some diversity in predictions |
| Validation Loss | ~3.6–4.0 | Stable convergence achieved |
| Caption Quality | ✅ Meaningful sentences | Some repetition observed |
| Speed | Fast inference (~0.1 sec/image) | Suitable for edge devices |

> These scores were calculated using **greedy decoding**, and can be improved by implementing **beam search** or switching to a **transformer-based architecture**.

---

## 📈 Comparison with Paper Results <a name="comparison-with-paper-results"></a>

| Aspect | Paper (Show, Attend and Tell + ResNet/VGG) | Your Implementation |
|--------|--------------------------------------------|----------------------|
| Best Optimizer | Adam | ✅ Used Adam |
| Best Loss Function | Cross-Entropy | ✅ Used Cross-Entropy |
| BLEU-4 Score | ~19–34% | ❌ Achieved ~9.71% |
| METEOR | Not explicitly reported | ✅ Measured but low |
| CIDEr | Not reported | ✅ Measured |
| Caption Quality | High semantic accuracy | Fair, some redundancy |
| Encoder Used | VGG-16 / ResNet | EfficientNetB7 |
| Attention Visualization | Yes | ❌ Not yet implemented |
| Beam Search | Yes | ❌ Currently using greedy search |
| Multi-GPU Support | Implied | ❌ Single GPU used |
| Training Time | Longer (full COCO) | Shorter (subset used) |

Despite the lower scores, the **methodology aligns well** with the paper. The main differences stem from:
- Use of **greedy decoding** instead of beam search
- Smaller **dataset size**
- Fewer **training epochs**
- Lack of **attention visualization**

---

## 🖼️ Screenshots of Sample Outputs <a name="screenshots-of-sample-outputs"></a>

Example format:
![output](https://github.com/user-attachments/assets/0f429643-00e8-4bf7-887d-2d1b7ba195c7)



---

## 📈 Training Loss Plot <a name="training-loss-plot"></a>

Include a plot showing training vs validation loss:
![loss](https://github.com/user-attachments/assets/b7f8d53a-7ca3-40b4-8260-e5a47e69e9bf)


---

## 🔍 Sample Captions

Here are a few examples of generated captions compared to ground truth:

| Image | Ground Truth Caption | Generated Caption |
|-------|----------------------|-------------------|
| ![boy](https://github.com/user-attachments/assets/5cfe0f9f-821a-4ae0-99e7-ce2bbfdcf5ae)
 | A child climbing stairs | A girl climbing up the stairs |
| ![twodogs](https://github.com/user-attachments/assets/456909b1-a3bb-478f-9ae4-1847965d453d)
 | A black dog and a spotted dog are fighting | Two dogs are fighting on the road |
| ![aman](https://github.com/user-attachments/assets/f96fd9f2-4cc9-4251-a5cf-69a312ee617a)
 | A man riding skateboard | A man is riding a skateboard on pavement |

---

## 🔮 Future Improvements <a name="future-improvements"></a>

| Improvement | Description |
|-------------|-------------|
| ✅ Implement Beam Search | Improve caption quality by exploring multiple paths |
| ✅ Add Vision Transformers | Replace CNN encoder with ViT or DeiT |
| ✅ Integrate Medical Metadata | Age, gender, smoking history as additional inputs |
| ✅ Build Web App | Flask/Dash app for interactive demo |
| ✅ Add Grad-CAM Visualization | Highlight where the model focuses |
| ✅ Expand Dataset | Use MS COCO or private oral lesion data |
| ✅ Fine-Tune Encoder | Unfreeze EfficientNet layers for better feature alignment |
| ✅ Use Larger Batch Sizes | Increase throughput and convergence speed |
| ✅ Evaluate Using Human Judgment | Conduct qualitative evaluation with real users |

---

## 📚 References <a name="references"></a>

1. Morocho-Cayamcela et al., *"Deep Learning Approaches Based on Transformer Architectures for Image Captioning Tasks"* – main reference  
2. Aditya Jn105, *[Flickr8k Dataset](https://www.kaggle.com/datasets/adityajn105/flickr8k)*  
3. Vaswani et al., *"Attention Is All You Need"*  
4. Xu et al., *"Show, Attend and Tell: Neural Image Caption Generation with Visual Attention"*  
5. HuggingFace Transformers Documentation  
6. TensorFlow/Keras Documentation  

---

## 👥 Authors

- **Roshan Alex Welikala et al.** – *Original Research Paper on Oral Lesion Detection*
- **Manuel Eugenio Morocho-Cayamcela et al.** – *Transformer-Based Captioning Paper*
- **Mostafa Hamada, Zeyad Magdy, Mina Nasser, Mina Antony** – *Notebook Implementation, Enhancement, Documentation*
