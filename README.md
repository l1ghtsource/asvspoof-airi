# Audio Spoof Detection: A Comprehensive Analysis of Multi-Model Approaches in SafeSpeak-2024

## Overview
This project investigates advanced techniques for detecting audio spoofing using state-of-the-art self-supervised learning (SSL) and transformer architectures. It evaluates various models on ASVspoof benchmark datasets, demonstrating significant improvements in distinguishing between genuine and synthesized speech.

## Models Evaluated
1. **Wav2Vec 2.0**
2. **HuBERT**
3. **SSL Wav2Vec 2.0 with PSFAN Backend**
4. **Audio Spectral Transformer (AST)**
5. **Sound Event Detection Model (EfficientNet-B0)**
6. **WavLM Base**

## Key Data Augmentation Techniques
- Gaussian noise injection
- Signal-to-noise ratio modifications
- Dynamic gain variations
- Background noise injection

## Loss Functions & Optimization
- Focal Loss, Cross-Entropy Loss, BCEWithLogits Loss
- Optimizer: AdamW with linear/cosine schedulers

## Experimental Results
The Audio Spectral Transformer and SSL Wav2Vec outperformed other models with near-perfect precision and recall, demonstrating the power of transformer architectures in spoof detection.

| Model                     | Public LB EER | Precision | Recall | F1-Score |
|---------------------------|---------------|-----------|--------|----------|
| Wav2Vec 2.0               | 0.46516       | 0.888     | 0.788  | 0.835    |
| SSL Wav2Vec               | 0.02925       | -         | -      | -        |
| Audio Spectral Transformer | 0.01384       | 0.999     | 0.999  | 0.999    |
| HuBERT                    | 8.11672       | 0.877     | 0.764  | 0.817    |
| Pretrained Wav2Vec        | 0.77492       | 0.845     | 0.725  | 0.780    |
| WavLM Base                | 1.87658       | 0.820     | 0.690  | 0.750    |

## Links
- [AST Spoofing Model (HuggingFace)](https://huggingface.co/lightsource/ast-spoofing-airi)
- [SSL Wav2Vec Model (HuggingFace)](https://huggingface.co/lightsource/ssl-w2v-psfan-spoofing-airi)
- [GitHub Repository](https://github.com/l1ghtsource/asvspoof-airi)

## Notebooks
- [AST](notebooks/ast-inference%20[0.01384].ipynb) - 0.01384 EER on Public LB
- [SSL](notebooks/ssl-inference-3plus2-epochs%20[0.02925].ipynb) - 0.02925 EER on Public LB
