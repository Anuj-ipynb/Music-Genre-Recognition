# 🎵 Music Genre Classification with CNN

This project classifies audio files into music genres using Convolutional Neural Networks (CNNs) trained on Mel Spectrograms of the audio. It uses the [GTZAN dataset](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification), a popular benchmark dataset for music genre classification.

## 📁 Dataset

- **Source**: GTZAN Dataset on Kaggle  
- **Genres**: blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock  
- **Format**: `.wav` audio files, 30 seconds each

## 🧠 Model Overview

- **Input**: Mel Spectrograms extracted from raw audio (shape: 128 × time frames)
- **Architecture**:
  - 3 × Convolution + MaxPooling layers
  - Flatten + Dense layers
  - Dropout regularization
  - Output layer with Softmax for multi-class classification
- **Loss**: Sparse Categorical Crossentropy  
- **Optimizer**: Adam  
- **Accuracy**: ~75–80% on test set (can be improved with tuning)

## 📦 Dependencies

- Python 3.8+
- TensorFlow / Keras
- Librosa
- Matplotlib
- scikit-learn
- Kaggle API
- Pydub & FFmpeg

Install with:
```bash
pip install kaggle librosa matplotlib scikit-learn tensorflow pydub
apt-get install ffmpeg -y
