# Spatio-Temporal Deep Learning for Water Stress Detection

## 🌱 Introduction
Water stress, a prominent abiotic factor, poses significant threats to crop growth and overall yield. Recognizing its importance, our framework offers timely and consistent monitoring solutions, aiming for effective interventions. This repository encapsulates a non-invasive deep learning approach that predicts water stresses by analyzing long-term crop growth patterns. Central to our solution is the ConvLSTM architecture, adept at grasping spatio-temporal nuances, and our innovative Random Sequence Splicing (RSS) algorithm, addressing data limitations and imbalances.

## 🌟 Highlights
- **ConvLSTM**: A deep learning backbone capturing intricate spatio-temporal patterns.
- **RSS Algorithm**: Our bespoke data augmentation strategy, mitigating training data challenges.
- **Achievements**: Even with data challenges, our model flaunts an 81.5% accuracy. Remarkably, with down-sampled data, it still maintains a 74.6% accuracy using a lean 10k parameters.

## 📂 Dataset
Our experiments harness the *Eschikon Plant Stress Phenotyping Dataset*. This public treasure trove provides a comprehensive look into sugar beet crop growth, affected by various environmental dynamics.

## 🔍 Repository Insight

### 📘 Notebooks
- `BTP (2).ipynb`: A holistic notebook covering EDA, preprocessing, RSS integration, CNN_LSTM modeling, training, ablation studies, and testing.
- `BTP_CONV (1).ipynb`: Similar to the above but focusing on the ConvLSTM model.

### 📄 Scripts
- `CNN_LSTM.py`: All functionalities related to the CNN_LSTM model, encapsulated.
- `ConvLSTM.py`: The ConvLSTM model's architecture and functionalities, detailed.

### 📊 Dataset Files
- `Readings.xlsx`: Raw dataset readings.

### 🖥️ Models
- `Conv_LSTM`: Post-training model files, primed for further evaluations or deployment.

## 🎨 Preprocessing
Our preprocessing blueprint is designed to morph raw images into structured sequences, encompassing:
- Image transformations.
- Center cropping to zone into key regions.
- Sequence creation via RSS.
- Dataset diversification through the Albumentations Library.

## 🚀 Training Paradigm
We explored two distinct avenues:
1. Enriching data using RSS.
2. A traditional approach, sans RSS.

Training employed the Adam optimizer, set at a 0.001 learning rate, and hinged on binary cross-entropy for loss calculation, given the binary nature of our classification task.

## 📏 Evaluation Metrics
A comprehensive evaluation suite:
- **Accuracy**: Capturing the overall prediction accuracy.
- **Precision**: Precision in positive predictions.
- **Recall**: Span of true positive samples covered.
- **F-1 Score**: Balancing Precision and Recall.

## 💡 Key Insights
- The ConvLSTM and RSS duo shines as a beacon for early water stress detection in agriculture.
- RSS proves its mettle in handling data challenges, ensuring optimal model performance.
- The ability to detect stress early, even with shorter sequences, underlines the model's practical potential.

## 🖋️ Conclusion
Our research underscores the potential of fusing the ConvLSTM framework with the RSS algorithm for agricultural water stress prediction. The findings emphasize the value of continuous monitoring, paving the path towards sustainable agriculture and fortifying food security.
