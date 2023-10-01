# Spatio-Temporal Deep Learning Analysis for Water Stresses Detection

## Introduction
Water stress is a crucial abiotic factor that significantly hampers crop growth and yield. Timely and consistent monitoring can pave the way for necessary interventions, ensuring optimal crop health. This repository introduces a non-invasive deep learning framework designed to predict water stresses in crops by analyzing their long-term growth progression. At its core, the framework leverages the ConvLSTM architecture for understanding spatio-temporal patterns and a novel Random Sequence Splicing (RSS) algorithm to mitigate the challenges posed by limited and skewed training data.

## Highlights
- **Deep Learning Framework**: Uses ConvLSTM to capture the spatio-temporal patterns in sequential plant growth data.
- **RSS Algorithm**: A proprietary technique that assists in data augmentation, addressing data scarcity and imbalance issues.
- **Impressive Results**: Despite the data imbalance, the model boasts an accuracy of 81.5%. Even with down-sampled sequences, it retains an accuracy of 74.6% with a compact model size of just 10k parameters.

## Dataset
The core evaluations were carried out using the *Eschikon Plant Stress Phenotyping Dataset*. This public dataset encompasses spatio-temporal-spectral data, detailing the growth of sugar beet crops under varied environmental conditions.

## Repository Structure

### 📔 Notebooks
- `BTP (2).ipynb`: EDA, Preprocessing,RSS,Modeling of CNN_LSTM, Training, ablation studies and Testing
- `BTP_CONV (1).ipynb`: EDA, Preprocessing, RSS, Modeling of ConvLSTM, Training, ablation studies and Testing

### 📜 Python Scripts
- `CNN_LSTM.py`: Script to encompassing the CNN_LSTM model architecture and related functionalities.
- `ConvLSTM.py`: Script encompassing the ConvLSTM model architecture and related functionalities.

### 📂 Readings
- `Readings.xlsx`

### 🌐 Models
- `Conv_LSTM`: Saved model files post-training. Ready for deployment or further evaluations.

## Preprocessing
The preprocessing pipeline is meticulously crafted to transform the raw image dataset into a structured sequential format. It encompasses:
- Image conversion processes.
- Center cropping techniques to focus on the region of interest.
- Sequence generation using the RSS technique.
- Data augmentation using the Albumentations Library to enrich the dataset variability.

## Training Pipeline
Two distinct pipelines were tested: 
1. Incorporating RSS for data augmentation.
2. Without RSS.

Model training employed the Adam optimizer with a learning rate set at 0.001. Binary cross-entropy was chosen as the loss criterion due to the binary classification nature of the task.

## Evaluation Metrics
Performance evaluation was comprehensive, covering:
- Accuracy: Overall proportion of correct predictions.
- Precision: Accuracy of positive predictions.
- Recall: Coverage of actual positive samples.
- F-1 Score: Harmonic mean of Precision and Recall.

## Key Takeaways
- The ConvLSTM-RSS combo offers a promising solution for early and consistent water stress detection in crops.
- The RSS algorithm effectively addresses data scarcity and imbalance, paving the way for robust model performance.
- The early detection capability, even with shorter image sequences, emphasizes the model's efficiency and potential for real-world applications.

## Conclusion
Through this study, we've illustrated the profound potential of integrating the ConvLSTM framework with the RSS algorithm for predicting water stress levels in agriculture. Our results underscore the importance of continual monitoring and proactive interventions, spotlighting a path forward for sustainable agricultural practices and ensuring food security.
