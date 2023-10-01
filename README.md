# Spatio-Temporal Analysis for Water Stresses

## Introduction
Abiotic stresses, particularly water stress, significantly impact crop growth and yield. Early and continual monitoring of water stress can assist in mitigating its adverse effects. This repository presents a non-invasive deep learning framework that predicts water stresses by analyzing the long-term progression of crops. The framework utilizes a novel Random Sequence Splicing (RSS) algorithm to handle limited training data and high class skewness. It also employs the ConvLSTM architecture to learn spatio-temporal patterns in progressive plant growth cycles.

## Key Results
- Achieved an accuracy of 81.5% despite an imbalance ratio of 1.83 in the original dataset.
- Attained an impressive accuracy of 74.6% on down-sampled sequences, while using only 10k parameters.

## Dataset
Experiments and tests were conducted using the Eschikon Plant Stress Phenotyping Dataset. This dataset is a public resource containing spatio-temporal-spectral data of sugar beet crop growth under various environmental factors.

## Files and Directories
- **Notebooks**:
  - `BTP (2).ipynb`
  - `BTP_CONV (1).ipynb`
- **Python Scripts**:
  - `CNN_LSTM.py`
  - `ConvLSTM.py`
- **Dataset**:
  - `Readings.xlsx`
- **Models Saved**
  - `Conv_LSTM`


## Preprocessing
A detailed preprocessing pipeline has been followed to convert the image dataset into sequential data. This includes image conversion, center cropping, sequence generation using RSS, and various augmentations using the Albumentations Library.

## Training
Two separate processing pipelines were utilized, one employing RSS and the other without. The adam optimizer was used with a learning rate of 0.001 and binary cross-entropy as the loss criterion.

## Evaluation
Evaluation metrics such as Accuracy, Precision, Recall, and F-1 score were used to assess the model's performance.

## Conclusion
The study showcases the potential of using the ConvLSTM framework and the RSS algorithm in forecasting water stress levels in agriculture. The results indicate that early and continual monitoring can be beneficial for agricultural practices and food security.
