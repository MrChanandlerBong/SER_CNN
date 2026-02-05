 Speech Emotion Recognition (SER) using CNN and RCNN

This project implements a Speech Recognition system using the RAVDESS Dataset.  

Dataset used :

. RAVDESS
. Had 8 emotion classes
. Used Mel Spectrograms for the CNN's to feed upon

Preprocessing :

Since had very less data , and had some noise in the data , preprocessing of the data was very much required 
The preprocessing pipeline is shared across all models to ensure fair comparison 
  Preprocessing implemented:

. Silence trimming using energy thresholding
. Amplitude normalization
. Log-Mel spectrogram extraction
. Fixed-size padding/truncation to 
. Labels converted to 0-indexed format


Models Implemented

  1️. Baseline CNN (Plain CNN)
      . Conv2D + MaxPooling
      . Flatten + Dense classifier
      . Achieves higher apparent accuracy
      . Prone to overfitting and speaker 
      . Used as a  baseline.
  
  
  2️. Regularized CNN  - V2 (BatchNorm + Dropout)
      . Conv2D + MaxPooling + Batch Normalisation and some dropout
      . Why this design ?
          - BatchNormalization stabilizes training across speakers
          - Dropout reduces overfitting
          - Flatten preserves time–frequency detail critical for SER
          - Pooling provides mild invariance without collapsing temporal information
      . Not a Big Change in apparent accuracy , in fact alightly dropped 
      . Key Point is had a much higher entropy signifying it was truly unsure about its choices and hence learnt much more than the Plain CNN
  
  3. Regularized CNN  - V3 (GAP - Global Average Pooling)
       . Conv2D + MaxPooling + GAP
       . Why This Design and what happened ? :
          - Replaces Flatten with GlobalAveragePooling2D
          - Significantly reduces parameters
          - Improves regularization and stability
          - However, performance drops due to loss of temporal emotion cues
      . Global Average Pooling enforces spatial invariance, which is beneficial in vision tasks but overly compressive for speech emotion recognition.
  
  4. RCNN + LSTM (Long Short Term Memory)
      . CNN used as a feature extractor
      . LSTM models temporal evolution of speech
      . Significantly outperforms CNN-based models

Evaluation Metrics
  . Accuracy
  . Macro F1 Score (primary metric due to class imbalance)
  . Confusion Matrix
  . Prediction Entropy
  . Gender-wise accuracy and Macro F1
  

Findings Made :
  . Plain CNN achieves higher accuracy due to overfitting
  . GAP-based CNN underperforms due to excessive temporal compression
  . Regularized CNN offers a balanced baseline
  . RCNN provides the strongest and most reliable performance
