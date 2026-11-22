# Multi-model-Disease-Progression-Detection
1. Overview

Skin cancer is mostly diagnosed from how a skin spot looks, so images are the main data. But images alone do not carry all the information. This project builds a machine learning system that classifies skin cancer using two kinds of data together:

Images of skin lesions, processed by a CNN
Text / non-image data, processed with NLP techniques

The features from both are joined (multimodal feature fusion) and passed to a classifier. A later goal is to study how the disease progresses using this multimodal information.

This is a support tool for a first check. It is not a replacement for a doctor's diagnosis.

2. Problem Statement
GLOBOCAN 2022 estimated about 331,722 new melanoma cases and 58,667 deaths worldwide.
Melanoma found early has over 99% five-year survival (American Cancer Society). When spread to distant parts, this falls to 36%.
Diagnosis is hard: visual check, then dermoscopy, then biopsy and lab tests. Many lesions look alike.
Tracking progression needs different kinds of information collected over time.
Esteva et al. (2017) showed a deep learning model can classify skin lesions about as well as dermatologists in their test.
3. Objectives
Build an ML pipeline that classifies skin cancer using both image and text features.
Compare multimodal performance against image-only (and text-only) baselines.
Study disease progression using the multimodal information (method not yet decided).

Expected outcome: a working classification model and a study of how useful it is to combine image and text features.

4. Proposed Pipeline
START
  |
DATA COLLECTION (skin lesion images + textual data)
  |
DATA PREPROCESSING
  |-- Images: resize -> noise removal -> normalization      -> CNN feature extraction
  |-- Text:   tokenize -> stopword removal -> lemmatize
  |           -> feature vectorization                      -> NLP feature extraction
  |
MULTIMODAL FEATURE FUSION (combine image + text features)
  |
CLASSIFICATION (skin cancer classification)
  |
FINAL RESULT (evaluation + final class)
4.1 Image branch (OpenCV + CNN)
Resize all images to one size, remove noise, normalize pixel values.
CNN: convolution filters -> ReLU -> pooling -> fully connected layer -> image feature vector.
4.2 Text branch (NLP with NLTK)
Tokenization: split text into words.
Stopword removal: drop common words.
Lemmatization: reduce words to base form.
Feature vectorization: convert text to numeric vectors (e.g. TF-IDF or embeddings).
4.3 Fusion and classification
Concatenate the image vector and text vector into one feature set.
Train a classifier on the fused features.
Evaluate and produce the final classification.
5. Tech Stack
Item	Choice	Why
Language	Python 3.10+	Simple syntax, largest ML ecosystem
Environment	Google Colab	No local setup, free GPU/TPU, easy sharing, saves to Drive
Data	Pandas, NumPy	Loading, cleaning, array operations
Plots	Matplotlib	Accuracy/loss curves, results
Text	NLTK	Tokenization, stopwords, lemmatization
Images	OpenCV	Resize, denoise, normalize
Deep learning	CNN (TensorFlow/Keras or PyTorch)	Image feature extraction
6. Literature Survey
#	Year	Paper	Method	Result
1	2024	Hussein & Abdulazeez, A Review on Utilizing ML Classification Algorithms for Skin Cancer	Review of SVM, KNN, CNN on ISIC, HAM10000, PH2	SVM 90-95%, KNN 85-90%, CNN 95-98%
2	2025	Abohashish, Amin, Elsedimy, Hybrid LSTM-CNN for melanoma / non-melanoma	ABCDE preprocessing, patch sequences, time-distributed CNN + LSTM, HAM10000	96.21% accuracy
3	2021	Tumpa & Kabir, ANN based detection of melanoma using hybrid texture features	MGI, Otsu, ABCD + GLCM + LBP features, ANN	97.7% accuracy (binary only)
4	2022	Ou et al., Multimodal fusion model using smartphone images and metadata	CNN (images) + MLP (metadata) + attention-guided fusion, PAD-UFES-20	76.8% accuracy, 0.947 AUC

Takeaways: CNNs lead on image-only classification. Papers 1-3 are image-only and use few lesion classes; only paper 4 is truly multimodal, and it fuses images with clinical metadata.

7. Open Issues to Resolve with the Supervisor
Paired data. The report proposes Twitter text, but tweets are not linked to specific lesion images, so the two modalities cannot be paired per sample. Option: use clinical metadata (age, site, symptoms) as the second modality, as in paper 4, with datasets like PAD-UFES-20 or HAM10000.
Dataset choice. Not yet selected.
Progression. Most skin datasets have no time-series data. Options: a longitudinal dataset, or reframe as risk / severity staging.
8. Roadmap
Phase	Work
1	Choose dataset, load in Colab, EDA (class balance, missing values)
2	Preprocessing, train/val/test split (split by patient/lesion to avoid leakage)
3	Baselines: image-only CNN (transfer learning), second-modality-only model
4	Fusion model, compare against baselines
5	Handle class imbalance, tune, evaluate (accuracy, balanced accuracy, F1, AUC), Grad-CAM
6	Progression study
7	Final report and presentation
9. Suggested Repo Structure
.
├── README.md
├── data/            # raw and processed data (not committed)
├── notebooks/       # 01_eda, 02_preprocess, 03_image_cnn, 04_text_or_meta, 05_fusion
├── src/             # reusable code
├── results/         # plots, metrics
└── report/          # progress reports
10. References
A. A. Hussein and A. M. Abdulazeez, "A review on utilizing machine learning classification algorithms for skin cancer," Journal of Applied Science and Technology Trends, 5(2), 2024, pp. 60-71.
S. M. M. Abohashish, H. H. Amin, E. I. Elsedimy, "Enhanced melanoma and non-melanoma skin cancer classification using a hybrid LSTM-CNN model," Scientific Reports, 15(1), 24994, 2025.
P. P. Tumpa and M. A. Kabir, "An artificial neural network based detection and classification of melanoma skin cancer using hybrid texture features," Sensors International, vol. 2, 2021, art. 100128.
C. Ou et al., "A deep learning based multimodal fusion model for skin lesion diagnosis using smartphone collected clinical images and metadata," Frontiers in Surgery, vol. 9, 2022.
P. Tschandl, C. Rosendahl, H. Kittler, "The HAM10000 dataset," Scientific Data, 5, 180161, 2018.
11. Study Material

Video

DigitalSreeni, Skin Cancer Lesion Classification Using the HAM10000 Dataset (YouTube; listed on Class Central): https://classcentral.com/classroom/youtube-skin-cancer-lesion-classification-using-the-ham10000-dataset-128258

Written tutorials

MICCAI Educational Initiative, Skin Lesion Classification - An Educational Guide (PyTorch, HAM10000): https://medium.com/miccai-educational-initiative/skin-cancer-image-classification-an-educational-guide-2a043a1beb59
Integrating image and tabular data for deep learning (fusion on SIIM-ISIC melanoma): https://naity.github.io/integrating-image-and-tabular-data-for-deep-learning
Hands-on Keras tutorial on HAM10000: https://kannan-kasthuri.github.io/teaching/Lecture_Files/ML_AI/AI.html

YouTube search terms

HAM10000 skin cancer classification transfer learning
multimodal deep learning image and tabular data fusion
transfer learning ResNet50 Keras tutorial
Grad-CAM explained
NLTK tokenization stopwords lemmatization tutorial
TF-IDF explained 
