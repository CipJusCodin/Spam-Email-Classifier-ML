# Spam Email Classifier

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.6%2B-blue.svg)](https://www.python.org/)
[![pandas](https://img.shields.io/badge/pandas-1.0%2B-150458?logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![numpy](https://img.shields.io/badge/numpy-1.18%2B-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-0.24%2B-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![nltk](https://img.shields.io/badge/nltk-3.5%2B-9F5C0D?logo=python&logoColor=white)](https://www.nltk.org/)
[![seaborn](https://img.shields.io/badge/seaborn-0.11%2B-3776AB?logo=python&logoColor=white)](https://seaborn.pydata.org/)
[![matplotlib](https://img.shields.io/badge/matplotlib-3.2%2B-11557C?logo=python&logoColor=white)](https://matplotlib.org/)
[![wordcloud](https://img.shields.io/badge/wordcloud-1.8%2B-46B5A8?logo=python&logoColor=white)](https://github.com/amueller/word_cloud)
[![xgboost](https://img.shields.io/badge/xgboost-1.3%2B-AA0A0A?logo=python&logoColor=white)](https://xgboost.readthedocs.io/)
[![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)](https://jupyter.org/)

---

## Table of Contents

- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Key Features](#key-features)
- [Dependencies](#dependencies)
- [Getting Started](#getting-started)
- [Screenshots](#screenshots)
- [Achievements](#achievements)
- [Feature Engineering](#feature-engineering)
- [Conclusion](#conclusion)
- [License](#license)
- [Contributing](#contributing)

---

## Introduction

The Spam Email Classifier leverages machine learning to distinguish between spam and non-spam (ham) messages in emails or SMS. With ever-increasing volumes of unsolicited messages, this project provides a robust and automated solution to filter out spam, improving user productivity and security.

---

## Project Structure

```
Spam-Email-Classifier-ML/
│
├── Dataset/
│   └── spam.csv
├── README.md
└── spam_classifier.ipynb
```

- **Dataset/spam.csv** — The main dataset used for training and testing the classifier.
- **spam_classifier.ipynb** — Jupyter Notebook containing code for data exploration, preprocessing, model training, evaluation, etc.
- **README.md** — Project documentation.

---

## Key Features

- Text Preprocessing: Cleaning, tokenization, stopword & punctuation removal, stemming.
- Exploratory Data Analysis (EDA): Visualization of data distribution and word frequencies.
- Model Building: Implementation of Naive Bayes, SVM, Decision Trees, Random Forest, XGBoost, etc.
- High Accuracy: Achieved over 97% accuracy with Random Forest; perfect precision with Random Forest & Naive Bayes.
- Jupyter Notebook: Well-documented workflow for reproducibility.

---

## Dependencies

- Python >= 3.6
- pandas
- numpy
- scikit-learn
- nltk
- seaborn
- matplotlib
- wordcloud
- xgboost

```sh
pip install pandas numpy scikit-learn nltk seaborn matplotlib wordcloud xgboost
```
Additionally, download NLTK resources:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

---

## Getting Started

1. Clone the repository:
   ```sh
   git clone https://github.com/CipJusCodin/Spam-Email-Classifier-ML.git
   cd Spam-Email-Classifier-ML
   ```

2. Install dependencies (see above).

3. Open the notebook:
   - Launch Jupyter and open `spam_classifier.ipynb` to explore the workflow, run experiments, or retrain models.

---

## Screenshots

Below are screenshots of the user interface demonstrating the model in action:

![Screenshot 2024-04-06 233925](https://github.com/CipJusCodin/Spam-Email-Classifier-ML/assets/112339466/434004e7-4eca-4833-8d76-fcf6c95da7d6)
![Screenshot 2024-04-06 233906](https://github.com/CipJusCodin/Spam-Email-Classifier-ML/assets/112339466/3d4c2c04-6d70-47dc-88dc-5736a2392ca8)

---

## Achievements

- 97%+ Accuracy with Random Forest classifier
- 100% Precision with Multinomial Naive Bayes & Random Forest
- Feature engineering for optimal model performance

---

## Feature Engineering

- Lowercasing, tokenization, stopword/punctuation removal, stemming
- Vectorization (BoW, TF-IDF)
- EDA for pattern discovery and data understanding
- Hyperparameter tuning for improved accuracy

---

## Conclusion

The Spam Classifier demonstrates effective ML techniques for spam detection. The provided notebook allows for easy experimentation and continued learning and improvement.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contributing

Contributions are welcome! Please open issues or submit pull requests for improvements or bug fixes.
