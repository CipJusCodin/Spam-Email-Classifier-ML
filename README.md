# Spam Email Classifier Using Machine Learning
## Introduction
+ ### Brief overview of the project
  The Spam Classifier is a machine learning model deployed to distinguish between spam and non-spam (ham) messages in emails or SMS.
+ ### Importance and relevance of the project
  With the increasing volume of unsolicited messages, such as spam emails and texts, it is crucial to have an efficient system that can automatically filter out such messages, saving time and resources for users.
## Key Features
+ **Preprocessing:** Text cleaning, tokenization, removal of stopwords and punctuation, and stemming.
+ **Exploratory Data Analysis (EDA):** Understanding the data distribution, visualizing word frequencies, and analyzing message characteristics.
+ **Model Building:** Utilizing various classification algorithms including Naive Bayes, SVM, Decision Trees, etc.
+ **Deployment:** Using Streamlit to create an interactive web application for real-time spam detection.
## Dependencies

1. **Python (>=3.6)** - Download and install Python from the official website (https://www.python.org/). Make sure to select the option to add Python to your PATH during installation.
2. **pandas** - `pip install pandas`
3. **numpy** - `pip install numpy`
4. **scikit-learn** - `pip install scikit-learn`
5. **nltk** - `pip install nltk`
6. **seaborn** - `pip install seaborn`
7. **matplotlib** - `pip install matplotlib`
8. **streamlit** - `pip install streamlit`
9. **wordcloud** - `pip install wordcloud`
10. **xgboost** - `pip install xgboost`
11. **NLTK Resources:** After installing NLTK, download required resources by running the following code in Python:
```
import nltk
nltk.download('punkt')
nltk.download('stopwords') 
```
## Achievements
+ Achieved an accuracy score of over 97% with the Random Forest classifier.
+ Precision scores were consistently high across multiple models, with Multinomial Naive Bayes and Random Forest achieving perfect precision scores of 100%.
+ Implemented feature engineering techniques such as text preprocessing and exploratory data analysis for better model performance.
+ Successfully deployed the best performing model using Streamlit, allowing for easy usage and accessibility.

## Feature Engineering
+ Text Preprocessing: Lowercasing, tokenization, removal of stopwords and punctuation, and stemming are performed to clean and normalize the text data.
+ Exploratory Data Analysis (EDA): Analyzing data distribution, word frequencies, and message characteristics helps in understanding the dataset better and identifying patterns.
+ Vectorization: Text data is transformed into numerical vectors using techniques like Bag of Words (BoW) or TF-IDF (Term Frequency-Inverse Document Frequency) to make it suitable for machine learning algorithms.
+ Model Building: Various classification algorithms are employed, and hyperparameter tuning is performed to optimize model performance.

## Conclusion
+ The Spam Classifier project demonstrates the effective utilization of machine learning techniques for spam detection.
+ The interactive web application allows users to easily classify messages as spam or not spam in real-time, enhancing user experience and productivity.
+ Continual improvements and updates can be made to the model to adapt to evolving spamming techniques and improve accuracy further.

## User Interface and Interaction
1. ![Screenshot 2024-04-06 233925](https://github.com/CipJusCodin/Spam-Email-Classifier-ML/assets/112339466/434004e7-4eca-4833-8d76-fcf6c95da7d6)
2. ![Screenshot 2024-04-06 233906](https://github.com/CipJusCodin/Spam-Email-Classifier-ML/assets/112339466/3d4c2c04-6d70-47dc-88dc-5736a2392ca8)


