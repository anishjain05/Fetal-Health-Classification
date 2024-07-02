# Fetal-Health-Classification

*(Made as part of the Statistical Machine Learning (SML) coursework at IIIT-Delhi)*

## Description

This project aims to apply statistical machine-learning techniques to classify fetal health based on
features extracted from Cardiotocogram (CTG) exams. Fetal health classification is a critical task
in obstetrics, **aiding in early detection and intervention of potential issues during pregnancy**.
The problem we're addressing involves multi-class classification, with the dataset categorized
into three classes: Normal, Suspect, and Pathological, as labeled by domain experts.

## Method

We employ and test various statistical machine learning algorithms such as logistic
regression, decision trees, random forests, support vector machines, and possibly neural
networks. We begin with **exploratory data analysis** to gain insights into the distribution and
relationships among the features and the target classes.

*Code available in main.py (uploaded)*

## Dataset

The dataset consists of 2126 records of features obtained from CTG exams, including fetal heart
rate, uterine contractions, fetal movements, and various other physiological parameters. **Each
instance in the dataset is labeled with one of three classes: Normal, Suspect, or
Pathological**, based on expert assessment. The dataset is rich in features and contains a sufficient
number of instances to train and evaluate our models effectively.

*Dataset available in fetal_health_dataset.csv (uploaded)*

## Metric

We will evaluate the performance of our classification models using standard metrics such as
**accuracy, precision, recall, and F1-score**.

*Check report for results, analysis and takeaways (uploaded)*

## Contributions
- **Anish Jain**
- **Dhawal Bansal**
