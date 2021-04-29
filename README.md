
# loan-classification
House Loan Prediction (Python)

**Jupyter Notebook: [Model EDA, Training and Testing](https://github.com/vbabashov/loan-classification/blob/main/notebooks/loan_prediction.ipynb)**

### Problem Statement:

Data Source: https://datahack.analyticsvidhya.com/contest/practice-problem-loan-prediction-iii/

Dream Housing Finance company provides mortgage lending solutions to home buyers. Using this partial dataset provided, the company wants to automate the loan eligibility process (in real-time) based on customer information provided while filling online application form. These details are Gender, Marital Status, Education, Number of Dependents, Income, Loan Amount, Credit History and others. To this end, I'll explore three classification models in this notebook. 


### Exploratory Data Analysis (EDA)
***
<p align="center">
  <img src="https://user-images.githubusercontent.com/26305084/116580358-4a255280-a8e1-11eb-8ebd-35378cc2c8c3.jpeg" />
</p>

**Summary** 

- distribution of values for each categorical feature are close given the loan status class labels! 
- credit History seems to have an explanatory power because
- vast majority (95%+) of rejected applications have no credit history
- approximately, 80% of approved applications have a prior credit history

<p align="center">
  <img src="https://user-images.githubusercontent.com/26305084/116582282-31b63780-a8e3-11eb-8063-c6c7a15de8a7.jpeg" />
</p>


Summary:

- Distributions are skewed to the left and right suggesting many outliers
- Average Applicant Income for approved and rejected loans are about the same.
- Those who are approved had either zero or ~ 2500 of Coapplicant income.
- Requested loan amounts for approved and rejected loans are equal on average
- Loan applications are either for 180 or 360 months term.

<p align="center">
  <img src="https://user-images.githubusercontent.com/26305084/116581016-ebaca400-a8e1-11eb-80c8-0c319426a659.jpeg" />
</p>

                                    
Summary

- There is no collinearity between the features
- Highest correlation coefficient is 0.57 between the Loan Amount and Applicant Income which makes sense


### Model Development
***

I'll explore the following three classifiers to predict loan eligibility

    clf1 = LogisticRegression(random_state=1)
    clf2 = RandomForestClassifier(random_state=1)
    clf3 = XGBClassifier(random_state=1)   

First, we need to deal with the class imbalance of 70% (Y) and 30% (N) before model development process.

     Y    0.687296
     N    0.312704
    Name: Loan_Status, dtype: float64




### Results
***


<p align="center">
  <img src="https://user-images.githubusercontent.com/26305084/116587977-00d90100-a8e9-11eb-857f-c21f91d14dd8.jpeg" />
</p>


<p align="center">
  <img src="https://user-images.githubusercontent.com/26305084/116588128-29f99180-a8e9-11eb-865b-9cac6de214db.jpeg" />
</p>



