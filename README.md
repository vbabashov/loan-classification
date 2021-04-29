
# loan-classification
House Loan Prediction (Python)

**Jupyter Notebook: [Model EDA, Training and Testing](https://github.com/vbabashov/loan-classification/blob/main/notebooks/loan_prediction.ipynb)**

### Problem Statement:

Data Source: https://datahack.analyticsvidhya.com/contest/practice-problem-loan-prediction-iii/

Dream Housing Finance company provides mortgage lending solutions to home buyers. Using this partial dataset provided, the company wants to automate the loan eligibility process (in real-time) based on customer information provided while filling online application form. These details are Gender, Marital Status, Education, Number of Dependents, Income, Loan Amount, Credit History and others. To this end, I'll explore three classification models in this notebook. 


### Exploratory Data Analysis (EDA)
***
   ![proportion](https://user-images.githubusercontent.com/26305084/116580358-4a255280-a8e1-11eb-8ebd-35378cc2c8c3.jpeg)

**Summary** 

- distribution of values for each categorical feature are close given the loan status class labels! 
- credit History seems to have an explanatory power because
- vast majority (95%+) of rejected applications have no credit history
- approximately, 80% of approved applications have a prior credit history

    ![density](https://user-images.githubusercontent.com/26305084/116582282-31b63780-a8e3-11eb-8063-c6c7a15de8a7.jpeg)

Summary:

- Distributions are skewed to the right suggesting many outliers
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
