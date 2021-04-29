# loan-classification
House Loan Prediction (Python)



- **Jupyter Notebook: [Model EDA, Training and Testing](https://github.com/vbabashov/loan-classification/blob/main/notebooks/loan_prediction.ipynb)**

### Problem Statement:

Data Source: https://datahack.analyticsvidhya.com/contest/practice-problem-loan-prediction-iii/

Dream Housing Finance company provides mortgage lending solutions to home buyers. Using this partial dataset provided, the company wants to automate the loan eligibility process (in real-time) based on customer information provided while filling online application form. These details are Gender, Marital Status, Education, Number of Dependents, Income, Loan Amount, Credit History and others. To this end, I'll explore three classification models in this notebook. 


### 1.EDA
***

   ![proportion](https://user-images.githubusercontent.com/26305084/116580358-4a255280-a8e1-11eb-8ebd-35378cc2c8c3.jpeg)

Summary Distribution of values for each categorical feature are close given the Loan Status class labels! However, Credit History seems to have an explanatory power because

- vast majority (95%+) of rejected applications have no credit history
- approximately, 80% of approved applications have a prior credit history
