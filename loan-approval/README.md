# Loan Eligibility Checker

The Loan Eligibility Checker project aims to develop a machine learning model (classification) to predict whether a user will be approved or not for a loan. By analyzing a dataset of customer information provided during the online application process, the project seeks to automate the loan eligibility process and target specific customer segments.

## Overview

The loan process involves several steps from application submission to approval or rejection like application submission, documentation review, credit check, underwriting, decision-making, loan disbursement, and repayment. The criteria when assesing a borrower's eligibility for a loan vary depending on the type of loan and the lender's policies but commom considerations includes credit history, income, employment status, debt-to-income ratio, collateral, loan purpose, amount, and terms.

<p align="center">
<img src="https://github.com/guilhermegarcia-ai/ml-classification-models/assets/62107649/620ca12c-dfbb-41d8-a775-0d1dee2d1c15" width=600 height=300>
</p>

## The Objective

The primary objective of this project is to develop a robust classification model capable of accurately predicting loan eligibility based on customer details. By analyzing historical customer data and identifying significant characteristics, the model aims to automate and streamline the loan approval process for the company.

## The Data

The dataset contains 614 records detailing customer attributes collected during the online application process. It includes the following features:

- **Loan_Status:**	Status of loan, is the prediction target.
- **Loan_ID:** Loan reference number.
- **Gender:** Applicant gender (Male or Female).
- **Married:** Applicant marital status (Married or not married).
- **Dependents:** Number of family members.
- **Education:** Applicant education/qualification (graduate or not graduate).
- **Self_Employed:** Applicant employment status (yes for self-employed, no for employed/others).
- **ApplicantIncome:** Applicant's monthly salary/income.
- **CoapplicantIncome:** Additional applicant's monthly salary/income.
- **LoanAmount:** Loan amount.
- **Loan_Amount_Term:** The loan's repayment period (in days).
- **Credit_History:** Records of previous credit history (0: bad credit history, 1: good credit history).
- **Property_Area:** The location of property (Rural/Semiurban/Urban).

## Business Impact

A successful model can have significant implications for the company's loan approval process:

- **Efficiency:** Automate and streamline the loan eligibility process, reducing manual effort and processing time.
- **Targeting:** Identify specific customer segments eligible for loan approval, enabling targeted marketing and customer acquisition strategies.
- **Risk Management:** Mitigate the risk of loan defaults by accurately assessing customer eligibility based on predictive modeling.
