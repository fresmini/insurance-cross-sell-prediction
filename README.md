# **Problem Definition:**

(Disclaimer: all information found here is _fictitious_) - A **health insurance company** is attempting to expand their business by branching into the **vehicle insurance business**. The company conducted a survey with about **380,000 of its customers** on their willingness in purchasing this new product. The idea behind the survey is to gather enough data so it would be possible to foresee which customers from another **group of 127,037 people** are more likely to be **interested in auto insurance**. However, due to financial limitations, **the company can only contact 20,000 customers**.

### **_Business Problem Statement / Answers the CEO of the company is seeking:_**
- What percentage of customers interested in purchasing auto insurance will we reach if we call 20000 customers?
- What if or sales team increases its capacity to 40000 calls, what percentage of customers interested in purchasing auto insurance will we reach?
- How many calls would the sales team need to make to reach 80% of interested customers in purchasing auto insurance?

# **Solution Planning:**

### **Proposed Solution:** 
We're building a machine learning model that will **rank customers** based on their estimated probability of being interested in acquiring vehicle insurance. Model input will be a **list of customers** and their known features. Once the predictions are done, the **probabilities will be added to the list** as an extra column. <br>
The ranking model will be hosted on a cloud server so it can be **accessed remotely at any time**.

### **Project Walkthrough:**
- _Collecting data from server:_ all data is stored in a database hosted on a cloud server;
- _Data description:_ check datatypes, null values and initial statistics;
- _Feature engineering:_ create business hypotheses and new features based on the original features;
- _Feature filtering:_ drop any non-helpful information;
- _Exploratory data analysis:_ dig deep into data to gather all possible information and attempt to validate our hypotheses;
- _Data preparation:_ get our data ready for machine learning models;
- _Feature selection:_ choose the most relevant features;
- _Machine learning models:_ train and compare several models and select the best performing ones;
- _Hyperparameter fine tuning:_ find optimal parameters for selected models and settle on the best one;
- _Business performance:_ evaluate how our model performs and answer the questions proposed by the CEO;
- _Model deployment into production:_ make our model available to the company.
