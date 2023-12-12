# ICR - Identifying Age-Related Conditions
2023-05-12 ~ 2023-08-11

![image](https://github.com/jasonheesanglee/kaggle/assets/123557477/5924192b-61cd-4681-974a-bf96376cff7b)
<p align="center">
    <img width="200" alt="image" src="https://github.com/jasonheesanglee/kaggle/assets/123557477/5d589558-286d-4623-831b-bdf3f9b5d3ee">
</p>

## Context of Mid-Project Presentation
1. The purpose of the competition is to identify the presence or absence of a disease with selected features, explained in the context of performance evaluation.
2. The data is personal information and encrypted, so the features of the data cannot be known.
3. The only thing we could know was whether the patient was sick or not.
4. The baseline code was 0.24.
5. When we first tried to include Greeks, we got 0.0002 (which was an amazing outcome) but got 3.22 for Public Leaderboard.<br> This means it was an overly over-fitted attempt.
6. Sampling was done by undersampling the data with class 0 to match the number of class 1.<br> The reason for undersampling was that the scores were not good when using other sampling, and there was a problem of overfitting.
7. There are about 60 columns.<br>We used feature importance to use 40 important columns because more columns would cause overfitting issues.
8. Combining LGBM and XGBoost resulted in an LB of 0.22.<br>
9. Finally, we used TabPFN to get an LB of 0.15.
10. Until the end of the competition, we can try using TabPFN, pseudo labeling, and Greeks.

## Analysis Methodology
### Defining what to analyze and target variables.<br>
We have 3 data sets: train set, test set, and Greeks.<br>
The training dataset has id, alphabetical column name, and class, with about 600 data.<br><br>
The test set is filled with zeros and is meaningless.<br>
The Greeks provided conditional information about the train.<br><br>
Based on feature engineering (Public LB 0.22)<br>
The sampling undersampled the data with Class 0 to match the number of Class 1s.<br>
We went with undersampling because we got bad scores and overfitting when using another sampling.<br>
We have about 60 columns. I used feature importance to use 40 important columns because, with more columns, you get overfitting issues.<br>


### Feature Engineerings we tried
We created new columns by arithmetic over each column.<br>We thought this would make the important columns more recognizable, but it didn't make much sense.<br>
We changed the important column values into categorical data by breaking them up into specific parts while looking at the chart.<br>The results were worse.<br>
We didn't have enough data to remove the outliers, so we changed the outliers to fall within the normal range as much as possible by looking at the chart.<br>The results were minimal.

###
Tutor's code base, performance checked with the above combination + catboost + lgbm.<br>
Tuning, Scaling = True, PCA = False, other conditions for preprocessing and training are the same.
What was different from the previous trials.<br>
- Unlike other attempts, which simply train the model with the Train Dataset, this attempt includes Greeks to train the model.

When merging Greeks, instead of Alpha: A, Beta: B, Gamma: C, Delta: D, I converted them to ABCD format. 
Merged_Greeks: ABCD format, and after merging, use LabelEncoder to label them, and then right-merge them into the train dataset to be included in X for model training.
→ The reason for doing this is that I thought it would be better to treat the combination of Greek letters for each patient as a group instead of the relationship between each Greek letter.

What I overlooked here is that all the column information in the Test Dataset was 0.0, so without any suspicion, I simply created a Merged_Greeks column in the Test Dataset, zooming all values to 0.0.
Also, the Alpha and Gamma Columns in the Greeks Dataset are just a refinement of the Class information in the Train Dataset, so the CV Score is below 0.1 for most models.

## Throwback
This was the first group work I have done as a Data Scientist.<br>
Even though the competition had a big shake-up for the final result, I learned a lot about Machine Learning Techniques, especially the `classification` techniques.<br><br>
After realizing that the competition would have a big shake-up, I decided to approach this competition personally using a rule-based technique (of course, with permission from my teammates).<br>
In this [Kaggle notebook](https://www.kaggle.com/code/jasonheesanglee/updated-beginner-eda-on-greeks) with a Gold Medal, I have recorded my progress in detail.<br>
I tried to focus on using Greeks (Meta Data) to figure out the final objective.<br>
<br>
