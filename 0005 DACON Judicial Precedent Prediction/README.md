![header_background](https://github.com/jasonheesanglee/kaggle/assets/123557477/521036f5-5da7-48c0-be4c-765966343de2)
# Judicial Precident Prediction
23-05-07 ~ 23-06-05
This competition has just started right after finishing the last competition in Acoustic - Emotion Classification.<br>
As there were no people competing yet, I tried my best to have the result as fast as I can to take the first place.<br>

## Requirement
### Background
  Welcome to the monthly Deacon court decision prediction AI competition.<br>
  The challenge for this Monthly Deacon is to develop an AI that predicts the outcome of a legal case.<br>
  We want you to create a model that utilizes the full power of AI to make accurate verdict predictions.<br>
  This will be an important step in exploring how AI can be effectively utilized in the legal field.<br>

### Topics
  Develop an AI model to predict court decisions<br>

### Description
  The provided dataset contains the case identifiers of US Supreme Court cases and the content of the case.<br>
  You need to develop an AI model that predicts whether the first party or the second party wins in a particular case.<br>

### Evaluation
  **Judging Criteria**: Accuracy<br>
  **Primary Evaluation (Public Score)**: Scored on a randomly sampled 50% of the test data, publicly available during the competition.<br>
  **Secondary Evaluation (Private Score)**: Scored with 100% of the test data, released immediately after the competition ends<br>
  The final ranking will be scored from the selected files, so participants must select the two files they would like to have scored in the submission window.<br>
  Final ranking is based on the highest score of the two selected files.<br>
  (If the final file is not selected, the first submitted file is automatically selected)<br>
  The Private Score ranking published immediately after the competition is not the final ranking; the final winners will be determined after code verification.<br>
  Determine the final ranking based on Private Score among the submitting teams that followed the competition evaluation rules.<br>
<br>
## Approach
We are given three csv files : `train`, `test`, `sample_submission`.<br>
`train` file includes columns below:<br>
`“ID”` : Index of each case<br>
`“first_party”` : Name of the person / party competing against `second_party`<br>
`“second_party”` : Name of the person / party competing against `first_party`<br>
`“facts”` : Brief facts about each case<br>
`“first_party_winner”` : Whether or not `first_party` won the case (0 or 1)<br><br>
As I was now quite familiar with the competitions, I knew what I had to do.<br>
<sub>(231213 edit: I was, and still am junior. Feels like there are still a lot to learn.)</sub><br><br>

1. Preprocess the data
2. Run the Machine Learning Model
3. Submit twice with different `train_accuracy_score` and `validation_accuracy_score` to check whether I should aim for higher accuracy_score or not.

<br>
### Achieving the First Place (Time-Attack)<br>
To achieve the first place, I directly worked on preprocessing.<br>
As it is a competition that we need to find either 0 or 1 for ‘`first_party_winner`’, I simply divided the original data into `trian` and `val` dataset then ran XG Boost.<br><br>

However, a crucial fact has blocked me on the way.<br>
XG Boost does not allow object type in neither `train` nor `val`.<br>
Then I quickly searched how to change them into numerical value.<br>
What I found was LabelEncoder() that groups the identical string and number them.<br><br>
After encoding them into numerical values, I ran XG Boost right away then submitted it.<br><br>
This method has given me the first place, but I wanted to develop more and protect my place.<br>
Moreover, I had to check the  system on the metric.<br><br>
Therefore, I have modified a bit of hyperparameter (=`n_estimator` : 50 ⇒ 25)<br><br>
The result of the first set and the second score were as below.<br><br>
I thought the result will be horrible, but I managed to take the first place and remain for 8 ~ 9 hours.<br>
<p align="center">
  <img width="350" alt="image" src="https://github.com/jasonheesanglee/kaggle/assets/123557477/cb63de4c-7066-425c-8eba-796937181071">
  <img width="325" alt="image" src="https://github.com/jasonheesanglee/kaggle/assets/123557477/32216fa3-82ed-4059-ab05-0586c3ddb1d3">
</p>

