![header_background](https://github.com/jasonheesanglee/kaggle/assets/123557477/521036f5-5da7-48c0-be4c-765966343de2)
# Judicial Precident Prediction
23-05-07 ~ 23-06-05<br>
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
</p><br>
While enjoying the glory of being the first place, I suddenly realized that this is not just a classification competition.<br>
But this was a NLP (Natural Language Processing) competition.<br><br>

There were only 2478 rows in the train dataset, and 1240 rows in the test dataset.<br><br>

I decided to join this competition aiming to master my skills for classification…<br>
But now I am destined to try NLP.<br>

I have heard about GPT, BERT, but am not yet familiar with them.<br>
So, I went on Google to first search about NLP, understood that what I am solving is a NLP Classification Problem.<br><br>

This is the first [article](https://pseudo-lab.github.io/Tutorial-Book/chapters/NLP/Ch1-Introduction.html) I found online, and it well informs different models and where each of them are strong at.<br><br>

It seemed to me that T5 and GPT were made for text-to-text conversations, while BERT seemed more suitable for predictions, classifications and such.<br>

So I have searched a bit more to learn more about BERT.<br>
As this was my first time actually looking into DeepLearning Models (except for the torch trial during the last competition), I thought there was one set of simple codes to activate the DeepLearning algorithms to work. <br>
But it was a huge mistake to assume it that way.<br><br>

There were hundreds of different models using BERT, and I had to choose which one to use.
I first looked into one [scholarly article](https://manuscriptlink-society-file.s3-ap-northeast-1.amazonaws.com/kips/conference/ack2022/presentation/KIPS_C2022B0013.pdf), that researched into Judicial Precedent Analysis with SBERT (Sentence BERT).<br><br>
After reviewing this article, I understood that I could also use BERT for preprocessing data.<br>
Then, I found this [article](https://www.tensorflow.org/text/tutorials/classify_text_with_bert?hl=ko) from TensorFlow about Text Classification using BERT, [SentenceTransformers](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).<br><br>
Had to do another set of DataFrame preprocessing to fit into model.<br>
I tried my best to understand, and implement it to my jupyter notebook.
Below is the first implementation, which I failed.
```
## BERT Failed

train_facts = pd.DataFrame(train_cleansed['facts'])
test_cleansed = pd.DataFrame(test_cleansed['facts'])
model_input_df = dlc.bert_tokenizer(train_facts, 'facts')

df_part_1, df_part_2, df_part_3, df_part_4, df_part_5, df_part_6, df_part_7, df_part_8, df_part_9, df_part_10, df_part_11, df_part_12, df_part_13, df_part_14, df_part_15, df_part_16, df_part_17, df_part_18, df_part_19, df_part_20, df_part_21, df_part_22, df_part_23, df_part_24, df_part_25, df_part_26 = so.df_divider(train_cleansed, 'facts')

df_part_1 = pd.DataFrame(df_part_1)
df_part_2 = pd.DataFrame(df_part_2)
df_part_3 = pd.DataFrame(df_part_3)
df_part_4 = pd.DataFrame(df_part_4)
df_part_5 = pd.DataFrame(df_part_5)
df_part_6 = pd.DataFrame(df_part_6)
df_part_7 = pd.DataFrame(df_part_7)
df_part_8 = pd.DataFrame(df_part_8)
df_part_9 = pd.DataFrame(df_part_9)
df_part_10 = pd.DataFrame(df_part_10)
df_part_11 = pd.DataFrame(df_part_11)
df_part_12 = pd.DataFrame(df_part_12)
df_part_13 = pd.DataFrame(df_part_13)
df_part_14 = pd.DataFrame(df_part_14)
df_part_15 = pd.DataFrame(df_part_15)
df_part_16 = pd.DataFrame(df_part_16)
df_part_17 = pd.DataFrame(df_part_17)
df_part_18 = pd.DataFrame(df_part_18)
df_part_19 = pd.DataFrame(df_part_19)
df_part_20 = pd.DataFrame(df_part_20)
df_part_21 = pd.DataFrame(df_part_21)
df_part_22 = pd.DataFrame(df_part_22)
df_part_23 = pd.DataFrame(df_part_23)
df_part_24 = pd.DataFrame(df_part_24)
df_part_25 = pd.DataFrame(df_part_25)
df_part_26 = pd.DataFrame(df_part_26)


embedded_df_1 = dlc.auto_tokenizer(train_cleansed, 'facts')
embedded_df_1
embedded_df_1 = embedded_df_1.rename(columns={0:'facts_berted'})
embedded_df_1.to_csv('./embeddings/facts_embedded.csv', index=False)
```
<br>
The main reason of the failure was that I did not have enough understanding of the NLP Preprocessing process.<br>
As per my current understanding, NLP Preprocessing has to go through 3 main steps.<br>
1. **Encoding**
2. **Tokenizing**
3. **Embedding**
<br>However, the steps I followed on the first trial was as below.

1. **Encoding**
2. **Tokenizing**
3. **Encoding**
4. **Tokenizing**
5. **Embedding**

I went through Encoding and Tokenizing process again, not knowing that I have already done it.<br>
This was why the attempt was keep failing, and I thought the problem was the amount of the data I had.<br>And that is why I tried to separate the rows into 26, which seemed like it wouldn’t cause any trouble.<br><br>
```
def auto_tokenizer(df, column_name):
    '''
    입력한 df의 문자 벡터를 수치화 합니다.

    :param df:문자 벡터를 수치화하고 DataFrame
    :return:
    '''
    bert_model = 'bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(bert_model)
    model = AutoModel.from_pretrained(bert_model)

    ei_total_list = []
    for i in tqdm(df[column_name]):
        ei_list = []
        for j in range(1):
            encoded_input = tokenizer(i, padding='max_length', max_length=512, truncation=True, return_tensors='pt')
            with torch.no_grad():
                model_output = model(**encoded_input)
            sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
            sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
            ei_list.append(sentence_embeddings)
        ei_total_list.append(ei_list)
    df_1 = pd.DataFrame(ei_total_list)
    return df_1

first_party_berted = dlc.auto_tokenizer(train_cleansed, 'first_party')
second_party_berted = dlc.auto_tokenizer(train_cleansed, 'second_party')
facts_berted = dlc.auto_tokenizer(train_cleansed, 'facts')

test_second_party_berted = dlc.auto_tokenizer(test_cleansed, 'test_second_party_berted')
test_second_party_berted = test_second_party_berted.rename(columns={0:'second_party_berted'})
test_second_party_berted.to_csv('./embeddings/test_second_party_berted.csv', index=False)

test_facts_berted = dlc.auto_tokenizer(test_cleansed, 'test_facts_berted')
test_facts_berted = test_facts_berted.rename(columns={0:'test_facts_berted'})
test_facts_berted.to_csv('./embeddings/test_facts_berted.csv', index=False)

# This process took so long, I had to export to csv and import it for later uses :)

test_first_party_berted = pd.read_csv('./embeddings/test_first_party_berted.csv')
test_second_party_berted = pd.read_csv('./embeddings/test_second_party_berted.csv')
test_facts_berted = pd.read_csv('./embeddings/test_facts_berted.csv')
```
***I was glad, it worked so well!!***
<br>
Below was the output for the next step.<br>
<p align="center">
  <img width="350" alt="image" src="https://github.com/jasonheesanglee/kaggle/assets/123557477/6bee32b9-c337-4172-bb76-a6f4758b1972">
</p><br>
However, I realized that most of the Machine Learning Models wouldn’t take this form as an input.<br>
***But what is TENSOR?!***<br>
After few minutes of research, I found out that these are multi-dimensional frame.<br>
I was suffering to convert these into 2-dimensional frame, but it was very confusing and harder than I thought.<br><br>
This was the first attempt on converting this 4-dimensional frame into 2-dimension.<br>

```
def tensor_2_2d(df, n):
    df_renamed = df.rename(columns={0: 'tbd', 1: 'hmm'})
    tensors = pd.DataFrame(df_renamed.groupby(by="tbd"))
    tensors1 = tensors[1]
    tensors1_df = pd.DataFrame(tensors1)
    tensors1_1 = pd.DataFrame(tensors1_df[1][n])
    target_name_temp = tensors1_1['tbd']
    target = tensors1_1['hmm']
    target_name_df = pd.DataFrame(target_name_temp)
    target_name = target_name_df.iat[0, 0]
    target_df = pd.DataFrame(target)
    target_df = target_df.reset_index()
    target_df = target_df.drop(columns='index')
    target_final_df = target_df.rename(columns={'hmm': target_name})

    temp = []
    for i in tqdm(range(len(target_final_df))):
        units = ['[', ']', 'tensor', '(', ')']

        for unit in units:
            s = str(target_final_df[target_name][i]).replace(unit, '')
        temp.append(s)

    temp_dict = {target_name: temp}

    final_df = pd.DataFrame(temp_dict)

    return final_df
```
It didn’t work well, and I still couldn’t use it as Xs and y.<br>
Then I thought, why don’t I simply convert these into string, process it, and convert them back to numerics.<br>
Here is the code I came up with, and it worked well!<br>

```
def tensor_separator(df, column_name):
    to_replace = ["t e n s o r", "[", "]", "(", ")", " "]
    full_tensor_list =[]
    for tensor in tqdm(df[column_name]):
        # tensor = tensor.astype(str) ## if tensor != str
        tensor = " ".join(tensor)
        list_per_row = []
        for i in to_replace:
            tensor = tensor.lower()
            tensor = tensor.replace(i, "")
        tensor_list = tensor.split(",")
        list_per_row.extend(tensor_list)
        full_tensor_list.append(list_per_row)
    full_tensor_df = pd.DataFrame(full_tensor_list)

    return full_tensor_df
```
`tensor = tensor.astype(str)`, `if tensor != str`<br>
I put this line of code just incase if each of the tensor’s data type was not read in str. <br>
But its type was str, so I skipped activating it.<br>
<p align="center">
  <img width="350" alt="image" src="https://github.com/jasonheesanglee/kaggle/assets/123557477/664d2183-0168-4044-9a79-5917efd07b79">
  <img width="350" alt="image" src="https://github.com/jasonheesanglee/kaggle/assets/123557477/16dfd996-67c8-4520-b726-2ebe0cf334b5">
</p><br>
After making these tensors into 2D, I could finally insert them into ML models!<br>
Also, unlikely to my previous attempts on hyperparameter tuning with for loop, I have used optuna, which I just learned from the course, to find the optimal parameters.<br>
Here is the output.<br>

```
X = to_be_X.drop(columns='first_party_winner')
y = pd.DataFrame(to_be_X['first_party_winner'])

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

test_x = to_be_test_x

def lgb_objective(trial):
    lgb_params = {
        'application': 'binary',
        'max_depth': -1,
        'metric': 'binary_logloss',
        'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt',  'dart']),
        'num_leaves': trial.suggest_int('num_leaves', 10, 2000),
        'lambda' : trial.suggest_float('lambda', 0.01, 0.5),
        'num_iteration': 500,
        'n_jobs': -1,
        'learning_rate': trial.suggest_float('learning_rate', 0.05, 0.1),
        'feature_fraction': trial.suggest_uniform('feature_fraction', 0.7, 0.9),
        'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.1, 0.8),
        'random_state': 42
    }
    
    lgb_model = lgb.LGBMClassifier(**lgb_params)
    lgb_model.fit(X_train, y_train)
    lgb_preds = lgb_model.predict(X_val)
    
    return accuracy_score(y_val, lgb_preds)

xgb_study = optuna.create_study(direction='minimize')
xgb_study.optimize(xgb_objective, show_progress_bar=True)

lgb_study = optuna.create_study(direction='minimize')
lgb_study.optimize(lgb_objective, n_trials=5, show_progress_bar=True)
```
<br>
Below is the output for validation set.<br>
<p align="center">
  <img width="430" alt="image" src="https://github.com/jasonheesanglee/kaggle/assets/123557477/56cd92bb-17b4-4e8f-8084-c14d0f971cd2">
  <img width="421" alt="image" src="https://github.com/jasonheesanglee/kaggle/assets/123557477/35e67a38-be58-4b55-ad34-69383147144e">
  <img width="177" alt="image" src="https://github.com/jasonheesanglee/kaggle/assets/123557477/657052d8-dde7-4ee1-aabf-c91a4136b97d">
</p>
<br>
Below is the output for test set.<br>
<p align="center">
  <img width="850" alt="image" src="https://github.com/jasonheesanglee/kaggle/assets/123557477/894e345f-b043-4038-aeda-b2db60c4bf23">
  <img width="366" alt="image" src="https://github.com/jasonheesanglee/kaggle/assets/123557477/3aa7a86c-065a-4752-83a7-ae4bf2393efd">
</p><br>
I have exported the final submission csv for 50+ times , but no score could exceed the result of my test_submission `0.5161`.<br>
While feeling helpless and being lazy on tuning hyperparameter for each model, I came up with a crazy idea.<br><br>

It was to run one `optuna` code to run all Classification models to get the best result.<br>
Simultaneously to when I thought of this idea, I thought of what we learned during the course.<br>
We learned that some parameters exists to regularize the model, give limits to the model so that it wouldn’t go overfitted.<br><br>

XGBoost for example, we were taught that when our model takes high `n_estimators`, it usually is better to have low `col_sample_bytree` / `col_sample_bylevel`.<br>
Then I thought, why don’t I simply create the instances for the parameters not to go beyond certain points.<br>

Even though I haven’t added the `cat_boost` yet, the `function` itself is working well.<br>

```
600 line code...
```
The limitation on this is that I am still not so clear about each parameter, and what their functions are.<br>
I have studied about how `optuna` works, and how each parameters work on training.<br>
However, I realized that `optuna` usually only helps a bit on improving the model performance; it wouldn’t boost up the model performance.<br>
As per many previous attempts of other people, optuna increase the score for only 0.001 to 0.01.<br>
> What impacts more on the model performance is how the dataset is preprocessed.<br>

Therefore, I tried to give each party name mask for bert.<br>
As there are many variances use cases of names in facts column, I separated the people’s name into Family and First name.<br>
Also, as some parties are governments, I had to give them a separate mask.<br><br>
I tried to run this code on Google Colab and Kaggle notebook using `gpu`, it wasn’t so successful and depleted all the limits.<br>
So I could only try more on local environment without `gpu`.<br>
However, as expected, estimated completion time was 999+ hours.<br><br>

It was weird to see such estimation as the dataset is not that extremely huge.<br>
> There should be something wrong with the code.<br>

While searching for the not detected error in the code, I had to start another competition (ICR - Identifying Age-Related Conditions) as the course I am taking asked me to.<br>
I will create another page for it. A lot to talk.<br><br>
Anyways, while working on this new competition, I learned about `Autogluon`.<br>
By using this `Autogluon`, it tries all possible methods (including ML and DL) on the Train dataset and predicts on Test dataset.<br>
So I decided to give it a try on this competition and not waste any submission chances.<br>
I have done 3 tries with `Autogluon`. (The length of each codes is surprisingly short)
