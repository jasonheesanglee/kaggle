![image](https://github.com/jasonheesanglee/kaggle/assets/123557477/ad6bc04f-5f29-47be-9ca2-0caecbec424a)
# Sound Emotion Classification 
23-05-07 ~ 23-06-05<br>

AI Competition held by DACON.<br>
For details of the competition and DACON, please refer to their [webpage](https://dacon.io/competitions/official/236105/overview/description).

This is my second time joining for a Data Science competition.<br>
For this challenge, we are permitted to use only the given dataset (TRAIN, TEST).<br>

There are 5001 .wav format files, 1881 `.wav` format files for each TRAIN and TEST, and `.csv` files with 3 columns `id`, `path`, `label` for each TRAIN and TEST.<br><br>

`id` column contains the name of each `.wav` file.<br>
`path` column contains the path to each `.wav` file.<br>
`label` column contains the label of each `.wav` file ⇒ This label indicates which emotion each `.wav` file is showing.<br><br>

Here are the specific requirements for the competition.<br>

## Requirements
**Background**<br>
Hello everyone! Welcome to the monthly Deacon Speech Emotion Recognition AI Competition.<br>
Speech emotion recognition is a technology that identifies people's emotional states.<br>
Create an AI model that can judge people's emotions based on acoustic data!<br><br>

**Theme**<br>
Developing emotion recognition AI models based on acoustic data<br>

**Description**<br>
Create a model to recognize emotions from acoustic data!<br>

**Evaluation**<br>
Judging Criteria: Accuracy<br>
Primary Evaluation (Public Score): Scored on a randomly sampled 30% of the test data, publicly available during the competition.<br>
Secondary Evaluation (Private Score): Scored on the remaining 70% of the test data, released immediately after the competition ends<br>
The final ranking will be scored from the selected files, so participants must select the two files they would like to have scored in the submission window.<br>
The final ranking is based on the highest score of the two selected files.<br>
(If the final file is not selected, the first submitted file is automatically selected)<br>
The Private Score ranking released immediately after the competition is not the final ranking; the final winners will be determined after code verification.<br>
Determine the final ranking based on Private Score among submitting teams that have complied with the competition evaluation rules.<br><br>

**Notes**<br>
Maximum number of submissions per day: 3 times<br>
Available languages: Python, R<br>
Utilization of evaluation datasets in model training (Data Leakage) is not eligible for awards.<br>
All learning, inference processes, and inference outputs must be based on legitimate code, and submissions obtained by abnormal means will be considered a rule violation if detected.<br>
The final ranking will be scored from the selected files, so participants must select the files they want to be scored in the submission window.<br>
Private rankings released immediately after the competition are not final and winners will be determined after code verification.<br>
Dacon strictly prohibits fraudulent submissions, and if you have a history of fraudulent submissions to Dacon competitions, your evaluation will be restricted.<br><br>

## Notes
It was already D-5 from final submission date when I found out about this competition.<br>
The remaining time was obviously insufficient, but I still decided to give it a try.<br>
The purpose for me to join this competition was simple.<br>
***I heard the acoustic dataset is rare, I want to experience it!***<br><br>

### Approach
For this competition what I have to do is to classify the TEST dataset into 6 different categories based on the TRAIN dataset which the categories are already marked for each data.<br>
Categories are as below.<br>

0: angry<br>
1: fear<br>
2: sad<br>
3: disgust<br>
4: neutral<br>
5: happy<br>

Let me first set up the `ipynb` file for this.<br>
There could be more imported library the more I get into this.<br>

```
# Importing
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from custom_dataset import (CustomDataSet,
                            speech_file_to_array_fn as sfaf,
                            collate_fn,
                            create_data_loader,
                            validation,
                            train)
from transformers import (Wav2Vec2FeatureExtractor,
                          Wav2Vec2Model,
                          Wav2Vec2Config,
                          Wav2Vec2ConformerForSequenceClassification,
                          AutoModelForAudioClassification)
import evaluate
import librosa
import random
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence 
from torch.utils.data import Dataset, DataLoader
from sklearn.tree import DecisionTreeClassifier

from datasets import load_dataset, Audio
import warnings
warnings.filterwarnings(action='ignore')
os.environ['CUDA_LAUNCH_BLOCKING']='1'
```

### Decision Tree Classifier
Below is my theory.<br>
1. Get the numerically expressed frequency data of each .wav TRAIN data
2. Find the trend of each category, and let the Machine Learn.
3. Then apply the trained machine to the TEST dataset.
4. Voila, here is the final submission .csv file.

As I was working alone for this competition, and no one around me had any experience in Acoustic data, I had to deal everything by myself.<br>
Then I have first set each csv into variable and checked the information.<br>

```
train_df = pd.read_csv('./train.csv')
print(train_df.info())

test_df = pd.read_csv('./test.csv')
print(test_df.info())

------------------------------------------

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 5001 entries, 0 to 5000
Data columns (total 3 columns):
 #   Column  Non-Null Count  Dtype 
---  ------  --------------  ----- 
 0   id      5001 non-null   object
 1   path    5001 non-null   object
 2   label   5001 non-null   int64 
dtypes: int64(1), object(2)
memory usage: 117.3+ KB
None

<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1881 entries, 0 to 1880
Data columns (total 2 columns):
 #   Column  Non-Null Count  Dtype 
---  ------  --------------  ----- 
 0   id      1881 non-null   object
 1   path    1881 non-null   object
dtypes: object(2)
memory usage: 29.5+ KB
None
```

To finalize the setting-up process, I have created the folders that will be used later, and set paths as variable.<br>
Also, have made dictionary to easily locate the files in each folders.<br><br>

```
# Folder Locations

dataset = "./"
TRAIN_WAV = dataset + "train/"
TEST_WAV = dataset + "test/"
PREPROCESSED = dataset + "preprocessed_data/"
TRAIN_LABEL_SEP = PREPROCESSED + "train_label_sep/"
WAV_TRAIN_LABEL_SEP = PREPROCESSED + "wav_train_label_sep/"
TEST_LABEL_SEP = PREPROCESSED + "test_label_sep/"
WAV_TEST_LABEL_SEP = PREPROCESSED + "wav_test_label_sep/"


if not os.path.exists(dataset + "preprocessed_data"):
    os.mkdir(dataset + "preprocessed_data")
    
if not os.path.exists(PREPROCESSED + "train_label_sep"):
    os.mkdir(PREPROCESSED + "train_label_sep")

if not os.path.exists(PREPROCESSED + "test_label_sep"):
    os.mkdir(PREPROCESSED + "test_label_sep")

if not os.path.exists(PREPROCESSED + "wav_train_label_sep"):
    os.mkdir(PREPROCESSED + "wav_train_label_sep")

if not os.path.exists(PREPROCESSED + "wav_test_label_sep"):
    os.mkdir(PREPROCESSED + "wav_test_label_sep")
```
```
wav_file_dict = {"train_wav" : TRAIN_WAV,
                "test_wav" : TEST_WAV,
                 "wav_sep" : WAV_TRAIN_LABEL_SEP
                 }
wav_file_locations = {}
for key, value in wav_file_dict.items():
    wav_file_locations[key] = glob.glob(value + "*.wav")
    
csv_file_dict = {"train_label_sep" : TRAIN_LABEL_SEP,
                 "wav_train_label_sep" : WAV_TRAIN_LABEL_SEP
                }

csv_file_location = {}
for key, value in csv_file_dict.items():
    csv_file_location[key] = glob.glob(value + "*.csv")
```

Before properly get started, I failed one attempt, and here is the record of it.<br>

I have searched a bit about on how to deal with the acoustic data.<br>
After searching for a while, I figured out that the `librosa` library was the latest technology to convert the .wav data into numeric values.<br>

`librosa.feature.mfcc` is what I used [link](https://librosa.org/doc/latest/feature.html) <br>

MFCC stands for Mel-Frequency Cepstral Coefficient, which is an algorithms that converts the Acoustic data into characteristics vector (features).<br>
MFCC takes each vector as 1 dimension.<br><br>
Also, for the Machine Learning Method, I decided to use Decision Tree Classifier.<br>
Simply, I thought it was easiest way to make it learn the trends per classification, and apply to the test `.wav` data.<br><br>

In order to work on it, I had to first set up CFG dictionary.<br>
`SR` refers to Sampling Rate<br>
`N_MFCC` refers to number of MFCC (dimensions)<br>
`SEED` refers to the seed for randomization<br><br>

```
CFG = {
    'SR':16000,
    'N_MFCC':32,
    'SEED':42
}
```

Then created a function definition <br>

1. It takes the DataFrame, in my case, train.csv, as an input.
2. It takes the `path` column to get the path of each .wav file.
3. It load the .wav file and separates the `y` and `sr`<br>`y` here refers to the audio
4. Made `mfcc` a variable by applying `librosa.feature.mfcc` to `y`, by using `sr` and `n_mfcc`
5. Appended numerical mean of each vector of `mfcc` into a list `y_feature`
6. Appended the list `y_feature` into list `features` which now is a 2 dimension array
7. Placed list `features` into a DataFrame `mfcc_df`, so that it can be accessible.
8. Returned `mfcc_df`

To make the `train_df` and `test_df` into the form that I can later use in Machine Learning, I have applied the `get_mfcc_feature` to each df.<br>

For future uses, I have assigned the label column of the train_df as train_y<br>
Preprocessing the data is done, and now it is time to let the Machine Learn!<br>

I have first took `DecisionTreeClassifier` and made it variable.<br>
Then trained the model with `train_x` and `train_y`<br>

Predicted test_x with the trained model.<br>
Made the predictions into csv file (codes) and submitted to the competition platform (manually)!<br><br>
![image](https://github.com/jasonheesanglee/kaggle/assets/123557477/3b433660-dca6-4ad4-8bf3-f0cf959d7e24)<br>
I was originally 170 out of 180 competitors when I first submitted yesterday, but now I am ranked 183 out of 194 competitors.<br>

After submitting twice more with different hyperparameters, I realized that the Decision Tree might not be the best-fit solution to this problem.<br>
So I decided to try with other models.<br>
<br>
### Transformer - wav2vec2
**(Failed - Lack of PyTorch Knowledge)** <br>
After trying with DT Classifier, I decided to go for the Transformer model.<br>
It took a bit of research to find which model should I use, and figured out to use the `wav2vec2-large-xlsr-53` by facebook.<br><br>

Here is the codes I modified after searching online.<br>
```
model = Wav2Vec2ForSequenceClassification.from_pretrained('facebook/wav2vec2-large-xlsr-53')

train_inputs = torch.tensor(train_x.values.astype(np.float32)).unsqueeze(1)
test_inputs = torch.tensor(test_x.values.astype(np.float32)).unsqueeze(1)

with torch.no_grad():
    train_logits = model(train_inputs).logits
    test_logits = model(test_inputs).logits
```
This was code was giving me so much identical error, no matter how much value I input into `.unsqueeze`<br>

So I have deleted the `.unsqueeze` itself.<br>
(Later found out that my `mfcc` method already took the `batch` into account and didn’t need to `unsqueeze` it in the first place)<br><br>
After solving this problem, another error was just aroused.<br>
I have googled, asked ChatGPT, but couldn’t find solution to this problem.<br>
```
RuntimeError: Calculated padded input size per channel: (2). Kernel size: (3). Kernel size can't be greater than actual input size
```
Instead of moving forward with `wav2vec2-large-xlsr-53`, they recommended me to go with wav2vec2<br>
It was Pickling Error, which I never encountered before.<br>
I had to ask Google to solve this problem, and they told me to move all the classes into a separate .py file then import it from there.<br>
Here is the full code I tried with, but had no luck moving forward from here.<br><br>
I tried to catch the error by inserting print to every single line in the custom_dataset.py, but it didn’t help.<br>
As I was lack of the torch knowledge, I decided to pause with the Transformation model, and to try with a different, but better understandable model.<br><br>

### Machine Learning Models
By using the preprocess data I made before, I am trying various Machine Learning models.<br>
- **Random Forest Classifier (Success)**
Below is the code, and I got 160th place with a score of 0.37411<br>
```
rf = RandomForestClassifier(n_estimators=50,
                           max_depth=4,
                           min_samples_split=2,
                           max_features=0.85,
                           n_jobs=-1,
                           random_state=CFG['SEED'])
X_train, X_val, y_train, y_val = train_test_split(train_x, train_y, test_size=0.2, random_state=CFG['SEED'])
print(X_train.shape, X_val.shape, y_train.shape, y_val.shape)

rf.fit(train_x, train_y)
print("-- Random Forest --")
print("Train ACC : %.3f" % accuracy_score(y_train, rf.predict(X_train)))
print("Val ACC : %.3f" % accuracy_score(y_val, rf.predict(X_val)))

------------------------------------------------------------------
# Output

-- Random Forest --
Train ACC : 0.423
Val ACC : 0.431

------------------------------------------------------------------

X_test = pd.get_dummies(data=test_x)
pred = rf.predict(X_test)
pred

submission = pd.read_csv(dataset + 'sample_submission.csv')
submission['label'] = pred
submission.to_csv(dataset + "baseline_submission.csv", index=False)
```
Now I will try with other parameters and other models as well.<br><br>

Then, I decided to use all 4 models I learned during the course.<br>
As I felt modifying the hyperparameters every single time was too much of work, I added counter to modify all values.<br><br>

As It was so hard to record all the data this `for` loop tried, and to prevent unexpected kernel shutdown, I added a code to automatically save the data to csv file.<br>(Which later created over 10,000 files)<br>
```
count = 1

for depth in range(1, 6):
  for estimator in range(1, 1000):
    for features in range(1,100):
      rf = RandomForestClassifier(max_depth=depth,
                                  n_estimators=estimator,
                                  min_samples_split=2,
                                  max_features=features/100,
                                  n_jobs=-1,
                                  random_state=CFG['SEED']
                                  )

      dt = DecisionTreeClassifier(max_depth=depth,
                                  min_samples_split=2,
                                  max_features=features/100,
                                  random_state=CFG['SEED']
                                  )

      xgboost = XGBClassifier(max_depth=depth,
                              n_estimators=estimator,
                              grow_policy='depthwise',
                              n_jobs=-1,
                              random_state=CFG['SEED'],
                              tree_method='auto'
                              )

      rf.fit(X_train, y_train)
      dt.fit(X_train, y_train)
      scaler = StandardScaler()
      x_trainScaled = scaler.fit_transform(X_train)
      xgboost.fit(x_trainScaled, y_train)
      Results = pd.DataFrame([["%.3f" % accuracy_score(y_train, rf.predict(X_train)), "%.3f" % accuracy_score(y_val, rf.predict(X_val))],
                              ["%.3f" % accuracy_score(y_train, dt.predict(X_train)), "%.3f" % accuracy_score(y_val, dt.predict(X_val))],
                              ["%.3f" % accuracy_score(y_train, xgboost.predict(X_train)), "%.3f" % accuracy_score(y_val, xgboost.predict(X_val))]],
                            index = ['Random Forest', "Decision Tree", "XG Boost"]
                            )
      get_parameters = pd.DataFrame([rf.get_params(), dt.get_params(), xgboost.get_params()],
                                    index = ['Random Forest', "Decision Tree", "XG Boost"])

      Results = Results.astype(float)
      ComResults = pd.concat([Results, get_parameters], axis=1)

      ComResults.to_csv(dataset + f"Results/Results{count}.csv")
      count += 1
      print(count)
```
After running the loops, I tried to find the best result.<br>
Based on my understandings, the best result is the result that has the highest `accuracy_score` for both `train` and `validation` sets.<br>
Therefore, I read all files and found only the sets that had the highest `accuracy_score` every round.<br><br>
```
result_files = glob.glob(dataset + "Results/*.csv")

max_temp = []
for file in result_files:
    filename = file.split(".")[1]
    filename = filename.split("/")[2]
    df = pd.read_csv(file)
    train_acc = pd.DataFrame(df["0"])
    val_acc = pd.DataFrame(df["1"])
    train_max = train_acc.idxmax()
    val_max = val_acc.idxmax()
    if train_max[-1] == val_max[-1]:
        max_temp.append([filename, train_acc.iloc[train_max[-1]][-1], val_acc.iloc[val_max[-1]][-1]])

max_temp_df = pd.DataFrame(max_temp)
max_temp_df = max_temp_df.rename(columns={0:'filename', 1:"train_acc", 2:"val_acc"})
```
With `max_temp_df`, a set of the best scores of each round, I found the highest `accuracy_score` among them again.<br>
It was to find the top-of-top `accuracy_score` to submit.<br><br>

As the file was the combination of 4 different methods, I had to separate the `fit`, `test` processes per method.<br>
(I later found out that I was not fully understanding `lgbm`, I had to mute the loop for `lgbm`.)<br><br>

To be precautious for the case where there is no top-of-top accuracy_score in this DataFrame, I added codes to select the best resulted validation_accuracy_score.<br>
I could also have added another elif or else statement to find train_accuracy_score, but I had no time to work on it as the deadline was only 30 minutes away.<br><br>
Moreover, as I was not sure if selecting the top-of-top accuracy_score would increase my competition score, I have also added a code to find the bottom-of-bottom accuracy_score.<br><br>

For the final submission, I have used two sets of parameter that resulted top-of-top accuracy_score and one set of parameter that resulted bottom-of-bottom accuracy_score.<br>

Which at last gave me these scores below.<br>

<p align="center">
  <img width="800" alt="image" src="https://github.com/jasonheesanglee/kaggle/assets/123557477/c61a9b31-47fd-4787-8f0a-654d7c8010b8">
  <img width="800" alt="image" src="https://github.com/jasonheesanglee/kaggle/assets/123557477/940e8da2-980b-487d-9e2f-343d49855daa">
  <img width="800" alt="image" src="https://github.com/jasonheesanglee/kaggle/assets/123557477/09522596-086f-4801-a36e-5d116d9bc292">
</p>

Below is my final Public and Private score.<br>

<p align="center">
  <img width="600" alt="image" src="https://github.com/jasonheesanglee/kaggle/assets/123557477/b94d7026-08b2-4a9f-b6b0-d1226f02f662">
  <img width="600" alt="image" src="https://github.com/jasonheesanglee/kaggle/assets/123557477/811a364c-502e-4015-bac8-022abdbc038d">
</p>

I couldn’t even made it to top 10%.<br><br>
However, this was my first time actually completed not only the preprocess process but the Machine Learning Process as well.<br>
I already feel that my skills are improved again, and I feel I will do better next time when I join a new competition.<br>
