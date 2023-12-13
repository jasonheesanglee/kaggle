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
