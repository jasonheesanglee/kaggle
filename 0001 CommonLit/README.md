![image](https://github.com/jasonheesanglee/kaggle/assets/123557477/7c271495-bfff-48de-8663-2e5aebe3b959)
 # CommonLit - Evaluating Students Summary
2023-07-13 ~ 2023-10-12

## Competition Overview
- I participated in the CommonLit – Evaluating Student Summary Competition, where the objective was to develop an automated system for evaluating the quality of summaries written by students in grades 3-12.<br>
- This involved building a Machine Learning / Deep Learning model capable of assessing how effectively a student captures the main idea and details of a prompt text, along with the clarity, precision, and fluency of their written summary.<br>
- A comprehensive dataset of real student summaries was given to the participants to train and refine the model.
  <br><br>
- This is the second Natural Language Processing competition I joined after Judicial Precedent Prediction by DACON.<br>
  For the previous competition, the names of the first and second parties and brief facts about multiple cases were given, and the competitors had to predict whether the first party had won the case or not.
- I learned the basics of the NLP technique while competing in the first competition and found my interest in Natural Language Processing.<br>
 Therefore, I have decided to join another NLP competition : CommonLit – Evaluating Student Summary.

## Scoring Metrics
- Submissions are scored using MCRMSE, mean column-wise root mean squared error.<br>Where $`𝑁_𝑡`$ is the number of scored ground truth target columns, and 𝑦 and 𝑦̂ are the actual and predicted values, respectively.<br>
``` math
MCRMSE=\frac{1}{N_t} \sum_{j=1}^{N_t}\left(\frac{1}{n} \sum_{i=1}^n\left(y_{i j}-\hat{y_{i j}}\right)^2\right)^{1 / 2}
```
- We were asked to find content & wording scores.
- The competition host has explained the differences between these two scores in rubric items.
<p align="center">
  <img width="788" alt="image" src="https://github.com/jasonheesanglee/kaggle/assets/123557477/6d47ad14-6017-42ef-9967-0871c534abf8">
</p>

## Dataset
<p align="center">
  <img width="600" alt="image" src="https://github.com/jasonheesanglee/kaggle/assets/123557477/44f564ec-9153-4ed0-bf6f-2548c576c05d">
  <img width="380" alt="image" src="https://github.com/jasonheesanglee/kaggle/assets/123557477/4634e0de-09ee-48cd-b3ec-524737086305">
</p>
- The organizers have provided 2 different CSV files for each train & test data.
  - **summaries**
    - This file includes below columns:
      - student_id :  The ID of the student, Unique to each student.
      - prompt_id :  The ID of the prompt, Unique to each prompt.
      - text :  Student composed summaries.
      - content_score :  Content score, First Target.
      - wording_score :  Wording score, Second Target.
  - **prompts_train**
    - This file includes below columns:
     - prompt_id : The ID of the prompt, Unique to each prompt.
     - prompt_question : The Questions unique to each prompt.<br>
       Students had to compose summaries based on these questions.
     - prompt_title		: Title of the prompt.
     - prompt_text		: Full prompt text

## Exploratory Data Analysis
- Before diving into the competition, it is always recommended to take a look at the data.
<p align="center">
  <img width="756" alt="image" src="https://github.com/jasonheesanglee/kaggle/assets/123557477/6b1560eb-14a4-42b6-9351-d3f7e852a678"> <br>
  <img width="800" alt="image" src="https://github.com/jasonheesanglee/kaggle/assets/123557477/7a4503e3-18b9-4a0c-8f1c-241280ec4598"> <br>
  <img width="800" alt="image" src="https://github.com/jasonheesanglee/kaggle/assets/123557477/9905a6ce-7f6e-4d8f-bcc5-55bc0c4b20d5">
</p>

## Preprocessing

### Text Cleansing
- Cleansing the text is a must to run the NLP model effectively.
- I first brought the `text_processor` module I composed while working on the previous competition.<br>It includes:
  - Regex pattern to replace all the words in all forms of brackets.
  - Converting `---n’t` to `--- not` format.
  - Removing `stopwords`.
<p align="center">
  <img width="800" alt="image" src="https://github.com/jasonheesanglee/kaggle/assets/123557477/4826d742-0829-4aef-bf70-039a5001786d">
</p>

### Spell Checking
- After going through the frequently used words on the word cloud,<br>I realized that there were many occasions where the students had misspelled words.<br>For example, many students have written “Pharaoh” as "Pharoh” or “Pharoah”.
- Considering that the task was to evaluate the summary, I believed that the misspelled words would affect the score.<br>Therefore, I decided not to replace the misspelled words with the correct words.
<p align="center">
  <img width="400" alt="image" src="https://github.com/jasonheesanglee/kaggle/assets/123557477/254be3f4-1c73-41ff-9adb-48d4a593d8c7">
</p>

- However, after finding out that the competition organizer mentioned that spelling is not the evaluation criteria, I have compared different spell-checking tools with Grammarly to find out which tool performs the best.
- Then SymSpellPy caught my attention as it was the only tool that correctly replaced “thenwent” with “then went”.
- The only struggle using SymSpellPy was that it had the slowest processing time among all the tools.<br>Therefore, I have decided to collaborate two tools: PySpellChecker for the misspelled word detecting tool and SymSpellPy as a correction tool.
- This as well was also included in the controller and changed the configuration per different training experiments.

<div align="center">
 
  | **Tool** | **Detected** | **Correctly Replaced** | **Misjudged** |
  | :-: | :-: | :-: | :-: |
  | Grammarly | 16 words | 16 words | Less 1 word |
  | TextBlob | 20 words | 14 words | 2 words | 2 names |
  | PySpellChecker | 18 words | 16 words | 2 words |
  | **SymSpellPy** | **18 words**| **17 words** | **1 name** |
  | AutoCorrect | 15 words | 12 words | 2 words | 1 name |

</div>

### Masking
- After reading some BERT-related research papers and watching videos about Natural Language Processing, I have learned that masking was a huge part of dealing with Natural Language data.
- When I tried to implement masking, I found that the tokens used for masking differed from model to model, although most of the models I tried were BERT-descendants.<br>Therefore, I decided only to include masking when using `debertav3base`.
- I have separated the masking module into two and let each module mask Keywords and Frequently-appeared words.
- However, when I asked for feedback from the Kaggle community, they told me that masking doesn’t work this way.
- Then, I tried to learn more and implemented in the other code, which I will introduce later in this document.

## Train & Infer

### MobileBERT
- As the purpose of this competition is to enhance the teaching and scoring experience of the teachers, I believed making this model lightweight is the key.<br>By implementing MobileBERT, the model would run on individual mobile devices; teachers at isolated locations with poor or no internet connection could also use this.<br>It would not need a simultaneous connection to the internet.
- Unfortunately, I had to bring a training code from the community, as the one I composed didn’t run successfully.
- When learning about Deep Learning modeling, one thing I learned was that, in many cases, it is better to have more data for the model to train on.
- Therefore, for the occasions where I turned the spellchecking module on, I made the model to be trained on the non-spellchecked data and the spellchecked data.<br>Which later, I found out that this tactic made the model overfitted to the training data.
- However, by doing so, I had a chance to look closer into the codes and to understand the logic.
- Then, I tried to implement MobileBERT to the code by simply changing the name of the model in `.from_pretrained()`, but it was giving an out-of-range Public LB score.<br><br>

#### Knowledge Distilation
- I realized something wrong when I saw the score and thought this is not the way how MobileBERT is implemented in the code.
- I read the [research paper of MobileBERT](https://arxiv.org/pdf/2004.02984.pdf) to find the solution.<br>Then, I realized that a teacher model needed to be taught before implementing MobileBERT, and the teacher model needs to teach MobileBERT.
- At first, I didn’t really understand the logic behind it.<br>Still, then I realized that the teacher model was the one containing the information, and MobileBERT only took the necessary information from the previous model (teacher model).
- The Public Leaderboard score after properly implementing the MobileBERT has been better than before, but it was still unsatisfying.<br>Which led me to revert back to other BERT-descendant models.

<p align="center">
  <img width="700" alt="image" src="https://github.com/jasonheesanglee/kaggle/assets/123557477/5377adc4-4796-438e-95c1-4d95f5b6b155">
</p>


### Other models
- Once the MobileBERT moved out of my scope, I wanted to try all the other available models in this [Dataset](https://www.kaggle.com/datasets/kozodoi/transformers).<br>Which are:

<div align="center">

  | Models | Models |
  | :-: | :-: |
  | bert-base-uncased | roberta-base |
  | bert-large-uncased | roberta-large |
  | distilroberta-base | t5-base |
  | distilbert-base-uncased | t5-large |
  | electra-base-discriminator | xlnet-base-cased |
  | bart-base | xlnet-large-cased |
  | bart-large | albert-large-v2 |

</div>

#### BART & T5
- I have tried all the non-large models above, but BART & T5.<br>When I tried to implement BART and T5 as I have done for other models, a large number of error messages occurred.<br>Later, I learned from the T5 research paper that unlike other encoder-only models listed above, T5 is an encoder-decoder model, and so is BART.<br>Therefore, I had to construct from scratch.
However, when I realized that I had to build a new set of code.<br>As it was already 3 days to the deadline, and I could only give up using BART and T5 for this time.

#### All the other encoder-only models.
- For other models than BART and T5, I have built a list to store the model location and, retrieved the model's name & location within the train & infer code structure I used for DeBERTa-v3-Base.<br>I understand that it wouldn’t be the best way to try out multiple models in a single structure, but as aforementioned, I did not have enough time to build again from ground zero.<br>After several experiments on Google Colab, I have figured out that `DistilRoBERTa-base` scored the best among the models.<br>Therefore, I have selected the DistilRoBERTa-version of the notebooks as one of my final submissions.

#### LightGBM
- After getting the result with NLP Deep Learning models, I used LightGBM to improve the score by feature engineering, as introduced in this [notebook](https://www.kaggle.com/code/tsunotsuno/updated-debertav3-lgbm-with-spell-autocorrect/notebook).<br>The original composer of the notebook has found the optimal hyperparameters through their experiment.<br>Still, as I have combined some columns on the way, I needed to find a better hyperparameter that would suit my version of the work.
- To do so, I added Optuna on LightGBM and ran multiple experiments on Google Colab.<br>This made me think of sending the result to my email, then auto shut-down the session (to save some quota).<br>Details on this are explained in this [notebook](https://www.kaggle.com/code/jasonheesanglee/email-notification-colab-session-autoshutdown).<br>However, it seemed like it didn’t work since the RMSE got higher (worse) than before running LightGBM.<br>Then, I decided not to fix onto one set of hyperparameter values but let the Optuna do its work for the hidden test data.

<p align="center">
  <img width="500" alt="image" src="https://github.com/jasonheesanglee/kaggle/assets/123557477/b7ea5264-4b78-4e51-bddd-4c2531a63191">
</p>

## 1st Place Solution
- Surprisingly or not, the first-place holder turned out to be using the same baseline that we were using.
- The differences between our solution and the first-place solution are:<br>
  1. They have barely modified the logic of the model.
  2. They not only have used the provided four prompts and generated even more by effectively using Large Language Model.
  3. They have excluded using LightGBM and solved the problem solely with DeBERTa-v3-Large.
- Per their discussion, they have spent more time generating new prompts and summaries with LLM than tuning the DeBERTa model.<br>I believe this was quite a smart move, as the competition overview – context has already mentioned utilizing LLM.
  - Even though I don’t think this is not the way the organizer expected the LLM to be used, they were the only team that actually used LLM for this competition.
<p align="center">
  <img width="500" alt="image" src="https://github.com/jasonheesanglee/kaggle/assets/123557477/e75be711-8dd4-41d1-8768-a9ec91d5e07f">
  <img width="800" alt="image" src="https://github.com/jasonheesanglee/kaggle/assets/123557477/33ab8dcd-b84b-43ac-8a5e-2cd629df4e16">
</p>

## Result & Relection
- I have teamed up with the study-group members who study NLP together.<br>As we were placed at the top 143 in the Public Leaderboard, we believed that we could make it to the Bronze medal when the Private Leaderboard was published.<br>Therefore, we tried our best until the very end to make minor changes to our existing solution.
  <img width="1140" alt="image" src="https://github.com/jasonheesanglee/kaggle/assets/123557477/2eb58c4d-5e2a-4961-8cff-3942786f052d">
- When the Private Leaderboard was published, we were pretty shocked with the result.
  <img width="1139" alt="image" src="https://github.com/jasonheesanglee/kaggle/assets/123557477/a7bd18c5-4de0-4c8d-bcdf-38a59645f668">
- We were expecting at least the top 170, which is the lower limit for the bronze medal.<br>After being depressed for 30 minutes, we started analyzing what have we done wrong.<br>There were quite some reasons we have detected.

- **First** : The proportion of the Public Data.<br>
  The organizer stated that the test data provided to the competitors is only 13% of the whole test data.<br>Overfitting was unavoidable as we focused on getting a better score on the Public Leaderboard.
  <p align="center">
    <img width="872" alt="image" src="https://github.com/jasonheesanglee/kaggle/assets/123557477/edf067d2-f3b3-4523-bc73-24d1a9268ddf">
  </p>
- **Second** : Using SymSpellPy for spellchecking.<br>I wouldn’t say this is totally wrong, but most competitors used PySpellCheck and AutoCorrect.<br>It was mainly AutoCorrect.
- **Third** : Focused too much on MobileBERT.<br>This is more likely to a self-reflection.<br>I have spent about 2 weeks implementing MobileBERT.<br>Obviously, it has been a great practice for me, but that was it.<br>It didn’t help in scoring.
- **Fourth** : Need more and better modeling techniques.<br>It would have been better if we could also try T5 and BART, as no one else was trying that measure.<br>Only if we had a better modeling technique, we could have also tried these two models.
