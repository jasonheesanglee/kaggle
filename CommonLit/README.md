![image](https://github.com/jasonheesanglee/kaggle/assets/123557477/bec38521-2723-43f1-a1d7-a304c763bea4)![image](https://github.com/jasonheesanglee/kaggle/assets/123557477/7e29c272-1c61-43f3-9ef2-c152c670f7ae)![image](https://github.com/jasonheesanglee/kaggle/assets/123557477/7c271495-bfff-48de-8663-2e5aebe3b959)
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

###Text Cleansing
- Cleansing the text is a must to run the NLP model effectively.
- I first brought the `text_processor` module I composed while working on the previous competition.<br>It includes:
  - Regex pattern to replace all the words in all forms of brackets.
  - Converting `---n’t` to `--- not` format.
  - Removing `stopwords`.
<p align="center">
  <img width="288" alt="image" src="https://github.com/jasonheesanglee/kaggle/assets/123557477/aeb8bd20-f578-4097-8068-d4354787bf79">
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
