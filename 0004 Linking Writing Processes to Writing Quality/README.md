![image](https://github.com/jasonheesanglee/kaggle/assets/123557477/ef5ea91d-88b8-4805-8702-f8510b8049b4)
# Linking Writing Processeses to Writing Quality
231003 ~ 240110

I was planning to stop joining competitions after CommonLit for job searching, but this competition looked very interesting and had to take a look.


Here are the brief information of the competition.

## Requirements
  I have skipped some part of the competition information as you can find them in the [Kaggle Website](https://www.kaggle.com/competitions/linking-writing-processes-to-writing-quality).
  - **Goal of the Competition**<br>
    The goal of this competition is to predict overall writing quality.<br>
    Does typing behavior affect the outcome of an essay?<br>
    You will develop a model trained on a large dataset of keystroke logs that have captured writing process features.<br>
    Your work will help explore the relationship between learners’ writing behaviors and writing performance, which could provide valuable insights for writing instruction, the development of automated writing evaluation techniques, and intelligent tutoring systems.<br><br>
  - **Context**<br>
    It’s difficult to summarize the complex set of behavioral actions and cognitive activities in the writing process.<br>
    Writers may use different techniques to plan and revise their work, demonstrate distinct pause patterns, or allocate time strategically throughout the writing process.<br><br>
    
  - **Evaluation**<br>
    We use the Root Mean Squared Error to score submissions, defined as:<br>
    ```math
        RMSE=\left(\frac{1}{n} \sum_{i=1}^n\left(y_i-\hat{y}_i\right)^2\right)^{1 / 2}
    ```
    where $`\hat y _i`$ is the predicted value and $`y_i`$ is the original value for each instance $`i`$ over $`n`$ total instances.<br>

## Keystroke Related
  - **Data Collection Procedure**
    Participants of this project were hired from Amazon Mechanical Turk, a crowdsourcing platform.<br>They were invited to log onto a website that housed a demographic survey, a series of typing tests, an argumentative writing task, and a vocabulary knowledge test.<br>Participants were required to use only computers with a keyboard.<br><br>
    During the argumentative writing task, participants were asked to write an argumentative essay within 30 minutes in response to a writing prompt adapted from a retired Scholastic Assessment Test (SAT) taken by high school students attempting to enter post-secondary institutions in the United States.<br>
    To control for potential prompt effects, four SAT-based writing prompts were used and each participant was randomly assigned one prompt.<br>
    Prior to the writing task, instructions were presented on the integral components in an argumentative essay (e.g., introduction, position, reasons and evidence) along with descriptions of their functions in argumentation.
    <br>The instructions pages also introduced a set of rules for the writing task.
    <br>These include that participants should write an essay of at least 200 words in 3 paragraphs and that they should not use any online or offline reference materials.<br>A screenshot of the writing task page is presented below.<br><br>
    
<p align="center">
  <img width="600" alt="image" src="https://github.com/jasonheesanglee/kaggle/assets/123557477/79f7353f-8379-408a-865b-bd7230bf1bc7">
</p>

  - **Keystroke Logging Program**
    To collect participants' keystroke information during the argumentative writing task, a keystroke logging program was written in vanilla JavaScript and was embedded in the script of the website built for this project.<br>
    The program listened to the keystroke and mouse events in the designated text input area using JavaScript’s addEventListener method.<br>
    It also logged the time stamp and cursor position information for each keystroke or mouse operation.<br>
    The table below provides an example output of keystroke logging information reported by the program.<br><br>
    <p align="center">
      <img width="800" alt="An Example Dataframe of Keystroke Logging Information" src="https://github.com/jasonheesanglee/kaggle/assets/123557477/10d5af02-1ba6-42aa-a00d-8445db798c1c">
    </p>

## Intro
The reason I found this competition interesting was that the data the organizer has prepared for, was a sequence of keyboard & mouse input data.<br>
When I first went through the data in the early phase of the competition, complete words could be found from the training data.<br>
For example, a part of the `down_activity` were : `Shift`, `b`, `e`, `f`, `o`, `r`, `e`, `Space`.<br>
However, for some reason, the organizers have replaced the files with anonymized alphabet characters.<br>
> To prevent reproduction of the essay text, all alphanumeric character inputs have been replaced with the ***"anonymous"*** character `q`
