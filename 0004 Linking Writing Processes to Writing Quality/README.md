![image](https://github.com/jasonheesanglee/kaggle/assets/123557477/ef5ea91d-88b8-4805-8702-f8510b8049b4)
# Linking Writing Processeses to Writing Quality
231003 ~ 240110

I was planning to stop joining competitions after CommonLit for job searching, but this competition looked very interesting and had to take a look.


Here are the brief information of the competition.
## Requirements
  - Goal of the Competition<br>
    The goal of this competition is to predict overall writing quality.<br>
    Does typing behavior affect the outcome of an essay?<br>
    You will develop a model trained on a large dataset of keystroke logs that have captured writing process features.<br>
    Your work will help explore the relationship between learners’ writing behaviors and writing performance, which could provide valuable insights for writing instruction, the development of automated writing evaluation techniques, and intelligent tutoring systems.<br><br>
  - Context<br>
    It’s difficult to summarize the complex set of behavioral actions and cognitive activities in the writing process.<br>
    Writers may use different techniques to plan and revise their work, demonstrate distinct pause patterns, or allocate time strategically throughout the writing process.<br> Many of these small actions may influence writing quality. Even so, most writing assessments focus on only the final product.<br>
    Data science may be able to uncover key aspects of the writing process.<br><br>
    Past research explored a number of process features related to behaviors, such as pausing, additions or deletions, and revisions.<br>
    However, previous studies have used relatively small datasets.<br>
    Additionally, only a small number of process features have been studied.<br><br>
    Competition host Vanderbilt University is a private research university in Nashville, Tennessee.<br>
    It offers 70 undergraduate majors and a full range of graduate and professional degrees across 10 schools and colleges, all on a beautiful campus—an accredited arboretum—complete with athletic facilities and state-of-the-art laboratories.<br>
    Together with The Learning Agency Lab, an independent nonprofit based in Arizona, Vanderbilt is optimized to inspire and nurture cross-disciplinary research that fosters discoveries that have a global impact.<br><br>
    Your work in this competition will use process features from keystroke log data to predict overall writing quality.<br>
    These efforts may identify relationships between learners’ writing behaviors and writing performance.<br>
    Additionally, given that most current writing assessment tools mainly focus on the final written products, this may help direct learners’ attention to their text production process and boost their autonomy, metacognitive awareness, and self-regulation in writing.<br><br>
  - Evaluation<br>
    We use the Root Mean Squared Error to score submissions, defined as:<br>
    ```math
        RMSE=\left(\frac{1}{n} \sum_{i=1}^n\left(y_i-\hat{y}_i\right)^2\right)^{1 / 2}
    ```
    where $`\hat y _i`$ is the predicted value and $`y_i`$ is the original value for each instance $`i`$ over $`n`$ total instances.<br>

  - Submission File
    For each `id` in the test set, you must predict the corresponding `score` (described on the Data page).<br>
    The file should contain a header and have the following format:
    ```text
    id,score
    0000aaaa,1.0
    2222bbbb,2.0
    4444cccc,3.0
    ...
## Keystroke Related
  - Data Collection Procedure
    Participants of this project were hired from Amazon Mechanical Turk, a crowdsourcing platform.<br>They were invited to log onto a website that housed a demographic survey, a series of typing tests, an argumentative writing task, and a vocabulary knowledge test.<br>Participants were required to use only computers with a keyboard.<br><br>
    During the argumentative writing task, participants were asked to write an argumentative essay within 30 minutes in response to a writing prompt adapted from a retired Scholastic Assessment Test (SAT) taken by high school students attempting to enter post-secondary institutions in the United States.<br>
    To control for potential prompt effects, four SAT-based writing prompts were used and each participant was randomly assigned one prompt.<br>
    Prior to the writing task, instructions were presented on the integral components in an argumentative essay (e.g., introduction, position, reasons and evidence) along with descriptions of their functions in argumentation.
    <br>The instructions pages also introduced a set of rules for the writing task.
    <br>These include that participants should write an essay of at least 200 words in 3 paragraphs and that they should not use any online or offline reference materials.<br>
    To make sure participants stayed focused on the task during writing and to track behavior, the writing task page issued warnings whenever the participant was detected inactive for more than 2 minutes or moved to a new window in the process of writing.<br> screenshot of the writing task page is presented below.<br><br>
