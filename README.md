# Natural Language Processing Projects

<p> This repo contains two NLP projects.</p>
<ol><li><a href="#1-news-article-classification"> News article classification</a></li>
<li><a href="#2-named-entity-recognition"> Named Entity Recognition</a></li></ol>

## 1. News article classification

### Background
<p> The data used in this project comes from the Sharing News Online project which investigates why people share news on social media. (It is not a NLP/machine learning related research, but if you're interested in political economics or journalism, go ahead and check out their <a href="https://www.google.com/books/edition/Sharing_News_Online/YQSjDwAAQBAJ?hl=en&gbpv=0">book</a>.) </br>
The authors also ask an interesting question: <b>which topics of articles get shared?</b></br>
The task of classifying news articles into different topics has to be done to perform this type of research.

</p>


### Objectives
<p>The goal for this project is to explore different feature extraction methods for supervised text classification of news articles. The workflow looks like:</p>
<ul>
<li>Train a statistical text classification system. 
<li>Evaluate the prediction and analyze errors.
<li>Imporve the model through experiments with different feature extraction methods.
</ul>

### Data
<p>2284 news articles which are annotated with one of ten categories: Business, Entertainment, Health, Politics, Science/Technology, Society, Sports, War, Other and Error. </p>

<p>Data is not shared in this repo.</p>

### Method
<p>I used logistic regression for the statistical model used different feature extraction methods for each experiment. 10-fold cross validation is used for testing. </p>
<ul>
<li>Baseline system: bag-of-words</li>
<li>Experiment 1: Cleaning and treating abbreviations</li>
<li>Experiment 2: N-grams</li>
<li>Experiment 3: Stop words, TF-IDF</li>
<li>Experiment 4: Stemming</li>
<li> Experiment 5: WordNet synsets, hypernyms</li>
</ul>


### Results

## 2. Named Entity Recognition


 
