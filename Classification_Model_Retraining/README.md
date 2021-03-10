# Project 3 - Subreddit Classification

<img src="./images/Word_Cloud_both.png" style="float: center; margin: 20px; height: 500px">

## Executive Summary

“A rose by any other name... is still a rose” (butchered Shakespeare phrase from Romeo and Juliet).

I am interested in exploring how a successful classification model holds up over time. This is relevant to the work that data scientists carry out, as there is always an element of timing to it. When a model is put into production, how long will it be useful and relevant for? Does it ever need to be retrained? If so, then how often?

I start with 2 categories: practical here's how you do stuff (r/LifeProTips), and deeper existential thoughts (r/Showerthoughts). They are among the largest subreddits, with memberships ~20M, ensuring large post volumes over time. 1000 posts are extracted from each subreddit, at midnight on the 20th of each month, starting Sept 2019 all the way to Feb 2021. Only post 'titles' were used for classification.

8 classification models are evaluated, and 2 are down-selected for comparison: a Logistic Regression and a Support Vector Classifier (SVC). Both perform similarly on test data during the model development phase, even though the SVC scored close to perfect on testing data.

Both models perform similarly throughout time, at accuracies close to that obtained during the model development phase (~85%), and with a remarkably linear relationship between their accuracies. This suggests that in this specific instance, these models are sufficiently robust to variations in posts over time.

The limits of the models are pushed by running them on different pairs of posts, on which the models were not trained. Happily, classification accuracy is still close to 80% when tried on 2 different subreddit pairings.

### Problem Statement
How well does a classification model hold up over time?

If a model is built based on a dataset collected at a specific point in time, how far into the future can it be applied, and still be successful (accurate)? Where success is defined as performing better than the baseline accuracy. It would be a bonus if I can identify whether covid specifically had an impact on accuracy, by creating my model from posts gathered from Sept 2019, and applying the model to posts gathered during subsequent months. In order to inject an added degree of difficulty to this problem, the subreddits need to have similar content. And in order to successfully fulfill the requirement to collect data over time, the subreddit memberships need to be very high, to generate enough volume. I am additionally curious about how generalizable the model will be when applied to posts from other subreddits.

Some references for retraining machine learning models:
[The Ultimate Guide to Model Retraining](https://mlinproduction.com/model-retraining/)
[What's Your ML Test Score? A rubric for ML Production Systems](https://storage.googleapis.com/pub-tools-public-publication-data/pdf/45742.pdf)
[A Gentle Introduction to Concept Drift in Machine Learning](https://machinelearningmastery.com/gentle-introduction-concept-drift-machine-learning/)

### Data Description
Data sets scraped from Reddit are stored in the /data folder.

|    | Folder       | Data file             | Description                                            |
|----|--------------|-----------------------|--------------------------------------------------------|
| 1  | ./data       | df_LifeProTips.csv    | 1000 posts from r/LifeProTips (Sept 2019)              |
| 2  | ./data       | df_Showerthoughts.csv | 1000 posts from r/Showerthoughts (Sept 2019)           |
| 3  | ./data       | df_model.csv          | Combined and cleaned up data, for modeling             |
| 4  | ./data/Raw   | LifeProTips_x.csv     | x: 0 -> 16. 1000 posts per month from Oct 2019 onwards |
| 5  | ./data/Raw   | Showerthoughts_x.csv  | x: 0 -> 16. 1000 posts per month from Oct 2019 onwards |
| 6  | ./data/extra | df_LifeProTips.csv    | 1000 posts from r/LifeProTips (Sept 2019)              |
| 7  | ./data/extra | df_Showerthoughts.csv | 1000 posts from r/Showerthoughts (Sept 2019)           |
| 8  | ./data/extra | df_space.csv          | 1000 posts from r/space (Sept 2019)                    |
| 9  | ./data/extra | df_stocks.csv         | 1000 posts from r/stocks (Sept 2019)                   |
| 10 | ./data/extra | df_todayilearned.csv  | 1000 posts from r/todayilearned (Sept 2019)            |

### Notebook Description
There are 5 notebooks in the /code folder, as described below:
1. Problem Statement and Webscraping
2. Exploratory Data Analysis
3. Logistic Regression Model
4. Random Forest, Extra Trees, Support Vector Machine Models
5. Apply Models to Datasets & Evaluate

### Methodology
The dataset used to build models (from Sept 2019) is composed of 2000 total posts, at a 50/50 split between r/LifeProTips and r/Showerthoughts. Duplicate rows were removed - this is virtually the only cleaning step that needed to be taken. The post titles were vectorized, using 'lpt' as a stopword since it is a 100% indicator of a post from r/LifeProTips. Various exploratory analysis steps were carried out to understand distributions of character and word counts. When building the different models, GridSearchCV and a pipeline were used to search for optimized hyperparameters and cross-validation. Model performance was compared to the baseline model (of 0.5) and 2 models were selected for use over time: the LogReg model with highest test score, and a SVC model, which had a similarly high test score. The details of these models are given in the table below:

| Model | Transformer | Estimator |            Details           | Accuracy (train) | Accuracy (train) |
|:-----:|:-----------:|:---------:|:----------------------------:|:----------------:|:----------------:|
|   1   |  TfidfVect  |   LogReg  |        'lpt' stopword        |       0.926      |       0.852      |
|   8   |   TfidVect  |    SVC    | GridSearchCV, 'lpt' stopword |       0.991      |       0.854      |

These models were rebuilt using the entire Sept 2019 dataset, pickled, and subsequently applied to 17 test sets. Accuracy scores are plotted over time and various comparisons are carried out to quantify similarities and differences in model performance over time.

### Key Findings
Here are the key findings of this project:

* When duplicated entries are removed, the dataset is still balanced between the 2 subreddits
* 90% of the 10 longest posts are from the 'LifeProTips' subreddit. 90% of the 10 shortest posts are from the 'Showerthoughts' subreddit. That's exactly opposite of each other! Maybe because a 'tip' needs some description (there's not really a 1-word tip other than maybe 'sleep'). Whereas in the shower, you can think of any random thing, which obviously could be 1-word
* 75% of r/LifeProTips posts have the identifier ‘LPT’ as part of the post. This was designated as a stop word for modeling
* There is a 37% overlap in words within the top 100 words of both subreddits. That means the model wouldn't have the easiest of times differentiating the 2 subreddits but it would still be very do-able
* From the 10 most common bigrams, it seems like r/LifeProTips discusses more practical, daily stuff while r/Showerthoughts discusses deeper thoughts
* Ultimately, both these models are overfit. The training accuracy was 93% for LogReg and 99% for SVC, while these test accuracies are around 83%. The degree of overfitting looks the same (same amount of difference between training score and average testing score).
* LogReg and SVC have the same performance for these datasets. This LogReg and SVC accuracies can be fit with a linear relationship
* There is not an obvious relationship between model accuracy and common word count
* Most posts are basically neutral according to sentiment analysis. A slightly higher fraction of the correctly classified posts have positive sentiments.
* When applied to post pairings from other subreddits, LogReg and SVC perform similarly (in accuracy trend) when trying to differentiate between subreddits. This is expected, from their linear relationship shown earlier. Other observations from comparing accuracies when subreddit pairs don't contain the original pair:
    * The model works best (just marginally) when classifying posts from the original 2 subreddits
    * The model sees posts from r/Showerthoughts to be similar to posts from r/todayilearned, and the accuracy rate when classifying these posts is barely 50%, which is just barely better than a coin toss, and barely above the baseline accuracy.
    * Another way we see this similarity is because the accuracy when r/LifeProTips is paired with r/todayilearned is similar to when r/LifeProTips is paired with its original partner, r/Showerthoughts
    * Any time r/LifeProTips is one of the subreddits, the model is well-able to classify the posts
    * The pair of posts with the worst accuracy even has r/Showerthoughts in it! (one of the 2 original subreddits)
    * Accuracy of classification is even higher when neither of the subreddits is from the original pair, compared to when r/Showerthoughts is one of the pair

The model is really learning "what is r/LifeProTips and what is NOT r/LifeProTips", instead of "what is r/LifeProTips and what is r/Showerthoughts"

### Conclusions and Recommendations
In conclusion:
* Models seem quite robust and hold up well over time
* Models are applicable to the same subreddits over time
* There may be some time-related underlying behaviors that are not captured here
* This demonstrates that the model captures what IS and IS NOT r/LifeProTips (instead of what IS r/LifeProTips and what IS r/Showerthoughts)

To further improve our understanding, we could:
* Switch the classification labels such that r/Showerthoughts is 1 and r/LifeProTips is 0, when building the model, to see whether the results are similar
* Train the model on r/LifeProTips, r/Showerthoughts, and r/todayilearned all together to see if it is possible to differentiate between these 3 subreddits
