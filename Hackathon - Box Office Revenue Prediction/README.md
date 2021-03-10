# **Project 4 - TMDB Box Office Prediction**
Aida Rahim, Okechukwu Ofili, Matt Bildzok, and Andy Roberts
## Problem Statement
Our analysis aims to analyze an initial dataset of 23 columns with the purpose of predicting movie revenue. Additionally, our model will provide detailed information regarding the greatest contributors towards box office revenue.
## Data Dictionary
|Feature|Type|Dataset|Description|
|---|---|---|---|
|id|int|train.csv|Film identification number|
|belongs_to_collection|object|train.csv|Specifies if a film is part of a collection|
|budget|int|train.csv|Cost to create the film|
|genres|object|train.csv|types of film genre each item belongs to|
|imdb_id|object|train.csv|identification on the imdb website|
|original_language|object|train.csv|language the film was orignally released in|
|original_title|object|train.csv|title the film was originally released under|
|overview|object|train.csv|Plot summary of the film|
|popularity|float|train.csv|Rating of the films popularity to the general public|
|poster_path|object|train.csv|Link to image of the film poster|
|production_companies|object|train.csv|name of companies who produced the film|
|production_countries|object|train.csv|country the film was produced in|
|release_date|object|train.csv|Date of film's initial release|
|runtime|float|train.csv|Length of film|
|spoken_language|object|train.csv|List of languages the film is available in|
|status|object|train.csv|Current status of the film|
|tagline|object|train.csv|Subheader of movie title|
|title|object|train.csv|Name of film|
|Keywords|object|train.csv|List of descriptive words pertaining to the film|
|cast|object|train.csv|Actors in the film|
|crew|object|train.csv|People who worked on film, excluding actors|
|revenue|int|train.csv|Amount of money the film has yielded|

# Summary of Analysis

## Data Cleaning & EDA

The first task in the analysis of this dataset consisted of cleaning the training dataset prior to our modeling. We began with dropping columns we deemed unnecessary, which generally consisted of text columns that lacked any predictive power. Dates were converted to a usable form with DatetimeIndex.

Many films were found to have zero or very small budget, which seemed unreasonable. Based on the small budget of hit film "the Blair Witch Project," we introduced a cutoff of $60000 to remove these outliers. Additionally, many films were found to have popularities that diverged dramatically from the bulk of the data, thus we removed films with a popularity greater than 100.

Summary statistics and basic visualizations were run on the data to better understand relative values observed in the dataset. Histograms, boxplots, cross plots, and correlation heat maps incorporating box office revenue provided better context prior to moving into our modeling phase.
## Preprocessing
Because several columns stored pertinent data as dictionaries within each cell, extensive data extraction was required. Data consisting of keyword, genre, production company, production country, cast, crew among others were extracted and converted to a usable format. Because the crew includes many members, we extracted only the director and director of photography. Finally, a new dataframe was created that contained this text data.

A new dataframe was then created that contained this text data, which was then fit and transformed using a CountVectorizer. Min_df was set at 2, so that dimensionality was reduced. This newly created dataframe containing our text data was then concatenated to our original numeric data.
## Model 1 - Simple Linear Regression Model
Our first attempt at modeling box office revenue utilized a linear regression model. We first utilized only the numeric data to accomplish a basic fit, which utilized a test/train split. Our model returned the following scores:

|R2 - Train|R2 - Test |RMSE - Test| RMSE - Baseline|
|---|---|---|---|
|0.61|0.50|78811591|111935927|

Our model was overfit, however it does perform better than the baseline score.

## Model 2 - Random Forest

Our second model utilized a random forest regressor. Here we deployed a GridSearch, so that modeling parameters would be optimized. Ultimately, our model was run on a max_depth of 10 and 100 estimators. This yielded the below results:

|R2 - Train|R2 - Test |RMSE - Test| RMSE - Baseline|
|---|---|---|---|
|0.91|0.56|74194954|111935927|

Model 2 exhibits an even greater overfit than Model 1, however we attained a slightly smaller RMSE.

## Model 3 - Linear Regression with Text Data

Our third model included the same numeric data as model one, while also incorporating the text data from the keyword, genre, production company, production country, cast, and crew columns. The aim here was too incorporate more data that may be indicative of box office revenue. This yielded the below results:

|R2 - Train|R2 - Test |RMSE - Test| RMSE - Baseline|
|---|---|---|---|
|0.99|0.36|89105473|111935927|

Despite extensive effort to create an all-encompassing model, our efforts proved too great, as we are massively overfit. This model also had the highest RMSE of the three models run.

# Conclusions and Recommendations

The analysis presented here succeeded in modeling box office movie revenue by three different methods. However, all models proved overfit. In terms of fit, our simple linear regression (Model 1), proved to have the best fit. Meanwhile, our random forest model (Model 2) had the lowest RMSE.

Several options were available for refining our model incorporating a Ridge or Lasso regularization to reduce our variance. Similarly, efforts were undertaken to manually input missing data that was not able to be utilized in our model. This additional step could potentially improve model performance.
