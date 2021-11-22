# Machine Learning Challenge of Tilburg University

In this challenge the task is to predict the number of citations a scientific paper
receives based on its abstract and metadata.

## ISSUES

Under [issues](https://github.com/happyfuntimegoup/machinelearning/issues), we can create about different tasks that we're working on and issues we have encountered (e.g. an funtion that supposedly works for one person, doesn't work for the other). 
If a function/assignment is completed, you can close the issues, so we all know it has been completed. An update on Whatsapp would also be appreciated! :)

## Tasks
We can update each other about current tasks that we came up with. Use strikethrough ('\~~(text)\~~') to show it has been completed.

#### Albert:
  -  ~~Function to split train dataset into train & validation set.~~
  -  ~~Open issues for individual feature extractions/creations.~~<br>
     In the upcoming days, I will open a number of issues; each issues specifically for an individual feature. Features will be based on the current literature (i.e. what current citation prediction models have been using as predictors). Some of them might end up not being used, as we will need to see if they are actually useful for our models (i.e. see if the addition of a feature explains a significant % of the data correctly).
  - Features:
    - ~~Team size (Team)~~
      - ~~Look at statistics~~
    - ~~Author productivity (ProAuth)~~
      - Look at statistics
    - Author H-index
      - ~~Citations per author function~~<br>
        This function is hella time-consuming, so I left it out of the main file. Maybe I should figure out how to use parallelization on this. As of now, it does not get included in the main file.
      - Look at statistics
    - ~~Age~~
      - ~~Look at statistics~~
    - Help out with making venue_citations and venue_frequency work in the main file.
  - Come up with some model


#### Melody:
  - Features:
    - Venue diversity (VenDiv)
    - Long term venue prestige (VenPresL) 
    - Word Length Title
  - Come up with some model

#### Regina:
  - EDA with Selin
  - Come up with some model  

#### Selin:
 -   EDA (if that counts?)
    - ~~Outlier function for numerical values~~
 -   Features:
    - Topic Diversity (TopDiv)
 -   Come up with some model 
