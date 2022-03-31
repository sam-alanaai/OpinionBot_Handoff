# OpinionBot_Handoff


##Opinion bot next steps and progression
#### the current steps to pursue with the sentiment model is to flesh out the training set  with properly extracted dialogues from emphateic dialogues , ubuntu and wizard of wikipedia.
#### the topic model requires adjustments to the model as well as testing on the ideal attributes to extract from wikidata for effectiveness. a lack of sufficient conversation data on each given topic is the current issue.
#### the code that was being used to extract from non redial datasets has turned out to be extremely  buggy in recent weeks despite remakes and will need to be recontruscted wholly
## models of t5 used in testing     t5-small

    t5-base

    t5-large

    t5-3b

    t5-11b.
    
##datasets used/to implement
Redial dataset
emphateic dilaogies
persona chat
ubuntu dialogue
wizard of wiki
### the code i had been using to implement a couple of these datasets has been found to be faulty and has been  removed from this final handover

## notable sections from wikidata found to be consistently useful 
  genre, set in enviroment , main subject
 ## notable sections from wikidata found to be obfuscating
  first 5,10,20 relevant cast members , country of origin ,narrative/filimg location , duration
 
 #papers a lst of papers and articles used in the construction of the model has been provided but the primary basis for the structure has been an attempt to recreate https://www.cs.sjtu.edu.cn/~li-fang/Ye2020_Chapter_KnowledgeEnhancedOpinionGenera.pdf   with T5  a final goal in emulation of this paper is being
 able to extract pertinent conversation pairs from large dialogue datasets
  
  ## Sentiment Analyser
  the structure of the program is that one of the 2 sentiment analysers needs to be ran which can be fed to train.py to create a model to examine with test.py this will produce a verification sheet for examining results . their is a results python file that can be used to create an examinable excel sheet 
 n-grams additionally need to be reimplemeneted as a metric for uniqueness in responses
  
