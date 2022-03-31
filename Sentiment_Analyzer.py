import csv

import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
import json

# Download and load JSON dataset
data_file = "train_data.jsonl" #takes the json file from redial datasets
data = []
nlp = spacy.load('en_core_web_lg')
nlp.add_pipe('spacytextblob')
with open(data_file, 'r') as json_file:
    json_list = list(json_file)
skipthisone = True


def sentimentthingy(sentiment, mode):
    facts = "error"
    if mode == 1:
        if 0 <= sentiment < 0.25:
            facts = "Polarity: Acceptable"
        if 0.25 <= sentiment < 0.5:
            facts = "Polarity: Good"
        if 0.5 <= sentiment < 0.75:
            facts = "Polarity: Great"
        if 0.75 <= sentiment <= 1:
            facts = "Polarity: Love"
        if -0.25 <= sentiment < 0:
            facts = "Polarity: Dislike"
        if -0.5 <= sentiment < -0.25:
            facts = "Polarity: Annoying"
        if -0.75 <= sentiment < -0.5:
            facts = "Polarity: Hate"
        if -1 <= sentiment <= -0.75:
            facts = "Polarity: Despise"

    if mode == 2:
        if 0 <= sentiment < 0.25:
            facts = "Polarity: This sentence is Acceptable"
        if 0.25 <= sentiment < 0.5:
            facts = "Polarity: This statement is Good"
        if 0.5 <= sentiment < 0.75:
            facts = "Polarity: This statement shows Great"
        if 0.75 <= sentiment <= 1:
            facts = "Polarity: This statement shows Love"
        if -0.25 <= sentiment < 0:
            facts = "Polarity: This sentance is shows Dislike"
        if -0.5 <= sentiment < -0.25:
            facts = "Polarity: This sentance shows annoyance"
        if -0.75 <= sentiment < -0.5:
            facts = "Polarity: This sentance shows hate"
        if -1 <= sentiment <= -0.75:
            facts = "Polarity: This sentance shows Spite and Vitriol"
    if mode == 3:
        if 0 <= sentiment < 0.25:
            facts = "Polarity: a"
        if 0.25 <= sentiment < 0.5:
            facts = "Polarity: b"
        if 0.5 <= sentiment < 0.75:
            facts = "Polarity: c"
        if 0.75 <= sentiment <= 1:
            facts = "Polarity: d"
        if -0.25 <= sentiment < 0:
            facts = "Polarity: e"
        if -0.5 <= sentiment < -0.25:
            facts = "Polarity: f"
        if -0.75 <= sentiment < -0.5:
            facts = "Polarity: g"
        if -1 <= sentiment <= -0.75:
            facts = "Polarity: h"
    return facts

for json_str in json_list:
    result = json.loads(json_str)
    askerman = ""
    askermanscoop = result["initiatorWorkerId"]
    recievermanscoop  = result["respondentWorkerId"]
    for i in result["messages"]:
        if i['senderWorkerId'] == askermanscoop:
            askerman = i['text']
        if i['senderWorkerId'] == recievermanscoop:

            text = i['text']
            doc = nlp(text)
            if ((doc._.polarity > 0.01 or doc._.polarity < -0.01) and (doc.ents)):
                csv_file = open("csv_file.csv", "a",encoding="utf8")
                source = ((sentimentthingy(doc._.polarity,1)) +  " Text: " + askerman)
                write_csvfile = csv.writer(csv_file)
                target = text.replace(",","")
                #+ str(doc.ents)).replace(",","")
                write_csvfile.writerow([source,"Text: " + target.replace("goodbye","")])#source  , target
                #save mst recent initiator message + sentiment




# need to save questions and only test doc

    # for key in result.keys():
    #     print(key)
    # for i in result["messages"]:
    #     print(i['text'])
    #     print(type(i))
    #print(result["movieMentions"].keys())
    #print(result["messages"][3])
    #print(f"result: {result}")
    #break
    #print(isinstance(result, dict))
    #break
#print(dataset["train"])
# the stage to reintroduce the topics needs to be done here before the dataset is fed to spaCy

with open("file.txt", 'w') as f:
    for s in data:
        f.write(str(s) + '\n')
# print("polarity")
# print(doc._.polarity)      # Polarity: -0.125
#
# print("subjectivity")
# print(doc._.subjectivity)  # Sujectivity: 0.9
#
# # print("assessements")
# # print(doc._.assessments)
#
# print("ents")
# print(doc.ents)