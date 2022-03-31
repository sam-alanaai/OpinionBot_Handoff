import csv

import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
import json

from SPARQLWrapper import SPARQLWrapper, JSON
def sparqleury(query):
    sparql = SPARQLWrapper(
        'https://query.wikidata.org/sparql',
        agent='Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36',
    )
    text = """
        SELECT ?a ?aLabel ?propLabel ?b ?bLabel
        WHERE
        {
          ?item rdfs:label """+ '"' + query +'"'+ """@en.
          ?item ?a ?b.
          SERVICE wikibase:label { bd:serviceParam wikibase:language "en". }
          ?prop wikibase:directClaim ?a .
        }
    """
    sparql.setQuery(text)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()
    for result in results["results"]["bindings"]:
        print(result["propLabel"]["value"] + " = " + result["bLabel"]["value"])
    return results   # remove labels from results as appropiate
sparqleury("Daffy Duck")
# Download and load JSON dataset
data_file = "train_data.jsonl"
data = []
nlp = spacy.load('en_core_web_lg')
nlp.add_pipe('spacytextblob')
with open(data_file, 'r') as json_file:
    json_list = list(json_file)
skipthisone = True
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
                csv_file = open("training_no_topic.csv", "a",encoding="utf8")
                source = ("Polarity: "+ str(doc._.polarity) +  " Text: " + askerman)
                write_csvfile = csv.writer(csv_file)
                target = text.replace(",","")
                topics = []
                for i in str(doc.ents):
                    topics.append(sparqleury(i))
                write_csvfile.writerow([source, "Text: " + target ,"Topics: " + str(topics)])#source  , target
                #save mst recent initiator message + sentiment + ratio
                #extract @34333 and such for topic dataset=


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