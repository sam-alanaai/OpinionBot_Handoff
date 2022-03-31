import csv

import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
import json

# Download and load JSON dataset
data_file = "train_data.jsonl"
data = []
nlp = spacy.load('en_core_web_lg')
nlp.add_pipe('spacytextblob')
with open(data_file, 'r') as json_file:
    json_list = list(json_file)
skipthisone = True
def comparator(nlp , filename):
    file = open(filename)

    csvreader = csv.reader(file)
    header = []
    header = next(csvreader)
    rows = []
    for row in csvreader:
        rows.append(row)
    file.close()
    csv_file = open("resultsanalysis.csv", "a", encoding="utf8")
    write_csvfile = csv.writer(csv_file)
    for row in rows:
        generated = nlp(row[1])
        actual = nlp(row[2])
        write_csvfile.writerow([generated._.polarity, actual._.polarity])



comparator(nlp,"predictionstorunanalycticon.csv")
