## Not checked
import json
with open("BioASQ-trainingDataset4b.json") as f:
    data = json.load(f)
total_docs = set()
for ques in data["questions"]:
    for docs in ques["documents"]:
        total_docs.add(docs[35:])
docs_connection = {}
total_connection = 0
docs_connection = {}
total_connection = 0
for ques in data["questions"]:
    for docs in ques["documents"]:
        if not docs[35:] in docs_connection:
            docs_connection[docs[35:]] = set()
        for docs_re in ques["documents"]:
            if not docs == docs_re:
                docs_connection[docs[35:]].add(docs_re[35:])
                if not docs_re[35:] in docs_connection:
                    docs_connection[docs_re[35:]] = set()
                docs_connection[docs_re[35:]].add(docs[35:])
                total_connection += 1

for a in docs_connection:
    docs_connection[a] = list(docs_connection[a])

json.dump(docs_connection, open('connection.json', 'w'), indent = 4)
