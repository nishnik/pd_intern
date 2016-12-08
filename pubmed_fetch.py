import json
from Bio import Entrez
from Bio.Entrez import efetch, read

Entrez.email = "nishantiam@gmail.com"
dict_data = {}
def fetch_data(pmid):
    handle = efetch(db='pubmed', id=pmid, retmode='xml')
    for xml_data in read(handle):
        try:
            pmid = xml_data['MedlineCitation']['PMID'].strip()
            dict_data[pmid] = {}
            dict_data[pmid]['title'] = xml_data['MedlineCitation']['Article']['ArticleTitle']
            dict_data[pmid]['abstract'] = xml_data['MedlineCitation']['Article']['Abstract']['AbstractText']
            # some are not there
            # dict_data[pmid]['smsh'] = xml_data['MedlineCitation']['MeshHeadingList']
        except Exception as e:
            print (e, pmid)
            pass

connection_dict = json.load(open('connection.json'))
unique_keys = list(connection_dict.keys())

def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]

stacked_keys = list(chunks(unique_keys, 100))

i = 0
for a in stacked_keys:
    print (i)
    i += 1
    fetch_data(str(a)[1:-1])

json.dump(dict_data, open('pubmed_fetch.json', 'w'))
