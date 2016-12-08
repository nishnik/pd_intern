import json
from Bio import Entrez
from Bio.Entrez import efetch, read

Entrez.email = "nishantiam@gmail.com"
pubmed_fetch = {}
def fetch_data(pmid):
    handle = efetch(db='pubmed', id=pmid, retmode='xml')
    for xml_data in read(handle):
        try:
            if ('MedlineCitation' in xml_data.keys()):
                pmid = xml_data['MedlineCitation']['PMID'].strip()
                pubmed_fetch[pmid] = {}
                pubmed_fetch[pmid]['title'] = xml_data['MedlineCitation']['Article']['ArticleTitle']
                pubmed_fetch[pmid]['abstract'] = xml_data['MedlineCitation']['Article']['Abstract']['AbstractText']
                # some are not there
                # pubmed_fetch[pmid]['smsh'] = xml_data['MedlineCitation']['MeshHeadingList']
            else:
                pmid = xml_data['BookDocument']['PMID'].strip()
                pubmed_fetch[pmid] = {}
                pubmed_fetch[pmid]['title'] = xml_data['BookDocument']['ArticleTitle']
                pubmed_fetch[pmid]['abstract'] = xml_data['BookDocument']['Abstract']['AbstractText']
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

json.dump(pubmed_fetch, open('pubmed_fetch.json', 'w'))
