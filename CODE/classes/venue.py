import pandas as pd
from sklearn import tree
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split 


def venue_frequency(source_file):
    papers = pd.read_json(source_file)
    venues_count = {}
    count = 1

    for i in range(len(papers)):
        venue_temp = papers.iloc[i]['venue']
        venue = ''

        for char in venue_temp:
            if char.isupper():
                venue += char
            
            if venue in venues_count.keys():
                venues_count[venue] += 1
            else:
                venues_count[venue] = 1
    
    return venues_count
    
def venues_citations(source_file):
    papers = pd.read_json(source_file)
    venues_count = {}
    citations = []

    for i in range(len(papers)):
        venue_temp = papers.iloc[i]['venue']
        venue = ''

        for char in venue_temp:
            if char.isupper():
                venue += char

            if venue in venues.keys():
                    # print(venue)
                citations = venues[venue]
                citations.append(papers.iloc[i]['citations'])
                venues[venue] = citations
            else:
                citations.append(papers.iloc[i]['citations'])
                venues[venue] = [(papers.iloc[i]['citations'])]


        for key, value in venues.items():
            print(key, ':', value)
            # sum_venue = sum(value)
            # print(key, ':', sum_venue)
            # venPresL = (sum_venue / len(value))
            # print(venPresL)
            
        # if venues['ACL'] == venues['INLG']:
        #     print('same')
        # else:
        #     print('not same')

