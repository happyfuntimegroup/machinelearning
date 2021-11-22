# def length_title(source_file):
#     """
#     This function is based on an online published article:
#         https://www.natureindex.com/news-blog/five-features-highly-cited-scientific-article
        
#         'Titles play an essential role in capturing the overall meaning of a paper. 
#         Previously published work investigating this question agrees that the title length can impact citation rates.'
        
#     """
#     import pandas as pd
#     papers = pd.read_json(source_file)
#     dict_wordlength = {}
#     dict_citations = {}
#     dict_length_citations = {}

#     for i in range(len(papers)):
#         title = papers.iloc[i]['title']
#         title_list = title.split()
#         dict_wordlength[title] = len(title_list)
#         dict_citations[title] = papers.iloc[i]['citations']
#         dict_length_citations[title[0]] = 'citations: '+ str(papers.iloc[i]['citations']) + ' and word length: ' + str(len(title_list))
    
#     # return dict_wordlength
#     return dict_length_citations



#    print(length_title('DATA/train-1.json'))




def length_title(source_file):
    """
    Slight modification to return a dictionary of lists so that we can grab the relevant number more easily
    key = title
    value = (length of title, number of citations)
    """
    import pandas as pd
    papers = pd.read_json(source_file)
    dict_title_length = {}
    
    for i in range(len(papers)):
        title = papers.iloc[i]['title']
        doi = papers.iloc[i]['doi']
        title_list = title.split()

        dict_title_length[doi] = (len(title_list))
    
    # return dict_wordlength
    return dict_title_length

