def length_title(source_file):
    """
    This function is based on an online published article:
        https://www.natureindex.com/news-blog/five-features-highly-cited-scientific-article
        
        'Titles play an essential role in capturing the overall meaning of a paper. 
        Previously published work investigating this question agrees that the title length can impact citation rates.'
        
    """
    import pandas as pd
    papers = pd.read_json(source_file)
    dict_wordlength = {}
    dict_citations = {}
    dict_length_citations = {}

    for i in range(len(papers)):
        title = papers.iloc[i]['title']
        title_list = title.split()
        dict_wordlength[title] = len(title_list)
        dict_citations[title] = papers.iloc[i]['citations']
        dict_length_citations[len(title_list)] = papers.iloc[i]['citations']
    
    # return dict_wordlength
    return dict_length_citations



    print(length_title('DATA/train-1.json'))
