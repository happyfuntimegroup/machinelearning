#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def best_keywords (data, words_per_paper, paper_quantile):
    import yake as yake
    # take the most highly cited papers from data set
    a = data['citations'].quantile(q = paper_quantile)
    best = data[data['citations'] > a]
    best['keywords'] = ''

    # setup for the keyword extractor
    extractor = yake.KeywordExtractor()
    language = 'en'
    max_ngram = 1
    deduplication_threshold = 0.1   # low value = duplication of keywords in ngrams not allowed
    num_keywords = words_per_paper
    
    # run the keyword extractor on the highly cited papers
    kwords = []
    for index, i_paper in best.iterrows(): # iterate over the dataframe 
        text = i_paper['abstract']
        custom_kw_extractor = yake.KeywordExtractor(lan=language, n=max_ngram, dedupLim=deduplication_threshold, 
                                                    top=num_keywords, features=None)
        keywords = custom_kw_extractor.extract_keywords(text)

        k = []
        for i in keywords:
            k.append(i[0].lower())
        kwords.append(k)

    # return a list of keywords
    kwords = [x for l in kwords for x in l]
    return set(kwords)

