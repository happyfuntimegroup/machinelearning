import yake
def best_keywords (data, words_per_paper, lower_bound, upper_bound):
    """
    Input: data, integer, float between 0 and 1, float between 0 and 1
    Output: a list of all unique keywords that were generated from qualifying papers
    """
    # source: https://towardsdatascience.com/keyword-extraction-process-in-python-with-natural-language-processing-nlp-d769a9069d5c
    # take the most highly cited papers from data set
    lower = data['citations'].quantile(q = lower_bound)
    best = data[data['citations'] > lower]

    upper = data['citations'].quantile(q = upper_bound)
    best = best[best['citations'] > upper]

    best['keywords'] = ""

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