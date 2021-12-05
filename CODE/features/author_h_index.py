def author_h_index(author_citation_dic):
    """
    Compute the h-index for each author.
    The h-index is defined as the value for which the author has h published papers with atleast h number of citations.
    Input:
        - author_citation_dic:  dataset (dict) of all authors (keys) in the known dataset 
                                 and a list with citations of papers they worked on (values)    [dict]
    Output:
        - h_index:              dataset (dict) of all authors (keys) in the known dataset
                                 and the h-index for that author (value)                        [dict]
    """
    def h_check(val):
        """
        Take a list with all the citations of a paper that one specific author worked on.
        Then binarizes that list it for a certain h-value. Stop until the condition is met where h-value = sum of binary list.
            i.e.:   - for a list of n published papers, take h-value = n
                    - binarize the list for that h-value: look for papers the citation # is equal to or greater than the h-value
                    - take the sum of the binarized list
                    - is the sum of the binarized list equal or greater than the h-value?
                        - if yes, then h = n
                        - if not, take h = h-1 and redo the steps until either h = a number or h = 0
        Input:
            val:    list of all citations for one author
        Output:
            h:      h-index for that author
        """
        n = len(val)
        for h in range(n,-1,-1):
            list_binary = [1 if citation >= h else 0 for citation in val]
            if sum(list_binary) >= h:   # Should change this condition, cause it returns 0 for a list of [18, 0, 16]
                return h        
    h_index = {}
    for key, val in author_citation_dic.items():
        h_index[key] = h_check(val)
    return(h_index)