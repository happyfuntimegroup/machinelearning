def author_h_index(author_citation_dic):
    """
    
    """
    def h_check(val):
        """
        
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