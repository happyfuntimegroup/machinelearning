def author_database(df):
    """
    Builds a database of all authors in df based on each last name in 'authors'. 
    Here, by last name, we mean the last word that appears for each author name in the author column.
    Also returns df['Authors'] reformatted.
    Input:
        - df:                   dataframe (dataset); 
                                can be only the 'authors' column                [pandas dataframe]
    Output:
        - author_db:            pd dataframe of unique authors in the dataset   [pandas dataframe]
        - reformatted_authors:  list of lists of authors groups for each paper
                                (with reformatted names)                        [list]
    """
    import pandas as pd
    import author_name
    author_db  = {}
    reformatted_authors = []
    for index, i_paper in df.iterrows():
        authors = i_paper['authors']    # Group of authors of the paper
        group = []
        for i_author in authors:
            name = author_name(i_author)
            group.append(name)
            if name not in author_db:
                author_db[name] = 1     # Number of published papers = 1
            else:
                author_db[name] += 1    # Add 1 to number of published papers
        reformatted_authors.append(group)
        
    # Turn output into pandas dataframe/series
    author_db = pd.DataFrame.from_dict(author_db, orient = 'index')
    author_db.columns = ['Total_papers_published']
    
    reformatted_authors = pd.Series(reformatted_authors)
    reformatted_authors.columns = ['Authors_reformatted']
    return author_db, reformatted_authors