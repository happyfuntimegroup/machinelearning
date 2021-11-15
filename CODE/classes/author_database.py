def author_database(df):
    """
    Builds a database of all authors in df based on each last name in 'authors'. 
    Here, by last name, we mean the last word that appears for each author name in the author column.
    Input:
        - df:               dataframe (dataset); 
                            can be only the 'authors' column            [pandas dataframe]
    Output:
        - author_db:        list of unique authors in the dataset       [pandas dataframe]
    """
    import pandas as pd
    import author_name
    author_db  = {}
    for index, i_paper in df[0:10].iterrows():
        authors = i_paper['authors']    # Group of authors of the paper
        for i_author in authors:
            name = author_name(i_author)
            if name not in author_db:
                author_db[name] = 1     # Number of published papers = 1
            else:
                author_db[name] += 1    # Add 1 to number of published papers
    author_db = pd.DataFrame.from_dict(author_db, orient = 'index')
    author_db.columns = ['Total_papers_published']
    return author_db