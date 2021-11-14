def author_database(df):
    """
    Builds a database of all authors in df based on each last name in 'authors'. 
    Here, by last name, we mean the last word that appears for each author name in the author column.
        e.g., an A. Zhang and Z. Zhang will be considered to be the same author, as some papers did not have a first name or first letter printed.
    Input:
        - df['authors']:    dataframe (dataset); 'authors' column       [pandas dataframe]
    Output:
        - unique_authors
    """
    unique_authors = []
    for i in df['authors'][0:10]:
        for j in i:
            if j.split()[-1] not in unique_authors:
                unique_authors.append(j.split()[-1])
    return unique_authors