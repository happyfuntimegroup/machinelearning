def author_database(df):
    """
    Builds a database of all authors in df based on each last name in 'authors'. 
    Here, by last name, we mean the last word that appears for each author name in the author column.
    Input:
        - df['authors']:    dataframe (dataset); 'authors' column      [pandas dataframe]
    Output:
        - unique_authors:   list of unique authors in the dataset      [list]
    """
    unique_authors = []
    for i in df['authors']:
        for j in i:
            first_raw = j.split()[0]
            last = j.split()[-1]
            if first_raw != last:       # First name is mentioned
                first = first_raw[0] + '. '
            else:                       # No first name mentioned
                first = 'X. '               # If not mentioned, default = X.
            name = first + last         # Combine first letter and last name
            if name not in unique_authors:
                unique_authors.append(name)
    return unique_authors
