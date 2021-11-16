def author_name(author):
    """
    Rewrite author name in a specific form:
        Take the last word that is present in 'author'. Set that as last name.
        If a different word is present, take the first letter. If not, set a default of X. 
        e.g.    the name 'Albert de Vries' gets turned into 'A. Vries';
                the name 'Vries' gets turned into 'X. Vries'
    Input:
        - Author name ((in)complete)            [string]
    Output:
        - Author name in specific form          [string]
    """
    first_raw = author.split()[0]
    last = author.split()[-1]
    if first_raw != last:       # First name is mentioned
        first = first_raw[0] + '. '
    else:                       # No first name mentioned
        first = 'X. '               # If not mentioned, default = X.
    name = first + last         # Combine first letter and last name
    return name