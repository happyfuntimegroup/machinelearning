import json

class Paper(): 
    """
    Reads database file and creates an instance of each paper. 
    """
    def __init__(self, source_file):
        self.paper = self.load_paper(source_file)

    def load_paper(self, source_file):

        with open(source_file, 'r') as file_object:  
            papers = json.load(file_object)  
