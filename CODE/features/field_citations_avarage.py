            if field in fields_dict.keys():
                citations.append(citation)
                fields_dict[field] = sum(citations) / len(citations)
            else:
                fields_dict[field] = citation