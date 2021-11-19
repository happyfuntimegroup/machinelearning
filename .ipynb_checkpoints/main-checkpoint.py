# from CODE.features import venue_citations
# from CODE.features import length_title
from CODE.features.venue_frequency import venue_frequency 
from CODE.features.venue_citations import venues_citations
from CODE.features.length_title import length_title

# print(venue_frequency('DATA/train-1.json'))

# print(venues_citations('DATA/train-1.json'))

print(length_title('DATA/train-1.json'))

