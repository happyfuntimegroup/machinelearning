import pandas as pd
from CODE.features.field_variety import field_variety
from CODE.features.topic_popularity import topic_popularity


data = pd.read_json('DATA/train-1.json') 
# field_variety(data)
topic_popularity(data)