import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from CODE.features.field_variety import field_variety
from CODE.features.topic_popularity import topic_popularity
from sklearn.linear_model import LinearRegression
from CODE.data_preprocessing.split_val import split_val
from CODE.features.topic_citations_avarage import topic_citations_avarage


data = pd.read_json('DATA/train.json') 

topic_citations_avarage(data)
# print(data['citations'])
# topic_popularity(data)

    # y = data['citations
    # correlaties = data.corr()
    # sns.heatmap(correlaties)
    # plt.savefig('test.png')
    # # print(fields_dict)
