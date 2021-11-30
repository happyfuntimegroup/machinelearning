import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from CODE.features.field_variety import field_variety
from CODE.features.topic_popularity import topic_popularity
from sklearn.linear_model import LinearRegression


data = pd.read_json('DATA/train.json') 
field_variety(data)
# topic_popularity(data)

    # y = data['citations']
    # correlaties = data.corr()
    # sns.heatmap(correlaties)
    # plt.savefig('test.png')
    # # print(fields_dict)
