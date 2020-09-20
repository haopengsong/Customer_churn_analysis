 ### Description
 
 This project utilized several supervised learning models to identify 
 customers who are likely to cancel services in the near future.
 Also, it extracted and analyzed top factors that contribute the most
 to user retention which could help companies to prevent from losing customers.
 
 The model formulated in this project would be suitable for predicting whether 
 a customer would or would not drop their subscriptions.

### Part 1 - Data Exploration 

#### 1.1 Load & understand the dataset


```python
from google.colab import drive
drive.mount("/content/drive")
```

    Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).



```python
!pip install seaborn --upgrade
!pip install xgboost
import pandas as pd
import numpy as np
import imblearn
import warnings
warnings.filterwarnings('ignore')

# to show all the columns
pd.set_option('display.max_columns', None)

data = pd.read_csv('/content/drive/My Drive/Colab Notebooks/churn.all')
data.head()
```

    Requirement already up-to-date: seaborn in /usr/local/lib/python3.6/dist-packages (0.11.0)
    Requirement already satisfied, skipping upgrade: numpy>=1.15 in /usr/local/lib/python3.6/dist-packages (from seaborn) (1.18.5)
    Requirement already satisfied, skipping upgrade: pandas>=0.23 in /usr/local/lib/python3.6/dist-packages (from seaborn) (1.0.5)
    Requirement already satisfied, skipping upgrade: scipy>=1.0 in /usr/local/lib/python3.6/dist-packages (from seaborn) (1.4.1)
    Requirement already satisfied, skipping upgrade: matplotlib>=2.2 in /usr/local/lib/python3.6/dist-packages (from seaborn) (3.2.2)
    Requirement already satisfied, skipping upgrade: python-dateutil>=2.6.1 in /usr/local/lib/python3.6/dist-packages (from pandas>=0.23->seaborn) (2.8.1)
    Requirement already satisfied, skipping upgrade: pytz>=2017.2 in /usr/local/lib/python3.6/dist-packages (from pandas>=0.23->seaborn) (2018.9)
    Requirement already satisfied, skipping upgrade: kiwisolver>=1.0.1 in /usr/local/lib/python3.6/dist-packages (from matplotlib>=2.2->seaborn) (1.2.0)
    Requirement already satisfied, skipping upgrade: cycler>=0.10 in /usr/local/lib/python3.6/dist-packages (from matplotlib>=2.2->seaborn) (0.10.0)
    Requirement already satisfied, skipping upgrade: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.1 in /usr/local/lib/python3.6/dist-packages (from matplotlib>=2.2->seaborn) (2.4.7)
    Requirement already satisfied, skipping upgrade: six>=1.5 in /usr/local/lib/python3.6/dist-packages (from python-dateutil>=2.6.1->pandas>=0.23->seaborn) (1.15.0)
    Requirement already satisfied: xgboost in /usr/local/lib/python3.6/dist-packages (0.90)
    Requirement already satisfied: scipy in /usr/local/lib/python3.6/dist-packages (from xgboost) (1.4.1)
    Requirement already satisfied: numpy in /usr/local/lib/python3.6/dist-packages (from xgboost) (1.18.5)


    /usr/local/lib/python3.6/dist-packages/sklearn/externals/six.py:31: FutureWarning: The module is deprecated in version 0.21 and will be removed in version 0.23 since we've dropped support for Python 2.7. Please rely on the official version of six (https://pypi.org/project/six/).
      "(https://pypi.org/project/six/).", FutureWarning)
    /usr/local/lib/python3.6/dist-packages/sklearn/utils/deprecation.py:144: FutureWarning: The sklearn.neighbors.base module is  deprecated in version 0.22 and will be removed in version 0.24. The corresponding classes / functions should instead be imported from sklearn.neighbors. Anything that cannot be imported from sklearn.neighbors is now part of the private API.
      warnings.warn(message, FutureWarning)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>state</th>
      <th>account_length</th>
      <th>area_code</th>
      <th>phone_number</th>
      <th>intl_plan</th>
      <th>voice_mail_plan</th>
      <th>number_vmail_messages</th>
      <th>total_day_minutes</th>
      <th>total_day_calls</th>
      <th>total_day_charge</th>
      <th>total_eve_minutes</th>
      <th>total_eve_calls</th>
      <th>total_eve_charge</th>
      <th>total_night_minutes</th>
      <th>total_night_calls</th>
      <th>total_night_charge</th>
      <th>total_intl_minutes</th>
      <th>total_intl_calls</th>
      <th>total_intl_charge</th>
      <th>number_customer_service_calls</th>
      <th>churned</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>KS</td>
      <td>128</td>
      <td>415</td>
      <td>382-4657</td>
      <td>no</td>
      <td>yes</td>
      <td>25</td>
      <td>265.1</td>
      <td>110</td>
      <td>45.07</td>
      <td>197.4</td>
      <td>99</td>
      <td>16.78</td>
      <td>244.7</td>
      <td>91</td>
      <td>11.01</td>
      <td>10.0</td>
      <td>3</td>
      <td>2.70</td>
      <td>1</td>
      <td>False.</td>
    </tr>
    <tr>
      <th>1</th>
      <td>OH</td>
      <td>107</td>
      <td>415</td>
      <td>371-7191</td>
      <td>no</td>
      <td>yes</td>
      <td>26</td>
      <td>161.6</td>
      <td>123</td>
      <td>27.47</td>
      <td>195.5</td>
      <td>103</td>
      <td>16.62</td>
      <td>254.4</td>
      <td>103</td>
      <td>11.45</td>
      <td>13.7</td>
      <td>3</td>
      <td>3.70</td>
      <td>1</td>
      <td>False.</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NJ</td>
      <td>137</td>
      <td>415</td>
      <td>358-1921</td>
      <td>no</td>
      <td>no</td>
      <td>0</td>
      <td>243.4</td>
      <td>114</td>
      <td>41.38</td>
      <td>121.2</td>
      <td>110</td>
      <td>10.30</td>
      <td>162.6</td>
      <td>104</td>
      <td>7.32</td>
      <td>12.2</td>
      <td>5</td>
      <td>3.29</td>
      <td>0</td>
      <td>False.</td>
    </tr>
    <tr>
      <th>3</th>
      <td>OH</td>
      <td>84</td>
      <td>408</td>
      <td>375-9999</td>
      <td>yes</td>
      <td>no</td>
      <td>0</td>
      <td>299.4</td>
      <td>71</td>
      <td>50.90</td>
      <td>61.9</td>
      <td>88</td>
      <td>5.26</td>
      <td>196.9</td>
      <td>89</td>
      <td>8.86</td>
      <td>6.6</td>
      <td>7</td>
      <td>1.78</td>
      <td>2</td>
      <td>False.</td>
    </tr>
    <tr>
      <th>4</th>
      <td>OK</td>
      <td>75</td>
      <td>415</td>
      <td>330-6626</td>
      <td>yes</td>
      <td>no</td>
      <td>0</td>
      <td>166.7</td>
      <td>113</td>
      <td>28.34</td>
      <td>148.3</td>
      <td>122</td>
      <td>12.61</td>
      <td>186.9</td>
      <td>121</td>
      <td>8.41</td>
      <td>10.1</td>
      <td>3</td>
      <td>2.73</td>
      <td>3</td>
      <td>False.</td>
    </tr>
  </tbody>
</table>
</div>




```python
# shape of the dataset
print('number of rows: {} and columns: {}'.format(data.shape[0], data.shape[1]))
```

    number of rows: 5000 and columns: 21


#### 1.2 Data cleaning


```python
# remove unnecessary whitespaces among features that we are about to use
print('before:' + str(data['voice_mail_plan'][0]))
print('before:' + str(data['intl_plan'][0]))
print('before:' + str(data['churned'][0]))
```

    before: yes
    before: no
    before: False.



```python
# print(data['voice_mail_plan'])
```


```python
# remove heading and trailing whitespaces
data['voice_mail_plan'] = data['voice_mail_plan'].map(lambda x : x.strip())
data['intl_plan'] = data['intl_plan'].map(lambda x : x.strip())
data['churned'] = data['churned'].map(lambda x : x.strip())
```


```python
print('after:' + str(data['voice_mail_plan'][0]))
print('after:' + str(data['intl_plan'][0]))
print('after:' + str(data['churned'][0]))
```

    after:yes
    after:no
    after:False.


#### 1.3 Explore features


```python
# plot feature values to show distribution
%matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns

sns.displot(data['intl_plan'])
sns.displot(data['total_night_charge'])
sns.displot(data['number_customer_service_calls'])
```




    <seaborn.axisgrid.FacetGrid at 0x7f5d5c8b5a20>




![png](/graph/output_12_1.png)



![png](/graph/output_12_2.png)



![png](/graph/output_12_3.png)



```python
# calculate the Pearson correlations between all continuous features
data['churned'] = data['churned'].map({'False.' : 0, 'True.': 1})
pcorr = data.corr()
#print(pcorr)
fig, ax = plt.subplots(figsize=(22,12))
sns.heatmap(pcorr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x7f5d5bea8f28>




![png](/graph/output_13_1.png)


After examining the heatmap, we can conclude that features that have a linear correlation with churned customers are **'total_day_minutes'**, **'total_day_charge'** and **'number_customer_service_calls'**

### Part 2 - Feature Preprocessing and Selection


```python
# calculate pearson correlation between two features
from scipy.stats import pearsonr
# linear correlation
print (pearsonr(data['total_intl_minutes'], data['total_intl_charge'])[0])
# no linear correlation
print (pearsonr(data['total_intl_minutes'], data['total_night_calls'])[0])
```

    0.9999926570208357
    0.00039075729962830945



```python
# get true Y output - ground truth
Y = data['churned']

# drop unnecessary features
data_train = data.drop(['state','area_code','phone_number','churned'], axis = 1)

# convert categorical features into boolean values
data_train[["intl_plan","voice_mail_plan"]] = data_train[["intl_plan","voice_mail_plan"]] == 'yes'
X = data_train
```


```python
#print(X_train)
```


```python
# check the propotion of churned customers
print('{}%'.format((Y.sum() / Y.shape * 100)[0]))
```

    14.14%



```python
X.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>account_length</th>
      <th>intl_plan</th>
      <th>voice_mail_plan</th>
      <th>number_vmail_messages</th>
      <th>total_day_minutes</th>
      <th>total_day_calls</th>
      <th>total_day_charge</th>
      <th>total_eve_minutes</th>
      <th>total_eve_calls</th>
      <th>total_eve_charge</th>
      <th>total_night_minutes</th>
      <th>total_night_calls</th>
      <th>total_night_charge</th>
      <th>total_intl_minutes</th>
      <th>total_intl_calls</th>
      <th>total_intl_charge</th>
      <th>number_customer_service_calls</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>128</td>
      <td>False</td>
      <td>True</td>
      <td>25</td>
      <td>265.1</td>
      <td>110</td>
      <td>45.07</td>
      <td>197.4</td>
      <td>99</td>
      <td>16.78</td>
      <td>244.7</td>
      <td>91</td>
      <td>11.01</td>
      <td>10.0</td>
      <td>3</td>
      <td>2.70</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>107</td>
      <td>False</td>
      <td>True</td>
      <td>26</td>
      <td>161.6</td>
      <td>123</td>
      <td>27.47</td>
      <td>195.5</td>
      <td>103</td>
      <td>16.62</td>
      <td>254.4</td>
      <td>103</td>
      <td>11.45</td>
      <td>13.7</td>
      <td>3</td>
      <td>3.70</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>137</td>
      <td>False</td>
      <td>False</td>
      <td>0</td>
      <td>243.4</td>
      <td>114</td>
      <td>41.38</td>
      <td>121.2</td>
      <td>110</td>
      <td>10.30</td>
      <td>162.6</td>
      <td>104</td>
      <td>7.32</td>
      <td>12.2</td>
      <td>5</td>
      <td>3.29</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>84</td>
      <td>True</td>
      <td>False</td>
      <td>0</td>
      <td>299.4</td>
      <td>71</td>
      <td>50.90</td>
      <td>61.9</td>
      <td>88</td>
      <td>5.26</td>
      <td>196.9</td>
      <td>89</td>
      <td>8.86</td>
      <td>6.6</td>
      <td>7</td>
      <td>1.78</td>
      <td>2</td>
    </tr>
    <tr>
      <th>4</th>
      <td>75</td>
      <td>True</td>
      <td>False</td>
      <td>0</td>
      <td>166.7</td>
      <td>113</td>
      <td>28.34</td>
      <td>148.3</td>
      <td>122</td>
      <td>12.61</td>
      <td>186.9</td>
      <td>121</td>
      <td>8.41</td>
      <td>10.1</td>
      <td>3</td>
      <td>2.73</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Encode and add categorical features (state)
data_encode_add = data.drop(['area_code', 'phone_number', 'churned'], axis = 1)

# convert categorical features into boolean values
data_encode_add[['intl_plan', 'voice_mail_plan']] = data_encode_add[['intl_plan', 'voice_mail_plan']] == 'yes'

# encode feature
data_encode_add = pd.get_dummies(data_encode_add, columns=['state'])

data_encode_add.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>account_length</th>
      <th>intl_plan</th>
      <th>voice_mail_plan</th>
      <th>number_vmail_messages</th>
      <th>total_day_minutes</th>
      <th>total_day_calls</th>
      <th>total_day_charge</th>
      <th>total_eve_minutes</th>
      <th>total_eve_calls</th>
      <th>total_eve_charge</th>
      <th>total_night_minutes</th>
      <th>total_night_calls</th>
      <th>total_night_charge</th>
      <th>total_intl_minutes</th>
      <th>total_intl_calls</th>
      <th>total_intl_charge</th>
      <th>number_customer_service_calls</th>
      <th>state_AK</th>
      <th>state_AL</th>
      <th>state_AR</th>
      <th>state_AZ</th>
      <th>state_CA</th>
      <th>state_CO</th>
      <th>state_CT</th>
      <th>state_DC</th>
      <th>state_DE</th>
      <th>state_FL</th>
      <th>state_GA</th>
      <th>state_HI</th>
      <th>state_IA</th>
      <th>state_ID</th>
      <th>state_IL</th>
      <th>state_IN</th>
      <th>state_KS</th>
      <th>state_KY</th>
      <th>state_LA</th>
      <th>state_MA</th>
      <th>state_MD</th>
      <th>state_ME</th>
      <th>state_MI</th>
      <th>state_MN</th>
      <th>state_MO</th>
      <th>state_MS</th>
      <th>state_MT</th>
      <th>state_NC</th>
      <th>state_ND</th>
      <th>state_NE</th>
      <th>state_NH</th>
      <th>state_NJ</th>
      <th>state_NM</th>
      <th>state_NV</th>
      <th>state_NY</th>
      <th>state_OH</th>
      <th>state_OK</th>
      <th>state_OR</th>
      <th>state_PA</th>
      <th>state_RI</th>
      <th>state_SC</th>
      <th>state_SD</th>
      <th>state_TN</th>
      <th>state_TX</th>
      <th>state_UT</th>
      <th>state_VA</th>
      <th>state_VT</th>
      <th>state_WA</th>
      <th>state_WI</th>
      <th>state_WV</th>
      <th>state_WY</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>128</td>
      <td>False</td>
      <td>True</td>
      <td>25</td>
      <td>265.1</td>
      <td>110</td>
      <td>45.07</td>
      <td>197.4</td>
      <td>99</td>
      <td>16.78</td>
      <td>244.7</td>
      <td>91</td>
      <td>11.01</td>
      <td>10.0</td>
      <td>3</td>
      <td>2.70</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>107</td>
      <td>False</td>
      <td>True</td>
      <td>26</td>
      <td>161.6</td>
      <td>123</td>
      <td>27.47</td>
      <td>195.5</td>
      <td>103</td>
      <td>16.62</td>
      <td>254.4</td>
      <td>103</td>
      <td>11.45</td>
      <td>13.7</td>
      <td>3</td>
      <td>3.70</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>137</td>
      <td>False</td>
      <td>False</td>
      <td>0</td>
      <td>243.4</td>
      <td>114</td>
      <td>41.38</td>
      <td>121.2</td>
      <td>110</td>
      <td>10.30</td>
      <td>162.6</td>
      <td>104</td>
      <td>7.32</td>
      <td>12.2</td>
      <td>5</td>
      <td>3.29</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>84</td>
      <td>True</td>
      <td>False</td>
      <td>0</td>
      <td>299.4</td>
      <td>71</td>
      <td>50.90</td>
      <td>61.9</td>
      <td>88</td>
      <td>5.26</td>
      <td>196.9</td>
      <td>89</td>
      <td>8.86</td>
      <td>6.6</td>
      <td>7</td>
      <td>1.78</td>
      <td>2</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>75</td>
      <td>True</td>
      <td>False</td>
      <td>0</td>
      <td>166.7</td>
      <td>113</td>
      <td>28.34</td>
      <td>148.3</td>
      <td>122</td>
      <td>12.61</td>
      <td>186.9</td>
      <td>121</td>
      <td>8.41</td>
      <td>10.1</td>
      <td>3</td>
      <td>2.73</td>
      <td>3</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



### Part 3 : Model Training and Evaluation
#### 3.1 Split data


```python
# train & test split
from sklearn.model_selection import train_test_split

# 25% for testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)

print('training data has %d rows and %d columns' % X_train.shape)
print('testing data has %d rows and %d columns' % X_test.shape)
```

    training data has 4000 rows and 17 columns
    testing data has 1000 rows and 17 columns



```python
# data normalization
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
```

#### 3.2 Model selection


```python
# evaluate and compare the different classifiers using cross-validated ROC-AUC 
from xgboost import  XGBClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

```


```python
# building a K-fold cross validation
random_state = 55
clfs = [SVC(random_state=random_state), 
      RandomForestClassifier(random_state=random_state),
      AdaBoostClassifier(n_estimators=100, random_state=random_state, learning_rate=0.5),
      GradientBoostingClassifier(random_state=random_state),
      KNeighborsClassifier(),
      LogisticRegression(random_state=random_state),
      XGBClassifier(random_state=random_state)]

# running a 10-fold cross validation to get accuracy for each models
cv_means = []
cv_std = []
for clf in clfs:
    cv_results = cross_val_score(clf, X_train, Y_train, scoring="roc_auc", cv = 10)
    cv_means.append(cv_results.mean())
    cv_std.append(cv_results.std())
    
cv_res = pd.DataFrame({"CV mean":cv_means,"CV std": cv_std,"Classifiers":["SVC","RandomForest","AdaBoost",
"GradientBoosting","KNeighboors","LogisticRegression","XGBoost"]})

sns.set_palette("Set2")
graph = sns.barplot(x='CV mean', y='Classifiers', data= cv_res, orient='h')
print(cv_res)
```

        CV mean    CV std         Classifiers
    0  0.909970  0.026767                 SVC
    1  0.915946  0.033483        RandomForest
    2  0.865312  0.024128            AdaBoost
    3  0.922301  0.024970    GradientBoosting
    4  0.847158  0.027540         KNeighboors
    5  0.826568  0.030604  LogisticRegression
    6  0.923936  0.023670             XGBoost



![png](/graph/output_27_1.png)


'''
Choose RandomForest, GradientBoosting, KNeighboors, LogisticRegression, and XGBoost for 
hyperparameters tunning
'''

#### 3.3 Hyperparameter tunning with grid search


```python
# print grid search results
def grid_search_output(gs):
    print('Best precision: %0.3f' % gs.best_score_)
    print('Best parameters set: \n', gs.best_params_)
```

##### Logistic Regression


```python
# possible hyperparameter options for logistic regression:
# regularization: L1 or L2, regularization parameter lambda c
lr_parameters = {
    'penalty' : ('l1', 'l2'),
    'C' : (1, 5, 10, 15, 20)
}
grid_lr = GridSearchCV(LogisticRegression(), lr_parameters, cv = 10)
grid_lr.fit(X_train, Y_train)
# Best parameters option
grid_search_output(grid_lr)
```

    Best precision: 0.863
    Best parameters set: 
     {'C': 1, 'penalty': 'l2'}


##### KNN


```python
# find the best k value
knn_parameter = {
    'n_neighbors' : [3,4,6, 7,9,12]
}
grid_knn = GridSearchCV(KNeighborsClassifier(), knn_parameter, cv=10)
grid_knn.fit(X_train, Y_train)
# best k
grid_search_output(grid_knn)
```

    Best precision: 0.900
    Best parameters set: 
     {'n_neighbors': 3}


##### Random Forest


```python
# find the best number of trees
rf_parameter = {
    'n_estimators' : [20, 50 ,80]
}
rf_grid = GridSearchCV(RandomForestClassifier(), rf_parameter, cv = 10)
rf_grid.fit(X_train, Y_train)
grid_search_output(rf_grid)
```

    Best precision: 0.956
    Best parameters set: 
     {'n_estimators': 50}


##### Gradient Boosting


```python
# possible parameters for tunning
# loss, n_estimators, learning_rate, max_depth, min_samples_leaf, max_features
gb_parameters = {
    'loss' : ['deviance', 'exponential'],
    'n_estimators' : [100, 150],
    'learning_rate' : [0.1, 0.2, 0.01],
    'max_depth' : [3, 6],
    'min_samples_leaf' : [5, 10, 15],
    'max_features' : [0.2, 0.5]
}
gb_grid = GridSearchCV(GradientBoostingClassifier(), gb_parameters, cv = 10)
gb_grid.fit(X_train, Y_train)
grid_search_output(gb_grid)
```

    Best precision: 0.961
    Best parameters set: 
     {'learning_rate': 0.1, 'loss': 'deviance', 'max_depth': 6, 'max_features': 0.5, 'min_samples_leaf': 5, 'n_estimators': 100}


##### XGBoost


```python
# xgboost tunning
xgb_parameters = {
    'learning_rate' : [0.01, 0.1, 0.05],
    'max_depth' : [2, 4, 6],
    'subsample' : [0.25, 0.5]
}
xgb_grid = GridSearchCV(XGBClassifier(), xgb_parameters, cv = 5)
xgb_grid.fit(X_train, Y_train)
grid_search_output(xgb_grid)
```

    Best precision: 0.958
    Best parameters set: 
     {'learning_rate': 0.05, 'max_depth': 4, 'subsample': 0.5}


#### 3.4 Model Evaluation 
##### 3.4.1 Confusion Matrix (Precision, Recall, Accuracy

Identify churned customers as positive samples

True Positive: churned

Precision = tp / tp + fp

high precision = not many retained users were predicated as churn users

Recall(sensitivity) = tp / tp + fn 

high recall = out of all churned customers, many of them were correctly identified


```python
from sklearn.metrics import confusion_matrix
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score

# calculate recall, precision and accuracy
def cal_rpa(clf, cm):
    tn = cm[0][0]
    fp = cm[0][1]
    fn = cm[1][0]
    tp = cm[1][1]
    accuracy  = (tp + tn) / (tp + fp + fn + tn + 0.0)
    precision = tp / (tp + fp + 0.0)
    recall = tp / (tp + fn + 0.0)
    print (clf)
    print ("Accuracy is: %0.3f" % accuracy)
    print ("precision is: %0.3f" % precision)
    print ("recall is: %0.3f" % recall)

# print confusion matrix
def draw_confusion_matrix(confusion_matrix):
    class_values = ['Not churned', 'churned']
    for cm in confusion_matrix:
        clf, cm = cm[0], cm[1]
        cal_rpa(clf, cm)
        fig = plt.figure()
        ax = fig.add_subplot()
        color_ax = ax.matshow(cm, interpolation = 'nearest', cmap=plt.get_cmap('Greens'))
        plt.title('Predicated')
        fig.colorbar(color_ax)
        ax.set_xticklabels([''] + class_values)
        ax.set_yticklabels([''] + class_values)
        plt.xlabel('Confusion Mattrix for %s' % clf)
        plt.ylabel('Actual')
        plt.show()
```


```python
# confusion matrix for RandomForest, GradientBoosting, KNeighboors, LogisticRegression, and XGBoost
cms = [
    ('RandomForest', confusion_matrix(Y_test, rf_grid.best_estimator_.predict(X_test))),
    ('GradientBoosting', confusion_matrix(Y_test, gb_grid.best_estimator_.predict(X_test))),
    ('KNeighboors', confusion_matrix(Y_test, grid_knn.best_estimator_.predict(X_test))),
    ('LogisticRegression', confusion_matrix(Y_test, grid_lr.best_estimator_.predict(X_test))),
    ('XGBoost', confusion_matrix(Y_test, xgb_grid.best_estimator_.predict(X_test))),
]
# cms = [
#     ('RandomForest', confusion_matrix(Y_test, rf_grid.best_estimator_.predict(X_test))),
#     ('KNeighboors', confusion_matrix(Y_test, grid_knn.best_estimator_.predict(X_test))),
#     ('LogisticRegression', confusion_matrix(Y_test, grid_lr.best_estimator_.predict(X_test))),
#     ('XGBoost', confusion_matrix(Y_test, xgb_grid.best_estimator_.predict(X_test))),
# ]
draw_confusion_matrix(cms)
```

    RandomForest
    Accuracy is: 0.955
    precision is: 0.938
    recall is: 0.734



![png](/graph/output_44_1.png)


    GradientBoosting
    Accuracy is: 0.957
    precision is: 0.939
    recall is: 0.748



![png](/graph/output_44_3.png)


    KNeighboors
    Accuracy is: 0.884
    precision is: 0.729
    recall is: 0.301



![png](/graph/output_44_5.png)


    LogisticRegression
    Accuracy is: 0.870
    precision is: 0.644
    recall is: 0.203



![png](/graph/output_44_7.png)


    XGBoost
    Accuracy is: 0.958
    precision is: 0.963
    recall is: 0.734



![png](/graph/output_44_9.png)


##### 3.4.2 ROC_AUC


```python
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

def draw_roc_auc(pred, clf):
    fpr, tpr, _ = roc_curve(Y_test, pred)
    plt.figure(1)
    plt.plot([0,1], [0,1], 'k--')
    plt.plot(fpr, tpr, label = clf)
    plt.xlabel('False Positive rate')
    plt.ylabel('True Positive rate')
    plt.title('ROC curve - %s' % clf)
    plt.legend(loc='best')
    plt.show()
    print('AUC score for {} is {}'.format(clf, roc_auc_score(Y_test, pred)))

    
# pick XGBoost, KNN and LR for comparsion as they are the best/moderate/worst performer
y_pred_xgb = xgb_grid.best_estimator_.predict_proba(X_test)[:,1]
y_pred_lr = grid_lr.best_estimator_.predict_proba(X_test)[:,1]
y_pred_knn = grid_knn.best_estimator_.predict_proba(X_test)[:,1]

draw_roc_auc(y_pred_lr, 'LR')
draw_roc_auc(y_pred_knn, 'KNN')
draw_roc_auc(y_pred_xgb, 'XGBoost')

```


![png](/graph/output_46_0.png)


    AUC score for LR is 0.8262845672413934



![png](/graph/output_46_2.png)


    AUC score for KNN is 0.795693221597539



![png](/graph/output_46_4.png)


    AUC score for XGBoost is 0.917528212744082


### Part 4. Feature Selection
#### Part 4.1 Feature selection for Logistic Regression
'''
By examining the correlation graph, features which are highly correlated 
are (total_day_minutes, total_day_charge), (total_eve_minutes, total_eve_charge) and (total_intl_minutes, total_intl_charge)
'''


```python
# use L1(LASSO) regularizaiton on LR and check for coefficient
scaler = StandardScaler()
lr_l1_x = scaler.fit_transform(X)
lr_l1 = LogisticRegression(solver = 'saga', penalty='l1', C = 0.05)
lr_l1.fit(lr_l1_x, Y)
print(lr_l1.coef_[0])

print('LR coefficients with L1 regularization: ')
for k, v in sorted(zip(data_train.columns,  
                       map(lambda x : round(x, 4), lr_l1.coef_[0])),
                  key=lambda kv : (-abs(kv[1]), kv[0])):
    print(k + ": " + str(v))
```

    [ 0.01568818  0.57140723 -0.40522927  0.          0.34954893  0.01046044
      0.31982883  0.14573546  0.          0.15284706  0.08721141  0.
      0.06441896  0.08658136 -0.12382309  0.10475627  0.61807958]
    LR coefficients with L1 regularization: 
    number_customer_service_calls: 0.6181
    intl_plan: 0.5714
    voice_mail_plan: -0.4052
    total_day_minutes: 0.3495
    total_day_charge: 0.3198
    total_eve_charge: 0.1528
    total_eve_minutes: 0.1457
    total_intl_calls: -0.1238
    total_intl_charge: 0.1048
    total_night_minutes: 0.0872
    total_intl_minutes: 0.0866
    total_night_charge: 0.0644
    account_length: 0.0157
    total_day_calls: 0.0105
    number_vmail_messages: 0.0
    total_eve_calls: 0.0
    total_night_calls: 0.0



```python
# use L2(ridge) regularization on LR and check for coefficient

lr_l2_x = scaler.fit_transform(X)
lr_l2 = LogisticRegression(penalty='l2', C = 0.05)
lr_l2.fit(lr_l2_x, Y)
print(lr_l2.coef_[0])

print('LR coefficients with L2 regularization: ')
for k, v in sorted(zip(data_train.columns,  
                       map(lambda x : round(x, 4), lr_l2.coef_[0])),
                  key=lambda kv : (-abs(kv[1]), kv[0])):
    print(k + ": " + str(v))
```

    [ 0.05385442  0.59306133 -0.49450361  0.05195019  0.35609257  0.04879417
      0.35572945  0.17269978 -0.02871012  0.17281055  0.09729187 -0.03095169
      0.09697765  0.117511   -0.16624232  0.11774349  0.64532563]
    LR coefficients with L2 regularization: 
    number_customer_service_calls: 0.6453
    intl_plan: 0.5931
    voice_mail_plan: -0.4945
    total_day_minutes: 0.3561
    total_day_charge: 0.3557
    total_eve_charge: 0.1728
    total_eve_minutes: 0.1727
    total_intl_calls: -0.1662
    total_intl_charge: 0.1177
    total_intl_minutes: 0.1175
    total_night_minutes: 0.0973
    total_night_charge: 0.097
    account_length: 0.0539
    number_vmail_messages: 0.052
    total_day_calls: 0.0488
    total_night_calls: -0.031
    total_eve_calls: -0.0287


#### 4.2 Random forest model - feature importance discussion


```python
# check feature importance of random forest for feature selection
forest = RandomForestClassifier()
forest.fit(X, Y)
importances = forest.feature_importances_
# print the feature ranking
for k, v in sorted(zip(map(lambda x : round(x, 4), importances),
                      X.columns), reverse = True):
    print(v + ": " + str(k))
```

    total_day_charge: 0.1493
    total_day_minutes: 0.1397
    number_customer_service_calls: 0.1166
    intl_plan: 0.0916
    total_eve_charge: 0.0661
    total_eve_minutes: 0.0654
    total_intl_calls: 0.0557
    total_night_minutes: 0.0421
    total_night_charge: 0.0395
    total_intl_charge: 0.039
    total_intl_minutes: 0.0374
    account_length: 0.0294
    number_vmail_messages: 0.0278
    total_day_calls: 0.0276
    total_night_calls: 0.0263
    total_eve_calls: 0.0244
    voice_mail_plan: 0.022


We can conclude that 'total_day_minutes' and 'total_day_charge' play a relatively important in making the classification decision

### Conclusion

After comparing different models, we can see that boosting and tree based model tend to perform well. For gradient boosting, the number of estimators can aggressively affect the classification performances, the more estimators we set, the better the precision. However, there is a good chance that gradient boosting might overfit the training data. This happened when gradient boosting underperformed on testing data compared with XGboost.

The factors that contribute the most on customer retention are 'total_day_charge', 'number_customer_service_calls' and 'total_day_minutes'. This is not unexpected and we can easily imply that
customers who spend more money on calls would expect a better service and reliable network connection ,however, if the service isn't as good as they expected, they are more likely to complain and the number of customer service calls would definitely go up. 


```python

```
