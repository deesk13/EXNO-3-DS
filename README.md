## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```
import pandas as pd
df=pd.read_csv("/content/Encoding Data.csv")
df
```
![Screenshot 2024-10-26 104752](https://github.com/user-attachments/assets/075c00b7-05e1-483f-a364-4a065cc9fa69)

```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
![Screenshot 2024-10-26 104951](https://github.com/user-attachments/assets/259bbe3a-ffa6-4d8c-89ef-ddd13e9a8411)
```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
![Screenshot 2024-10-26 105101](https://github.com/user-attachments/assets/0d65b1ba-b322-4d2b-adc0-211b88032e5a)
```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![Screenshot 2024-10-26 105201](https://github.com/user-attachments/assets/d17cdebb-8063-4c1c-8a6c-b0ee14c6043b)
```
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
```
![Screenshot 2024-10-26 105259](https://github.com/user-attachments/assets/b2cc05a8-b85b-47e4-babc-3e50990ef5c8)
```
df2=pd.concat([df2,enc],axis=1)
df2
```
![Screenshot 2024-10-26 105345](https://github.com/user-attachments/assets/619d61e1-22fd-4d00-922c-70b6d37853f6)
```
pd.get_dummies(df2,columns=["nom_0"])
```
![Screenshot 2024-10-26 105432](https://github.com/user-attachments/assets/ce6f6f2a-0314-4aa3-951c-e0fa779504f2)
```
pip install --upgrade category_encoders
```
![Screenshot 2024-10-26 105520](https://github.com/user-attachments/assets/04f79c02-5599-4c20-9f37-74b947ca0bea)
```
from category_encoders import BinaryEncoder
df=pd.read_csv("/content/data.csv")
df
```
![Screenshot 2024-10-26 105616](https://github.com/user-attachments/assets/94662bdb-7ce3-4331-b904-6521930f2ca2)
```
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
dfb1=df.copy()
dfb
```
![Screenshot 2024-10-26 105810](https://github.com/user-attachments/assets/718a218e-45d9-4d60-98dd-cab30d254c0b)
```
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
```
![Screenshot 2024-10-26 105851](https://github.com/user-attachments/assets/e52d4e9c-0cdd-4124-9179-ec51a87f2a4c)
```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/content/Data_to_Transform.csv")
df
```
![Screenshot 2024-10-26 105949](https://github.com/user-attachments/assets/dc3b803a-a1e4-45fb-a640-f2efb1c05600)
```
df.skew()
```
![Screenshot 2024-10-26 110038](https://github.com/user-attachments/assets/0975469b-6747-470a-bd0e-a027e4bbb680)
```
np.log(df["Highly Positive Skew"])
```
![Screenshot 2024-10-26 110116](https://github.com/user-attachments/assets/cd62052a-2812-42ff-a0c3-f49ebd1dc848)
```
np.reciprocal(df["Moderate Positive Skew"])
```
![Screenshot 2024-10-26 110210](https://github.com/user-attachments/assets/9f68c7d1-72ff-40a6-b243-2ac58799ae5e)
```
np.sqrt(df["Highly Positive Skew"])
```
![Screenshot 2024-10-26 110250](https://github.com/user-attachments/assets/c14883dc-d478-46da-83d6-e145e60c111c)
```
np.square(df["Highly Positive Skew"])
```
![Screenshot 2024-10-26 110336](https://github.com/user-attachments/assets/a9311d78-946b-4f9f-88a8-a5e722c62083)
```
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
![Screenshot 2024-10-26 110416](https://github.com/user-attachments/assets/2fb73401-fec7-40d2-8c79-505b7b4eebe2)
```
df["Moderate Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Moderate Negative Skew"])
```
![Screenshot 2024-10-26 110458](https://github.com/user-attachments/assets/cf8b211d-d3c0-42c7-8852-9ca7104f8f2a)
```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![Screenshot 2024-10-26 110545](https://github.com/user-attachments/assets/da1349d5-07f8-474e-9b2f-6b7d13151c5f)
```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew_1"]),line='45')
plt.show()
```
![Screenshot 2024-10-26 110625](https://github.com/user-attachments/assets/99522244-4398-4f08-b8be-ae0159aa8974)
```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)

df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])

sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![Screenshot 2024-10-26 110713](https://github.com/user-attachments/assets/f6615ec4-cf2c-4442-a4b5-867d9a99d2df)
```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```
![Screenshot 2024-10-26 110753](https://github.com/user-attachments/assets/7b0b5429-c38f-4642-819c-481460bfed6c)
```
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
```
![Screenshot 2024-10-26 110834](https://github.com/user-attachments/assets/90dbdd1b-e906-4e5e-91ff-9cbac6ab2869)
```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```
![Screenshot 2024-10-26 110920](https://github.com/user-attachments/assets/51957a2b-8562-45f2-8350-babbd0acd649)

# RESULT:
      Thus the given data, Feature Encoding, Transformation process and save the data to a file was performed successfully.


       
