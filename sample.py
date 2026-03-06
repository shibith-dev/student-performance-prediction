import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("E:/Career/Professional_ML/02_Machine_Learning/01_Projects/student_performance_prediction/data/ResearchInformation3.csv")
print(df.head(3))
'''
	Department	Gender	HSC	SSC	Income	Hometown	Computer	Preparation	Gaming	Attendance	Job	English	Extra	Semester	Last	Overall
0	Business Administration	Male	4.17	4.84	Low (Below 15,000)	Village	3	More than 3 Hours	0-1 Hour	80%-100%	No	3	Yes	6	3.220	3.350
1	Business Administration	Female	4.92	5.00	Upper middle (30,000-50,000)	City	3	0-1 Hour	0-1 Hour	80%-100%	No	3	Yes	7	3.467	3.467
2	Business Administration	Male	5.00	4.83	Lower middle (15,000-30,000)	Village	3	0-1 Hour	More than 3 Hours	80%-100%	No	4	Yes	3	4.000	3.720
'''
num_col = ['HSC', 'SSC', 'Computer', 'English', 'Semester', "last"]
Ordinal_cat_Col = ['Income','Preparation', 'Gaming', 'Attendance']
cat_col = ['Department', 'Hometown', 'Gender','Job', 'Extra']

x = df.drop(["Overall"], axis=1)
y = df["Overall"]
print(x.shape, y.shape)

x_train, x_test, y_train, y_test  = train_test_split(x, y, test_size=0.2, random_state=42)
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

income_order = ["Low (Below 15,000)","Lower middle (15,000-30,000)","Upper middle (30,000-50,000)","High (Above 50,000)"]
attendence_order = ['Below 40%','40%-59%','60%-79%','80%-100%']
preperation_order = ['0-1 Hour','2-3 Hours','More than 3 Hours']
gaming_order = ['0-1 Hour', '2-3 Hours','More than 3 Hours']

Ordinal_Encoder = OrdinalEncoder(categories=[income_order, preperation_order, gaming_order, attendence_order])

preprocessor = ColumnTransformer(
    transformers=[
        ("OHE", OneHotEncoder(drop="if_binary", handle_unknown="ignore"), cat_col),
        ("OrdinalEncoder", Ordinal_Encoder, Ordinal_cat_Col)],
    remainder="passthrough"
)

x_train_preprocessed = preprocessor.fit_transform(x_train)
x_test_preprocessed = preprocessor.transform(x_test)



