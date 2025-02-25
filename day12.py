# Assignment
# v0.9) v0.8 파일의 결측치 값을 산술평균으로 채워 넣는 다양한 방법을 적용하시오.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from sklearn.impute import SimpleInputer # Annotating the Syketrun Library

#Create Example Data
df = pd.DataFrame(
    {
        'A' : [1, 2,  np.nan, 4],
        'B' : [np.nan, 12, 3, 4],
        'C' : [1, 2, 3, 4]
    }
)

print("data")
print(df)

#Filling missing values by arithmetic mean
A_mean = df['A'].mean()
B_mean = df['B'].mean()
df['A'].fillna(A_mean, inplace=True)
df['B'].fillna(B_mean, inplace=True)

print("\n 결측치 평균값을 채운 후 데이터")
print(df)

# Visualization (A,B column data distribution)
plt.figure(figsize=(6, 4))

#Visualize column A
plt.scatter(range(len(df)), df['A'], color='blue', label='A (Age)')

#Visualize column B
plt.scatter(range(len(df)), df['B'], color='red', label='B (Other Variable)')

plt.title('A vs B Column Distribution After Filling Missing Values')
plt.xlabel('Index')
plt.ylabel('Values')
plt.legend()
plt.show()
