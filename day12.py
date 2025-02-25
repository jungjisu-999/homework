# Assignment
# v0.9) v0.8 파일의 결측치 값을 산술평균으로 채워 넣는 다양한 방법을 적용하시오.

import seaborn as sns
import matplotlib.pyplot as plt
# from sklearn.neighbors import KNeighborsRegressor  # 적용모델 : K 최근접 이웃 회귀 모델 (주석 처리)
# from sklearn.model_selection import train_test_split  # 훈련 / 검증 셋트 분할 함수 (주석 처리)

# 나이에 따른 생존율 계산
titanic = sns.load_dataset('titanic')  # 데이터 로딩
mean_age = titanic['age'].mean()  # 나이 평균값 산출
titanic_fill_row = titanic.fillna({'age': mean_age})  # 결측치 처리 (평균값으로 채움)

X = titanic_fill_row[['age']]  # 독립 변수 설정
y = titanic_fill_row[['survived']]  # 종속 변수 설정

# 시각화 (데이터 분포 확인용)
plt.figure(figsize=(5, 2))
plt.scatter(X, y, color='blue', label='Data Points')
plt.title('Titanic Age vs Survived')
plt.xlabel('Age')
plt.ylabel('Survived')
plt.legend()
plt.show()