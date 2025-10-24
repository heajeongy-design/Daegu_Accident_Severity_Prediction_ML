# 대구 교통사고 피해 예측 프로젝트 (Daegu Accident Severity Prediction)
https://dacon.io/competitions/official/236208/overview/description

---

## 1. 데이터 로드 및 기본 전처리

```python
import pandas as pd

# 데이터 로드
df = pd.read_csv('./train.csv')

# 피해운전자 관련 결측치 처리
victim_cols = ['피해운전자 차종', '피해운전자 성별', '피해운전자 연령', '피해운전자 상해정도']
for col in victim_cols:
    df[col] = df[col].fillna('없음')

# 분석용 주요 변수만 선택
eda_cols = ['사고일시', '요일', '기상상태', '시군구', '도로형태', '노면상태', '법규위반', '사고유형', '사고유형 - 세부분류', 'ECLO']
df_data = df[eda_cols].copy()
print(df_data.info())
```

**해석:** 전체 데이터는 39,609건으로, 사고일시·기상상태·도로형태·법규위반·사고유형·ECLO 중심으로 구성하였다. 피해운전자 결측치는 ‘없음’으로 처리해 데이터의 일관성을 확보했다.

---

## 2. 탐색적 데이터 분석 (EDA)

### (1) 요일별 평균 ECLO
<img width="831" height="546" alt="image" src="https://github.com/user-attachments/assets/a84a2567-7379-4949-aebb-f8336641c3a5" />


```python
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10,6))
sns.barplot(data=df_data, x='요일', y='ECLO', estimator='mean', ci=None)
plt.title('요일별 평균 ECLO')
plt.show()
```

**해석:** 일요일에 평균 ECLO가 가장 높고, 주말로 갈수록 피해 정도가 심화되는 경향이 있다.

### (2) 도로형태별 ECLO 분포 (IQR 이상치 제거)
<img width="1389" height="590" alt="image" src="https://github.com/user-attachments/assets/37971306-2240-46fa-925d-7ccdb5f1e11e" />


```python
Q1 = df['ECLO'].quantile(0.25)
Q3 = df['ECLO'].quantile(0.75)
IQR = Q3 - Q1
df_iqr = df[(df['ECLO'] >= Q1 - 1.5*IQR) & (df['ECLO'] <= Q3 + 1.5*IQR)]

plt.figure(figsize=(14,6))
sns.boxplot(data=df_iqr, x='도로형태', y='ECLO', palette='Blues')
plt.title('도로형태별 ECLO 분포 (IQR 이상치 제거 후)')
plt.xticks(rotation=20)
plt.show()
```

**해석:** 교차로·단일로 구간 등 교통 흐름이 복잡한 도로일수록 피해 강도가 높았다.

### (3) 기상상태별 사고 건수
<img width="989" height="590" alt="image" src="https://github.com/user-attachments/assets/8b0da5d6-d5c8-48d7-a1cb-0a0344910770" />

```python
weather_counts = df['기상상태'].value_counts().reset_index()
weather_counts.columns = ['기상상태', '사고건수']

plt.figure(figsize=(10,6))
sns.barplot(data=weather_counts, x='기상상태', y='사고건수', palette='Blues_d')
plt.title('기상상태별 사고건수')
plt.show()
```

**해석:** 맑은 날의 사고 건수가 압도적으로 많으며, 평상시 경계심 저하가 주요 요인으로 추정된다.

### (4) 군 단위 평균 ECLO 비교
<img width="1189" height="590" alt="image" src="https://github.com/user-attachments/assets/8c4410d8-d9a8-42a2-aaa5-5056af38bce7" />

```python
gun_df = df_data[df_data['시군구'].str.contains('군', na=False)].copy()
gun_df['군'] = gun_df['시군구'].apply(lambda x: x.split()[-1] if '군' in x else None)

gun_mean = gun_df.groupby('군')['ECLO'].mean().sort_values(ascending=False).reset_index()
plt.figure(figsize=(12,6))
sns.barplot(data=gun_mean, x='군', y='ECLO', palette='viridis')
plt.title('군 단위 평균 ECLO 비교')
plt.xticks(rotation=45)
plt.show()
```

**해석:** 논공읍·구지면에서 피해 수준이 높게 나타났으며, 외곽 지역일수록 비교적 안정적인 양상을 보였다.

---

## 3. 군집화 (KMeans)
<img width="1189" height="490" alt="image" src="https://github.com/user-attachments/assets/8016bbc0-7cf6-4491-baa7-83f9b32de298" />

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import numpy as np

# 시각화 변수 변환
df_cluster = df_data.copy()
df_cluster['사고시각'] = pd.to_datetime(df_cluster['사고일시']).dt.hour

# 원-핫 인코딩
df_cluster = pd.get_dummies(df_cluster, columns=['요일','기상상태','도로형태','노면상태'], drop_first=True)

# 스케일링
scaler = StandardScaler()
scaled = scaler.fit_transform(df_cluster.select_dtypes(include=np.number))

# Elbow / Silhouette
K_range = range(2, 11)
inertias, silhouettes = [], []
for k in K_range:
    model = KMeans(n_clusters=k, random_state=42, n_init=20)
    model.fit(scaled)
    inertias.append(model.inertia_)
    silhouettes.append(silhouette_score(scaled, model.labels_))

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(K_range, inertias, 'o-')
plt.title('Elbow Method')
plt.subplot(1,2,2)
plt.plot(K_range, silhouettes, 'o-', color='orange')
plt.title('Silhouette Score')
plt.show()

# 최적 군집 수 적용
df_cluster['cluster'] = KMeans(n_clusters=3, random_state=42, n_init=20).fit_predict(scaled)
print(df_cluster['cluster'].value_counts())
```

**해석:** 최적 군집 수는 3으로 도출되었으며, 사고 특성이 3가지 주요 패턴으로 구분됨을 시사한다.

---

## 4. 선형회귀 (PolynomialFeatures=2)
```python
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression

X = df_cluster.drop(columns=['ECLO', '사고일시'], errors='ignore')
y = pd.to_numeric(df_cluster['ECLO'], errors='coerce')

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X.select_dtypes(include=np.number))
X_poly = StandardScaler().fit_transform(X_poly)

model = LinearRegression()
model.fit(X_poly, y)
print(f'R² Score: {model.score(X_poly, y):.4f}')
```

**해석:** R² = 0.596 → 약 60%의 설명력 확보. ECLO는 복합적인 변수 상호작용의 영향을 받음을 의미한다.

---

## 5. ANOVA (유의 변수 검정)
```python
from scipy.stats import f_oneway

anova_results = []
for col in df_cluster.columns:
    if col == 'ECLO':
        continue
    if df_cluster[col].nunique() == 2:
        g1 = df_cluster.loc[df_cluster[col]==0, 'ECLO']
        g2 = df_cluster.loc[df_cluster[col]==1, 'ECLO']
        if len(g1)>3 and len(g2)>3:
            stat, p = f_oneway(g1, g2)
            anova_results.append({'변수명':col, 'F통계량':stat, 'p값':p})

anova_df = pd.DataFrame(anova_results).sort_values('p값')
print(anova_df.head(10))
```

**해석:** 요일(일요일), 도로형태(교차로), 노면상태(젖음/습기) 등이 ECLO에 유의미한 영향을 미쳤다.

---

## 6. 모델 학습 (ElasticNet, RandomForest, XGBoost)
```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

X = df_cluster.select_dtypes(include=np.number).drop(columns=['ECLO'], errors='ignore')
y = pd.to_numeric(df_cluster['ECLO'], errors='coerce')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    'ElasticNet': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42),
    'RandomForest': RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42),
    'XGBoost': XGBRegressor(n_estimators=300, learning_rate=0.1, max_depth=6, random_state=42)
}

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = mean_squared_error(y_test, preds, squared=False)
    r2 = r2_score(y_test, preds)
    print(f'[{name}] MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}')
```

**해석:** RandomForest가 R²=0.5896으로 가장 우수했으며, 비선형 구조를 효과적으로 학습함.

---

## 7. Stacking (CatBoost 메타모델)
```python
from sklearn.ensemble import StackingRegressor
from catboost import CatBoostRegressor

base_models = [
    ('elastic', ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)),
    ('rf', RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)),
    ('xgb', XGBRegressor(n_estimators=300, learning_rate=0.1, max_depth=6, random_state=42))
]

meta_model = CatBoostRegressor(iterations=500, learning_rate=0.05, depth=6, loss_function='RMSE', verbose=False)

stack = StackingRegressor(estimators=base_models, final_estimator=meta_model, passthrough=True, n_jobs=-1)
stack.fit(X_train, y_train)

preds = stack.predict(X_test)
print(f'Stacking → MAE={mean_absolute_error(y_test, preds):.4f}, RMSE={mean_squared_error(y_test, preds, squared=False):.4f}, R²={r2_score(y_test, preds):.4f}')
```

**해석:** CatBoost 기반 Stacking 모델은 R²=0.5932로 단일 모델보다 성능이 소폭 개선되었다.

---

## 8. Feature Importance (Permutation Importance)
<img width="990" height="590" alt="image" src="https://github.com/user-attachments/assets/64037e0e-5dc3-4e02-972b-be5f5571ecb5" />


```python
from sklearn.inspection import permutation_importance

result = permutation_importance(stack, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1)
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': result.importances_mean
}).sort_values('Importance', ascending=False)

print(importance_df.head(15))
```

**해석:** cluster 변수가 가장 높은 중요도를 보였으며, 사고시각·노면상태·도로형태·요일 순으로 영향을 미쳤다.

---

## 9. 종합 결론
이번 분석을 통해 대구 지역 교통사고 피해 정도(ECLO)는 **도로형태, 요일, 기상상태, 사고시각 등 외적 요인**에 의해 복합적으로 영향을 받는다는 점이 확인되었다.

군집화로 생성된 파생변수(cluster)는 단일 변수보다 높은 설명력을 보였으며, 이는 사고의 패턴적 특성이 피해 강도를 결정짓는 주요 요인임을 시사한다.

머신러닝 모델 중에서는 RandomForest와 CatBoost 기반 Stacking 모델이 가장 높은 예측 성능을 보였으며, 향후에는 사고 위치·차종 등 추가 공간 데이터를 결합하면 예측력을 더 향상시킬 수 있을 것이다.

---

**Author:** 윤해정 (Yoon Haejeong)  
heajeongy@naver.com  
Data Analytics & Visualization | Python · SQL · Power BI  
[GitHub](https://github.com/heajeongy-design)


