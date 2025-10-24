# 대구 교통사고 피해 예측 프로젝트 (Daegu Accident Severity Prediction)
https://dacon.io/competitions/official/236208/overview/description

---

## 1. 데이터 로드 & 기본 전처리

```python
import pandas as pd

df = pd.read_csv("./train.csv")

# 피해운전자 관련 결측치 처리
victim_cols = ['피해운전자 차종', '피해운전자 성별', '피해운전자 연령', '피해운전자 상해정도']
for col in victim_cols:
    df[col] = df[col].fillna("없음")

# 분석용 주요 변수만 선택
eda_cols = [
    '사고일시', '요일', '기상상태', '시군구',
    '도로형태', '노면상태', '법규위반', '사고유형',
    '사고유형 - 세부분류', 'ECLO'
]
df_data = df[eda_cols].copy()
```

**해석:**  
전체 데이터는 39,609건으로, 사고일시·기상상태·도로형태·법규위반·사고유형·ECLO 등 핵심 변수 중심으로 구성하였다.  
불필요한 컬럼을 제거하여 모델의 복잡도를 낮추고 다중공선성 문제를 방지했다.  
피해운전자 정보의 결측치는 `없음`으로 통일하여 실제 사고현장의 정보 누락 상황을 반영하면서 데이터 일관성을 확보하였다.

---

## 2. 탐색적 데이터 분석 (EDA)

### (1) 요일별 평균 ECLO
![weekday](https://github.com/user-attachments/assets/your_image_link_1)

**해석:**  
일요일에 사고 피해 정도가 가장 높고, 토요일·금요일 순으로 높게 나타났다.  
주말로 갈수록 ECLO가 상승하는 경향을 보이며, 주중보다 주말 사고가 더 심각하거나 인명피해로 이어질 가능성이 높음을 보여준다.

---

### (2) 도로형태별 ECLO 분포 (IQR 이상치 제거 후)
![roadtype](https://github.com/user-attachments/assets/your_image_link_2)

**해석:**  
‘단일로-교차로부근’, ‘교차로-기타도로형태’ 등 복잡한 도로 구조에서 피해 정도의 변동폭이 크게 나타났다.  
이는 교차로·분기점 등 시야 확보가 어려운 구간에서 사고 피해가 심각하게 나타나는 경향을 시사한다.

---

### (3) 기상상태별 사고 건수
![weather](https://github.com/user-attachments/assets/your_image_link_3)

**해석:**  
‘맑음’에서 압도적으로 많은 사고가 발생했다. 이는 날씨가 좋을 때 운전자의 경계심이 낮아지면서 사고 가능성이 오히려 높아지는 패턴으로 해석된다.

---

### (4) 군 단위 평균 ECLO 비교
![region](https://github.com/user-attachments/assets/your_image_link_4)

**해석:**  
논공읍·구지면에서 평균 ECLO가 가장 높게 나타났다.  
이는 물류차량 통행이 많은 산업단지 및 교통량 집중 지역에서 사고 심각도가 높다는 점을 보여준다.

---

## 3. 군집화 (KMeans 파생변수 추가)
![kmeans](https://github.com/user-attachments/assets/your_image_link_5)

**해석:**  
Elbow Method 및 Silhouette Score 결과 최적 군집 수는 **k=3**으로 확인되었다.  
이는 사고가 3개의 주요 패턴(도심형, 교차로형, 외곽형 등)으로 구분됨을 의미하며, 군집별 사고 특성은 향후 맞춤형 정책 수립에 활용 가능하다.

---

## 4. 회귀분석 (Polynomial Regression)

**결과:** R² = 0.5961  

**해석:**  
다항항을 추가하여 변수 간 비선형 관계를 반영한 결과, 약 60%의 설명력을 확보했다.  
도로형태·요일·기상상태 등이 단순 독립 변수가 아닌, 상호작용하며 피해 강도를 결정하는 구조임을 보여준다.

---

## 5. ANOVA (유의 변수 탐색)
![anova](https://github.com/user-attachments/assets/your_image_link_6)

**결과:**  
- 유의 변수: 요일_일요일, 도로형태_교차로-교차로안, 노면상태_젖음/습기 등  
- p<0.05 수준에서 ECLO 평균 차이 유의  

**해석:**  
시간적 요인(요일)과 공간적 요인(도로형태)이 교통사고 피해 강도에 구조적으로 영향을 미친다는 점을 확인하였다.

---

## 6. 모델 학습 (ElasticNet · RandomForest · XGBoost)
| 모델 | MAE | RMSE | R² |
|------|------|------|------|
| RandomForest | 1.4891 | 2.0208 | 0.5896 |
| XGBoost | 1.4950 | 2.0474 | 0.5787 |
| ElasticNet | 1.6938 | 2.4136 | 0.4145 |

**해석:**  
트리 기반 모델(RandomForest)이 가장 우수한 성능을 보였으며, 교통사고 피해 예측에는 비선형 관계 학습이 가능한 알고리즘이 적합함을 확인하였다.

---

## 7. Stacking (CatBoost Meta Model)
**결과:**  
- MAE: 1.4796  
- RMSE: 2.0118  
- R²: 0.5932  

**해석:**  
CatBoost를 메타모델로 활용한 스태킹 회귀모델은 기존 모델보다 성능이 개선되었다.  
다양한 알고리즘의 예측 결과를 결합하여 일반화 성능을 향상시켰으며, 오차 안정성과 예측 신뢰도가 높아졌다.

---

## 8. Feature Importance
![feature_importance](https://github.com/user-attachments/assets/your_image_link_7)

**상위 중요 변수:**  
- `cluster`  
- `사고시각`  
- `노면상태_젖음/습기`  
- `도로형태_단일로-기타`  
- `요일_수요일`  

**해석:**  
군집화로 생성된 `cluster` 변수가 피해 예측에 가장 강한 영향력을 가졌다.  
이는 사고의 패턴(시간대·위치·도로형태 조합)이 단일 변수보다 훨씬 설명력이 크다는 것을 보여준다.  
즉, 사고의 시공간적 맥락을 반영한 변수가 핵심 요인임을 확인하였다.

---

## 9. 종합 결론

이번 프로젝트를 통해 대구 지역의 교통사고 피해 정도(ECLO)는 **시간적·공간적 요인의 복합적 상호작용**에 의해 결정된다는 점을 확인하였다.  
요일·도로형태·노면상태는 단독 영향보다 조합 형태에서 피해 강도에 큰 차이를 만들어냈으며,  
KMeans 군집화로 생성된 사고 유형(cluster)이 가장 강력한 예측 변수로 작용했다.  

최종적으로 CatBoost 기반 Stacking 모델이 R²=0.593의 안정적 성능을 보였으며,  
본 분석은 **사고 발생 맥락 기반의 예측형 모델링**으로 확장 가능성을 보여주는 데이터 기반 연구라고 할 수 있다.

---

```
Daegu_Accident_Severity_Prediction
 ┣ train.csv
 ┣ daegu_analysis.ipynb
 ┣ README.md
 ┗ requirements.txt
```

---

**Author**  
**윤해정 (Yoon Haejeong)**  
heajeongy@naver.com  
Data Analytics & Visualization | Python, SQL, Power BI  
[GitHub](https://github.com/heajeongy-design)

