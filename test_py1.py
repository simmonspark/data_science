import pandas as pd
import numpy as np

# 예시 2D 데이터프레임
data = {
    'col1': [1, np.nan, 2, None, 4],
    'col2': [None, 5, np.nan, 7, 8],
    'col3': [9, np.nan, None, 12, None]
}
df = pd.DataFrame(data)
print(df)
# 1. None 값을 구분해서 별도의 마스크 생성 (열별로 구분)
none_mask = df.isnull()  # None 값을 추적하는 마스크

# 2. NaN 값을 ffill로 대체 (열 단위로 ffill 수행)
df = df.fillna(method='bfill')
print(df)
# 3. 원래 None이었던 값을 각 열의 평균으로 대체
for col in df.columns:
    df.loc[none_mask[col], col] = df[col].mean()

print(df)
