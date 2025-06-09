import streamlit as st
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression

# タイトル
st.title("Butu予測アプリ")

# 学習データの読み込み（初期ロード用）
@st.cache_data
def load_training_data():
    df = pd.read_excel("coating_defects_sampledata (1).xlsx", sheet_name="data")
    X = df.drop(columns=["Butu"])
    y = df["Butu"]
    return X, y

# モデルの学習
@st.cache_resource
def train_model(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model

X_train, y_train = load_training_data()
model = train_model(X_train, y_train)

# CSVアップロード
uploaded_file = st.file_uploader("CSVファイルをアップロードしてください", type="csv")

if uploaded_file is not None:
    new_data = pd.read_csv(uploaded_file)
    if all(col in new_data.columns for col in X_train.columns):
        predictions = model.predict(new_data[X_train.columns])
        new_data["Butu_予測値"] = predictions

        st.write("予測結果：")
        st.dataframe(new_data)

        # CSVダウンロードリンク
        csv = new_data.to_csv(index=False).encode("utf-8-sig")
        st.download_button("結果をCSVでダウンロード", csv, file_name="butu_predictions.csv", mime="text/csv")
    else:
        st.error("必要なカラムが不足しています。以下のカラムを含めてください：")
        st.write(list(X_train.columns))
