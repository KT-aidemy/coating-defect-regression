import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression

st.title("Butu予測アプリ")

# Excelファイルのアップロード（学習データ用）
st.subheader("① 学習用Excelファイル（Butuを含む）をアップロード")
train_file = st.file_uploader("Excelファイル（.xlsx）を選択してください", type="xlsx", key="train")

if train_file is not None:
    try:
        df_train = pd.read_excel(train_file, sheet_name="data")
        X_train = df_train.drop(columns=["Butu"])
        y_train = df_train["Butu"]

        # モデルの学習
        model = LinearRegression()
        model.fit(X_train, y_train)

        st.success("モデルを学習しました。次に予測用CSVをアップロードしてください。")

        # CSVアップロード（予測対象データ）
        st.subheader("② 予測用CSVファイルをアップロード")
        test_file = st.file_uploader("予測したいCSVファイルをアップロード", type="csv", key="predict")

        if test_file is not None:
            df_test = pd.read_csv(test_file)

            if all(col in df_test.columns for col in X_train.columns):
                predictions = model.predict(df_test[X_train.columns])
                df_test["Butu_予測値"] = predictions

                st.write("▼ 予測結果")
                st.dataframe(df_test)

                csv = df_test.to_csv(index=False).encode("utf-8-sig")
                st.download_button("結果をCSVでダウンロード", csv, file_name="butu_predictions.csv", mime="text/csv")
            else:
                st.error("CSVに必要なカラムがありません。以下のカラムを含めてください：")
                st.write(list(X_train.columns))
    except Exception as e:
        st.error(f"エクセルの読み込みに失敗しました: {e}")
else:
    st.info("まずは学習データとなるExcelファイル（dataシート）をアップロードしてください。")
