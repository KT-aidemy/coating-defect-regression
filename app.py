import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(layout="wide")
st.title("Butu予測アプリ（可視化 & モデル評価付き）")

# --- Sidebar ---
st.sidebar.header("モデル選択 & ファイルアップロード")
model_type = st.sidebar.selectbox("使用する回帰モデルを選んでください", ["Linear Regression", "Random Forest"])
train_file = st.sidebar.file_uploader("学習用Excelファイル（Butu含む）をアップロード", type="xlsx", key="train")

# --- メイン処理 ---
if train_file is not None:
    try:
        df_train = pd.read_excel(train_file)

        if "Butu" not in df_train.columns:
            st.error("'Butu'列が見つかりません。正しいファイルをアップロードしてください。")
        else:
            df_train = df_train[df_train["Butu"].notna()]
            X_train = df_train.drop(columns=["Butu"])
            y_train = df_train["Butu"]

            # モデルの選択
            if model_type == "Linear Regression":
                model = LinearRegression()
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42)

            model.fit(X_train, y_train)

            # モデル評価
            y_pred = model.predict(X_train)
            st.subheader("モデル評価結果")
            st.write(f"R2: {r2_score(y_train, y_pred):.3f}")
            st.write(f"MAE: {mean_absolute_error(y_train, y_pred):.2f}")
            st.write(f"RMSE: {np.sqrt(mean_squared_error(y_train, y_pred)):.2f}")

            # 散布図（任意の特徴量 vs Butu）
            st.subheader("特徴量とButuの関係（散布図＋回帰線）")
            selected_feature = st.selectbox("特徴量を選択", X_train.columns)
            fig, ax = plt.subplots()
            sns.regplot(data=df_train, x=selected_feature, y="Butu", ax=ax)
            st.pyplot(fig)

            # 特徴量重要度（RandomForestのみ）
            if model_type == "Random Forest":
                st.subheader("特徴量の重要度")
                importances = model.feature_importances_
                importance_df = pd.DataFrame({"特徴量": X_train.columns, "重要度": importances})
                importance_df = importance_df.sort_values(by="重要度", ascending=False)
                st.bar_chart(importance_df.set_index("特徴量"))

            # 予測データアップロード
            st.subheader("予測用CSVファイルをアップロード")
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
    st.info("まずは学習用Excelファイルを左のサイドバーからアップロードしてください。")

