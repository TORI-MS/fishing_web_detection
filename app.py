import streamlit as st
import pandas as pd
import re
from urllib.parse import urlparse
from sklearn.ensemble import RandomForestClassifier
import altair as alt
import os

# --- 모델 학습 및 반환 함수 ---
@st.cache_resource
def train_and_get_model(data_path):
    """
    지정된 데이터셋을 사용하여 모델을 학습하고 반환합니다.
    """
    try:
        df = pd.read_csv(data_path)

        # ✅ 타겟 라벨 처리
        if "status" in df.columns:
            y = df["status"].map({"phishing": 1, "legitimate": 0})
            X = df.drop(["status", "url"], axis=1, errors="ignore")
        else:
            st.error("데이터셋에 'status' 열이 없습니다.")
            st.stop()

        # 모델 학습
        model = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
        model.fit(X, y)

        return model, X, y, df

    except FileNotFoundError:
        st.error(f"오류: '{os.path.basename(data_path)}' 파일을 찾을 수 없습니다. Kaggle에서 다운로드하여 프로젝트 폴더에 넣어주세요.")
        st.stop()


# --- URL 특징 추출 함수 ---
def extract_features(url):
    """
    사용자가 입력한 URL에서 간단한 특징 추출.
    (데이터셋 feature columns과 최대한 매칭되도록 구성)
    """
    features = {}
    parsed_url = urlparse(url)

    features['length_url'] = len(url)
    features['length_hostname'] = len(parsed_url.netloc)
    features['ip'] = 1 if re.match(r'^\d{1,3}(\.\d{1,3}){3}$', parsed_url.netloc) else 0
    features['nb_dots'] = url.count('.')
    features['nb_hyphens'] = url.count('-')
    features['nb_at'] = url.count('@')
    features['nb_qm'] = url.count('?')
    features['nb_and'] = url.count('&')
    features['nb_or'] = url.count('|')
    features['https_token'] = 1 if 'https' in url else 0
    features['prefix_suffix'] = 1 if '-' in parsed_url.netloc else 0

    return pd.DataFrame([features])


# --- 스트림릿 애플리케이션 시작 ---
st.set_page_config(
    page_title="피싱 웹사이트 탐지기",
    page_icon="🎣",
    layout="centered"
)

st.title("🎣 피싱 웹사이트 탐지기")
st.markdown("### URL을 입력하여 피싱 웹사이트인지 확인해보세요.")

with st.expander("📖 사용법", expanded=False):
    st.markdown("""
    1. Kaggle에서 `dataset_phishing.csv` 파일을 다운로드하여 프로젝트 폴더에 넣으세요.
    2. 아래 입력창에 의심스러운 웹사이트 **URL**을 입력하세요.
    3. **'탐지하기'** 버튼을 클릭하세요.
    """)

# --- 모델 로드 ---
data_path = 'dataset_phishing.csv'
model, X_train, y_train, df_for_vis = train_and_get_model(data_path)

# --- URL 입력 필드 ---
user_input = st.text_input("🔗 URL 입력", "https://")

# --- 탐지 버튼 ---
if st.button("탐지하기", type="primary"):
    if not user_input or user_input == "https://":
        st.warning("URL을 입력해주세요.")
    else:
        with st.spinner("분석 중..."):
            try:
                # 학습한 feature set 기준으로 맞춰서 재구성
                feature_columns = model.feature_names_in_
                features_df = extract_features(user_input).reindex(columns=feature_columns, fill_value=0)

                prediction = model.predict(features_df)
                prediction_proba = model.predict_proba(features_df)

                phishing_prob = prediction_proba[0][1] * 100
                safe_prob = prediction_proba[0][0] * 100

                # --- 결과 출력 (안전 확률 기준) ---
                if phishing_prob >= 50:
                    st.error(f"⚠️ **위험!** 이 URL은 **피싱 웹사이트**일 가능성이 높습니다.\n\n예측 확률: **{phishing_prob:.2f}%**")
                else:
                    if safe_prob >= 70:
                        st.success(f"✅ **안전!** 이 URL은 **안전한 사이트**로 탐지되었습니다.\n\n예측 확률: **{safe_prob:.2f}%**")
                    else:
                        st.warning(f"⚠️ **안전하지 않음!** 이 URL은 안전 확률이 낮습니다.\n\n안전 확률: **{safe_prob:.2f}%**")

                # --- 탐지 근거 시각화 ---
                st.markdown("### 🔍 탐지 근거 (특징 비교)")
                important_features = ["length_url", "length_hostname", "nb_dots", "nb_hyphens", "nb_at", "https_token"]

                for feat in important_features:
                    if feat in df_for_vis.columns:
                        chart_data = df_for_vis[[feat, "status"]].copy()
                        chart_data["status"] = chart_data["status"].map({"phishing": "피싱", "legitimate": "정상"})

                        # Altair 분포 그래프
                        chart = (
                            alt.Chart(chart_data)
                            .mark_bar(opacity=0.5)
                            .encode(
                                x=alt.X(feat, bin=alt.Bin(maxbins=30)),
                                y="count()",
                                color="status"
                            )
                        )

                        # 사용자가 입력한 URL의 위치를 빨간 선으로 표시
                        user_value = features_df.iloc[0][feat]
                        rule = (
                            alt.Chart(pd.DataFrame({feat: [user_value]}))
                            .mark_rule(color="red", strokeWidth=2)
                            .encode(x=feat)
                        )

                        st.altair_chart(chart + rule, use_container_width=True)
                        st.caption(f"👉 입력한 URL의 **{feat} 값 = {user_value}**")

            except Exception as e:
                st.error(f"오류가 발생했습니다: {e}")
                st.info("입력한 URL 형식을 확인해주세요.")

# --- 간단한 데이터셋 통계 ---
with st.expander("📊 데이터셋 통계 보기", expanded=False):
    phishing_ratio = df_for_vis["status"].value_counts(normalize=True).reset_index()
    phishing_ratio.columns = ["status", "ratio"]

    chart = alt.Chart(phishing_ratio).mark_bar().encode(
        x="status",
        y="ratio",
        color="status"
    )
    st.altair_chart(chart, use_container_width=True)
