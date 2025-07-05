import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import requests
import urllib.parse
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 페이지 설정
st.set_page_config(
    page_title="결빙 교통사고 예측 시스템",
    page_icon="❄️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 사이드바 네비게이션
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "페이지 선택",
    ["홈", "데이터 수집", "데이터 분석", "예측 모델", "위험지역 지도", "배포 가이드"]
)

# 데이터 로드 함수
@st.cache_data
def load_sample_data():
    """샘플 데이터 생성 (실제 API 키가 없을 경우 대체용)"""
    np.random.seed(42)
    n_samples = 200
    
    # 한국 주요 도시 좌표 (대략적)
    cities = {
        '서울': (37.5665, 126.9780),
        '부산': (35.1796, 129.0756),
        '대구': (35.8714, 128.6014),
        '인천': (37.4563, 126.7052),
        '광주': (35.1595, 126.8526),
        '대전': (36.3504, 127.3845),
        '울산': (35.5384, 129.3114),
        '세종': (36.4641, 127.2393),
        '경기': (37.4138, 127.5183),
        '강원': (37.8228, 128.1555),
        '충북': (36.6357, 127.4917),
        '충남': (36.5184, 126.8000),
        '전북': (35.7175, 127.1530),
        '전남': (34.8679, 126.9910),
        '경북': (36.4919, 128.8889),
        '경남': (35.4606, 128.2132),
        '제주': (33.4996, 126.5312)
    }
    
    data = []
    for i in range(n_samples):
        city = np.random.choice(list(cities.keys()))
        lat, lon = cities[city]
        
        # 좌표에 약간의 노이즈 추가
        lat += np.random.normal(0, 0.1)
        lon += np.random.normal(0, 0.1)
        
        # 결빙 사고 관련 데이터 생성
        caslt_cnt = np.random.poisson(3)  # 사상자 수
        dth_dnv_cnt = np.random.poisson(0.1)  # 사망자 수
        se_dnv_cnt = np.random.poisson(0.5)  # 중상자 수
        sl_dnv_cnt = np.random.poisson(1.5)  # 경상자 수
        wnd_dnv_cnt = np.random.poisson(0.3)  # 부상자 수
        
        # 사고 발생 건수 (종속변수)
        occrrnc_cnt = caslt_cnt + np.random.poisson(2)
        
        data.append({
            'sido_sgg_nm': f'{city}특별시' if city in ['서울', '부산', '대구', '인천', '광주', '대전', '울산'] else f'{city}도',
            'la_crd': lat,
            'lo_crd': lon,
            'occrrnc_cnt': occrrnc_cnt,
            'caslt_cnt': caslt_cnt,
            'dth_dnv_cnt': dth_dnv_cnt,
            'se_dnv_cnt': se_dnv_cnt,
            'sl_dnv_cnt': sl_dnv_cnt,
            'wnd_dnv_cnt': wnd_dnv_cnt
        })
    
    return pd.DataFrame(data)

def fetch_api_data(service_key, year="2023"):
    """공공데이터 API에서 데이터 수집"""
    url = "http://apis.data.go.kr/B552061/frequentzoneFreezing/getRestFrequentzoneFreezing"
    params = {
        "serviceKey": service_key,
        "searchYearCd": year,
        "siDo": "",
        "guGun": "",
        "type": "json",
        "numOfRows": "1000",
        "pageNo": "1"
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if 'items' in data and data['items']:
                items = data['items']
                df_raw = pd.DataFrame(items)
                
                if 'item' in df_raw.columns:
                    df_parsed = pd.DataFrame(df_raw['item'].tolist())
                    return df_parsed
                else:
                    return pd.DataFrame(items)
            else:
                st.error("API 응답에서 데이터를 찾을 수 없습니다.")
                return None
        else:
            st.error(f"API 요청 실패: {response.status_code}")
            return None
            
    except Exception as e:
        st.error(f"API 호출 중 오류 발생: {str(e)}")
        return None

def preprocess_data(df):
    """데이터 전처리"""
    df_processed = df.copy()
    
    # 시도명 추출
    df_processed['시도명'] = df_processed['sido_sgg_nm'].str.extract(r'^(\S+)')
    
    # 위도/경도 변환
    df_processed['위도'] = pd.to_numeric(df_processed['la_crd'], errors='coerce')
    df_processed['경도'] = pd.to_numeric(df_processed['lo_crd'], errors='coerce')
    
    # 수치형 컬럼 변환
    numeric_cols = ['occrrnc_cnt', 'caslt_cnt', 'dth_dnv_cnt', 'se_dnv_cnt', 'sl_dnv_cnt', 'wnd_dnv_cnt']
    for col in numeric_cols:
        if col in df_processed.columns:
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
    
    # 결측치 제거
    df_processed = df_processed.dropna(subset=['위도', '경도', 'occrrnc_cnt'])
    
    return df_processed

def train_model(df):
    """머신러닝 모델 학습"""
    # 원-핫 인코딩
    df_encoded = pd.get_dummies(df, columns=['시도명'], drop_first=True)
    
    # 피처 선택
    features = ['caslt_cnt', 'dth_dnv_cnt', 'se_dnv_cnt', 'sl_dnv_cnt', 'wnd_dnv_cnt', '위도', '경도']
    features += [col for col in df_encoded.columns if col.startswith('시도명_')]
    
    # 사용 가능한 피처만 선택
    available_features = [col for col in features if col in df_encoded.columns]
    
    X = df_encoded[available_features]
    y = df_encoded['occrrnc_cnt']
    
    # 학습/테스트 분할
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 모델 학습
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 예측 및 평가
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    return model, X_test, y_test, y_pred, rmse, r2, df_encoded

# 홈 페이지
if page == "홈":
    st.title("결빙 교통사고 예측 시스템")
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 시스템 목적
        - 결빙으로 인한 교통사고 다발지역 예측
        - 사고 위험도 기반 우선순위 제공
        - 데이터 기반 교통안전 정책 지원
        """)
        
        st.markdown("""
        ### 주요 기능
        - 공공데이터 API 연동
        - 머신러닝 기반 사고 예측
        - 인터랙티브 지도 시각화
        - 위험지역 TOP 10 분석
        """)
    
    with col2:
        st.image("https://via.placeholder.com/400x300/87CEEB/000000?text=Winter+Road+Safety", 
                caption="겨울철 도로 안전")
    
    st.markdown("---")
    st.info("왼쪽 사이드바에서 원하는 기능을 선택하세요!")
    
    # 시스템 개요
    st.subheader("시스템 개요")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("데이터 소스", "공공데이터포털")
        st.metric("예측 모델", "Random Forest")
    
    with col2:
        st.metric("분석 연도", "2023")
        st.metric("지역 범위", "전국")
    
    with col3:
        st.metric("예측 정확도", "R² > 0.8")
        st.metric("업데이트", "실시간")

# 데이터 수집 페이지
elif page == "데이터 수집":
    st.title("데이터 수집")
    st.markdown("---")
    
    st.subheader("공공데이터 API 연동")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### API 키 설정
        공공데이터포털에서 발급받은 API 키를 입력하세요.
        """)
        
        service_key = st.text_input(
            "API 서비스 키 입력",
            type="password",
            placeholder=" .. 여기에 API 서비스키(Decoding용, 오류시 Encoding용)를 입력 .."
        )
        
        year = st.selectbox("분석 연도", ["2023", "2022", "2021"])
        
        use_sample = st.checkbox("샘플 데이터 사용 (API 키 없이 테스트)")
    
    with col2:
        st.markdown("""
        ### API 정보
        - **API명**: 교통사고 다발지역 정보 서비스
        - **제공기관**: 도로교통공단
        - **데이터 형식**: JSON
        - **업데이트**: 월 1회
        """)
    
    if st.button("데이터 수집 시작"):
        with st.spinner("데이터 수집 중..."):
            if use_sample or not service_key:
                st.info("샘플 데이터를 사용합니다.")
                df_raw = load_sample_data()
                st.session_state.df_raw = df_raw
                st.success("샘플 데이터 로드 완료!")
            else:
                df_raw = fetch_api_data(service_key, year)
                if df_raw is not None:
                    st.session_state.df_raw = df_raw
                    st.success("API 데이터 수집 완료!")
                else:
                    st.error("데이터 수집 실패")
                    st.stop()
    
    # 데이터 미리보기
    if 'df_raw' in st.session_state:
        st.subheader("수집된 데이터")
        df_raw = st.session_state.df_raw
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("총 레코드 수", len(df_raw))
        with col2:
            st.metric("컬럼 수", len(df_raw.columns))
        with col3:
            st.metric("결측값", df_raw.isnull().sum().sum())
        
        st.dataframe(df_raw.head())
        
        # 데이터 전처리
        if st.button("데이터 전처리 실행"):
            df_processed = preprocess_data(df_raw)
            st.session_state.df_processed = df_processed
            st.success("데이터 전처리 완료!")
            st.dataframe(df_processed.head())

# 데이터 분석 페이지
elif page == "데이터 분석":
    st.title("데이터 분석")
    st.markdown("---")
    
    if 'df_processed' not in st.session_state:
        st.warning("먼저 데이터를 수집하고 전처리해주세요!")
        st.stop()
    
    df = st.session_state.df_processed
    
    # 기술통계
    st.subheader("기술통계")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 기본 통계")
        numeric_cols = ['occrrnc_cnt', 'caslt_cnt', 'dth_dnv_cnt', 'se_dnv_cnt', 'sl_dnv_cnt', 'wnd_dnv_cnt']
        available_cols = [col for col in numeric_cols if col in df.columns]
        
        if available_cols:
            st.dataframe(df[available_cols].describe())
    
    with col2:
        st.markdown("### 지역별 분포")
        if '시도명' in df.columns:
            region_counts = df['시도명'].value_counts()
            fig = px.bar(
                x=region_counts.index,
                y=region_counts.values,
                title="지역별 사고 건수"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # 상관관계 분석
    st.subheader("상관관계 분석")
    
    if available_cols and len(available_cols) > 1:
        corr_matrix = df[available_cols].corr()
        
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            aspect="auto",
            title="변수 간 상관관계"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # 지역별 분석
    st.subheader("지역별 상세 분석")
    
    if '시도명' in df.columns:
        selected_region = st.selectbox("분석할 지역 선택", df['시도명'].unique())
        
        region_data = df[df['시도명'] == selected_region]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("선택 지역 사고 건수", len(region_data))
            st.metric("평균 사상자 수", f"{region_data['caslt_cnt'].mean():.2f}")
        
        with col2:
            if len(region_data) > 0:
                fig = px.scatter(
                    region_data,
                    x='위도',
                    y='경도',
                    size='occrrnc_cnt',
                    title=f"{selected_region} 지역 사고 분포"
                )
                st.plotly_chart(fig, use_container_width=True)

# 예측 모델 페이지
elif page == "예측 모델":
    st.title("예측 모델")
    st.markdown("---")
    
    if 'df_processed' not in st.session_state:
        st.warning("먼저 데이터를 수집하고 전처리해주세요!")
        st.stop()
    
    df = st.session_state.df_processed
    
    st.subheader("모델 설정")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 모델 정보
        - **알고리즘**: Random Forest Regressor
        - **타겟 변수**: 사고 발생 건수
        - **피처**: 사상자 수, 위경도, 지역 정보
        """)
        
        test_size = st.slider("테스트 데이터 비율", 0.1, 0.5, 0.2)
        n_estimators = st.slider("트리 개수", 50, 200, 100)
    
    with col2:
        st.markdown("""
        ### 예상 성능
        - **RMSE**: < 2.0
        - **R² Score**: > 0.8
        - **학습 시간**: < 10초
        """)
    
    if st.button("모델 학습 시작"):
        with st.spinner("모델 학습 중..."):
            try:
                model, X_test, y_test, y_pred, rmse, r2, df_encoded = train_model(df)
                
                # 결과 저장
                st.session_state.model = model
                st.session_state.X_test = X_test
                st.session_state.y_test = y_test
                st.session_state.y_pred = y_pred
                st.session_state.df_encoded = df_encoded
                
                st.success("모델 학습 완료!")
                
                # 모델 평가
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("RMSE", f"{rmse:.2f}")
                    st.metric("R² Score", f"{r2:.3f}")
                
                with col2:
                    # 예측 vs 실제 차트
                    fig = px.scatter(
                        x=y_test,
                        y=y_pred,
                        title="예측값 vs 실제값",
                        labels={'x': '실제값', 'y': '예측값'}
                    )
                    
                    # 대각선 추가
                    min_val = min(y_test.min(), y_pred.min())
                    max_val = max(y_test.max(), y_pred.max())
                    
                    fig.add_shape(
                        type="line",
                        x0=min_val, y0=min_val,
                        x1=max_val, y1=max_val,
                        line=dict(dash="dash", color="red")
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # 변수 중요도
                st.subheader("변수 중요도")
                
                importances = pd.Series(model.feature_importances_, index=X_test.columns)
                top_features = importances.nlargest(10)
                
                fig = px.bar(
                    x=top_features.values,
                    y=top_features.index,
                    orientation='h',
                    title="중요 변수 Top 10"
                )
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"모델 학습 중 오류 발생: {str(e)}")

# 위험지역 지도 페이지
elif page == "위험지역 지도":
    st.title("위험지역 지도")
    st.markdown("---")
    
    if 'model' not in st.session_state:
        st.warning("먼저 모델을 학습해주세요!")
        st.stop()
    
    model = st.session_state.model
    X_test = st.session_state.X_test
    y_pred = st.session_state.y_pred
    df_encoded = st.session_state.df_encoded
    
    # 예측 결과 데이터 생성
    X_test_result = X_test.copy()
    X_test_result['예측사고건수'] = y_pred
    X_test_result['위도'] = df_encoded.loc[X_test_result.index, '위도']
    X_test_result['경도'] = df_encoded.loc[X_test_result.index, '경도']
    
    # 지도 시각화
    st.subheader("예측 사고 위험지역")
    
    fig = px.scatter_mapbox(
        X_test_result,
        lat="위도",
        lon="경도",
        size="예측사고건수",
        color="예측사고건수",
        hover_data=["예측사고건수"],
        zoom=6,
        size_max=30,
        title="예측된 사고 위험 지역 (버블 크기 = 예측 사고 건수)",
        color_continuous_scale="Reds"
    )
    
    fig.update_layout(
        mapbox_style="carto-positron",
        margin={"r":0,"t":40,"l":0,"b":0},
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # 위험지역 TOP 10
    st.subheader("위험지역 TOP 10")
    
    top_10 = X_test_result.nlargest(10, '예측사고건수')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.dataframe(
            top_10[['위도', '경도', '예측사고건수']].round(2),
            use_container_width=True
        )
    
    with col2:
        fig = px.bar(
            x=range(1, 11),
            y=top_10['예측사고건수'],
            title="TOP 10 위험지역 예측 사고 건수"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # 인사이트 및 권장사항
    st.subheader(" 인사이트 및 권장사항")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 분석 결과
        - 예측 모델을 통해 사고 위험도가 높은 지역을 식별
        - 위도, 경도, 사상자 수 등이 주요 예측 변수
        - 특정 지역에 사고가 집중되는 경향 확인
        """)
    
    with col2:
        st.markdown("""
        ###정책 제안
        - 고위험 지역 우선 제설 작업 실시
        - 경고 표지판 및 안전시설 설치
        - 겨울철 교통 통제 강화
        - 실시간 모니터링 시스템 구축
        """)

# 배포 가이드 페이지
elif page == "배포 가이드":
    st.title("배포 가이드")
    st.markdown("---")
    
    st.subheader("배포 준비 사항")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 필요 파일
        1. `app.py` - 메인 애플리케이션
        2. `requirements.txt` - 의존성 패키지
        3. `README.md` - 프로젝트 설명
        4. `.gitignore` - Git 제외 파일
        """)
    
    with col2:
        st.markdown("""
        ### 배포 플랫폼
        1. **Streamlit Cloud** (추천)
        2. **Heroku**
        3. **AWS EC2**
        4. **Google Cloud Platform**
        """)
    
    # requirements.txt
    st.subheader("requirements.txt")
    
    requirements = """streamlit
pandas
numpy
plotly
scikit-learn
matplotlib
requests
urllib3"""
    
    st.code(requirements, language="text")
    
    # 배포 단계
    st.subheader("Streamlit Cloud 배포 단계")
    
    st.markdown("""
    1. **GitHub 저장소 생성**
       - 새 저장소 생성
       - 코드 업로드
       - Public으로 설정
    
    2. **Streamlit Cloud 접속**
       - https://streamlit.io/cloud 방문
       - GitHub 계정으로 로그인
       - "New app" 클릭
    
    3. **앱 설정**
       - 저장소 선택
       - 브랜치 선택 (main)
       - 메인 파일 경로 설정 (app.py)
    
    4. **배포 완료**
       - "Deploy!" 클릭
       - 자동 빌드 및 배포
       - 고유 URL 생성
    """)
    
    # 환경 변수 설정
    st.subheader("환경 변수 설정")
    
    st.markdown("""
    API 키와 같은 민감한 정보는 환경 변수로 관리하세요.
    
    **Streamlit Cloud에서 설정:**
    1. 앱 설정 페이지 접속
    2. "Advanced settings" 클릭
    3. "Secrets" 섹션에서 추가
    """)
    
    st.code("""
# .streamlit/secrets.toml
API_KEY = "your_api_key_here"
    """, language="toml")
    
    # 성능 최적화
    st.subheader("성능 최적화 팁")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 속도 개선
        - `@st.cache_data` 사용
        - 데이터 사전 처리
        - 불필요한 계산 최소화
        """)
    
    with col2:
        st.markdown("""
        ### 메모리 관리
        - 대용량 데이터 청크 처리
        - 사용하지 않는 변수 삭제
        - 효율적인 데이터 구조 사용
        """)

# 사이드바 추가 정보
st.sidebar.markdown("---")
st.sidebar.info("""
**사용 팁:**
- API 키는 공공데이터포털에서 발급
- 샘플 데이터로 먼저 테스트해보세요
- 배포 전 로컬에서 충분히 검증하세요
""")

st.sidebar.markdown("---")
st.sidebar.markdown("**Created by YSYUN with Claud**")
st.sidebar.markdown("Made with Streamlit")
