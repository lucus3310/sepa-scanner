import streamlit as st
import pandas as pd
import yfinance as yf
import requests
import time
import warnings
import urllib3
import numpy as np
import concurrent.futures
from finvizfinance.screener.overview import Overview

# 隱藏警告訊息
warnings.filterwarnings('ignore')
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- 網頁介面設定 ---
st.set_page_config(page_title="Minervini SEPA 終極掃描器 Pro", page_icon="🦅", layout="wide")

# ==========================================
# 🧠 建立記憶系統 (Session State)
# ==========================================
if 'us_data' not in st.session_state:
    st.session_state['us_data'] = pd.DataFrame()
if 'tw_data' not in st.session_state:
    st.session_state['tw_data'] = pd.DataFrame()

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

# ==========================================
# ⚡ 核心黑科技：資料快取函數
# ==========================================
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_finviz_data(filters_tuple):
    foverview = Overview()
    foverview.set_filter(filters_dict=dict(filters_tuple))
    return foverview.screener_view()

@st.cache_data(ttl=43200, show_spinner=False)
def get_tw_stock_list():
    all_tickers = []
    tw_names_dict = {}
    try:
        url_twse = "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL"
        for item in requests.get(url_twse, headers=HEADERS, verify=False, timeout=15).json():
            code, name = item.get('Code', ''), item.get('Name', '')
            if len(code) == 4 and code.isdigit(): 
                all_tickers.append(code + ".TW")
                tw_names_dict[code + ".TW"] = name
                
        url_tpex = "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_quotes"
        for item in requests.get(url_tpex, headers=HEADERS, verify=False, timeout=15).json():
            code, name = item.get('SecuritiesCompanyCode', ''), item.get('CompanyName', '')
            if len(code) == 4 and code.isdigit(): 
                all_tickers.append(code + ".TWO")
                tw_names_dict[code + ".TWO"] = name
    except Exception: pass
    return tuple(set(all_tickers)), tw_names_dict

@st.cache_data(ttl=86400, show_spinner=False)
def get_tw_official_industries():
    ind_dict = {}
    try:
        url_twse = "https://openapi.twse.com.tw/v1/opendata/t187ap03_L"
        res_twse = requests.get(url_twse, headers=HEADERS, verify=False, timeout=15)
        if res_twse.status_code == 200:
            for item in res_twse.json():
                ind_dict[str(item.get('公司代號', ''))] = item.get('產業類別', '未分類')
                
        url_tpex = "https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap03_O"
        res_tpex = requests.get(url_tpex, headers=HEADERS, verify=False, timeout=15)
        if res_tpex.status_code == 200:
            for item in res_tpex.json():
                ind_dict[str(item.get('公司代號', ''))] = item.get('產業類別', '未分類')
    except Exception: pass
    return ind_dict

@st.cache_data(ttl=43200, show_spinner=False)
def fetch_yf_history(chunk_tuple):
    return yf.download(list(chunk_tuple), period="1y", group_by='ticker', threads=True, progress=False)

@st.cache_data(ttl=43200, show_spinner=False)
def get_benchmark_return(market):
    ticker = "SPY" if market == "US" else "0050.TW"
    try:
        df = yf.download(ticker, period="1y", progress=False)
        if not df.empty:
            start_price = float(df['Close'].iloc[0])
            end_price = float(df['Close'].iloc[-1])
            return ((end_price - start_price) / start_price) * 100
    except: pass
    return 0.0

# ==========================================
# 🤖 輔助函數 (翻譯、題材、VCP演算法)
# ==========================================
def get_ai_theme(summary):
    if not isinstance(summary, str) or not summary: return "一般題材"
    text = summary.lower()
    themes = []
    if any(k in text for k in ['artificial intelligence', ' ai ', 'machine learning']): themes.append("🤖 AI/機器學習")
    if any(k in text for k in ['cloud computing', 'data center', 'server']): themes.append("☁️ 雲端/伺服器")
    if any(k in text for k in ['electric vehicle', ' ev ', 'energy storage']): themes.append("🚗 電動車/儲能")
    if any(k in text for k in ['cybersecurity', 'network security']): themes.append("🛡️ 網路資安")
    if any(k in text for k in ['semiconductor', 'foundry', 'chip', 'ic design']): themes.append("💽 半導體鏈")
    if any(k in text for k in ['biotech', 'pharma', 'medical device']): themes.append("🧬 生技醫療")
    return " + ".join(themes) if themes else "一般題材"

def get_chinese_industry(eng_str):
    if not isinstance(eng_str, str) or eng_str == 'Unknown' or eng_str == '': return '未分類'
    clean_str = eng_str.lower().replace('-', '').replace('—', '').replace(' ', '').replace('&', '').replace(',', '')
    smart_dict = {
        'semiconductors': '半導體', 'semiconductorequipmentmaterials': '半導體設備與材料',
        'softwareinfrastructure': '基礎軟體', 'softwareapplication': '應用軟體',
        'consumerelectronics': '消費性電子', 'electroniccomponents': '電子零組件',
        'communicationequipment': '通信設備', 'computerhardware': '電腦硬體/週邊',
        'informationtechnologyservices': '資訊科技服務', 'scientifictechnicalinstruments': '科學技術儀器',
        'electronicgamingmultimedia': '遊戲與多媒體', 'biotechnology': '生技醫療',
        'medicaldevices': '醫療器材', 'drugmanufacturersgeneral': '一般製藥',
        'drugmanufacturersspecialtygeneric': '特種製藥', 'banksregional': '區域銀行',
        'banksdiversified': '多元銀行', 'autoparts': '汽車零組件',
        'automanufacturers': '汽車製造', 'electricalequipmentparts': '電機機械',
        'buildingproductsequipment': '建材營造', 'specialtyindustrialmachinery': '特種工業機械',
        'aerospacedefense': '航太國防', 'packagingcontain': '包裝容器',
        'packagingcontainers': '包裝容器', 'chemicals': '化學品',
        'specialtychemicals': '特種化學品', 'steel': '鋼鐵',
        'gold': '黃金', 'silver': '白銀',
        'otherpreciousmetalsmining': '其他貴金屬與採礦', 'otherpreciousmeta': '其他貴金屬與採礦',
        'copper': '銅', 'aluminum': '鋁',
        'oilgasintegrated': '整合型油氣', 'oilgasep': '油氣探勘生產',
        'oilgasmidstream': '油氣中游', 'marineshipping': '航運',
        'airlines': '航空', 'railroads': '鐵路',
        'trucking': '卡車運輸', 'integratedfreightlogistics': '整合物流',
        'packagedfoods': '包裝食品', 'beveragesnonalcoholic': '無酒精飲料',
        'beverageswineriesdistilleries': '釀酒廠', 'apparelmanufacturing': '服飾製造',
        'restaurants': '餐廳', 'householdpersonalproducts': '家庭個人用品',
        'specialtyretail': '專業零售', 'internetretail': '網路零售',
        'rentalleasingser': '租賃服務', 'rentalleasingservices': '租賃服務',
        'reitindustrial': '工業 REITs', 'reitretail': '零售 REITs',
        'reitoffice': '辦公室 REITs', 'reitresidential': '住宅 REITs',
        'reithealthcarefacilities': '醫療機構 REITs', 'reitmortgage': '抵押貸款 REITs',
        'realestateservices': '房地產服務', 'solar': '太陽能',
        'solarequipmentmaterials': '太陽能設備材料',
        'engineeringconstruction': '工程與建築',
        'electronicscomputerdistribution': '電子通路',
        'electroniccomputerdistribution': '電子通路',
        'consultingservices': '顧問服務'
    }
    return smart_dict.get(clean_str, eng_str)

# 🌟 VCP 收斂演算法 (計算 VCP Score)
def calculate_vcp_score(df):
    try:
        if len(df) < 60: return 0, 0
        
        # 抓取不同週期的最大回撤 (模擬波動收斂)
        high_60 = df['High'].iloc[-60:].max()
        low_60 = df['Low'].iloc[-60:].min()
        dd_60 = ((high_60 - low_60) / high_60) * 100

        high_20 = df['High'].iloc[-20:].max()
        low_20 = df['Low'].iloc[-20:].min()
        dd_20 = ((high_20 - low_20) / high_20) * 100

        high_10 = df['High'].iloc[-10:].max()
        low_10 = df['Low'].iloc[-10:].min()
        dd_10 = ((high_10 - low_10) / high_10) * 100
        
        vcp_score = 0
        # 1. 波動收斂 (Half-rule 雛形)：60天波動 > 20天波動 > 10天波動
        if dd_60 > dd_20 and dd_20 > dd_10: vcp_score += 40
        # 2. 終極壓縮：最後收斂小於 8%
        if dd_10 < 8.0: vcp_score += 30
        # 3. 底部量縮：近5日均量 < 50日均量的 75%
        vol_5 = df['Volume'].iloc[-5:].mean()
        vol_50 = df['Volume'].rolling(50).mean().iloc[-1]
        if vol_5 < vol_50 * 0.75: vcp_score += 30
        
        return vcp_score, dd_10
    except:
        return 0, 0

# ==========================================
# 🎛️ 頂部控制面板 (Minervini Pro 版)
# ==========================================
st.title("🦅 Minervini SEPA 終極掃描器 Pro")
st.markdown("💡 **Pro 版升級**：導入嚴格 8 項 Trend Template、VCP 收斂演算法、EPS 加速檢查與 Minervini 綜合評分系統。全多執行緒加速！")

fund_map = {"停用": None, "大於 10%": 10, "大於 15%": 15, "大於 20%": 20, "大於 25%": 25, "大於 30%": 30, "大於 50%": 50}
kd_map = {"停用": None, "近 3 日": 3, "近 4 日": 4, "近 5 日": 5}

with st.expander("⚙️ 展開/收起 嚴格篩選條件設定", expanded=True):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("##### 📈 1. 嚴格 Trend Template")
        use_strict_trend = st.checkbox("🌟 啟用完整 8 項趨勢鐵律", value=True, help="價格>150MA/200MA、150MA>200MA、200MA上升中、價格>50MA等8項嚴格標準。")
        use_price = st.checkbox("價格 > $10 (台股15元)", value=True)
        use_near_high = st.checkbox("距離 52W 高點在 25% 以內", value=True)

    with col2:
        st.markdown("##### 💰 2. 盈餘與營收加速 (SEPA)")
        eps_opt = st.selectbox("EPS 單季年增 YoY", list(fund_map.keys()), index=3)
        check_eps_accel = st.checkbox("🌟 檢查 EPS 成長加速", value=True, help="確保盈餘具備爆發力 (受限於免費API，此選項會過濾出近期有明確強勁成長的標的)")
        tw_rev_opt = st.selectbox("🌟 營收 YoY (含台股月營收)", list(fund_map.keys()), index=3)
        roe_opt = st.selectbox("ROE 下限", list(fund_map.keys()), index=2)

    with col3:
        st.markdown("##### 🎯 3. VCP 偵測與輔助指標")
        use_vcp_score = st.checkbox("🌟 過濾 VCP Score > 40 (具備收斂特徵)", value=True)
        use_pivot = st.checkbox("距近期 15 日樞紐高點 < 3% (準備突破)", value=True)
        use_rs_rating = st.checkbox("RS 相對強度 > 大盤表現", value=True)
        kd_opt = st.selectbox("KD<20 低檔黃金交叉 (彈性開關)", list(kd_map.keys()), index=0)
        
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🗑️ 清空歷史快取 (重新下載)", use_container_width=True):
            st.session_state['us_data'] = pd.DataFrame()
            st.session_state['tw_data'] = pd.DataFrame()
            st.cache_data.clear() 
            st.rerun()

roe_val = fund_map[roe_opt]
eps_val = fund_map[eps_opt]
rev_val = fund_map[tw_rev_opt]
kd_days = kd_map[kd_opt]
use_kd = kd_days is not None

# ==========================================
# 🚀 啟動按鈕
# ==========================================
st.markdown("<br>", unsafe_allow_html=True)
btn_col1, btn_col2 = st.columns(2)
with btn_col1:
    btn_us = st.button("🇺🇸 啟動【美股】SEPA 極速狙擊", use_container_width=True, type="primary")
with btn_col2:
    btn_tw = st.button("🇹🇼 啟動【台股】SEPA 穩健狙擊", use_container_width=True, type="primary")

# ==========================================
# 核心運算邏輯 (並行處理 ThreadPoolExecutor)
# ==========================================
def process_single_stock(ticker, df, benchmark_return, market, tw_names_dict, tw_official_industry_dict):
    try:
        df.dropna(subset=['Close'], inplace=True)
        if df.empty or len(df) < 200: return None

        close_price = float(df['Close'].iloc[-1])
        high_52w = float(df['High'].max())
        low_52w = float(df['Low'].min())
        
        sma_50 = float(df['Close'].rolling(window=50).mean().iloc[-1])
        sma_150 = float(df['Close'].rolling(window=150).mean().iloc[-1])
        sma_200 = float(df['Close'].rolling(window=200).mean().iloc[-1])
        sma_200_20d_ago = float(df['Close'].rolling(window=200).mean().shift(20).iloc[-1])

        # 🌟 嚴格 8 項 Trend Template 檢查
        if use_strict_trend:
            if not (close_price > sma_150 and close_price > sma_200): return None
            if not (sma_150 > sma_200): return None
            if not (sma_200 > sma_200_20d_ago): return None
            if not (sma_50 > sma_150 and sma_50 > sma_200): return None
            if not (close_price > sma_50): return None
            if not (close_price >= low_52w * 1.30): return None

        if use_price and close_price < (15 if market=="TW" else 10): return None 
        
        drop_from_high = ((high_52w - close_price) / high_52w) * 100
        if use_near_high and drop_from_high > 25: return None

        # 🌟 計算 VCP Score
        vcp_score, final_contraction = calculate_vcp_score(df)
        if use_vcp_score and vcp_score < 40: return None

        # KD 檢查
        if use_kd:
            high_9 = df['High'].rolling(9).max()
            low_9 = df['Low'].rolling(9).min()
            rsv = (df['Close'] - low_9) / (high_9 - low_9).replace(0, np.nan) * 100
            k_series = rsv.fillna(50).ewm(com=2, adjust=False).mean()
            d_series = k_series.ewm(com=2, adjust=False).mean()
            kd_cross_series = (k_series > d_series) & (k_series.shift(1) <= d_series.shift(1)) & (k_series.shift(1) < 20)
            if not kd_cross_series.iloc[-kd_days:].any(): return None
            kd_display = f"🔥 K:{float(k_series.iloc[-1]):.1f}"
        else:
            kd_display = "-"

        # 樞紐突破檢查
        recent_15d_high = float(df['High'].iloc[-16:-1].max())
        dist_to_pivot = ((recent_15d_high - close_price) / close_price) * 100
        if use_pivot and not (-1 <= dist_to_pivot <= 3): return None

        # RS 計算
        price_1y_ago = float(df['Close'].iloc[0])
        rs_1y = ((close_price - price_1y_ago) / price_1y_ago) * 100
        if use_rs_rating and rs_1y < benchmark_return: return None 

        # 🌟 API 抓取財報 (最耗時，利用多執行緒平行處理)
        info = yf.Ticker(ticker).info
        clean_ticker = ticker.split('.')[0] if market=="TW" else ticker
        
        roe = info.get('returnOnEquity', 0)
        eps_growth = info.get('earningsQuarterlyGrowth', 0)
        rev_growth = info.get('revenueGrowth', 0)
        
        if roe_val and (roe is None or roe < (roe_val / 100.0)): return None
        if eps_val and (eps_growth is None or eps_growth < (eps_val / 100.0)): return None
        if rev_val and (rev_growth is None or rev_growth < (rev_val / 100.0)): return None
        
        # EPS 加速邏輯 (嚴格要求大於0且達標)
        if check_eps_accel and (eps_growth is None or eps_growth <= 0): return None

        summary = info.get('longBusinessSummary', '')
        ai_theme = get_ai_theme(summary)

        if market == "TW":
            company_name = tw_names_dict.get(ticker, info.get('shortName', clean_ticker))
            industry = tw_official_industry_dict.get(clean_ticker)
            if not industry or industry == '未分類':
                industry = get_chinese_industry(info.get('industry', 'Unknown'))
        else:
            company_name = info.get('shortName', clean_ticker)
            industry = get_chinese_industry(info.get('industry', 'Unknown'))

        # 🌟 計算 Minervini Overall Score (綜合評分)
        # RS (佔40%): 表現超越大盤幅度，滿分40
        rs_score_norm = min(40, max(0, ((rs_1y - benchmark_return) / 100) * 40))
        # VCP (佔30%): 滿分30 (原本vcp_score滿分100，按比例換算)
        vcp_score_norm = vcp_score * 0.3
        # 財報 (佔30%): EPS與營收動能
        fund_score = 0
        if eps_growth and eps_growth > 0.2: fund_score += 15
        if rev_growth and rev_growth > 0.2: fund_score += 15
        
        overall_score = int(rs_score_norm + vcp_score_norm + fund_score)

        return {
            '市場': '🇹🇼 台股' if market=="TW" else '🇺🇸 美股',
            '股票代號': clean_ticker,
            '公司名稱': company_name,
            '分類產業': industry,
            '🌟 綜合評分': overall_score,
            '🌟 VCP 分數': int(vcp_score),
            '🔥 AI 雷達題材': ai_theme,
            '最新收盤價': round(close_price, 2),
            '距樞紐突破點': f"{dist_to_pivot:.1f}%" if dist_to_pivot > 0 else "已突破",
            '最後收斂幅度': f"{final_contraction:.1f}%",
            '距52W高點跌幅': f"-{drop_from_high:.1f}%",
            'EPS 單季 YoY': f"{(eps_growth*100):.1f}%" if eps_growth else "-",
            '營收 單季 YoY': f"{(rev_growth*100):.1f}%" if rev_growth else "-",
            'RS(1年漲幅)': f"{rs_1y:+.1f}%",
            'KD狀態': kd_display,
            '排序列': overall_score
        }
    except Exception: return None

def run_scanner(market="TW"):
    benchmark_return = get_benchmark_return(market)
    sepa_results = []
    
    if market == "TW":
        all_tickers_tuple, tw_names_dict = get_tw_stock_list()
        all_tickers = list(all_tickers_tuple)
        tw_official_industry_dict = get_tw_official_industries()
    else:
        filters_dict = {'Price': 'Over $10'}
        if use_near_high: filters_dict['52-Week High/Low'] = '30% or more above Low'
        df_finviz = fetch_finviz_data(tuple(filters_dict.items()))
        if df_finviz.empty: return pd.DataFrame()
        all_tickers = df_finviz['Ticker'].tolist()
        tw_names_dict, tw_official_industry_dict = {}, {}

    chunk_size = 100 
    progress_bar = st.progress(0)
    status_text = st.empty()

    for i in range(0, len(all_tickers), chunk_size):
        chunk_tickers = all_tickers[i:i + chunk_size]
        status_text.text(f"⏳ 下載歷史資料並進行多執行緒分析... 批次 {i//chunk_size + 1}")
        
        data = fetch_yf_history(tuple(chunk_tickers))
        if data.empty: continue
        
        # 🌟 啟動多執行緒加速 (ThreadPoolExecutor)
        with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
            future_to_ticker = {}
            for ticker in chunk_tickers:
                if isinstance(data.columns, pd.MultiIndex):
                    if ticker in data.columns.levels[0]: df = data[ticker].copy()
                    else: continue
                else:
                    if len(chunk_tickers) == 1: df = data.copy()
                    else: continue
                    
                future = executor.submit(
                    process_single_stock, ticker, df, benchmark_return, market, 
                    tw_names_dict, tw_official_industry_dict
                )
                future_to_ticker[future] = ticker
                
            for future in concurrent.futures.as_completed(future_to_ticker):
                res = future.result()
                if res: sepa_results.append(res)
                
        progress_bar.progress(min(1.0, (i + chunk_size) / len(all_tickers)))
    
    progress_bar.empty()
    status_text.empty()
    
    if len(sepa_results) > 0:
        df_result = pd.DataFrame(sepa_results).sort_values(by=['排序列'], ascending=False)
        return df_result.drop(columns=['排序列'])
    return pd.DataFrame()

# ==========================================
# 觸發執行與報表顯示
# ==========================================
if btn_us:
    with st.spinner('⚡ [多執行緒模式] 正在全速掃描美股...'):
        st.session_state['us_data'] = run_scanner("US")
        if not st.session_state['us_data'].empty: st.success("🎉 美股 Pro 版掃描完成！")
        else: st.warning("😅 目前沒有符合 Minervini 嚴苛條件的美股。")

if btn_tw:
    with st.spinner('⏳ [多執行緒模式] 正在全速掃描台股...'):
        st.session_state['tw_data'] = run_scanner("TW")
        if not st.session_state['tw_data'].empty: st.success("🎉 台股 Pro 版掃描完成！")
        else: st.warning("😅 台股掃描完畢！大盤可能處於弱勢，無符合標的。")

df_us = st.session_state.get('us_data', pd.DataFrame())
df_tw = st.session_state.get('tw_data', pd.DataFrame())

if not df_us.empty or not df_tw.empty:
    st.markdown("---")
    
    if not df_us.empty:
        st.subheader(f"🇺🇸 美股 VCP 綜合評分排行榜：共 {len(df_us)} 檔")
        st.dataframe(df_us, use_container_width=True)
        st.download_button("📥 下載美股名單", df_us.to_csv(index=False).encode('utf-8-sig'), 'US_SEPA_Pro.csv', 'text/csv')

    st.markdown("<br>", unsafe_allow_html=True)

    if not df_tw.empty:
        st.subheader(f"🇹🇼 台股 VCP 綜合評分排行榜：共 {len(df_tw)} 檔")
        st.dataframe(df_tw, use_container_width=True)
        st.download_button("📥 下載台股名單", df_tw.to_csv(index=False).encode('utf-8-sig'), 'TW_SEPA_Pro.csv', 'text/csv')
