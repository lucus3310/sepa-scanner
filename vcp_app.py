import streamlit as st
import pandas as pd
import yfinance as yf
import requests
import time
import warnings
import urllib3
import numpy as np
from finvizfinance.screener.overview import Overview

# 隱藏警告訊息
warnings.filterwarnings('ignore')
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# --- 網頁介面設定 ---
st.set_page_config(page_title="SEPA 全球極速終極版", page_icon="🌍", layout="wide")

# ==========================================
# 🧠 建立記憶系統 (Session State)
# ==========================================
if 'us_data' not in st.session_state:
    st.session_state['us_data'] = pd.DataFrame()
if 'tw_data' not in st.session_state:
    st.session_state['tw_data'] = pd.DataFrame()

# ==========================================
# ⚡ 核心黑科技：資料快取函數 (Cache)
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
    url_twse = "https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL"
    for item in requests.get(url_twse, verify=False).json():
        code, name = item.get('Code', ''), item.get('Name', '')
        if len(code) == 4 and code.isdigit(): 
            all_tickers.append(code + ".TW")
            tw_names_dict[code + ".TW"] = name
            
    url_tpex = "https://www.tpex.org.tw/openapi/v1/tpex_mainboard_quotes"
    for item in requests.get(url_tpex, verify=False).json():
        code, name = item.get('SecuritiesCompanyCode', ''), item.get('CompanyName', '')
        if len(code) == 4 and code.isdigit(): 
            all_tickers.append(code + ".TWO")
            tw_names_dict[code + ".TWO"] = name
            
    return tuple(set(all_tickers)), tw_names_dict

@st.cache_data(ttl=86400, show_spinner=False)
def get_tw_official_industries():
    ind_dict = {}
    try:
        url_twse = "https://openapi.twse.com.tw/v1/opendata/t187ap03_L"
        for item in requests.get(url_twse, verify=False, timeout=10).json():
            ind_dict[str(item.get('公司代號', ''))] = item.get('產業類別', '未分類')
            
        url_tpex = "https://www.tpex.org.tw/openapi/v1/mopsfin_t187ap03_O"
        for item in requests.get(url_tpex, verify=False, timeout=10).json():
            ind_dict[str(item.get('公司代號', ''))] = item.get('產業類別', '未分類')
    except Exception as e:
        print("TWSE Industry fetch failed:", e)
    return ind_dict

@st.cache_data(ttl=43200, show_spinner=False)
def fetch_yf_history(chunk_tuple):
    return yf.download(list(chunk_tuple), period="1y", group_by='ticker', threads=True, progress=False)

@st.cache_data(ttl=43200, show_spinner=False)
def fetch_yf_info(ticker):
    try:
        return yf.Ticker(ticker).info
    except:
        return {}

# ==========================================
# 🌐 產業翻譯字典 (美股與備用)
# ==========================================
def get_chinese_industry(eng_str):
    if not isinstance(eng_str, str) or eng_str == 'Unknown' or eng_str == '':
        return '未分類'
    
    clean_str = eng_str.lower().replace('-', '').replace('—', '').replace(' ', '').replace('&', '').replace(',', '')
    
    smart_dict = {
        'semiconductors': '半導體', 'semiconductorequipmentmaterials': '半導體設備與材料',
        'softwareinfrastructure': '基礎軟體', 'softwareapplication': '應用軟體',
        'consumerelectronics': '消費性電子', 'electroniccomponents': '電子零組件',
        'communicationequipment': '通信設備', 'computerhardware': '電腦硬體',
        'informationtechnologyservices': '資訊科技服務', 'scientifictechnicalinstruments': '科學與技術儀器',
        'electronicgamingmultimedia': '遊戲與多媒體', 'biotechnology': '生技醫療',
        'medicaldevices': '醫療器材', 'drugmanufacturersgeneral': '一般製藥',
        'drugmanufacturersspecialtygeneric': '特種製藥', 'banksregional': '區域銀行',
        'banksdiversified': '多元銀行', 'autoparts': '汽車零組件',
        'automanufacturers': '汽車製造', 'electricalequipmentparts': '電機與零件',
        'buildingproductsequipment': '建材與設備', 'specialtyindustrialmachinery': '特種工業機械',
        'aerospacedefense': '航太與國防', 'packagingcontain': '包裝與容器',
        'packagingcontainers': '包裝與容器', 'chemicals': '化學品',
        'specialtychemicals': '特種化學品', 'steel': '鋼鐵',
        'gold': '黃金', 'silver': '白銀',
        'otherpreciousmetalsmining': '其他貴金屬與採礦', 'otherpreciousmeta': '其他貴金屬與採礦',
        'copper': '銅', 'aluminum': '鋁',
        'oilgasintegrated': '整合型油氣', 'oilgasep': '油氣探勘與生產',
        'oilgasmidstream': '油氣中游', 'marineshipping': '航運',
        'airlines': '航空', 'railroads': '鐵路',
        'trucking': '卡車運輸', 'integratedfreightlogistics': '整合物流',
        'packagedfoods': '包裝食品', 'beveragesnonalcoholic': '無酒精飲料',
        'beverageswineriesdistilleries': '釀酒廠', 'apparelmanufacturing': '服飾製造',
        'restaurants': '餐廳', 'householdpersonalproducts': '家庭與個人用品',
        'specialtyretail': '專業零售', 'internetretail': '網路零售',
        'rentalleasingser': '租賃服務', 'rentalleasingservices': '租賃服務',
        'reitindustrial': '工業 REITs', 'reitretail': '零售 REITs',
        'reitoffice': '辦公室 REITs', 'reitresidential': '住宅 REITs',
        'reithealthcarefacilities': '醫療機構 REITs', 'reitmortgage': '抵押貸款 REITs',
        'realestateservices': '房地產服務', 'solar': '太陽能',
        'solarequipmentmaterials': '太陽能設備與材料'
    }
    
    if clean_str in smart_dict: 
        return smart_dict[clean_str]
    return eng_str

def get_finviz_tier(val, is_roe=False):
    if val >= 30: tier = 30
    elif val >= 25: tier = 25
    elif val >= 20: tier = 20
    elif val >= 15: tier = 15
    elif val >= 10: tier = 10
    elif val >= 5: tier = 5
    else: return 'Positive (>0%)'

    if is_roe:
        return f'Over +{tier}%'
    else:
        return f'Over {tier}%'

# ==========================================
# 🎛️ 左側控制中心 (Sidebar)
# ==========================================
st.sidebar.title("🎯 SEPA 策略控制台")

st.sidebar.subheader("1. 趨勢模板 (Trend Template)")
use_price = st.sidebar.toggle("價格 > $10 (台股15元)", value=True)
use_sma_trend = st.sidebar.toggle("長線多頭 (150MA > 200MA)", value=True)
use_sma50 = st.sidebar.toggle("中短線爆發 (收盤價 > 50MA > 150MA)", value=True)
use_off_low = st.sidebar.toggle("脫離底部 (高於52週低點 30% 以上)", value=True)

st.sidebar.subheader("2. 超級績效催化劑")
use_roe = st.sidebar.toggle("啟用 ROE 過濾", value=True)
if use_roe:
    roe_val = st.sidebar.slider("🌍 全球 ROE 下限 (%)", min_value=0, max_value=50, value=15, step=1)

use_eps = st.sidebar.toggle("啟用 EPS 季增率過濾", value=True)
if use_eps:
    eps_val = st.sidebar.slider("🌍 全球 EPS 單季年增率 YoY (%)", min_value=0, max_value=100, value=25, step=1)

use_rev = st.sidebar.toggle("啟用 營收季增率過濾", value=True)
if use_rev:
    rev_val = st.sidebar.slider("🌍 全球 營收單季年增率 YoY (%)", min_value=0, max_value=100, value=25, step=1)

st.sidebar.subheader("3. VCP 壓縮與量縮")
use_vcp = st.sidebar.toggle("啟用 VCP 距高點跌幅區間", value=True)
if use_vcp:
    vcp_drop_range = st.sidebar.slider("距離最高點跌幅區間 (%)", min_value=0, max_value=50, value=(0, 25), step=1)
    min_drop, max_drop = vcp_drop_range

use_vol_dry = st.sidebar.toggle("底部量縮 (近期成交量 < 月均量 75%)", value=True)

# 🌟 更新：擴大偵測到「近5日內」的超賣轉折點
st.sidebar.subheader("4. 短線起漲時機 (Timing)")
use_kd = st.sidebar.toggle("啟用「近5日內」KD < 20 黃金交叉", value=True)
st.sidebar.caption("💡 開啟後，只挑選近 5 個交易日內，在 K 值小於 20 的極度超賣區剛發生黃金交叉的股票。")

if st.sidebar.button("🗑️ 清空所有歷史快取 (重新下載資料)", type="secondary"):
    st.session_state['us_data'] = pd.DataFrame()
    st.session_state['tw_data'] = pd.DataFrame()
    st.cache_data.clear() 
    st.rerun()

# ==========================================
# 主畫面
# ==========================================
st.title("🌍 全球 SEPA 終極掃描器：低檔超賣轉折版")
st.markdown("💡 **操作秘訣**：透過精準捕捉近 5 日內的 KD < 20 黃金交叉，我們專挑大趨勢向上、但短線被超賣錯殺的絕佳買點！")

col1, col2 = st.columns(2)
with col1:
    btn_us = st.button("🇺🇸 啟動【美股】SEPA 極速狙擊", use_container_width=True, type="primary")
with col2:
    btn_tw = st.button("🇹🇼 啟動【台股】SEPA 穩健狙擊", use_container_width=True, type="primary")

# ==========================================
# 🇺🇸 美股掃描邏輯
# ==========================================
if btn_us:
    with st.spinner('⚡ [階段一] 呼叫 Finviz 進行美股初篩...'):
        try:
            filters_dict = {}
            if use_price: filters_dict['Price'] = 'Over $10'
            if use_sma_trend: filters_dict['200-Day Simple Moving Average'] = 'SMA200 below SMA50' 
            if use_sma50: filters_dict['50-Day Simple Moving Average'] = 'Price above SMA50'
            if use_off_low: filters_dict['52-Week High/Low'] = '30% or more above Low'
            
            if use_roe: filters_dict['Return on Equity'] = get_finviz_tier(roe_val, is_roe=True)
            if use_eps: filters_dict['EPS growthqtr over qtr'] = get_finviz_tier(eps_val, is_roe=False)
            if use_rev: filters_dict['Sales growthqtr over qtr'] = get_finviz_tier(rev_val, is_roe=False)
            
            finviz_filters_tuple = tuple(filters_dict.items())
            df_finviz = fetch_finviz_data(finviz_filters_tuple)
            
            if not df_finviz.empty:
                us_tickers = df_finviz['Ticker'].tolist()
                sepa_results = []
                
                chunk_size = 50 
                us_progress = st.progress(0)
                status_msg = st.empty()

                for i in range(0, len(us_tickers), chunk_size):
                    chunk_tickers = us_tickers[i:i + chunk_size]
                    status_msg.text("📦 美股精算批次: " + str(i//chunk_size + 1) + "/" + str((len(us_tickers)//chunk_size)+1))
                    
                    data = fetch_yf_history(tuple(chunk_tickers))
                    
                    if data.empty: continue
                    
                    for ticker in chunk_tickers:
                        try:
                            if isinstance(data.columns, pd.MultiIndex):
                                if ticker in data.columns.levels[0]: df = data[ticker].copy()
                                else: continue
                            else:
                                if len(chunk_tickers) == 1: df = data.copy()
                                else: continue

                            df.dropna(subset=['Close'], inplace=True)
                            if df.empty or len(df) < 150: continue

                            close_price = float(df['Close'].iloc[-1])
                            high_52w = float(df['High'].max())
                            sma_50 = float(df['Close'].rolling(window=50).mean().iloc[-1])
                            sma_150 = float(df['Close'].rolling(window=150).mean().iloc[-1])
                            sma_200 = float(df['Close'].rolling(window=200).mean().iloc[-1])
                            vol_sma50 = float(df['Volume'].rolling(window=50).mean().iloc[-1])
                            recent_vol_min = float(df['Volume'].iloc[-5:].min())

                            if use_sma_trend and not (sma_150 > sma_200): continue
                            if use_sma50 and not (sma_50 > sma_150): continue
                            if use_vol_dry and not (recent_vol_min < vol_sma50 * 0.75): continue

                            drop_pct = ((high_52w - close_price) / high_52w) * 100
                            if use_vcp and not (min_drop <= drop_pct <= max_drop): continue
                                
                            high_9 = df['High'].rolling(9).max()
                            low_9 = df['Low'].rolling(9).min()
                            rsv = (df['Close'] - low_9) / (high_9 - low_9).replace(0, np.nan) * 100
                            rsv = rsv.fillna(50)
                            k_series = rsv.ewm(com=2, adjust=False).mean()
                            d_series = k_series.ewm(com=2, adjust=False).mean()
                            
                            # 🌟 嚴格判定：K>D 且 昨天K<=D 且 昨天K<20
                            kd_cross_series = (k_series > d_series) & (k_series.shift(1) <= d_series.shift(1)) & (k_series.shift(1) < 20)
                            
                            # 🌟 改為檢查最近 5 天內是否發生過黃金交叉
                            recent_cross = kd_cross_series.iloc[-5:].any()

                            k_val = float(k_series.iloc[-1])
                            d_val = float(d_series.iloc[-1])

                            if use_kd and not recent_cross: continue

                            price_1y_ago = float(df['Close'].iloc[0])
                            rs_1y = ((close_price - price_1y_ago) / price_1y_ago) * 100

                            high_low = df['High'] - df['Low']
                            high_close = np.abs(df['High'] - df['Close'].shift())
                            low_close = np.abs(df['Low'] - df['Close'].shift())
                            ranges = pd.concat([high_low, high_close, low_close], axis=1)
                            true_range = np.max(ranges, axis=1)
                            atr_14 = true_range.rolling(14).mean().iloc[-1]
                            atr_pct = (atr_14 / close_price) * 100

                            company = df_finviz[df_finviz['Ticker'] == ticker]['Company'].values[0]
                            industry_eng = df_finviz[df_finviz['Ticker'] == ticker]['Industry'].values[0]
                            industry_tw = get_chinese_industry(industry_eng)

                            kd_display = f"🔥 K:{k_val:.1f} / D:{d_val:.1f}" if recent_cross else f"K:{k_val:.1f} / D:{d_val:.1f}"

                            stats = {
                                '市場': '🇺🇸 美股',
                                '股票代號': ticker,
                                '公司名稱': company,
                                '分類產業': industry_tw,
                                '最新收盤價': round(close_price, 2),
                                'KD狀態': kd_display,
                                '距高點跌幅': f"-{drop_pct:.1f}%",
                                'RS(1年漲幅)': f"{rs_1y:+.1f}%",
                                'ATR日均振幅': f"{atr_pct:.1f}%",
                                '停損價(-7%)': round(close_price * 0.93, 2),
                                '排序列': rs_1y 
                            }
                            sepa_results.append(stats)
                        except: pass
                    
                    us_progress.progress(min(1.0, (i + chunk_size) / len(us_tickers)))
                    time.sleep(0.5)

                status_msg.empty()
                us_progress.empty()

                if len(sepa_results) > 0:
                    df_result = pd.DataFrame(sepa_results)
                    df_result = df_result.sort_values(by=['分類產業', '排序列'], ascending=[True, False])
                    st.session_state['us_data'] = df_result.drop(columns=['排序列'])
                    st.success("🎉 美股極速掃描完成！")
                else:
                    st.warning("😅 目前沒有符合條件的美股。超賣轉折非常稀有，耐心等待大盤回檔出現機會！")
            else:
                st.warning("😅 Finviz 初篩未找到股票，請嘗試放寬條件！")
        except Exception as e:
            st.error("❌ 抓取失敗：" + str(e))

# ==========================================
# 🇹🇼 台股掃描邏輯
# ==========================================
if btn_tw:
    with st.spinner('⏳ 正在執行台股掃描 (若已快取將瞬間完成)...'):
        try:
            all_tickers_tuple, tw_names_dict = get_tw_stock_list()
            all_tickers = list(all_tickers_tuple)
            
            tw_official_industry_dict = get_tw_official_industries()
            
            passed_technical_tickers = []
            chunk_size = 80 
            sepa_tech_data = {}
            
            tech_progress = st.progress(0)

            for i in range(0, len(all_tickers), chunk_size):
                chunk_tickers = all_tickers[i:i + chunk_size]
                data = fetch_yf_history(tuple(chunk_tickers))
                
                if data.empty: continue
                for ticker in chunk_tickers:
                    try:
                        if isinstance(data.columns, pd.MultiIndex):
                            if ticker in data.columns.levels[0]: df = data[ticker].copy()
                            else: continue 
                        else:
                            if len(chunk_tickers) == 1: df = data.copy()
                            else: continue

                        df.dropna(subset=['Close'], inplace=True)
                        if df.empty or len(df) < 150: continue

                        close_price = float(df['Close'].iloc[-1])
                        high_52w = float(df['High'].max())
                        low_52w = float(df['Low'].min())
                        sma_50 = float(df['Close'].rolling(window=50).mean().iloc[-1])
                        sma_150 = float(df['Close'].rolling(window=150).mean().iloc[-1])
                        sma_200 = float(df['Close'].rolling(window=200).mean().iloc[-1])
                        vol_sma50 = float(df['Volume'].rolling(window=50).mean().iloc[-1])
                        recent_vol_min = float(df['Volume'].iloc[-5:].min())

                        if use_price and close_price < 15: continue 
                        if use_sma_trend and not (sma_150 > sma_200): continue
                        if use_sma50 and not (sma_50 > sma_150 and close_price > sma_50): continue
                        if use_off_low and not (close_price >= low_52w * 1.30): continue 
                        if use_vol_dry and not (recent_vol_min < vol_sma50 * 0.75): continue
                        
                        drop_pct = ((high_52w - close_price) / high_52w) * 100
                        if use_vcp and not (min_drop <= drop_pct <= max_drop): continue

                        high_9 = df['High'].rolling(9).max()
                        low_9 = df['Low'].rolling(9).min()
                        rsv = (df['Close'] - low_9) / (high_9 - low_9).replace(0, np.nan) * 100
                        rsv = rsv.fillna(50)
                        k_series = rsv.ewm(com=2, adjust=False).mean()
                        d_series = k_series.ewm(com=2, adjust=False).mean()
                        
                        kd_cross_series = (k_series > d_series) & (k_series.shift(1) <= d_series.shift(1)) & (k_series.shift(1) < 20)
                        
                        # 🌟 改為檢查最近 5 天內是否發生過黃金交叉
                        recent_cross = kd_cross_series.iloc[-5:].any()

                        k_val = float(k_series.iloc[-1])
                        d_val = float(d_series.iloc[-1])

                        if use_kd and not recent_cross: continue

                        price_1y_ago = float(df['Close'].iloc[0])
                        rs_1y = ((close_price - price_1y_ago) / price_1y_ago) * 100

                        high_low = df['High'] - df['Low']
                        high_close = np.abs(df['High'] - df['Close'].shift())
                        low_close = np.abs(df['Low'] - df['Close'].shift())
                        ranges = pd.concat([high_low, high_close, low_close], axis=1)
                        true_range = np.max(ranges, axis=1)
                        atr_14 = true_range.rolling(14).mean().iloc[-1]
                        atr_pct = (atr_14 / close_price) * 100

                        passed_technical_tickers.append(ticker)
                        sepa_tech_data[ticker] = {
                            'drop_pct': drop_pct, 'rs_1y': rs_1y, 'atr_pct': atr_pct, 
                            'close_price': close_price, 'k_val': k_val, 'd_val': d_val,
                            'recent_cross': recent_cross 
                        }
                    except Exception: continue
                tech_progress.progress(min(1.0, (i + chunk_size) / len(all_tickers)))

            final_superstars = []
            if len(passed_technical_tickers) > 0:
                fund_progress = st.progress(0)
                total_survivors = len(passed_technical_tickers)
                
                for i, ticker in enumerate(passed_technical_tickers):
                    fund_progress.progress((i + 1) / total_survivors)
                    try:
                        info = fetch_yf_info(ticker)
                        clean_ticker = ticker.split('.')[0]
                        
                        roe = info.get('returnOnEquity', 0)
                        eps_growth = info.get('earningsQuarterlyGrowth', 0)
                        rev_growth = info.get('revenueGrowth', 0)
                        
                        official_industry = tw_official_industry_dict.get(clean_ticker)
                        if official_industry and official_industry != '未分類':
                            industry_tw = official_industry
                        else:
                            industry_eng = info.get('industry', 'Unknown')
                            industry_tw = get_chinese_industry(industry_eng)

                        if use_roe and (roe is None or roe < (roe_val / 100.0)): continue
                        if use_eps and (eps_growth is None or eps_growth < (eps_val / 100.0)): continue
                        if use_rev and (rev_growth is None or rev_growth < (rev_val / 100.0)): continue

                        company_name = tw_names_dict.get(ticker, info.get('shortName', clean_ticker))
                        tech_stats = sepa_tech_data[ticker]
                        
                        kd_display = f"🔥 K:{tech_stats['k_val']:.1f} / D:{tech_stats['d_val']:.1f}" if tech_stats['recent_cross'] else f"K:{tech_stats['k_val']:.1f} / D:{tech_stats['d_val']:.1f}"

                        stats = {
                            '市場': '🇹🇼 台股',
                            '股票代號': clean_ticker,
                            '公司名稱': company_name,
                            '分類產業': industry_tw,
                            '最新收盤價': round(tech_stats['close_price'], 2),
                            'KD狀態': kd_display,
                            '距高點跌幅': f"-{tech_stats['drop_pct']:.1f}%",
                            'RS(1年漲幅)': f"{tech_stats['rs_1y']:+.1f}%",
                            'ATR日均振幅': f"{tech_stats['atr_pct']:.1f}%",
                            '停損價(-7%)': round(tech_stats['close_price'] * 0.93, 2),
                            '排序列': tech_stats['rs_1y']
                        }
                        final_superstars.append(stats)
                    except Exception: continue

            if len(final_superstars) > 0:
                df_result = pd.DataFrame(final_superstars)
                df_result = df_result.sort_values(by=['分類產業', '排序列'], ascending=[True, False])
                st.session_state['tw_data'] = df_result.drop(columns=['排序列'])
                st.success("🎉 台股極速掃描完成！")
            else:
                st.session_state['tw_data'] = pd.DataFrame()
                st.warning("😅 台股掃描完畢！沒有股票滿足所有的 SEPA 嚴苛條件。可能短線大盤走弱，沒有剛轉折起漲的標的！")
                
        except Exception as e:
            st.error("❌ 抓取失敗：" + str(e))

# ==========================================
# 🌍 報表顯示區塊 (包含檔數統計)
# ==========================================
df_us = st.session_state.get('us_data', pd.DataFrame())
df_tw = st.session_state.get('tw_data', pd.DataFrame())

if not df_us.empty or not df_tw.empty:
    st.markdown("---")
    st.header("🌍 全球 SEPA 決策面板")
    
    frames = []
    if not df_us.empty: frames.append(df_us[['分類產業']])
    if not df_tw.empty: frames.append(df_tw[['分類產業']])
    df_combined_ind = pd.concat(frames, ignore_index=True)
    
    total_count = len(df_us) + len(df_tw)

    st.subheader(f"🔥 全球資金細分產業熱力榜 (共入選 {total_count} 檔)")
    industry_counts = df_combined_ind['分類產業'].value_counts().reset_index()
    industry_counts.columns = ['分類產業', '入選飆股數量']
    st.dataframe(industry_counts, use_container_width=True)

    st.markdown("---")
    
    if not df_us.empty:
        st.subheader(f"🇺🇸 美股精選名單：共 {len(df_us)} 檔")
        st.caption("💡 已依官方分類產業排序，方便同儕比較。帶有 🔥 符號代表該股於近 5 日內發生「KD < 20 超賣區黃金交叉」！")
        st.dataframe(df_us, use_container_width=True)
        csv_us = df_us.to_csv(index=False).encode('utf-8-sig')
        st.download_button(label="📥 下載美股名單", data=csv_us, file_name='US_SEPA_Grouped.csv', mime='text/csv')
    else:
        st.subheader("🇺🇸 美股精選名單：共 0 檔")
        st.info("尚無美股資料，或條件過於嚴苛，請點擊上方按鈕重新掃描。")

    st.markdown("<br>", unsafe_allow_html=True)

    if not df_tw.empty:
        st.subheader(f"🇹🇼 台股精選名單：共 {len(df_tw)} 檔")
        st.caption("💡 已依官方分類產業排序，方便同儕比較。帶有 🔥 符號代表該股於近 5 日內發生「KD < 20 超賣區黃金交叉」！")
        st.dataframe(df_tw, use_container_width=True)
        csv_tw = df_tw.to_csv(index=False).encode('utf-8-sig')
        st.download_button(label="📥 下載台股名單", data=csv_tw, file_name='TW_SEPA_Grouped.csv', mime='text/csv')
    else:
        st.subheader("🇹🇼 台股精選名單：共 0 檔")
        st.info("尚無台股資料，或條件過於嚴苛，請點擊上方按鈕重新掃描。")
