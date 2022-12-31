#####################
# stocks page setup #
#####################

#pip install yfinance
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import webbrowser
from PIL import Image

########################
# Technical Indicators #
########################

# Create a Mixin of interdependent non-side-effect-free code that is shared between components.
class IndicatorMixin:

    _fillna = False
    # check nulls
    def _check_fillna(self, series: pd.Series, value: int = 0) -> pd.Series:

        if self._fillna:
            series_output = series.copy(deep=False)
            series_output = series_output.replace([np.inf, -np.inf], np.nan)
            if isinstance(value, int) and value == -1:
                series = series_output.fillna(method="ffill").fillna(value=-1)
            else:
                series = series_output.fillna(method="ffill").fillna(value)
        return series

    @staticmethod
    # define the True Range which is the greatest distance you can find between any two of these three prices.
    def _true_range(
        high: pd.Series, low: pd.Series, prev_close: pd.Series
    ) -> pd.Series:
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        true_range = pd.DataFrame(data={"tr1": tr1, "tr2": tr2, "tr3": tr3}).max(axis=1)
        return true_range

# define dropna function
def dropna(df: pd.DataFrame) -> pd.DataFrame:
    # Drop rows with null values
    df = df.copy()
    number_cols = df.select_dtypes("number").columns.to_list()
    df[number_cols] = df[number_cols][df[number_cols] < math.exp(709)]  # big number
    df[number_cols] = df[number_cols][df[number_cols] != 0.0]
    df = df.dropna()
    return df

# define simple moving average (SMA)
def _sma(series, periods: int, fillna: bool = False):
    min_periods = 0 if fillna else periods
    return series.rolling(window=periods, min_periods=min_periods).mean()

# define exponential moving average (EMA) that places a greater weight and significance on the most recent data points.
def _ema(series, periods, fillna=False):
    min_periods = 0 if fillna else periods
    return series.ewm(span=periods, min_periods=min_periods, adjust=False).mean()

# Calling min() and max() With a Single Iterable Argument
def _get_min_max(series1: pd.Series, series2: pd.Series, function: str = "min"):
    # Find min or max value between two lists for each index
    series1 = np.array(series1)
    series2 = np.array(series2)
    if function == "min":
        output = np.amin([series1, series2], axis=0)
    elif function == "max":
        output = np.amax([series1, series2], axis=0)
    else:
        raise ValueError('"f" variable value should be "min" or "max"')

    return pd.Series(output)

# Setup BollingerBands
class BollingerBands(IndicatorMixin):
    # The __init__ function is called every time an object is created from a class
    def __init__(
        self,
        close: pd.Series,
        window: int = 20,
        window_dev: int = 2,
        fillna: bool = False,
    ):
        self._close = close
        self._window = window
        self._window_dev = window_dev
        self._fillna = fillna
        self._run()
    # define the BollingerBands run function
    def _run(self):
        min_periods = 0 if self._fillna else self._window
        self._mavg = self._close.rolling(self._window, min_periods=min_periods).mean()
        self._mstd = self._close.rolling(self._window, min_periods=min_periods).std(
            ddof=0
        )
        self._hband = self._mavg + self._window_dev * self._mstd
        self._lband = self._mavg - self._window_dev * self._mstd
    # define bollinger moving average Bollinger, channel middle band and returns pandas.Series new feature generated
    def bollinger_mavg(self) -> pd.Series:
        mavg = self._check_fillna(self._mavg, value=-1)
        return pd.Series(mavg, name="mavg")
    # Add bollinger band high indicator filling nans values, channel middle band and returns pandas.Series new feature generated
    def bollinger_hband(self) -> pd.Series:
        hband = self._check_fillna(self._hband, value=-1)
        return pd.Series(hband, name="hband")
    # Add bollinger band low indicator filling nans values, channel lower band and returns pandas.Series new feature generated
    def bollinger_lband(self) -> pd.Series:
        lband = self._check_fillna(self._lband, value=-1)
        return pd.Series(lband, name="lband")
    # Add bollinger band width indicator filling nans values, channel width band and returns pandas.Series new feature generated
    def bollinger_wband(self) -> pd.Series:
        wband = ((self._hband - self._lband) / self._mavg) * 100
        wband = self._check_fillna(wband, value=0)
        return pd.Series(wband, name="bbiwband")
    # Add bollinger band percentage indicator filling nans values, channel percentage band and returns pandas.Series new feature generated
    def bollinger_pband(self) -> pd.Series:
        pband = (self._close - self._lband) / (self._hband - self._lband)
        pband = self._check_fillna(pband, value=0)
        return pd.Series(pband, name="bbipband")
    # Bollinger Channel Indicator Crossing High Band (binary). It returns 1, if close is higher than bollinger_hband. Else, it returns 0 and returns pandas.Series new feature generated
    def bollinger_hband_indicator(self) -> pd.Series:
        hband = pd.Series(
            np.where(self._close > self._hband, 1.0, 0.0), index=self._close.index
        )
        hband = self._check_fillna(hband, value=0)
        return pd.Series(hband, index=self._close.index, name="bbihband")
    # Bollinger Channel Indicator Crossing Low Band (binary). It returns 1, if close is lower than bollinger_lband. Else, it returns 0 and returns pandas.Series new feature generated
    def bollinger_lband_indicator(self) -> pd.Series:
        lband = pd.Series(
            np.where(self._close < self._lband, 1.0, 0.0), index=self._close.index
        )
        lband = self._check_fillna(lband, value=0)
        return pd.Series(lband, name="bbilband")

# Relative Strength Index (RSI) Compares the magnitude of recent gains and losses over a specified time period to measure speed and change of price movements of a security. It is primarily used to attempt to identify overbought or oversold conditions in the trading of an asset.
class RSIIndicator(IndicatorMixin):
    # The __init__ function is called every time an object is created from a class
    def __init__(self, close: pd.Series, window: int = 14, fillna: bool = False):
        self._close = close
        self._window = window
        self._fillna = fillna
        self._run()
    # define the Relative Strength Index run function
    def _run(self):
        diff = self._close.diff(1)
        up_direction = diff.where(diff > 0, 0.0)
        down_direction = -diff.where(diff < 0, 0.0)
        min_periods = 0 if self._fillna else self._window
        emaup = up_direction.ewm(
            alpha=1 / self._window, min_periods=min_periods, adjust=False
        ).mean()
        emadn = down_direction.ewm(
            alpha=1 / self._window, min_periods=min_periods, adjust=False
        ).mean()
        relative_strength = emaup / emadn
        self._rsi = pd.Series(
            np.where(emadn == 0, 100, 100 - (100 / (1 + relative_strength))),
            index=self._close.index,
        )
    # define Relative Strength Index (RSI) check nulls and returns pandas.Series new feature generated
    def rsi(self) -> pd.Series:
        rsi_series = self._check_fillna(self._rsi, value=50)
        return pd.Series(rsi_series, name="rsi")

# Moving Average Convergence Divergence (MACD) Is a trend-following momentum indicator that shows the relationship between two moving averages of prices.
class MACD(IndicatorMixin):
    # The __init__ function is called every time an object is created from a class
    def __init__(
        self,
        close: pd.Series,
        window_slow: int = 26,
        window_fast: int = 12,
        window_sign: int = 9,
        fillna: bool = False,
    ):
        self._close = close
        self._window_slow = window_slow
        self._window_fast = window_fast
        self._window_sign = window_sign
        self._fillna = fillna
        self._run()
    # define the Moving Average Convergence Divergence (MACD) run function
    def _run(self):
        self._emafast = _ema(self._close, self._window_fast, self._fillna)
        self._emaslow = _ema(self._close, self._window_slow, self._fillna)
        self._macd = self._emafast - self._emaslow
        self._macd_signal = _ema(self._macd, self._window_sign, self._fillna)
        self._macd_diff = self._macd - self._macd_signal
    # define MACD line, check nulls and returns pandas.Series new feature generated
    def macd(self) -> pd.Series:
        macd_series = self._check_fillna(self._macd, value=0)
        return pd.Series(
            macd_series, name=f"MACD_{self._window_fast}_{self._window_slow}"
        )
    # define MACD signal, check nulls and returns pandas.Series new feature generated
    def macd_signal(self) -> pd.Series:
        macd_signal_series = self._check_fillna(self._macd_signal, value=0)
        return pd.Series(
            macd_signal_series,
            name=f"MACD_sign_{self._window_fast}_{self._window_slow}",
        )
    # define MACD histogram, check nulls and returns pandas.Series new feature generated
    def macd_diff(self) -> pd.Series:
        macd_diff_series = self._check_fillna(self._macd_diff, value=0)
        return pd.Series(
            macd_diff_series, name=f"MACD_diff_{self._window_fast}_{self._window_slow}"
        )

# The Rate-of-Change (ROC) indicator, which is also referred to as simply Momentum, is a pure momentum oscillator. The ROC calculation compares the current price with the price "n" periods ago
class ROCIndicator(IndicatorMixin):
    # The __init__ function is called every time an object is created from a class
    def __init__(self, close: pd.Series, window: int = 12, fillna: bool = False):
        self._close = close
        self._window = window
        self._fillna = fillna
        self._run()
    # define the rate of change run function
    def _run(self):
        self._roc = (
            (self._close - self._close.shift(self._window))
            / self._close.shift(self._window)
        ) * 100
    # define rate of change, check nulls and returns pandas.Series new feature generated
    def roc(self) -> pd.Series:
        roc_series = self._check_fillna(self._roc)
        return pd.Series(roc_series, name="roc")

# The true strength index (TSI) is a technical momentum oscillator used to identify trends and reversals. The indicator may be useful for determining overbought and oversold conditions, indicating potential trend direction changes via centerline or signal line crossovers, and warning of trend weakness through divergence.
class TSIIndicator(IndicatorMixin):
    # The __init__ function is called every time an object is created from a class
    def __init__(
        self,
        close: pd.Series,
        window_slow: int = 25,
        window_fast: int = 13,
        fillna: bool = False,
    ):
        self._close = close
        self._window_slow = window_slow
        self._window_fast = window_fast
        self._fillna = fillna
        self._run()
    # define the true strength index run function
    def _run(self):
        diff_close = self._close - self._close.shift(1)
        min_periods_r = 0 if self._fillna else self._window_slow
        min_periods_s = 0 if self._fillna else self._window_fast
        smoothed = (
            diff_close.ewm(
                span=self._window_slow, min_periods=min_periods_r, adjust=False
            )
            .mean()
            .ewm(span=self._window_fast, min_periods=min_periods_s, adjust=False)
            .mean()
        )
        smoothed_abs = (
            abs(diff_close)
            .ewm(span=self._window_slow, min_periods=min_periods_r, adjust=False)
            .mean()
            .ewm(span=self._window_fast, min_periods=min_periods_s, adjust=False)
            .mean()
        )
        self._tsi = smoothed / smoothed_abs
        self._tsi *= 100
    # define the true strength index, check nulls and returns pandas.Series new feature generated
    def tsi(self) -> pd.Series:
        tsi_series = self._check_fillna(self._tsi, value=0)
        return pd.Series(tsi_series, name="tsi")

##################
# Set up sidebar #
##################
# set sidebar title 
st.sidebar.title('Stocks Dashboard :moneybag:')
url = 'https://finance.yahoo.com/most-active'
# add a button to open the yahoo finance website
if st.sidebar.button('Yahoo! Stocks'):
    webbrowser.open_new_tab(url)
    
# from PIL import Image 
image = Image.open('./images/investor.jpg')
st.sidebar.image(image)
# load stock symbols list
option = st.sidebar.selectbox('Select a Stock', ('AAPL','A', 'AA', 'AABA', 'AAC', 'AAL', 'AAME', 'AAN', 'AAOI', 'AAON', 'AAP', 'AAT', 'AAV', 'AAWW', 'AAXJ', 'AAXN', 'AB', 'ABAC', 'ABAX', 'ABB', 'ABBV', 'ABC', 'ABCB', 'ABCD', 'ABDC', 'ABEO', 'ABEOW', 'ABEV', 'ABG', 'ABIL', 'ABIO', 'ABLX', 'ABM', 'ABMD', 'ABR', 'ABRN', 'ABR^A', 'ABR^B', 'ABR^C', 'ABT', 'ABTX', 'ABUS', 'ABX', 'AC', 'ACAD', 'ACBI', 'ACC', 'ACCO', 'ACER', 'ACERW', 'ACET', 'ACFC', 'ACGL', 'ACGLO', 'ACGLP', 'ACH', 'ACHC', 'ACHN', 'ACHV', 'ACIA', 'ACIU', 'ACIW', 'ACLS', 'ACM', 'ACMR', 'ACN', 'ACNB', 'ACOR', 'ACP', 'ACRE', 'ACRS', 'ACRX', 'ACSF', 'ACST', 'ACT', 'ACTG', 'ACV', 'ACWI', 'ACWX', 'ACXM', 'ADAP', 'ADBE', 'ADC', 'ADES', 'ADI', 'ADM', 'ADMA', 'ADMP', 'ADMS', 'ADNT', 'ADOM', 'ADP', 'ADRA', 'ADRD', 'ADRE', 'ADRO', 'ADRU', 'ADS', 'ADSK', 'ADSW', 'ADT', 'ADTN', 'ADUS', 'ADVM', 'ADX', 'ADXS', 'ADXSW', 'AEB', 'AED', 'AEE', 'AEG', 'AEGN', 'AEH', 'AEHR', 'AEIS', 'AEK', 'AEL', 'AEM', 'AEMD', 'AEO', 'AEP', 'AER', 'AERI', 'AES', 'AET', 'AETI', 'AEUA', 'AEY', 'AEZS', 'AFAM', 'AFB', 'AFC', 'AFG', 'AFGE', 'AFGH', 'AFH', 'AFHBL', 'AFI', 'AFL', 'AFMD', 'AFSI', 'AFSI^A', 'AFSI^B', 'AFSI^C', 'AFSI^D', 'AFSI^E', 'AFSI^F', 'AFSS', 'AFST', 'AFT', 'AG', 'AGC', 'AGCO', 'AGD', 'AGEN', 'AGFS', 'AGFSW', 'AGI', 'AGII', 'AGIIL', 'AGIO', 'AGLE', 'AGM', 'AGM.A', 'AGM^A', 'AGM^B', 'AGM^C', 'AGN', 'AGNC', 'AGNCB', 'AGNCN', 'AGND', 'AGO', 'AGO^B', 'AGO^E', 'AGO^F', 'AGR', 'AGRO', 'AGRX', 'AGS', 'AGTC', 'AGX', 'AGYS', 'AGZD', 'AHC', 'AHGP', 'AHH', 'AHL', 'AHL^C', 'AHL^D', 'AHP', 'AHPA', 'AHPAU', 'AHPAW', 'AHPI', 'AHP^B', 'AHT', 'AHT^D', 'AHT^F', 'AHT^G', 'AHT^H', 'AHT^I', 'AI', 'AIA', 'AIC', 'AIF', 'AIG', 'AIG.WS', 'AIMC', 'AIMT', 'AIN', 'AINV', 'AIPT', 'AIR', 'AIRG', 'AIRR', 'AIRT', 'AIT', 'AIV', 'AIV^A', 'AIW', 'AIY', 'AIZ', 'AI^B', 'AJG', 'AJRD', 'AJX', 'AJXA', 'AKAM', 'AKAO', 'AKBA', 'AKCA', 'AKER', 'AKO.A', 'AKO.B', 'AKP', 'AKR', 'AKRX', 'AKS', 'AKTS', 'AKTX', 'AL', 'ALB', 'ALBO', 'ALCO', 'ALDR', 'ALDX', 'ALE', 'ALEX', 'ALG', 'ALGN', 'ALGT', 'ALIM', 'ALJJ', 'ALK', 'ALKS', 'ALL', 'ALLE', 'ALLT', 'ALLY', 'ALLY^A', 'ALL^A', 'ALL^B', 'ALL^C', 'ALL^D', 'ALL^E', 'ALL^F', 'ALNA', 'ALNY', 'ALOG', 'ALOT', 'ALPN', 'ALP^Q', 'ALQA', 'ALRM', 'ALRN', 'ALSK', 'ALSN', 'ALT', 'ALTR', 'ALTY', 'ALV', 'ALX', 'ALXN', 'AM', 'AMAG', 'AMAT', 'AMBA', 'AMBC', 'AMBCW', 'AMBR', 'AMC', 'AMCA', 'AMCN', 'AMCX', 'AMD', 'AMDA', 'AME', 'AMED', 'AMEH', 'AMG', 'AMGN', 'AMGP', 'AMH', 'AMH^C', 'AMH^D', 'AMH^E', 'AMH^F', 'AMH^G', 'AMID', 'AMKR', 'AMMA', 'AMN', 'AMNB', 'AMOT', 'AMOV', 'AMOV', 'AMP', 'AMPH', 'AMR', 'AMRB', 'AMRC', 'AMRH', 'AMRHW', 'AMRK', 'AMRN', 'AMRS', 'AMRWW', 'AMSC', 'AMSF', 'AMSWA', 'AMT', 'AMTD', 'AMTX', 'AMWD', 'AMX', 'AMZN', 'AN', 'ANAB', 'ANAT', 'ANCB', 'ANCX', 'ANDA', 'ANDAR', 'ANDAU', 'ANDAW', 'ANDE', 'ANDV', 'ANDX', 'ANET', 'ANF', 'ANFI', 'ANGI', 'ANGO', 'ANH', 'ANH^A', 'ANH^B', 'ANH^C', 'ANIK', 'ANIP', 'ANSS', 'ANTH', 'ANTM', 'ANTX', 'ANW', 'ANY', 'AOBC', 'AOD', 'AOI', 'AON', 'AOS', 'AOSL', 'AP', 'APA', 'APAM', 'APB', 'APC', 'APD', 'APDN', 'APDNW', 'APEI', 'APEN', 'APF', 'APH', 'APLE', 'APLP', 'APLS', 'APO', 'APOG', 'APOP', 'APOPW', 'APO^A', 'APPF', 'APPN', 'APPS', 'APRI', 'APRN', 'APTI', 'APTO', 'APTS', 'APTV', 'APU', 'APVO', 'APWC', 'AQ', 'AQB', 'AQMS', 'AQN', 'AQUA', 'AQXP', 'AR', 'ARA', 'ARAY', 'ARC', 'ARCB', 'ARCC', 'ARCH', 'ARCI', 'ARCO', 'ARCT', 'ARCW', 'ARD', 'ARDC', 'ARDM', 'ARDX', 'ARE', 'ARES', 'ARES^A', 'AREX', 'ARE^D', 'ARGS', 'ARGX', 'ARI', 'ARII', 'ARI^C', 'ARKR', 'ARL', 'ARLP', 'ARLZ', 'ARMK', 'ARMO', 'ARNA', 'ARNC', 'AROC', 'AROW', 'ARQL', 'ARR', 'ARRS', 'ARRY', 'ARR^A', 'ARR^B', 'ARTNA', 'ARTW', 'ARTX', 'ARW', 'ARWR', 'ASA', 'ASB', 'ASB^C', 'ASB^D', 'ASC', 'ASCMA', 'ASET', 'ASFI', 'ASG', 'ASGN', 'ASH', 'ASIX', 'ASMB', 'ASML', 'ASNA', 'ASND', 'ASNS', 'ASPN', 'ASPS', 'ASPU', 'ASR', 'ASRV', 'ASRVP', 'ASTC', 'ASTE', 'ASUR', 'ASV', 'ASX', 'ASYS', 'AT', 'ATAC', 'ATACR', 'ATACU', 'ATAI', 'ATAX', 'ATEC', 'ATEN', 'ATGE', 'ATH', 'ATHM', 'ATHN', 'ATHX', 'ATI', 'ATKR', 'ATLC', 'ATLO', 'ATNI', 'ATNX', 'ATO', 'ATOM', 'ATOS', 'ATR', 'ATRA', 'ATRC', 'ATRI', 'ATRO', 'ATRS', 'ATSG', 'ATTO', 'ATTU', 'ATU', 'ATUS', 'ATV', 'ATVI', 'ATXI', 'AU', 'AUBN', 'AUDC', 'AUO', 'AUPH', 'AUTO', 'AUY', 'AVA', 'AVAL', 'AVAV', 'AVB', 'AVD', 'AVDL', 'AVEO', 'AVGO', 'AVGR', 'AVH', 'AVHI', 'AVID', 'AVK', 'AVNW', 'AVP', 'AVT', 'AVX', 'AVXL', 'AVXS', 'AVY', 'AVYA', 'AWF', 'AWI', 'AWK', 'AWP', 'AWR', 'AWRE', 'AXAS', 'AXDX', 'AXE', 'AXGN', 'AXL', 'AXON', 'AXP', 'AXR', 'AXS', 'AXSM', 'AXS^D', 'AXS^E', 'AXTA', 'AXTI', 'AY', 'AYI', 'AYR', 'AYTU', 'AYX', 'AZN', 'AZO', 'AZPN', 'AZRE', 'AZRX', 'AZUL', 'AZZ', 'B', 'BA', 'BABA', 'BABY', 'BAC', 'BAC.WS.A', 'BAC.WS.B', 'BAC^A', 'BAC^C', 'BAC^D', 'BAC^E', 'BAC^I', 'BAC^L', 'BAC^W', 'BAC^Y', 'BAF', 'BAH', 'BAK', 'BAM', 'BANC', 'BANC^C', 'BANC^D', 'BANC^E', 'BAND', 'BANF', 'BANFP', 'BANR', 'BANX', 'BAP', 'BAS', 'BASI', 'BATRA', 'BATRK', 'BAX', 'BB', 'BBBY', 'BBC', 'BBD', 'BBDO', 'BBF', 'BBG', 'BBGI', 'BBH', 'BBK', 'BBL', 'BBN', 'BBOX', 'BBP', 'BBRG', 'BBSI', 'BBT', 'BBT^D', 'BBT^E', 'BBT^F', 'BBT^G', 'BBT^H', 'BBU', 'BBVA', 'BBW', 'BBX', 'BBY', 'BC', 'BCAC', 'BCACR', 'BCACU', 'BCACW', 'BCBP', 'BCC', 'BCE', 'BCEI', 'BCH', 'BCLI', 'BCO', 'BCOM', 'BCOR', 'BCOV', 'BCPC', 'BCRH', 'BCRX', 'BCS', 'BCS^D', 'BCTF', 'BCX', 'BDC', 'BDC^B', 'BDGE', 'BDJ', 'BDN', 'BDSI', 'BDX', 'BDXA', 'BEAT', 'BECN', 'BEDU', 'BEL', 'BELFA', 'BELFB', 'BEN', 'BEP', 'BERY', 'BF.A', 'BF.B', 'BFAM', 'BFIN', 'BFIT', 'BFK', 'BFO', 'BFR', 'BFRA', 'BFS', 'BFS^C', 'BFS^D', 'BFZ', 'BG', 'BGB', 'BGC', 'BGCA', 'BGCP', 'BGFV', 'BGG', 'BGH', 'BGIO', 'BGNE', 'BGR', 'BGS', 'BGT', 'BGX', 'BGY', 'BH', 'BHAC', 'BHACR', 'BHACU', 'BHACW', 'BHBK', 'BHE', 'BHF', 'BHGE', 'BHK', 'BHLB', 'BHP', 'BHVN', 'BIB', 'BICK', 'BID', 'BIDU', 'BIF', 'BIG', 'BIIB', 'BIO', 'BIO.B', 'BIOC', 'BIOL', 'BIOS', 'BIP', 'BIS', 'BIT', 'BITA', 'BJRI', 'BJZ', 'BK', 'BKCC', 'BKD', 'BKE', 'BKEP', 'BKEPP', 'BKH', 'BKHU', 'BKI', 'BKK', 'BKN', 'BKNG', 'BKS', 'BKSC', 'BKT', 'BKU', 'BKYI', 'BK^C', 'BL', 'BLBD', 'BLCM', 'BLCN', 'BLD', 'BLDP', 'BLDR', 'BLFS', 'BLH', 'BLIN', 'BLK', 'BLKB', 'BLL', 'BLMN', 'BLMT', 'BLNK', 'BLNKW', 'BLPH', 'BLRX', 'BLUE', 'BLW', 'BLX', 'BMA', 'BMCH', 'BME', 'BMI', 'BMLP', 'BML^G', 'BML^H', 'BML^I', 'BML^J', 'BML^L', 'BMO', 'BMRA', 'BMRC', 'BMRN', 'BMS', 'BMTC', 'BMY', 'BNCL', 'BNDX', 'BNED', 'BNFT', 'BNJ', 'BNS', 'BNSO', 'BNTC', 'BNTCW', 'BNY', 'BOCH', 'BOE', 'BOFI', 'BOFIL', 'BOH', 'BOJA', 'BOKF', 'BOKFL', 'BOLD', 'BOMN', 'BOOM', 'BOOT', 'BORN', 'BOSC', 'BOTJ', 'BOTZ', 'BOX', 'BOXL', 'BP', 'BPFH', 'BPFHP', 'BPFHW', 'BPI', 'BPK', 'BPL', 'BPMC', 'BPMP', 'BPOP', 'BPOPM', 'BPOPN', 'BPRN', 'BPT', 'BPTH', 'BPY', 'BQH', 'BR', 'BRAC', 'BRACR', 'BRACU', 'BRACW', 'BRC', 'BREW', 'BRFS', 'BRID', 'BRK.A', 'BRK.B', 'BRKL', 'BRKR', 'BRKS', 'BRO', 'BRPA', 'BRPAR', 'BRPAU', 'BRPAW', 'BRQS', 'BRS', 'BRSS', 'BRT', 'BRX', 'BSAC', 'BSBR', 'BSD', 'BSE', 'BSET', 'BSF', 'BSL', 'BSM', 'BSMX', 'BSPM', 'BSQR', 'BSRR', 'BST', 'BSTC', 'BSTI', 'BSX', 'BT', 'BTA', 'BTAI', 'BTE', 'BTEC', 'BTI', 'BTO', 'BTT', 'BTU', 'BTZ', 'BUD', 'BUFF', 'BUI', 'BUR', 'BURG', 'BURL', 'BUSE', 'BVN', 'BVNSC', 'BVSN', 'BVXV', 'BVXVW', 'BW', 'BWA', 'BWEN', 'BWFG', 'BWG', 'BWINA', 'BWINB', 'BWP', 'BWXT', 'BX', 'BXC', 'BXE', 'BXG', 'BXMT', 'BXMX', 'BXP', 'BXP^B', 'BXS', 'BY', 'BYBK', 'BYD', 'BYFC', 'BYM', 'BYSI', 'BZH', 'BZUN', 'C', 'C.WS.A', 'CA', 'CAAP', 'CAAS', 'CABO', 'CAC', 'CACC', 'CACG', 'CACI', 'CADC', 'CADE', 'CAE', 'CAF', 'CAFD', 'CAG', 'CAH', 'CAI', 'CAJ', 'CAKE', 'CAL', 'CALA', 'CALD', 'CALI', 'CALL', 'CALM', 'CALX', 'CAMP', 'CAMT', 'CAPL', 'CAPR', 'CAR', 'CARA', 'CARB', 'CARG', 'CARO', 'CARS', 'CART', 'CARV', 'CARZ', 'CASA', 'CASC', 'CASH', 'CASI', 'CASM', 'CASS', 'CASY', 'CAT', 'CATB', 'CATC', 'CATH', 'CATM', 'CATO', 'CATS', 'CATY', 'CATYW', 'CAVM', 'CB', 'CBA', 'CBAK', 'CBAN', 'CBAY', 'CBB', 'CBB^B', 'CBD', 'CBFV', 'CBG', 'CBH', 'CBI', 'CBIO', 'CBK', 'CBL', 'CBLI', 'CBL^D', 'CBL^E', 'CBM', 'CBMG', 'CBO', 'CBOE', 'CBPO', 'CBPX', 'CBRL', 'CBS', 'CBS.A', 'CBSH', 'CBSHP', 'CBT', 'CBTX', 'CBU', 'CBX', 'CBZ', 'CC', 'CCBG', 'CCCL', 'CCCR', 'CCD', 'CCE', 'CCI', 'CCIH', 'CCI^A', 'CCJ', 'CCK', 'CCL', 'CCLP', 'CCM', 'CCMP', 'CCNE', 'CCO', 'CCOI', 'CCR', 'CCRC', 'CCRN', 'CCS', 'CCT', 'CCU', 'CCUR', 'CCXI', 'CCZ', 'CDC', 'CDE', 'CDEV', 'CDK', 'CDL', 'CDLX', 'CDMO', 'CDMOP', 'CDNA', 'CDNS', 'CDOR', 'CDR', 'CDR^B', 'CDR^C', 'CDTI', 'CDTX', 'CDW', 'CDXC', 'CDXS', 'CDZI', 'CE', 'CEA', 'CECE', 'CECO', 'CEE', 'CEIX', 'CEL', 'CELC', 'CELG', 'CELGZ', 'CELH', 'CELP', 'CEM', 'CEMI', 'CEN', 'CENT', 'CENTA', 'CENX', 'CEO', 'CEPU', 'CEQP', 'CERC', 'CERCW', 'CERN', 'CERS', 'CETV', 'CETX', 'CETXP', 'CETXW', 'CEVA', 'CEY', 'CEZ', 'CF', 'CFA', 'CFBI', 'CFBK', 'CFC^B', 'CFFI', 'CFFN', 'CFG', 'CFMS', 'CFO', 'CFR', 'CFRX', 'CFR^A', 'CFX', 'CG', 'CGA', 'CGBD', 'CGEN', 'CGG', 'CGI', 'CGIX', 'CGNT', 'CGNX', 'CGO', 'CHA', 'CHCI', 'CHCO', 'CHCT', 'CHD', 'CHDN', 'CHE', 'CHEF', 'CHEK', 'CHEKW', 'CHFC', 'CHFN', 'CHFS', 'CHGG', 'CHH', 'CHI', 'CHK', 'CHKE', 'CHKP', 'CHKR', 'CHK^D', 'CHL', 'CHMA', 'CHMG', 'CHMI', 'CHMI^A', 'CHN', 'CHNR', 'CHRS', 'CHRW', 'CHS', 'CHSCL', 'CHSCM', 'CHSCN', 'CHSCO', 'CHSCP', 'CHSP', 'CHT', 'CHTR', 'CHU', 'CHUBA', 'CHUBK', 'CHUY', 'CHW', 'CHY', 'CI', 'CIA', 'CIB', 'CIBR', 'CIC', 'CIC.U', 'CIC.WS', 'CID', 'CIDM', 'CIEN', 'CIF', 'CIFS', 'CIG', 'CIG.C', 'CIGI', 'CII', 'CIL', 'CIM', 'CIM^A', 'CIM^B', 'CINF', 'CINR', 'CIO', 'CIO^A', 'CIR', 'CISN', 'CIT', 'CIU', 'CIVB', 'CIVBP', 'CIVI', 'CIZ', 'CIZN', 'CJ', 'CJJD', 'CKH', 'CKPT', 'CL', 'CLAR', 'CLB', 'CLBS', 'CLCT', 'CLD', 'CLDC', 'CLDR', 'CLDT', 'CLDX', 'CLF', 'CLFD', 'CLGN', 'CLGX', 'CLH', 'CLI', 'CLIR', 'CLIRW', 'CLLS', 'CLMT', 'CLNC', 'CLNE', 'CLNS', 'CLNS^B', 'CLNS^D', 'CLNS^E', 'CLNS^G', 'CLNS^H', 'CLNS^I', 'CLNS^J', 'CLPR', 'CLR', 'CLRB', 'CLRBW', 'CLRBZ', 'CLRG', 'CLRO', 'CLS', 'CLSD', 'CLSN', 'CLUB', 'CLVS', 'CLW', 'CLWT', 'CLX', 'CLXT', 'CM', 'CMA', 'CMA.WS', 'CMC', 'CMCM', 'CMCO', 'CMCSA', 'CMCT', 'CMCTP', 'CMD', 'CME', 'CMFN', 'CMG', 'CMI', 'CMO', 'CMO^E', 'CMP', 'CMPR', 'CMRE', 'CMRE^B', 'CMRE^C', 'CMRE^D', 'CMRE^E', 'CMRX', 'CMS', 'CMSS', 'CMSSR', 'CMSSU', 'CMSSW', 'CMS^B', 'CMTA', 'CMTL', 'CMU', 'CNA', 'CNAC', 'CNACR', 'CNACU', 'CNACW', 'CNAT', 'CNBKA', 'CNC', 'CNCE', 'CNCR', 'CNDT', 'CNET', 'CNFR', 'CNHI', 'CNI', 'CNIT', 'CNK', 'CNMD', 'CNNE', 'CNO', 'CNOB', 'CNP', 'CNQ', 'CNS', 'CNSL', 'CNTF', 'CNTY', 'CNX', 'CNXM', 'CNXN', 'CO', 'COBZ', 'CODA', 'CODI', 'CODI^A', 'CODX', 'COE', 'COF', 'COF.WS', 'COF^C', 'COF^D', 'COF^F', 'COF^G', 'COF^H', 'COF^P', 'COG', 'COGT', 'COHR', 'COHU', 'COKE', 'COL', 'COLB', 'COLD', 'COLL', 'COLM', 'COMM', 'COMT', 'CONE', 'CONN', 'COO', 'COOL', 'COP', 'COR', 'CORE', 'CORI', 'CORR', 'CORR^A', 'CORT', 'COST', 'COT', 'COTV', 'COTY', 'COUP', 'COWN', 'COWNZ', 'CP', 'CPA', 'CPAC', 'CPAH', 'CPB', 'CPE', 'CPE^A', 'CPF', 'CPG', 'CPHC', 'CPIX', 'CPK', 'CPL', 'CPLA', 'CPLP', 'CPRT', 'CPRX', 'CPS', 'CPSH', 'CPSI', 'CPSS', 'CPST', 'CPT', 'CPTA', 'CPTAG', 'CPTAL', 'CR', 'CRAI', 'CRAY', 'CRBP', 'CRC', 'CRCM', 'CRD.A', 'CRD.B', 'CRED', 'CREE', 'CREG', 'CRESY', 'CRH', 'CRI', 'CRIS', 'CRK', 'CRL', 'CRM', 'CRME', 'CRMT', 'CRNT', 'CRON', 'CROX', 'CRR', 'CRS', 'CRSP', 'CRT', 'CRTO', 'CRUS', 'CRUSC', 'CRVL', 'CRVS', 'CRWS', 'CRY', 'CRZO', 'CS', 'CSA', 'CSB', 'CSBK', 'CSBR', 'CSCO', 'CSF', 'CSFL', 'CSGP', 'CSGS', 'CSII', 'CSIQ', 'CSJ', 'CSL', 'CSLT', 'CSML', 'CSOD', 'CSPI', 'CSQ', 'CSRA', 'CSS', 'CSSE', 'CSTE', 'CSTM', 'CSTR', 'CSU', 'CSV', 'CSWC', 'CSWCL', 'CSWI', 'CSX', 'CTAA', 'CTAS', 'CTB', 'CTBB', 'CTBI', 'CTDD', 'CTG', 'CTHR', 'CTIB', 'CTIC', 'CTL', 'CTLT', 'CTMX', 'CTR', 'CTRE', 'CTRL', 'CTRN', 'CTRP', 'CTRV', 'CTS', 'CTSH', 'CTSO', 'CTT', 'CTU', 'CTV', 'CTW', 'CTWS', 'CTX', 'CTXR', 'CTXRW', 'CTXS', 'CTY', 'CTZ', 'CUB', 'CUBA', 'CUBE', 'CUBI', 'CUBI^C', 'CUBI^D', 'CUBI^E', 'CUBI^F', 'CUBS', 'CUE', 'CUI', 'CUK', 'CULP', 'CUR', 'CURO', 'CUTR', 'CUZ', 'CVA', 'CVBF', 'CVCO', 'CVCY', 'CVE', 'CVEO', 'CVG', 'CVGI', 'CVGW', 'CVI', 'CVLT', 'CVLY', 'CVNA', 'CVON', 'CVONW', 'CVRR', 'CVS', 'CVTI', 'CVV', 'CVX', 'CW', 'CWAY', 'CWBC', 'CWBR', 'CWCO', 'CWH', 'CWST', 'CWT', 'CX', 'CXDC', 'CXE', 'CXH', 'CXO', 'CXP', 'CXRX', 'CXSE', 'CXW', 'CY', 'CYAD', 'CYAN', 'CYBE', 'CYBR', 'CYCC', 'CYCCP', 'CYD', 'CYH', 'CYHHZ', 'CYOU', 'CYRN', 'CYRX', 'CYRXW', 'CYS', 'CYS^A', 'CYS^B', 'CYTK', 'CYTR', 'CYTX', 'CYTXW', 'CZFC', 'CZNC', 'CZR', 'CZWI', 'CZZ', 'C^C', 'C^J', 'C^K', 'C^L', 'C^N', 'C^S', 'D', 'DAC', 'DAIO', 'DAKT', 'DAL', 'DAN', 'DAR', 'DARE', 'DATA', 'DAVE', 'DAX', 'DB', 'DBD', 'DBL', 'DBVT', 'DCAR', 'DCF', 'DCI', 'DCIX', 'DCM', 'DCO', 'DCOM', 'DCP', 'DCPH', 'DCT', 'DCUD', 'DDBI', 'DDD', 'DDE', 'DDF', 'DDR', 'DDR^A', 'DDR^J', 'DDR^K', 'DDS', 'DDT', 'DD^A', 'DD^B', 'DE', 'DEA', 'DECK', 'DEI', 'DELT', 'DENN', 'DEO', 'DEPO', 'DERM', 'DESP', 'DEST', 'DEX', 'DF', 'DFBG', 'DFBHU', 'DFFN', 'DFIN', 'DFNL', 'DFP', 'DFRG', 'DFS', 'DFVL', 'DFVS', 'DG', 'DGICA', 'DGICB', 'DGII', 'DGLD', 'DGLY', 'DGRE', 'DGRS', 'DGRW', 'DGX', 'DHCP', 'DHF', 'DHG', 'DHI', 'DHIL', 'DHR', 'DHT', 'DHX', 'DHXM', 'DIAX', 'DIN', 'DINT', 'DIOD', 'DIS', 'DISCA', 'DISCB', 'DISCK', 'DISH', 'DJCO', 'DK', 'DKL', 'DKS', 'DKT', 'DL', 'DLB', 'DLBL', 'DLBS', 'DLHC', 'DLNG', 'DLNG^A', 'DLPH', 'DLPN', 'DLPNW', 'DLR', 'DLR^C', 'DLR^G', 'DLR^H', 'DLR^I', 'DLR^J', 'DLTH', 'DLTR', 'DLX', 'DM', 'DMB', 'DMLP', 'DMO', 'DMPI', 'DMRC', 'DNB', 'DNBF', 'DNI', 'DNKN', 'DNLI', 'DNOW', 'DNP', 'DNR', 'DO', 'DOC', 'DOGZ', 'DOOR', 'DORM', 'DOTA', 'DOTAR', 'DOTAU', 'DOTAW', 'DOV', 'DOVA', 'DOX', 'DPG', 'DPLO', 'DPS', 'DPZ', 'DQ', 'DRAD', 'DRD', 'DRE', 'DRH', 'DRI', 'DRIO', 'DRIOW', 'DRNA', 'DRQ', 'DRRX', 'DRUA', 'DRYS', 'DS', 'DSE', 'DSGX', 'DSKE', 'DSKEW', 'DSL', 'DSLV', 'DSM', 'DSPG', 'DST', 'DSU', 'DSW', 'DSWL', 'DSX', 'DSXN', 'DSX^B', 'DS^B', 'DS^C', 'DS^D', 'DTE', 'DTEA', 'DTF', 'DTJ', 'DTLA^', 'DTQ', 'DTRM', 'DTUL', 'DTUS', 'DTV', 'DTW', 'DTY', 'DTYL', 'DTYS', 'DUC', 'DUK', 'DUKH', 'DUSA', 'DVA', 'DVAX', 'DVCR', 'DVD', 'DVMT', 'DVN', 'DVY', 'DWAC', 'DWAQ', 'DWAS', 'DWAT', 'DWCH', 'DWCR', 'DWDP', 'DWFI', 'DWIN', 'DWLD', 'DWLV', 'DWPP', 'DWSN', 'DWTR', 'DX', 'DXB', 'DXC', 'DXCM', 'DXGE', 'DXJS', 'DXLG', 'DXPE', 'DXPS', 'DXYN', 'DX^A', 'DX^B', 'DY', 'DYN', 'DYN.WS.A', 'DYNC', 'DYNT', 'DYSL', 'DZSI', 'E', 'EA', 'EAB', 'EACQ', 'EACQU', 'EACQW', 'EAE', 'EAGL', 'EAGLU', 'EAGLW', 'EAI', 'EARN', 'EARS', 'EAST', 'EASTW', 'EAT', 'EBAY', 'EBAYL', 'EBF', 'EBIO', 'EBIX', 'EBMT', 'EBR', 'EBR.B', 'EBS', 'EBSB', 'EBTC', 'EC', 'ECA', 'ECC', 'ECCA', 'ECCB', 'ECCY', 'ECCZ', 'ECHO', 'ECL', 'ECOL', 'ECOM', 'ECPG', 'ECR', 'ECT', 'ECYT', 'ED', 'EDAP', 'EDBI', 'EDD', 'EDF', 'EDGE', 'EDGW', 'EDI', 'EDIT', 'EDN', 'EDR', 'EDU', 'EDUC', 'EE', 'EEA', 'EEFT', 'EEI', 'EEMA', 'EEP', 'EEQ', 'EEX', 'EFAS', 'EFBI', 'EFC', 'EFF', 'EFII', 'EFL', 'EFOI', 'EFR', 'EFSC', 'EFT', 'EFX', 'EGAN', 'EGBN', 'EGF', 'EGHT', 'EGHT', 'EGIF', 'EGL', 'EGLE', 'EGLT', 'EGN', 'EGO', 'EGOV', 'EGP', 'EGRX', 'EGY', 'EHC', 'EHI', 'EHIC', 'EHT', 'EHTH', 'EIG', 'EIGI', 'EIGR', 'EIX', 'EKSO', 'EL', 'ELC', 'ELEC', 'ELECU', 'ELECW', 'ELF', 'ELGX', 'ELJ', 'ELLI', 'ELON', 'ELP', 'ELS', 'ELSE', 'ELTK', 'ELU', 'ELVT', 'ELY', 'EMB', 'EMCB', 'EMCF', 'EMCG', 'EMCI', 'EMD', 'EME', 'EMES', 'EMF', 'EMIF', 'EMITF', 'EMKR', 'EML', 'EMMS', 'EMN', 'EMO', 'EMP', 'EMR', 'EMXC', 'ENB', 'ENBL', 'ENDP', 'ENFC', 'ENG', 'ENIA', 'ENIC', 'ENJ', 'ENLC', 'ENLK', 'ENO', 'ENPH', 'ENR', 'ENS', 'ENSG', 'ENT', 'ENTA', 'ENTG', 'ENV', 'ENVA', 'ENZ', 'ENZL', 'EOCC', 'EOD', 'EOG', 'EOI', 'EOLS', 'EOS', 'EOT', 'EPAM', 'EPAY', 'EPC', 'EPD', 'EPE', 'EPIX', 'EPR', 'EPR^C', 'EPR^E', 'EPR^G', 'EPZM', 'EP^C', 'EQBK', 'EQC', 'EQC^D', 'EQFN', 'EQGP', 'EQIX', 'EQM', 'EQR', 'EQRR', 'EQS', 'EQT', 'ERA', 'ERF', 'ERI', 'ERIC', 'ERIE', 'ERII', 'ERJ', 'EROS', 'ERYP', 'ES', 'ESBK', 'ESCA', 'ESE', 'ESEA', 'ESES', 'ESG', 'ESGD', 'ESGE', 'ESGG', 'ESGR', 'ESGU', 'ESIO', 'ESL', 'ESLT', 'ESND', 'ESNT', 'ESPR', 'ESQ', 'ESRT', 'ESRX', 'ESS', 'ESSA', 'ESTE', 'ESTR', 'ESTRW', 'ESV', 'ESXB', 'ETB', 'ETE', 'ETFC', 'ETG', 'ETH', 'ETJ', 'ETM', 'ETN', 'ETO', 'ETP', 'ETR', 'ETSY', 'ETV', 'ETW', 'ETX', 'ETY', 'EUFN', 'EURN', 'EV', 'EVA', 'EVBG', 'EVC', 'EVEP', 'EVF', 'EVFM', 'EVFTC', 'EVG', 'EVGBC', 'EVGN', 'EVH', 'EVHC', 'EVK', 'EVLMC', 'EVLV', 'EVN', 'EVOK', 'EVOL', 'EVR', 'EVRI', 'EVSTC', 'EVT', 'EVTC', 'EW', 'EWBC', 'EWZS', 'EXAS', 'EXC', 'EXD', 'EXEL', 'EXFO', 'EXG', 'EXK', 'EXLS', 'EXP', 'EXPD', 'EXPE', 'EXPO', 'EXPR', 'EXR', 'EXTN', 'EXTR', 'EXXI', 'EYE', 'EYEG', 'EYEGW', 'EYEN', 'EYES', 'EYESW', 'EZPW', 'EZT', 'F', 'FAAR', 'FAB', 'FAC', 'FAD', 'FAF', 'FALN', 'FAM', 'FAMI', 'FANG', 'FANH', 'FARM', 'FARO', 'FAST', 'FAT', 'FATE', 'FB', 'FBC', 'FBHS', 'FBIO', 'FBIOP', 'FBIZ', 'FBK', 'FBM', 'FBMS', 'FBNC', 'FBNK', 'FBP', 'FBR', 'FBSS', 'FBZ', 'FC', 'FCA', 'FCAL', 'FCAN', 'FCAP', 'FCAU', 'FCB', 'FCBC', 'FCCO', 'FCCY', 'FCE.A', 'FCEF', 'FCEL', 'FCF', 'FCFS', 'FCFS', 'FCN', 'FCNCA', 'FCPT', 'FCRE', 'FCSC', 'FCT', 'FCVT', 'FCX', 'FDBC', 'FDC', 'FDEF', 'FDEU', 'FDIV', 'FDP', 'FDS', 'FDT', 'FDTS', 'FDUS', 'FDUSL', 'FDX', 'FE', 'FEDU', 'FEI', 'FEIM', 'FELE', 'FELP', 'FEM', 'FEMB', 'FEMS', 'FENC', 'FENG', 'FEO', 'FEP', 'FET', 'FEUZ', 'FEX', 'FEYE', 'FF', 'FFA', 'FFBC', 'FFBCW', 'FFBW', 'FFC', 'FFG', 'FFHL', 'FFIC', 'FFIN', 'FFIV', 'FFKT', 'FFNW', 'FFWM', 'FG', 'FG.WS', 'FGB', 'FGBI', 'FGEN', 'FGM', 'FGP', 'FHB', 'FHK', 'FHN', 'FHN^A', 'FHY', 'FI', 'FIBK', 'FICO', 'FIF', 'FII', 'FINL', 'FINX', 'FIS', 'FISI', 'FISV', 'FIT', 'FITB', 'FITBI', 'FIV', 'FIVE', 'FIVN', 'FIX', 'FIXD', 'FIZZ', 'FJP', 'FKO', 'FKU', 'FL', 'FLAG', 'FLAT', 'FLC', 'FLDM', 'FLEX', 'FLGT', 'FLIC', 'FLIR', 'FLKS', 'FLL', 'FLN', 'FLO', 'FLOW', 'FLR', 'FLS', 'FLT', 'FLWS', 'FLXN', 'FLXS', 'FLY', 'FMAO', 'FMB', 'FMBH', 'FMBI', 'FMC', 'FMHI', 'FMI', 'FMK', 'FMN', 'FMNB', 'FMO', 'FMS', 'FMSA', 'FMX', 'FMY', 'FN', 'FNB', 'FNBG', 'FNB^E', 'FNCB', 'FND', 'FNF', 'FNGN', 'FNHC', 'FNJN', 'FNK', 'FNKO', 'FNLC', 'FNSR', 'FNTE', 'FNTEU', 'FNTEW', 'FNV', 'FNWB', 'FNX', 'FNY', 'FOANC', 'FOE', 'FOF', 'FOGO', 'FOLD', 'FOMX', 'FONE', 'FONR', 'FOR', 'FORD', 'FORK', 'FORM', 'FORR', 'FORTY', 'FOSL', 'FOX', 'FOXA', 'FOXF', 'FPA', 'FPAY', 'FPF', 'FPH', 'FPI', 'FPI^B', 'FPL', 'FPRX', 'FPXI', 'FR', 'FRA', 'FRAC', 'FRAN', 'FRBA', 'FRBK', 'FRC', 'FRC^D', 'FRC^E', 'FRC^F', 'FRC^G', 'FRC^H', 'FRED', 'FRGI', 'FRME', 'FRO', 'FRPH', 'FRPT', 'FRSH', 'FRSX', 'FRT', 'FRTA', 'FRT^C', 'FSAC', 'FSACU', 'FSACW', 'FSB', 'FSBC', 'FSBW', 'FSCT', 'FSD', 'FSFG', 'FSIC', 'FSLR', 'FSM', 'FSNN', 'FSS', 'FSTR', 'FSV', 'FSZ', 'FT', 'FTA', 'FTAG', 'FTAI', 'FTC', 'FTCS', 'FTD', 'FTEK', 'FTEO', 'FTFT', 'FTGC', 'FTHI', 'FTI', 'FTK', 'FTLB', 'FTNT', 'FTR', 'FTRI', 'FTRPR', 'FTS', 'FTSI', 'FTSL', 'FTSM', 'FTV', 'FTW', 'FTXD', 'FTXG', 'FTXH', 'FTXL', 'FTXN', 'FTXO', 'FTXR', 'FUL', 'FULT', 'FUN', 'FUNC', 'FUND', 'FUSB', 'FUV', 'FV', 'FVC', 'FVE', 'FWONA', 'FWONK', 'FWP', 'FWRD', 'FYC', 'FYT', 'FYX', 'G', 'GAB', 'GABC', 'GAB^D', 'GAB^G', 'GAB^H', 'GAB^J', 'GAIA', 'GAIN', 'GAINM', 'GAINN', 'GAINO', 'GALT', 'GAM', 'GAM^B', 'GARS', 'GASS', 'GATX', 'GBAB', 'GBCI', 'GBDC', 'GBL', 'GBLI', 'GBLIL', 'GBLIZ', 'GBNK', 'GBT', 'GBX', 'GCAP', 'GCBC', 'GCH', 'GCI', 'GCO', 'GCP', 'GCV', 'GCVRZ', 'GCV^B', 'GD', 'GDDY', 'GDEN', 'GDI', 'GDL', 'GDL^B', 'GDO', 'GDOT', 'GDS', 'GDV', 'GDV^A', 'GDV^D', 'GDV^G', 'GE', 'GEC', 'GECC', 'GECCL', 'GECCM', 'GEF', 'GEF.B', 'GEK', 'GEL', 'GEMP', 'GEN', 'GENC', 'GENE', 'GENY', 'GEO', 'GEOS', 'GER', 'GERN', 'GES', 'GEVO', 'GF', 'GFA', 'GFED', 'GFF', 'GFI', 'GFN', 'GFNCP', 'GFNSL', 'GFY', 'GG', 'GGAL', 'GGB', 'GGG', 'GGM', 'GGP', 'GGP^A', 'GGT', 'GGT^B', 'GGT^E', 'GGZ', 'GGZ^A', 'GHC', 'GHDX', 'GHL', 'GHM', 'GHY', 'GIB', 'GIFI', 'GIG', 'GIG.U', 'GIG.WS', 'GIGM', 'GIG~', 'GIII', 'GIL', 'GILD', 'GILT', 'GIM', 'GIS', 'GJH', 'GJO', 'GJP', 'GJR', 'GJS', 'GJT', 'GJV', 'GKOS', 'GLAD', 'GLADN', 'GLBS', 'GLBZ', 'GLDD', 'GLDI', 'GLMD', 'GLNG', 'GLOB', 'GLOG', 'GLOG^A', 'GLOP', 'GLOP^A', 'GLOP^B', 'GLP', 'GLPG', 'GLPI', 'GLRE', 'GLT', 'GLUU', 'GLW', 'GLYC', 'GM', 'GM.WS.B', 'GME', 'GMED', 'GMLP', 'GMLPP', 'GMRE', 'GMRE^A', 'GMS', 'GMTA', 'GMZ', 'GNBC', 'GNC', 'GNCA', 'GNE', 'GNE^A', 'GNK', 'GNL', 'GNL^A', 'GNMA', 'GNMK', 'GNMX', 'GNRC', 'GNRT', 'GNRX', 'GNT', 'GNTX', 'GNTY', 'GNT^A', 'GNUS', 'GNW', 'GOF', 'GOGL', 'GOGO', 'GOL', 'GOLD', 'GOLF', 'GOOD', 'GOODM', 'GOODO', 'GOODP', 'GOOG', 'GOOGL', 'GOOS', 'GOV', 'GOVNI', 'GPAQU', 'GPC', 'GPI', 'GPIC', 'GPJA', 'GPK', 'GPM', 'GPMT', 'GPN', 'GPOR', 'GPP', 'GPRE', 'GPRK', 'GPRO', 'GPS', 'GPT', 'GPT^A', 'GPX', 'GRA', 'GRAM', 'GRBIC', 'GRBK', 'GRC', 'GRFS', 'GRID', 'GRIF', 'GRMN', 'GROW', 'GRP.U', 'GRPN', 'GRR', 'GRUB', 'GRVY', 'GRX', 'GRX^A', 'GRX^B', 'GS', 'GSBC', 'GSBD', 'GSH', 'GSHT', 'GSHTU', 'GSHTW', 'GSIT', 'GSK', 'GSL', 'GSL^B', 'GSM', 'GSUM', 'GSVC', 'GS^A', 'GS^B', 'GS^C', 'GS^D', 'GS^J', 'GS^K', 'GS^N', 'GT', 'GTES', 'GTHX', 'GTIM', 'GTLS', 'GTN', 'GTN.A', 'GTS', 'GTT', 'GTXI', 'GTY', 'GTYH', 'GTYHU', 'GTYHW', 'GULF', 'GURE', 'GUT', 'GUT^A', 'GUT^C', 'GVA', 'GWB', 'GWGH', 'GWPH', 'GWR', 'GWRE', 'GWRS', 'GWW', 'GXP', 'GYB', 'GYC', 'GYRO', 'GZT', 'H', 'HA', 'HABT', 'HAE', 'HAFC', 'HAIN', 'HAIR', 'HAL', 'HALL', 'HALO', 'HAS', 'HASI', 'HAWK', 'HAYN', 'HBAN', 'HBANN', 'HBANO', 'HBB', 'HBCP', 'HBHC', 'HBHCL', 'HBI', 'HBIO', 'HBK', 'HBM', 'HBM.WS', 'HBMD', 'HBNC', 'HBP', 'HCA', 'HCAP', 'HCAPZ', 'HCC', 'HCCI', 'HCHC', 'HCI', 'HCKT', 'HCLP', 'HCM', 'HCOM', 'HCP', 'HCSG', 'HD', 'HDB', 'HDNG', 'HDP', 'HDS', 'HDSN', 'HE', 'HEAR', 'HEBT', 'HEES', 'HEI', 'HEI.A', 'HELE', 'HEP', 'HEQ', 'HES', 'HESM', 'HES^A', 'HEWG', 'HE^U', 'HF', 'HFBC', 'HFBL', 'HFC', 'HFGIC', 'HFRO', 'HFWA', 'HGH', 'HGSH', 'HGT', 'HGV', 'HHC', 'HHS', 'HI', 'HIBB', 'HIE', 'HIFR', 'HIFS', 'HIG', 'HIG.WS', 'HIHO', 'HII', 'HIIQ', 'HIL', 'HIMX', 'HIO', 'HIVE', 'HIW', 'HIX', 'HJV', 'HK', 'HK.WS', 'HL', 'HLF', 'HLG', 'HLI', 'HLIT', 'HLNE', 'HLT', 'HLX', 'HL^B', 'HMC', 'HMHC', 'HMI', 'HMLP', 'HMLP^A', 'HMN', 'HMNF', 'HMNY', 'HMST', 'HMSY', 'HMTA', 'HMTV', 'HMY', 'HNDL', 'HNI', 'HNNA', 'HNP', 'HNRG', 'HOFT', 'HOG', 'HOLI', 'HOLX', 'HOMB', 'HOME', 'HON', 'HONE', 'HOPE', 'HOS', 'HOV', 'HOVNP', 'HP', 'HPE', 'HPF', 'HPI', 'HPJ', 'HPP', 'HPQ', 'HPS', 'HPT', 'HQCL', 'HQH', 'HQL', 'HQY', 'HR', 'HRB', 'HRC', 'HRG', 'HRI', 'HRL', 'HRS', 'HRTG', 'HRTX', 'HRZN', 'HSBC', 'HSBC^A', 'HSC', 'HSEA', 'HSEB', 'HSGX', 'HSIC', 'HSII', 'HSKA', 'HSON', 'HST', 'HSTM', 'HSY', 'HT', 'HTA', 'HTBI', 'HTBK', 'HTBX', 'HTD', 'HTFA', 'HTGC', 'HTGM', 'HTGX', 'HTH', 'HTHT', 'HTLD', 'HTLF', 'HTY', 'HTZ', 'HT^C', 'HT^D', 'HT^E', 'HUBB', 'HUBG', 'HUBS', 'HUD', 'HUM', 'HUN', 'HUNT', 'HUNTU', 'HUNTW', 'HURC', 'HURN', 'HVBC', 'HVT', 'HVT.A', 'HWBK', 'HWCC', 'HWKN', 'HX', 'HXL', 'HY', 'HYAC', 'HYACU', 'HYACW', 'HYB', 'HYGS', 'HYH', 'HYI', 'HYLS', 'HYND', 'HYT', 'HYXE', 'HYZD', 'HZN', 'HZNP', 'HZO', 'I', 'IAC', 'IAE', 'IAG', 'IAM', 'IAMXR', 'IAMXW', 'IART', 'IBA', 'IBB', 'IBCP', 'IBKC', 'IBKCO', 'IBKCP', 'IBKR', 'IBM', 'IBN', 'IBOC', 'IBP', 'IBTX', 'IBUY', 'ICAD', 'ICB', 'ICBK', 'ICCC', 'ICCH', 'ICD', 'ICE', 'ICFI', 'ICHR', 'ICL', 'ICLK', 'ICLN', 'ICLR', 'ICON', 'ICPT', 'ICUI', 'IDA', 'IDCC', 'IDE', 'IDLB', 'IDRA', 'IDSA', 'IDSY', 'IDT', 'IDTI', 'IDXG', 'IDXX', 'IEF', 'IEI', 'IEP', 'IESC', 'IEUS', 'IEX', 'IFEU', 'IFF', 'IFGL', 'IFMK', 'IFN', 'IFON', 'IFRX', 'IFV', 'IGA', 'IGD', 'IGF', 'IGI', 'IGLD', 'IGOV', 'IGR', 'IGT', 'IHC', 'IHD', 'IHG', 'IHIT', 'IHTA', 'IID', 'IIF', 'III', 'IIIN', 'IIJI', 'IIM', 'IIN', 'IIPR', 'IIPR^A', 'IIVI', 'IJT', 'IKNX', 'ILG', 'ILMN', 'ILPT', 'IMAX', 'IMDZ', 'IMGN', 'IMI', 'IMKTA', 'IMMP', 'IMMR', 'IMMU', 'IMMY', 'IMNP', 'IMOS', 'IMPV', 'IMRN', 'IMRNW', 'IMTE', 'INAP', 'INB', 'INBK', 'INBKL', 'INCY', 'INDB', 'INDU', 'INDUU', 'INDUW', 'INDY', 'INF', 'INFI', 'INFN', 'INFO', 'INFR', 'INFY', 'ING', 'INGN', 'INGR', 'INN', 'INNT', 'INN^C.CL', 'INN^D', 'INN^E', 'INO', 'INOD', 'INOV', 'INPX', 'INSE', 'INSG', 'INSI', 'INSM', 'INST', 'INSW', 'INSY', 'INT', 'INTC', 'INTG', 'INTL', 'INTU', 'INTX', 'INVA', 'INVE', 'INVH', 'INWK', 'INXN', 'IO', 'IONS', 'IOSP', 'IOTS', 'IOVA', 'IP', 'IPAR', 'IPAS', 'IPCC', 'IPCI', 'IPDN', 'IPG', 'IPGP', 'IPHI', 'IPHS', 'IPI', 'IPIC', 'IPKW', 'IPL^D', 'IPOA', 'IPOA.U', 'IPOA.WS', 'IPWR', 'IPXL', 'IQI', 'IQV', 'IR', 'IRBT', 'IRCP', 'IRDM', 'IRDMB', 'IRET', 'IRET^C', 'IRIX', 'IRL', 'IRM', 'IRMD', 'IROQ', 'IRR', 'IRS', 'IRT', 'IRTC', 'IRWD', 'ISBC', 'ISCA', 'ISD', 'ISF', 'ISG', 'ISHG', 'ISIG', 'ISNS', 'ISRG', 'ISRL', 'ISSC', 'ISTB', 'ISTR', 'IT', 'ITCB', 'ITCI', 'ITEQ', 'ITG', 'ITGR', 'ITI', 'ITIC', 'ITRI', 'ITRN', 'ITT', 'ITUB', 'ITUS', 'ITW', 'IUSB', 'IUSG', 'IUSV', 'IVAC', 'IVC', 'IVENC', 'IVFGC', 'IVFVC', 'IVH', 'IVR', 'IVR^A', 'IVR^B', 'IVR^C', 'IVTY', 'IVZ', 'IX', 'IXUS', 'IZEA', 'JACK', 'JAG', 'JAGX', 'JAKK', 'JASN', 'JASNW', 'JASO', 'JAX', 'JAZZ', 'JBGS', 'JBHT', 'JBK', 'JBL', 'JBLU', 'JBN', 'JBR', 'JBSS', 'JBT', 'JCAP', 'JCAP^B', 'JCE', 'JCI', 'JCO', 'JCOM', 'JCP', 'JCS', 'JCTCF', 'JD', 'JDD', 'JE', 'JEC', 'JELD', 'JEMD', 'JEQ', 'JE^A', 'JFR', 'JGH', 'JHA', 'JHB', 'JHD', 'JHG', 'JHI', 'JHS', 'JHX', 'JHY', 'JILL', 'JJSF', 'JKHY', 'JKI', 'JKS', 'JLL', 'JLS', 'JMBA', 'JMEI', 'JMF', 'JMLP', 'JMM', 'JMP', 'JMPB', 'JMPD', 'JMT', 'JMU', 'JNCE', 'JNJ', 'JNP', 'JNPR', 'JOBS', 'JOE', 'JOF', 'JONE', 'JOUT', 'JP', 'JPC', 'JPI', 'JPM', 'JPM.WS', 'JPM^A', 'JPM^B', 'JPM^E', 'JPM^F', 'JPM^G', 'JPM^H', 'JPS', 'JPT', 'JQC', 'JRI', 'JRJC', 'JRO', 'JRS', 'JRVR', 'JSD', 'JSM', 'JSMD', 'JSML', 'JSYN', 'JSYNR', 'JSYNU', 'JSYNW', 'JT', 'JTA', 'JTD', 'JTPY', 'JVA', 'JW.A', 'JW.B', 'JWN', 'JXSB', 'JYNT', 'K', 'KAAC', 'KAACU', 'KAACW', 'KAI', 'KALA', 'KALU', 'KALV', 'KAMN', 'KANG', 'KAP', 'KAR', 'KB', 'KBAL', 'KBH', 'KBLM', 'KBLMR', 'KBLMU', 'KBLMW', 'KBR', 'KBSF', 'KBWB', 'KBWD', 'KBWP', 'KBWR', 'KBWY', 'KCAP', 'KCAPL', 'KDMN', 'KE', 'KED', 'KEG', 'KELYA', 'KELYB', 'KEM', 'KEN', 'KEP', 'KEQU', 'KERX', 'KEX', 'KEY', 'KEYS', 'KEYW', 'KEY^I', 'KF', 'KFFB', 'KFRC', 'KFS', 'KFY', 'KGC', 'KGJI', 'KHC', 'KIDS', 'KIM', 'KIM^I', 'KIM^J', 'KIM^K', 'KIM^L', 'KIM^M', 'KIN', 'KINS', 'KIO', 'KIRK', 'KKR', 'KKR^A', 'KKR^B', 'KL', 'KLAC', 'KLIC', 'KLXI', 'KMB', 'KMDA', 'KMF', 'KMG', 'KMI', 'KMI^A', 'KMM', 'KMPA', 'KMPH', 'KMPR', 'KMT', 'KMX', 'KN', 'KND', 'KNDI', 'KNL', 'KNOP', 'KNSL', 'KNX', 'KO', 'KODK', 'KODK.WS', 'KODK.WS.A', 'KOF', 'KONA', 'KONE', 'KOOL', 'KOP', 'KOPN', 'KORS', 'KOS', 'KOSS', 'KPTI', 'KR', 'KRA', 'KRC', 'KREF', 'KRG', 'KRMA', 'KRNT', 'KRNY', 'KRO', 'KRP', 'KRYS', 'KS', 'KSM', 'KSS', 'KST', 'KSU', 'KSU^', 'KT', 'KTCC', 'KTEC', 'KTF', 'KTH', 'KTN', 'KTOS', 'KTOV', 'KTOVW', 'KTP', 'KTWO', 'KURA', 'KVHI', 'KW', 'KWEB', 'KWR', 'KYE', 'KYN', 'KYN^F', 'KYO', 'KZIA', 'L', 'LABL', 'LAC', 'LACQ', 'LACQU', 'LACQW', 'LAD', 'LADR', 'LAKE', 'LALT', 'LAMR', 'LANC', 'LAND', 'LANDP', 'LARK', 'LAUR', 'LAWS', 'LAYN', 'LAZ', 'LB', 'LBAI', 'LBC', 'LBCC', 'LBIX', 'LBRDA', 'LBRDK', 'LBRT', 'LBTYA', 'LBTYB', 'LBTYK', 'LC', 'LCA', 'LCAHU', 'LCAHW', 'LCI', 'LCII', 'LCM', 'LCNB', 'LCUT', 'LDF', 'LDL', 'LDOS', 'LDP', 'LDRI', 'LE', 'LEA', 'LECO', 'LEDS', 'LEE', 'LEG', 'LEGR', 'LEJU', 'LEN', 'LEN.B', 'LENS', 'LEO', 'LEXEA', 'LEXEB', 'LFC', 'LFGR', 'LFIN', 'LFUS', 'LFVN', 'LGC', 'LGC.U', 'LGC.WS', 'LGCY', 'LGCYO', 'LGCYP', 'LGF.A', 'LGF.B', 'LGI', 'LGIH', 'LGND', 'LH', 'LHC.U', 'LHCG', 'LHO', 'LHO^I', 'LHO^J', 'LIFE', 'LII', 'LILA', 'LILAK', 'LINC', 'LIND', 'LINDW', 'LINK', 'LINU', 'LION', 'LITB', 'LITE', 'LIVE', 'LIVN', 'LIVX', 'LJPC', 'LKFN', 'LKOR', 'LKQ', 'LKSD', 'LL', 'LLEX', 'LLIT', 'LLL', 'LLNW', 'LLY', 'LM', 'LMAT', 'LMB', 'LMBS', 'LMFA', 'LMFAW', 'LMHA', 'LMHB', 'LMNR', 'LMNX', 'LMRK', 'LMRKO', 'LMRKP', 'LMT', 'LN', 'LNC', 'LNC.WS', 'LNCE', 'LND', 'LNDC', 'LNGR', 'LNN', 'LNT', 'LNTH', 'LOAN', 'LOB', 'LOCO', 'LOGI', 'LOGM', 'LOMA', 'LONE', 'LOOP', 'LOPE', 'LOR', 'LORL', 'LOW', 'LOXO', 'LPCN', 'LPG', 'LPI', 'LPL', 'LPLA', 'LPNT', 'LPSN', 'LPT', 'LPTH', 'LPTX', 'LPX', 'LQ', 'LQDT', 'LRAD', 'LRCX', 'LRGE', 'LRN', 'LSBK', 'LSCC', 'LSI', 'LSTR', 'LSXMA', 'LSXMB', 'LSXMK', 'LTBR', 'LTC', 'LTM', 'LTN.U', 'LTRPA', 'LTRPB', 'LTRX', 'LTXB', 'LUB', 'LUK', 'LULU', 'LUNA', 'LUNG', 'LUV', 'LVHD', 'LVNTA', 'LVNTB', 'LVS', 'LW', 'LWAY', 'LX', 'LXFR', 'LXFT', 'LXP', 'LXP^C', 'LXRX', 'LXU', 'LYB', 'LYG', 'LYL', 'LYTS', 'LYV', 'LZB', 'M', 'MA', 'MAA', 'MAA^I', 'MAC', 'MACK', 'MACQ', 'MACQU', 'MACQW', 'MAGS', 'MAIN', 'MAMS', 'MAN', 'MANH', 'MANT', 'MANU', 'MAR', 'MARA', 'MARK', 'MARPS', 'MAS', 'MASI', 'MAT', 'MATR', 'MATW', 'MATX', 'MAV', 'MAXR', 'MAYS', 'MB', 'MBB', 'MBCN', 'MBFI', 'MBFIO', 'MBI', 'MBII', 'MBIN', 'MBIO', 'MBOT', 'MBRX', 'MBSD', 'MBT', 'MBTF', 'MBUU', 'MBVX', 'MBWM', 'MC', 'MCA', 'MCB', 'MCBC', 'MCC', 'MCD', 'MCEF', 'MCEP', 'MCFT', 'MCHI', 'MCHP', 'MCHX', 'MCI', 'MCK', 'MCN', 'MCO', 'MCR', 'MCRB', 'MCRI', 'MCRN', 'MCS', 'MCV', 'MCX', 'MCY', 'MD', 'MDB', 'MDC', 'MDCA', 'MDCO', 'MDGL', 'MDGS', 'MDIV', 'MDLQ', 'MDLX', 'MDLY', 'MDLZ', 'MDP', 'MDR', 'MDRX', 'MDSO', 'MDT', 'MDU', 'MDWD', 'MDXG', 'MED', 'MEDP', 'MEET', 'MEI', 'MEIP', 'MELI', 'MELR', 'MEN', 'MEOH', 'MERC', 'MER^K', 'MER^P', 'MESO', 'MET', 'METC', 'MET^A', 'MFA', 'MFA^B', 'MFC', 'MFCB', 'MFD', 'MFG', 'MFGP', 'MFIN', 'MFINL', 'MFL', 'MFM', 'MFNC', 'MFO', 'MFSF', 'MFT', 'MFV', 'MG', 'MGA', 'MGEE', 'MGEN', 'MGF', 'MGI', 'MGIC', 'MGLN', 'MGM', 'MGNX', 'MGP', 'MGPI', 'MGRC', 'MGU', 'MGYR', 'MHD', 'MHF', 'MHI', 'MHK', 'MHLA', 'MHLD', 'MHN', 'MHNC', 'MHO', 'MH^A', 'MH^C', 'MH^D', 'MIC', 'MICT', 'MICTW', 'MIDD', 'MIE', 'MIII', 'MIIIU', 'MIIIW', 'MIK', 'MILN', 'MIME', 'MIN', 'MIND', 'MINDP', 'MINI', 'MITK', 'MITL', 'MITT', 'MITT^A', 'MITT^B', 'MIXT', 'MIY', 'MKC', 'MKC.V', 'MKGI', 'MKL', 'MKSI', 'MKTX', 'MLAB', 'MLCO', 'MLHR', 'MLI', 'MLM', 'MLNT', 'MLNX', 'MLP', 'MLR', 'MLVF', 'MMAC', 'MMC', 'MMD', 'MMDM', 'MMDMR', 'MMDMU', 'MMDMW', 'MMI', 'MMLP', 'MMM', 'MMP', 'MMS', 'MMSI', 'MMT', 'MMU', 'MMYT', 'MN', 'MNDO', 'MNE', 'MNGA', 'MNK', 'MNKD', 'MNLO', 'MNOV', 'MNP', 'MNR', 'MNRO', 'MNR^C', 'MNST', 'MNTA', 'MNTX', 'MO', 'MOBL', 'MOD', 'MODN', 'MOFG', 'MOG.A', 'MOG.B', 'MOGLC', 'MOH', 'MOMO', 'MON', 'MORN', 'MOS', 'MOSC', 'MOSC.U', 'MOSC.WS', 'MOSY', 'MOTS', 'MOV', 'MOXC', 'MPA', 'MPAA', 'MPAC', 'MPACU', 'MPACW', 'MPB', 'MPC', 'MPCT', 'MPLX', 'MPO', 'MPV', 'MPVD', 'MPW', 'MPWR', 'MPX', 'MP^D', 'MQT', 'MQY', 'MRAM', 'MRBK', 'MRC', 'MRCC', 'MRCY', 'MRDN', 'MRDNW', 'MRIN', 'MRK', 'MRLN', 'MRNS', 'MRO', 'MRSN', 'MRT', 'MRTN', 'MRTX', 'MRUS', 'MRVL', 'MS', 'MSA', 'MSB', 'MSBF', 'MSBI', 'MSCA.CL', 'MSCC', 'MSCI', 'MSD', 'MSEX', 'MSF', 'MSFG', 'MSFT', 'MSG', 'MSG', 'MSGN', 'MSI', 'MSL', 'MSM', 'MSON', 'MSP', 'MSTR', 'MS^A', 'MS^E', 'MS^F', 'MS^G', 'MS^I', 'MS^K', 'MT', 'MTB', 'MTB.WS', 'MTBC', 'MTBCP', 'MTB^', 'MTB^C', 'MTCH', 'MTD', 'MTDR', 'MTEC', 'MTECU', 'MTECW', 'MTEM', 'MTEX', 'MTFB', 'MTFBW', 'MTG', 'MTGE', 'MTGEP', 'MTH', 'MTL', 'MTLS', 'MTL^', 'MTN', 'MTOR', 'MTP', 'MTR', 'MTRN', 'MTRX', 'MTSC', 'MTSI', 'MTSL', 'MTT', 'MTU', 'MTW', 'MTX', 'MTZ', 'MU', 'MUA', 'MUC', 'MUDSU', 'MUE', 'MUH', 'MUI', 'MUJ', 'MULE', 'MUR', 'MUS', 'MUSA', 'MUX', 'MVBF', 'MVC', 'MVCD', 'MVIS', 'MVO', 'MVT', 'MWA', 'MX', 'MXE', 'MXF', 'MXIM', 'MXL', 'MXWL', 'MYC', 'MYD', 'MYE', 'MYF', 'MYGN', 'MYI', 'MYJ', 'MYL', 'MYN', 'MYND', 'MYNDW', 'MYOK', 'MYOS', 'MYOV', 'MYRG', 'MYSZ', 'MZF', 'MZOR', 'NAC', 'NAD', 'NAII', 'NAKD', 'NAN', 'NANO', 'NAO', 'NAOV', 'NAP', 'NAT', 'NATH', 'NATI', 'NATR', 'NAUH', 'NAV', 'NAVG', 'NAVI', 'NAV^D', 'NAZ', 'NBB', 'NBD', 'NBEV', 'NBHC', 'NBIX', 'NBL', 'NBLX', 'NBN', 'NBR', 'NBRV', 'NBTB', 'NC', 'NCA', 'NCB', 'NCBS', 'NCI', 'NCLH', 'NCLH', 'NCMI', 'NCNA', 'NCOM', 'NCR', 'NCS', 'NCSM', 'NCTY', 'NCV', 'NCZ', 'NDAQ', 'NDLS', 'NDP', 'NDRA', 'NDRAW', 'NDRO', 'NDSN', 'NE', 'NEA', 'NEBU', 'NEBUU', 'NEBUW', 'NEE', 'NEE^I', 'NEE^J', 'NEE^K', 'NEE^Q', 'NEE^R', 'NEM', 'NEO', 'NEOG', 'NEON', 'NEOS', 'NEP', 'NEPT', 'NERV', 'NESR', 'NESRW', 'NETE', 'NETS', 'NEU', 'NEV', 'NEWA', 'NEWM', 'NEWR', 'NEWT', 'NEWTI', 'NEWTL', 'NEWTZ', 'NEXA', 'NEXT', 'NFBK', 'NFEC', 'NFG', 'NFJ', 'NFLX', 'NFX', 'NGG', 'NGHC', 'NGHCN', 'NGHCO', 'NGHCP', 'NGHCZ', 'NGL', 'NGLS^A', 'NGL^B', 'NGS', 'NGVC', 'NGVT', 'NH', 'NHA', 'NHF', 'NHI', 'NHLD', 'NHLDW', 'NHTC', 'NI', 'NICE', 'NICK', 'NID', 'NIE', 'NIHD', 'NIM', 'NINE', 'NIQ', 'NITE', 'NJR', 'NJV', 'NK', 'NKE', 'NKG', 'NKSH', 'NKTR', 'NKX', 'NL', 'NLNK', 'NLS', 'NLSN', 'NLST', 'NLY', 'NLY^C', 'NLY^D', 'NLY^F', 'NLY^G', 'NM', 'NMFC', 'NMI', 'NMIH', 'NMK^B', 'NMK^C', 'NMM', 'NMR', 'NMRD', 'NMRK', 'NMS', 'NMT', 'NMY', 'NMZ', 'NM^G', 'NM^H', 'NNA', 'NNBR', 'NNC', 'NNDM', 'NNI', 'NNN', 'NNN^E', 'NNN^F', 'NNY', 'NOA', 'NOAH', 'NOC', 'NODK', 'NOK', 'NOM', 'NOMD', 'NOV', 'NOVN', 'NOVT', 'NOW', 'NP', 'NPK', 'NPN', 'NPO', 'NPTN', 'NPV', 'NQ', 'NQP', 'NR', 'NRCIA', 'NRCIB', 'NRE', 'NRG', 'NRIM', 'NRK', 'NRP', 'NRT', 'NRZ', 'NS', 'NSA', 'NSA^A', 'NSC', 'NSEC', 'NSH', 'NSIT', 'NSL', 'NSM', 'NSP', 'NSS', 'NSSC', 'NSTG', 'NSYS', 'NS^A', 'NS^B', 'NS^C', 'NTAP', 'NTB', 'NTC', 'NTCT', 'NTEC', 'NTES', 'NTEST', 'NTEST.A', 'NTEST.B', 'NTEST.C', 'NTG', 'NTGR', 'NTIC', 'NTLA', 'NTNX', 'NTP', 'NTR', 'NTRA', 'NTRI', 'NTRP', 'NTRS', 'NTRSP', 'NTWK', 'NTX', 'NTZ', 'NUAN', 'NUE', 'NUM', 'NUO', 'NURO', 'NUROW', 'NUS', 'NUV', 'NUVA', 'NUW', 'NVAX', 'NVCN', 'NVCR', 'NVDA', 'NVEC', 'NVEE', 'NVFY', 'NVG', 'NVGS', 'NVIV', 'NVLN', 'NVMI', 'NVMM', 'NVO', 'NVR', 'NVRO', 'NVS', 'NVTA', 'NVTR', 'NVUS', 'NWBI', 'NWE', 'NWFL', 'NWHM', 'NWL', 'NWLI', 'NWN', 'NWPX', 'NWS', 'NWSA', 'NWY', 'NX', 'NXC', 'NXEO', 'NXEOU', 'NXEOW', 'NXJ', 'NXN', 'NXP', 'NXPI', 'NXQ', 'NXR', 'NXRT', 'NXST', 'NXTD', 'NXTDW', 'NXTM', 'NYCB', 'NYCB^A', 'NYCB^U', 'NYLD', 'NYLD.A', 'NYMT', 'NYMTN', 'NYMTO', 'NYMTP', 'NYMX', 'NYNY', 'NYRT', 'NYT', 'NYV', 'NZF', 'O', 'OA', 'OAK', 'OAKS', 'OAKS^A', 'OAS', 'OASM', 'OBAS', 'OBCI', 'OBE', 'OBLN', 'OBSV', 'OC', 'OCC', 'OCFC', 'OCIP', 'OCLR', 'OCN', 'OCSI', 'OCSL', 'OCSLL', 'OCUL', 'ODC', 'ODFL', 'ODP', 'ODT', 'OEC', 'OESX', 'OFC', 'OFED', 'OFG', 'OFG^A', 'OFG^B', 'OFG^D', 'OFIX', 'OFLX', 'OFS', 'OGE', 'OGS', 'OHAI', 'OHGI', 'OHI', 'OHRP', 'OI', 'OIA', 'OIBR.C', 'OII', 'OIIM', 'OIS', 'OKDCC', 'OKE', 'OKTA', 'OLBK', 'OLD', 'OLED', 'OLLI', 'OLN', 'OLP', 'OMAA', 'OMAB', 'OMAD', 'OMAD.U', 'OMAD.WS', 'OMAM', 'OMC', 'OMCL', 'OMED', 'OMER', 'OMEX', 'OMF', 'OMI', 'OMN', 'OMNT', 'OMP', 'ON', 'ONB', 'ONCE', 'ONCS', 'ONDK', 'ONEQ', 'ONS', 'ONSIW', 'ONSIZ', 'ONTX', 'ONTXW', 'ONVO', 'OOMA', 'OPB', 'OPGN', 'OPGNW', 'OPHC', 'OPHT', 'OPK', 'OPNT', 'OPOF', 'OPP', 'OPTN', 'OPTT', 'OPY', 'OR', 'ORA', 'ORAN', 'ORBC', 'ORBK', 'ORC', 'ORCL', 'OREX', 'ORG', 'ORI', 'ORIG', 'ORIT', 'ORLY', 'ORMP', 'ORN', 'ORPN', 'ORRF', 'OSB', 'OSBC', 'OSBCP', 'OSG', 'OSIS', 'OSK', 'OSLE', 'OSN', 'OSPR', 'OSPRU', 'OSPRW', 'OSS', 'OSTK', 'OSUR', 'OTEL', 'OTEX', 'OTIC', 'OTIV', 'OTTR', 'OTTW', 'OUT', 'OVAS', 'OVBC', 'OVID', 'OVLY', 'OXBR', 'OXBRW', 'OXFD', 'OXLC', 'OXLCM', 'OXLCO', 'OXM', 'OXY', 'OZM', 'OZRK', 'P', 'PAA', 'PAAS', 'PAC', 'PACB', 'PACW', 'PAG', 'PAGG', 'PAGP', 'PAGS', 'PAH', 'PAHC', 'PAI', 'PAM', 'PANL', 'PANW', 'PAR', 'PARR', 'PATI', 'PATK', 'PAVM', 'PAVMW', 'PAY', 'PAYC', 'PAYX', 'PB', 'PBA', 'PBB', 'PBBI', 'PBCT', 'PBCTP', 'PBF', 'PBFX', 'PBH', 'PBHC', 'PBI', 'PBIB', 'PBIP', 'PBI^B', 'PBPB', 'PBR', 'PBR.A', 'PBSK', 'PBT', 'PBYI', 'PCAR', 'PCF', 'PCG', 'PCH', 'PCI', 'PCK', 'PCM', 'PCMI', 'PCN', 'PCOM', 'PCQ', 'PCRX', 'PCSB', 'PCTI', 'PCTY', 'PCYG', 'PCYO', 'PDBC', 'PDCE', 'PDCO', 'PDEX', 'PDFS', 'PDI', 'PDLB', 'PDLI', 'PDM', 'PDP', 'PDS', 'PDT', 'PDVW', 'PE', 'PEB', 'PEBK', 'PEBO', 'PEB^C', 'PEB^D', 'PEG', 'PEGA', 'PEGI', 'PEI', 'PEIX', 'PEI^B', 'PEI^C', 'PEI^D', 'PEN', 'PENN', 'PEO', 'PEP', 'PER', 'PERI', 'PERY', 'PES', 'PESI', 'PETQ', 'PETS', 'PETX', 'PETZ', 'PEY', 'PEZ', 'PF', 'PFBC', 'PFBI', 'PFD', 'PFE', 'PFF', 'PFG', 'PFGC', 'PFH', 'PFI', 'PFIE', 'PFIN', 'PFIS', 'PFK', 'PFL', 'PFLT', 'PFM', 'PFMT', 'PFN', 'PFO', 'PFPT', 'PFS', 'PFSI', 'PFSW', 'PG', 'PGC', 'PGEM', 'PGH', 'PGJ', 'PGLC', 'PGNX', 'PGP', 'PGR', 'PGRE', 'PGTI', 'PGTI', 'PGZ', 'PH', 'PHD', 'PHG', 'PHH', 'PHI', 'PHII', 'PHIIK', 'PHK', 'PHM', 'PHO', 'PHT', 'PHX', 'PI', 'PICO', 'PID', 'PIE', 'PIH', 'PII', 'PIM', 'PINC', 'PIO', 'PIR', 'PIRS', 'PIXY', 'PIY', 'PIZ', 'PJC', 'PJH', 'PJT', 'PK', 'PKBK', 'PKD', 'PKE', 'PKG', 'PKI', 'PKO', 'PKOH', 'PKW', 'PKX', 'PLAB', 'PLAY', 'PLBC', 'PLCE', 'PLD', 'PLNT', 'PLOW', 'PLPC', 'PLSE', 'PLT', 'PLUG', 'PLUS', 'PLW', 'PLXP', 'PLXS', 'PLYA', 'PM', 'PMBC', 'PMD', 'PME', 'PMF', 'PML', 'PMM', 'PMO', 'PMOM', 'PMPT', 'PMT', 'PMTS', 'PMT^A', 'PMT^B', 'PMX', 'PNBK', 'PNC', 'PNC.WS', 'PNC^P', 'PNC^Q', 'PNF', 'PNFP', 'PNI', 'PNK', 'PNM', 'PNNT', 'PNQI', 'PNR', 'PNRG', 'PNTR', 'PNW', 'PODD', 'POL', 'POLA', 'POOL', 'POPE', 'POR', 'POST', 'POWI', 'POWL', 'PPBI', 'PPC', 'PPDF', 'PPG', 'PPH', 'PPIH', 'PPL', 'PPR', 'PPSI', 'PPT', 'PPX', 'PQ', 'PQG', 'PRA', 'PRAA', 'PRAH', 'PRAN', 'PRCP', 'PRE^F', 'PRE^G', 'PRE^H', 'PRE^I', 'PRFT', 'PRFZ', 'PRGO', 'PRGS', 'PRGX', 'PRH', 'PRI', 'PRIM', 'PRKR', 'PRLB', 'PRMW', 'PRN', 'PRO', 'PROV', 'PRPH', 'PRPL', 'PRPLW', 'PRPO', 'PRQR', 'PRSC', 'PRSS', 'PRTA', 'PRTK', 'PRTO', 'PRTS', 'PRTY', 'PRU', 'PSA', 'PSAU', 'PSA^A', 'PSA^B', 'PSA^C', 'PSA^D', 'PSA^E', 'PSA^F', 'PSA^G', 'PSA^U', 'PSA^V', 'PSA^W', 'PSA^X', 'PSA^Y', 'PSA^Z', 'PSB', 'PSB^U', 'PSB^V', 'PSB^W', 'PSB^X', 'PSB^Y', 'PSC', 'PSCC', 'PSCD', 'PSCE', 'PSCF', 'PSCH', 'PSCI', 'PSCM', 'PSCT', 'PSCU', 'PSDO', 'PSDV', 'PSEC', 'PSET', 'PSF', 'PSL', 'PSMT', 'PSO', 'PSTG', 'PSTI', 'PSX', 'PSXP', 'PTC', 'PTCT', 'PTEN', 'PTF', 'PTGX', 'PTH', 'PTI', 'PTIE', 'PTLA', 'PTNR', 'PTR', 'PTSI', 'PTX', 'PTY', 'PUB', 'PUI', 'PUK', 'PUK^', 'PUK^A', 'PULM', 'PUMP', 'PVAC', 'PVAL', 'PVBC', 'PVG', 'PVH', 'PWOD', 'PWR', 'PX', 'PXD', 'PXI', 'PXLW', 'PXS', 'PXUS', 'PY', 'PYDS', 'PYN', 'PYPL', 'PYS', 'PYT', 'PYZ', 'PZC', 'PZE', 'PZN', 'PZZA', 'QABA', 'QADA', 'QADB', 'QAT', 'QBAK', 'QCLN', 'QCOM', 'QCP', 'QCRH', 'QD', 'QDEL', 'QEP', 'QES', 'QGEN', 'QGEN', 'QHC', 'QINC', 'QIWI', 'QLC', 'QLYS', 'QNST', 'QQEW', 'QQQ', 'QQQC', 'QQQX', 'QQXT', 'QRHC', 'QRVO', 'QSII', 'QSR', 'QTEC', 'QTM', 'QTNA', 'QTNT', 'QTRH', 'QTRX', 'QTS', 'QTWO', 'QUAD', 'QUIK', 'QUMU', 'QUOT', 'QURE', 'QVCA', 'QVCB', 'QYLD', 'R', 'RA', 'RACE', 'RAD', 'RADA', 'RAIL', 'RAND', 'RARE', 'RARX', 'RAS', 'RAS^A', 'RAS^B', 'RAS^C', 'RAVE', 'RAVN', 'RBA', 'RBB', 'RBBN', 'RBC', 'RBCAA', 'RBCN', 'RBNC', 'RBS', 'RBS^S', 'RCI', 'RCII', 'RCKT', 'RCKY', 'RCL', 'RCM', 'RCMT', 'RCON', 'RCS', 'RDC', 'RDCM', 'RDFN', 'RDHL', 'RDI', 'RDIB', 'RDN', 'RDNT', 'RDS.A', 'RDS.B', 'RDUS', 'RDVY', 'RDWR', 'RDY', 'RE', 'RECN', 'REDU', 'REFR', 'REG', 'REGI', 'REGN', 'REIS', 'RELL', 'RELV', 'RELX', 'REN', 'RENN', 'RENX', 'REPH', 'RES', 'RESI', 'RESN', 'RETA', 'RETO', 'REV', 'REVG', 'REX', 'REXR', 'REXR^A', 'REXR^B', 'REXX', 'RF', 'RFAP', 'RFDI', 'RFEM', 'RFEU', 'RFI', 'RFIL', 'RFP', 'RFT', 'RFTA', 'RF^A', 'RF^B', 'RGA', 'RGCO', 'RGEN', 'RGLD', 'RGLS', 'RGNX', 'RGR', 'RGS', 'RGSE', 'RGT', 'RH', 'RHI', 'RHP', 'RHT', 'RIBT', 'RIBTW', 'RICK', 'RIG', 'RIGL', 'RILY', 'RILYG', 'RILYL', 'RILYZ', 'RING', 'RIO', 'RIOT', 'RIV', 'RJF', 'RKDA', 'RL', 'RLGY', 'RLH', 'RLI', 'RLJ', 'RLJE', 'RLJ^A', 'RM', 'RMAX', 'RMBL', 'RMBS', 'RMCF', 'RMD', 'RMGN', 'RMNI', 'RMP', 'RMPL^', 'RMR', 'RMT', 'RMTI', 'RNDB', 'RNDM', 'RNDV', 'RNEM', 'RNET', 'RNG', 'RNGR', 'RNLC', 'RNMC', 'RNP', 'RNR', 'RNR^C', 'RNR^E', 'RNSC', 'RNST', 'RNWK', 'ROBO', 'ROBT', 'ROCK', 'ROG', 'ROIC', 'ROK', 'ROKU', 'ROL', 'ROLL', 'ROP', 'ROSE', 'ROSEU', 'ROSEW', 'ROSG', 'ROST', 'ROYT', 'RP', 'RPAI', 'RPD', 'RPIBC', 'RPM', 'RPT', 'RPT^D', 'RPXC', 'RQI', 'RRC', 'RRD', 'RRD', 'RRGB', 'RRR', 'RRTS', 'RS', 'RSG', 'RSLS', 'RSO', 'RSO^B.CL', 'RSO^C', 'RSPP', 'RST', 'RSYS', 'RTEC', 'RTIX', 'RTN', 'RTRX', 'RTTR', 'RUBI', 'RUN', 'RUSHA', 'RUSHB', 'RUTH', 'RVEN', 'RVLT', 'RVNC', 'RVSB', 'RVT', 'RWGE', 'RWGE.U', 'RWGE.WS', 'RWLK', 'RWT', 'RXII', 'RXIIW', 'RXN', 'RXN^A', 'RY', 'RYAAY', 'RYAM', 'RYAM^A', 'RYB', 'RYI', 'RYN', 'RYTM', 'RY^T', 'RZA', 'RZB', 'S', 'SA', 'SAB', 'SABR', 'SAEX', 'SAFE', 'SAFM', 'SAFT', 'SAGE', 'SAH', 'SAIA', 'SAIC', 'SAIL', 'SAL', 'SALM', 'SALT', 'SAM', 'SAMG', 'SAN', 'SANM', 'SANW', 'SAN^A', 'SAN^B', 'SAN^C', 'SAN^I', 'SAP', 'SAR', 'SASR', 'SATS', 'SAUC', 'SAVE', 'SAVE', 'SB', 'SBAC', 'SBBC', 'SBBP', 'SBBX', 'SBCF', 'SBFG', 'SBFGP', 'SBGI', 'SBGL', 'SBH', 'SBI', 'SBLK', 'SBLKZ', 'SBNA', 'SBNY', 'SBNYW', 'SBOT', 'SBOW', 'SBPH', 'SBR', 'SBRA', 'SBRAP', 'SBS', 'SBSI', 'SBT', 'SBUX', 'SB^C', 'SB^D', 'SC', 'SCA', 'SCAC', 'SCACU', 'SCACW', 'SCCO', 'SCD', 'SCE^G', 'SCE^H', 'SCE^J', 'SCE^K', 'SCE^L', 'SCG', 'SCHL', 'SCHN', 'SCHW', 'SCHW^C', 'SCHW^D', 'SCI', 'SCKT', 'SCL', 'SCM', 'SCON', 'SCPH', 'SCS', 'SCSC', 'SCVL', 'SCWX', 'SCX', 'SCYX', 'SCZ', 'SD', 'SDLP', 'SDR', 'SDRL', 'SDT', 'SDVY', 'SE', 'SEAC', 'SEAS', 'SECO', 'SEDG', 'SEE', 'SEED', 'SEIC', 'SEII', 'SELB', 'SELF', 'SEM', 'SEMG', 'SEND', 'SENEA', 'SENEB', 'SEP', 'SERV', 'SES', 'SF', 'SFB', 'SFBC', 'SFBS', 'SFE', 'SFIX', 'SFL', 'SFLY', 'SFM', 'SFNC', 'SFS', 'SFST', 'SFUN', 'SF^A', 'SGBX', 'SGC', 'SGEN', 'SGF', 'SGH', 'SGLB', 'SGLBW', 'SGMA', 'SGMO', 'SGMS', 'SGOC', 'SGQI', 'SGRP', 'SGRY', 'SGU', 'SGY', 'SGYP', 'SGZA', 'SHAK', 'SHBI', 'SHEN', 'SHG', 'SHI', 'SHIP', 'SHIPW', 'SHLD', 'SHLDW', 'SHLM', 'SHLO', 'SHLX', 'SHO', 'SHOO', 'SHOP', 'SHOS', 'SHO^E', 'SHO^F', 'SHPG', 'SHSP', 'SHV', 'SHW', 'SHY', 'SID', 'SIEB', 'SIEN', 'SIFI', 'SIFY', 'SIG', 'SIGI', 'SIGM', 'SILC', 'SIMO', 'SINA', 'SINO', 'SIR', 'SIRI', 'SITE', 'SITO', 'SIVB', 'SIX', 'SJI', 'SJM', 'SJR', 'SJT', 'SJW', 'SKIS', 'SKM', 'SKOR', 'SKT', 'SKX', 'SKYS', 'SKYW', 'SKYY', 'SLAB', 'SLB', 'SLCA', 'SLCT', 'SLD', 'SLDA', 'SLDB', 'SLF', 'SLG', 'SLGL', 'SLGN', 'SLG^I', 'SLIM', 'SLM', 'SLMBP', 'SLNO', 'SLNOW', 'SLP', 'SLQD', 'SLRC', 'SLS', 'SLTB', 'SLVO', 'SM', 'SMBC', 'SMBK', 'SMCI', 'SMCP', 'SMED', 'SMFG', 'SMG', 'SMHI', 'SMI', 'SMIT', 'SMLP', 'SMM', 'SMMF', 'SMMT', 'SMP', 'SMPL', 'SMPLW', 'SMRT', 'SMSI', 'SMTC', 'SMTX', 'SN', 'SNA', 'SNAP', 'SNBR', 'SNCR', 'SND', 'SNDE', 'SNDR', 'SNDX', 'SNE', 'SNES', 'SNFCA', 'SNGX', 'SNGXW', 'SNH', 'SNHNI', 'SNHNL', 'SNHY', 'SNLN', 'SNMX', 'SNN', 'SNNA', 'SNOA', 'SNOAW', 'SNP', 'SNPS', 'SNR', 'SNSR', 'SNSS', 'SNV', 'SNV^C', 'SNX', 'SNY', 'SO', 'SOCL', 'SODA', 'SOFO', 'SOGO', 'SOHO', 'SOHOB', 'SOHOK', 'SOHOO', 'SOHU', 'SOI', 'SOJA', 'SOJB', 'SOJC', 'SOL', 'SON', 'SONA', 'SONC', 'SOR', 'SORL', 'SOV^C', 'SOXX', 'SP', 'SPA', 'SPAR', 'SPB', 'SPCB', 'SPE', 'SPEX', 'SPE^B', 'SPG', 'SPGI', 'SPG^J', 'SPH', 'SPHS', 'SPI', 'SPIL', 'SPKE', 'SPKEP', 'SPLK', 'SPLP', 'SPLP^A', 'SPN', 'SPNE', 'SPNS', 'SPOK', 'SPPI', 'SPR', 'SPRO', 'SPRT', 'SPSC', 'SPTN', 'SPWH', 'SPWR', 'SPXC', 'SPXX', 'SQ', 'SQBG', 'SQLV', 'SQM', 'SQNS', 'SQQQ', 'SQZZ', 'SR', 'SRAX', 'SRC', 'SRCE', 'SRCL', 'SRCLP', 'SRC^A', 'SRDX', 'SRE', 'SRET', 'SREV', 'SRE^A', 'SRF', 'SRG', 'SRG^A', 'SRI', 'SRLP', 'SRNE', 'SRPT', 'SRRA', 'SRT', 'SRTS', 'SRTSW', 'SRV', 'SSB', 'SSBI', 'SSC', 'SSD', 'SSFN', 'SSI', 'SSKN', 'SSL', 'SSLJ', 'SSNC', 'SSNT', 'SSP', 'SSRM', 'SSTI', 'SSTK', 'SSW', 'SSWA', 'SSWN', 'SSW^D', 'SSW^E', 'SSW^G', 'SSW^H', 'SSYS', 'ST', 'STAA', 'STAF', 'STAG', 'STAG^B', 'STAG^C', 'STAR', 'STAR^D', 'STAR^G', 'STAR^I', 'STAY', 'STB', 'STBA', 'STBZ', 'STC', 'STCN', 'STDY', 'STE', 'STFC', 'STI', 'STI.WS.A', 'STI.WS.B', 'STI^A', 'STI^E.CL', 'STK', 'STKL', 'STKS', 'STL', 'STLD', 'STLR', 'STLRU', 'STLRW', 'STLY', 'STL^A', 'STM', 'STML', 'STMP', 'STN', 'STNG', 'STNL', 'STNLU', 'STNLW', 'STO', 'STON', 'STOR', 'STPP', 'STRA', 'STRL', 'STRM', 'STRS', 'STRT', 'STT', 'STT^C', 'STT^D', 'STT^E', 'STT^G', 'STWD', 'STX', 'STZ', 'STZ.B', 'SU', 'SUI', 'SUM', 'SUMR', 'SUN', 'SUNS', 'SUNW', 'SUP', 'SUPN', 'SUPV', 'SUSB', 'SUSC', 'SVA', 'SVBI', 'SVRA', 'SVU', 'SVVC', 'SWCH', 'SWIN', 'SWIR', 'SWJ', 'SWK', 'SWKS', 'SWM', 'SWN', 'SWP', 'SWX', 'SWZ', 'SXC', 'SXCP', 'SXE', 'SXI', 'SXT', 'SYBT', 'SYBX', 'SYF', 'SYK', 'SYKE', 'SYMC', 'SYNA', 'SYNC', 'SYNH', 'SYNL', 'SYNT', 'SYPR', 'SYRS', 'SYX', 'SYY', 'SZC', 'SZC~', 'T', 'TA', 'TAC', 'TACO', 'TACOW', 'TACT', 'TAHO', 'TAIT', 'TAL', 'TANH', 'TANNI', 'TANNL', 'TANNZ', 'TAP', 'TAP.A', 'TAPR', 'TARO', 'TAST', 'TATT', 'TAX', 'TAYD', 'TBB', 'TBBK', 'TBI', 'TBK', 'TBNK', 'TBPH', 'TCAP', 'TCBI', 'TCBIL', 'TCBIP', 'TCBIW', 'TCBK', 'TCCA', 'TCCB', 'TCCO', 'TCF', 'TCF.WS', 'TCFC', 'TCF^D', 'TCGP', 'TCI', 'TCMD', 'TCO', 'TCON', 'TCO^J', 'TCO^K', 'TCP', 'TCPC', 'TCRD', 'TCRX', 'TCRZ', 'TCS', 'TCX', 'TD', 'TDA', 'TDC', 'TDE', 'TDF', 'TDG', 'TDI', 'TDIV', 'TDJ', 'TDOC', 'TDS', 'TDW', 'TDW.WS.A', 'TDW.WS.B', 'TDY', 'TEAM', 'TECD', 'TECH', 'TECK', 'TEDU', 'TEF', 'TEGP', 'TEI', 'TEL', 'TELL', 'TEN', 'TENX', 'TEO', 'TEP', 'TER', 'TERP', 'TESS', 'TEVA', 'TEX', 'TFSL', 'TFX', 'TG', 'TGA', 'TGEN', 'TGH', 'TGI', 'TGLS', 'TGNA', 'TGP', 'TGP^A', 'TGP^B', 'TGS', 'TGT', 'TGTX', 'THC', 'THFF', 'THG', 'THGA', 'THO', 'THQ', 'THR', 'THRM', 'THS', 'THST', 'THW', 'TI', 'TI.A', 'TICC', 'TICCL', 'TIER', 'TIF', 'TIG', 'TIL', 'TILE', 'TIPT', 'TISA', 'TISI', 'TITN', 'TIVO', 'TJX', 'TK', 'TKC', 'TKR', 'TLF', 'TLGT', 'TLI', 'TLK', 'TLND', 'TLP', 'TLRA', 'TLRD', 'TLT', 'TLYS', 'TM', 'TMHC', 'TMK', 'TMK^C', 'TMO', 'TMSR', 'TMSRW', 'TMST', 'TMUS', 'TNAV', 'TNC', 'TNDM', 'TNET', 'TNH', 'TNK', 'TNP', 'TNP^B', 'TNP^C', 'TNP^D', 'TNP^E', 'TNTR', 'TNXP', 'TOCA', 'TOL', 'TOO', 'TOO^A', 'TOO^B', 'TOO^E', 'TOPS', 'TORC', 'TOT', 'TOUR', 'TOWN', 'TOWR', 'TPB', 'TPC', 'TPGE', 'TPGE.U', 'TPGE.WS', 'TPGH', 'TPGH.U', 'TPGH.WS', 'TPH', 'TPIC', 'TPIV', 'TPL', 'TPR', 'TPRE', 'TPVG', 'TPVY', 'TPX', 'TPZ', 'TQQQ', 'TR', 'TRC', 'TRCB', 'TRCH', 'TRCO', 'TREC', 'TREE', 'TREX', 'TRGP', 'TRHC', 'TRI', 'TRIB', 'TRIL', 'TRIP', 'TRK', 'TRMB', 'TRMD', 'TRMK', 'TRMT', 'TRN', 'TRNC', 'TRNO', 'TRNS', 'TROV', 'TROW', 'TROX', 'TRP', 'TRPX', 'TRQ', 'TRS', 'TRST', 'TRTN', 'TRTX', 'TRU', 'TRUE', 'TRUP', 'TRV', 'TRVG', 'TRVN', 'TS', 'TSBK', 'TSC', 'TSCO', 'TSE', 'TSEM', 'TSG', 'TSI', 'TSLA', 'TSLF', 'TSLX', 'TSM', 'TSN', 'TSQ', 'TSRI', 'TSRO', 'TSS', 'TST', 'TSU', 'TTC', 'TTD', 'TTEC', 'TTEK', 'TTGT', 'TTI', 'TTM', 'TTMI', 'TTNP', 'TTOO', 'TTP', 'TTPH', 'TTS', 'TTWO', 'TU', 'TUES', 'TUP', 'TUR', 'TURN', 'TUSA', 'TUSK', 'TV', 'TVC', 'TVE', 'TVIX', 'TVIZ', 'TVPT', 'TVTY', 'TWI', 'TWIN', 'TWLO', 'TWMC', 'TWN', 'TWNK', 'TWNKW', 'TWO', 'TWOU', 'TWO^A', 'TWO^B', 'TWO^C', 'TWTR', 'TWX', 'TX', 'TXMD', 'TXN', 'TXRH', 'TXT', 'TY', 'TYG', 'TYHT', 'TYL', 'TYME', 'TYPE', 'TY^', 'TZOO', 'UA', 'UAA', 'UAE', 'UAL', 'UAN', 'UBA', 'UBCP', 'UBFO', 'UBIO', 'UBNK', 'UBNT', 'UBOH', 'UBP', 'UBP^G', 'UBP^H', 'UBS', 'UBSH', 'UBSI', 'UCBA', 'UCBI', 'UCFC', 'UCTT', 'UDBI', 'UDR', 'UE', 'UEIC', 'UEPS', 'UFCS', 'UFI', 'UFPI', 'UFPT', 'UFS', 'UG', 'UGI', 'UGLD', 'UGP', 'UHAL', 'UHS', 'UHT', 'UIHC', 'UIS', 'UL', 'ULBI', 'ULH', 'ULTA', 'ULTI', 'UMBF', 'UMC', 'UMH', 'UMH^B', 'UMH^C', 'UMH^D', 'UMPQ', 'UN', 'UNAM', 'UNB', 'UNF', 'UNFI', 'UNH', 'UNIT', 'UNM', 'UNP', 'UNT', 'UNTY', 'UNVR', 'UONE', 'UONEK', 'UPL', 'UPLD', 'UPS', 'URBN', 'URGN', 'URI', 'USA', 'USAC', 'USAK', 'USAP', 'USAT', 'USATP', 'USAU', 'USB', 'USB^A', 'USB^H', 'USB^M', 'USB^O', 'USCR', 'USDP', 'USEG', 'USFD', 'USG', 'USLB', 'USLM', 'USLV', 'USM', 'USMC', 'USNA', 'USOI', 'USPH', 'UTF', 'UTHR', 'UTI', 'UTL', 'UTMD', 'UTSI', 'UTX', 'UVE', 'UVSP', 'UVV', 'UZA', 'UZB', 'UZC', 'V', 'VAC', 'VALE', 'VALU', 'VALX', 'VAR', 'VBF', 'VBFC', 'VBIV', 'VBLT', 'VBND', 'VBTX', 'VC', 'VCEL', 'VCIT', 'VCLT', 'VCO', 'VCRA', 'VCSH', 'VCTR', 'VCV', 'VCYT', 'VDSI', 'VDTH', 'VEAC', 'VEACU', 'VEACW', 'VEC', 'VECO', 'VEDL', 'VEEV', 'VEON', 'VER', 'VERI', 'VERU', 'VER^F', 'VET', 'VFC', 'VG', 'VGI', 'VGIT', 'VGLT', 'VGM', 'VGR', 'VGSH', 'VHI', 'VIA', 'VIAB', 'VIAV', 'VICI', 'VICL', 'VICR', 'VIDI', 'VIGI', 'VIIX', 'VIIZ', 'VIPS', 'VIRC', 'VIRT', 'VIV', 'VIVE', 'VIVO', 'VJET', 'VKQ', 'VKTX', 'VKTXW', 'VLGEA', 'VLO', 'VLP', 'VLRS', 'VLRX', 'VLT', 'VLY', 'VLY.WS', 'VLY^A', 'VLY^B', 'VMBS', 'VMC', 'VMI', 'VMO', 'VMW', 'VNCE', 'VNDA', 'VNET', 'VNO', 'VNOM', 'VNO^K', 'VNO^L', 'VNO^M', 'VNQI', 'VNTR', 'VOC', 'VOD', 'VONE', 'VONG', 'VONV', 'VOXX', 'VOYA', 'VPG', 'VPV', 'VR', 'VRA', 'VRAY', 'VREX', 'VRIG', 'VRML', 'VRNA', 'VRNS', 'VRNT', 'VRS', 'VRSK', 'VRSN', 'VRTS', 'VRTSP', 'VRTU', 'VRTV', 'VRTX', 'VRX', 'VR^A', 'VR^B', 'VSAR', 'VSAT', 'VSDA', 'VSEC', 'VSH', 'VSI', 'VSLR', 'VSM', 'VSMV', 'VST', 'VSTM', 'VSTO', 'VTA', 'VTC', 'VTGN', 'VTHR', 'VTIP', 'VTL', 'VTN', 'VTNR', 'VTR', 'VTRB', 'VTVT', 'VTWG', 'VTWO', 'VTWV', 'VUSE', 'VUZI', 'VVC', 'VVI', 'VVPR', 'VVR', 'VVUS', 'VVV', 'VWOB', 'VXRT', 'VXUS', 'VYGR', 'VYMI', 'VZ', 'VZA', 'W', 'WAAS', 'WAB', 'WABC', 'WAFD', 'WAFDW', 'WAGE', 'WAIR', 'WAL', 'WALA', 'WASH', 'WAT', 'WATT', 'WB', 'WBA', 'WBAI', 'WBC', 'WBK', 'WBS', 'WBS^F', 'WBT', 'WCC', 'WCFB', 'WCG', 'WCN', 'WD', 'WDAY', 'WDC', 'WDFC', 'WDR', 'WEA', 'WEB', 'WEBK', 'WEC', 'WELL', 'WELL^I', 'WEN', 'WERN', 'WES', 'WETF', 'WEX', 'WEYS', 'WF', 'WFC', 'WFC.WS', 'WFC^J', 'WFC^L', 'WFC^N', 'WFC^O', 'WFC^P', 'WFC^Q', 'WFC^R', 'WFC^T', 'WFC^V', 'WFC^W', 'WFC^X', 'WFC^Y', 'WFE^A', 'WFT', 'WG', 'WGL', 'WGO', 'WGP', 'WHD', 'WHF', 'WHFBL', 'WHG', 'WHLM', 'WHLR', 'WHLRD', 'WHLRP', 'WHLRW', 'WHR', 'WIA', 'WIFI', 'WILC', 'WIN', 'WINA', 'WING', 'WINS', 'WIRE', 'WIT', 'WIW', 'WIX', 'WK', 'WKHS', 'WLB', 'WLDN', 'WLFC', 'WLH', 'WLK', 'WLKP', 'WLL', 'WLTW', 'WM', 'WMB', 'WMC', 'WMGI', 'WMGIZ', 'WMIH', 'WMK', 'WMLP', 'WMS', 'WMT', 'WNC', 'WNEB', 'WNS', 'WOOD', 'WOR', 'WOW', 'WP', 'WPC', 'WPG', 'WPG^H', 'WPG^I', 'WPM', 'WPP', 'WPRT', 'WPX', 'WPXP', 'WPZ', 'WR', 'WRB', 'WRB^B', 'WRB^C', 'WRB^D', 'WRD', 'WRE', 'WRI', 'WRK', 'WRLD', 'WRLS', 'WRLSR', 'WRLSU', 'WRLSW', 'WSBC', 'WSBF', 'WSC', 'WSCI', 'WSCWW', 'WSFS', 'WSM', 'WSO', 'WSO.B', 'WSR', 'WST', 'WSTG', 'WSTL', 'WTBA', 'WTFC', 'WTFCM', 'WTFCW', 'WTI', 'WTM', 'WTR', 'WTS', 'WTTR', 'WTW', 'WU', 'WUBA', 'WVE', 'WVFC', 'WVVI', 'WVVIP', 'WWD', 'WWE', 'WWR', 'WWW', 'WY', 'WYN', 'WYNN', 'X', 'XBIO', 'XBIT', 'XCRA', 'XEC', 'XEL', 'XELA', 'XELB', 'XENE', 'XENT', 'XFLT', 'XGTI', 'XGTIW', 'XHR', 'XIN', 'XL', 'XLNX', 'XLRN', 'XNCR', 'XNET', 'XOG', 'XOM', 'XOMA', 'XON', 'XONE', 'XOXO', 'XPER', 'XPLR', 'XPO', 'XRAY', 'XRF', 'XRM', 'XRX', 'XSPA', 'XT', 'XTLB', 'XYL', 'Y', 'YDIV', 'YECO', 'YELP', 'YEXT', 'YGE', 'YGYI', 'YIN', 'YLCO', 'YLDE', 'YNDX', 'YOGA', 'YORW', 'YPF', 'YRCW', 'YRD', 'YRIV', 'YTEN', 'YTRA', 'YUM', 'YUMC', 'YY', 'Z', 'ZAGG', 'ZAIS', 'ZAYO', 'ZBH', 'ZBIO', 'ZBK', 'ZBRA', 'ZB^A', 'ZB^G', 'ZB^H', 'ZEAL', 'ZEN', 'ZEUS', 'ZF', 'ZFGN', 'ZG', 'ZGNX', 'ZION', 'ZIONW', 'ZIONZ', 'ZIOP', 'ZIV', 'ZIXI', 'ZKIN', 'ZLAB', 'ZN', 'ZNGA', 'ZNH', 'ZNWAA', 'ZOES', 'ZSAN', 'ZTO', 'ZTR', 'ZTS', 'ZUMZ', 'ZX', 'ZYME', 'ZYNE'))


# set date and calendar params with error detection
import datetime

today = datetime.date.today()
before = today - datetime.timedelta(days=730)
start_date = st.sidebar.date_input('Start date', before) 
end_date = st.sidebar.date_input('End date', today)
if start_date < end_date:
    st.sidebar.success('Start date: `%s`\n\nEnd date:`%s`' % (start_date, end_date))
else:
    st.sidebar.error('Error: End date must fall after start date.')
# add creator information
st.sidebar.caption('Presented by Jeff, Thomas and Ray :hotsprings:')

##############
# Stock data #
##############
# setup of the main body window
# create dataframe to get data from yahoo finance
df = yf.download(option,start= start_date,end= end_date, progress=False)
st.title(option)
st.caption("note: previous day's closing data")
st.dataframe(df.tail(1))
# display buy and sell recommendations
st.markdown('##### Buy & Sell Recommendations')
tickerData = yf.Ticker(option)
tickerData.recommendations
# add a progress bar
progress_bar = st.progress(0)
st.subheader('_Technical Indicators_')
st.markdown('##### Bollinger Bands®')
indicator_bb = BollingerBands(df['Close'])
# create the bollinger bands df
bb = df
bb['Bollinger_Band_High'] = indicator_bb.bollinger_hband()
bb['Bollinger_Band_Low'] = indicator_bb.bollinger_lband()
bb = bb[['Close','Bollinger_Band_High','Bollinger_Band_Low']]
# create the Moving Average Convergence Divergence (MACD) df
macd = MACD(df['Close']).macd()
# create the Relative Strength Index (RSI) df
rsi = RSIIndicator(df['Close']).rsi()
# create the True Strength Index (TSI) df
tsi = TSIIndicator(df['Close']).tsi()
# create the Rate of Change (ROC) df
roc = ROCIndicator(df['Close']).roc()

###################
# Set up main app #
###################
# plot the bollinger bands line chart
st.line_chart(bb)
# set the chickable button url detail
url = 'https://www.investopedia.com/articles/technical/102201.asp'
# create a button
if st.button('Bollinger Bands® FAQs'):
    webbrowser.open_new_tab(url)
# add a seperator line
progress_bar = st.progress(0)
# create a 2 column view
col1, col2 = st.columns(2)
# plot the Moving Average Convergence Divergence (MACD) line chart
with col1: 
    st.markdown('##### Moving Average Convergence Divergence (MACD)')
    st.area_chart(macd)
    # set the chickable button url detail
    url = 'https://www.investopedia.com/terms/m/macd.asp'
    # create a button
    if st.button('MACD FAQs'):
        webbrowser.open_new_tab(url)
# plot the Relative Strength Index (RSI) line chart     
with col2:
    st.markdown("##### Relative Strength Index (RSI)")
    st.line_chart(rsi)
    st.markdown(" ")
    # set the chickable button url detail
    url = 'https://www.investopedia.com/terms/r/rsi.asp'
    # create a button
    if st.button('Relative Strength Index (RSI) FAQs'):
        webbrowser.open_new_tab(url)
# add a seperator line
progress_bar = st.progress(0)
# create a 2 column view     
col1, col2 = st.columns(2)
# plot the True Strength Index (TSI) line chart
with col1: 
    st.markdown("##### True Strength Index (TSI)")
    st.line_chart(tsi)
    # set the chickable button url detail
    url = 'https://www.investopedia.com/terms/t/tsi.asp'
    # create a button
    if st.button('True Strength Index (TSI) FAQs'):
        webbrowser.open_new_tab(url)
# plot the Rate of Change (ROC) line chart
with col2:
    st.markdown("##### Rate of Change (ROC)")
    st.line_chart(roc)
    # set the chickable button url detail
    url = 'https://www.investopedia.com/terms/r/rateofchange.asp'
    # create a button
    if st.button('Rate of Change (ROC) FAQs'):
        webbrowser.open_new_tab(url)
# add a seperator line
progress_bar = st.progress(0)
# display a snapshot of the df data        
st.markdown("##### 10 Day Snapshot :chart_with_upwards_trend:")
st.write(option)
st.dataframe(df.tail(10))
# add a seperator line
progress_bar = st.progress(0)
# display Additional Corporate Data
st.markdown('##### Additional Corporate Data')
# display Institutional Holders
st.caption('Institutional Holders')
tickerData.institutional_holders
# display Major Share Holders
st.caption('Major Share Holders')
tickerData.major_holders
# display Financials
st.caption('Financials')
tickerData.financials
# display Balance Sheet
st.caption('Balance Sheet')
tickerData.balance_sheet
# display Cashflow
st.caption('Cashflow')
tickerData.cashflow
st.caption('Provided by Yahoo! finance, results were generated a few mins ago. Pricing data is updated frequently. Currency in USD.')
# add a seperator line
progress_bar = st.progress(0)

################
# Download csv #
################

# load imports
import base64
from io import BytesIO
# define the excel function
def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1')
    writer.save()
    processed_data = output.getvalue()
    return processed_data

def get_table_download_link(df):
    val = to_excel(df)
    b64 = base64.b64encode(val)
    return f'<a href="data:application/octet-stream;base64,{b64.decode()}" download="download.xlsx">Download excel file</a>'
# define section title and download link
st.markdown(" ")
st.markdown("##### Create Stock Report :pencil:")
st.markdown(get_table_download_link(df), unsafe_allow_html=True)
# define the csv dataframe function
@st.cache
def convert_df(df):
    return df.to_csv().encode('utf-8')

csv = convert_df(df)
# create the csv file button and download details
st.download_button(
    label="Download data as CSV",
    data=csv,
    file_name='stocks.csv',
    mime='text/csv',
)

