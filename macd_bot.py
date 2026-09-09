import os
import requests
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timezone


# ============================================================
# 설정
# ============================================================

SYMBOLS = ["EWY", "SOXX", "QQQ", "TSLA"]

FAST_LENGTH = 12
SLOW_LENGTH = 26
SIGNAL_LENGTH = 9
LOOKBACK = 50

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")


# ============================================================
# EMA
# Pine Script의 ta.ema와 동일한 방식
# ============================================================

def ema(series, length):
    return series.ewm(
        span=length,
        adjust=False
    ).mean()


# ============================================================
# MACD 계산
#
# Pine Script:
# macd = EMA(close, 12) - EMA(close, 26)
# signal = EMA(macd, 9)
# hist = macd - signal
# ============================================================

def calculate_macd(close):

    macd = ema(close, FAST_LENGTH) - ema(close, SLOW_LENGTH)

    signal = ema(macd, SIGNAL_LENGTH)

    hist = macd - signal

    return macd, signal, hist


# ============================================================
# EMA Histogram 계산
#
# Pine Script:
# ema_hist = ta.ema(hist_current, signal_length)
# ============================================================

def calculate_ema_hist(close):

    macd, signal, hist = calculate_macd(close)

    ema_hist = ema(hist, SIGNAL_LENGTH)

    return macd, signal, hist, ema_hist


# ============================================================
# Pine Script의 스케일링
#
# max_abs_range =
# max(abs(hist_max), abs(hist_min))
# ============================================================

def calculate_scaled_values(close):

    macd, signal, hist, ema_hist = calculate_ema_hist(close)

    hist_recent = hist.tail(LOOKBACK)

    hist_min = hist_recent.min()
    hist_max = hist_recent.max()

    max_abs_range = max(
        abs(hist_max),
        abs(hist_min),
        1e-10
    )

    macd_scaled = (
        macd / max_abs_range
    ) * 100

    signal_scaled = (
        signal / max_abs_range
    ) * 100

    hist_scaled = (
        hist / max_abs_range
    ) * 100

    ema_scaled = (
        ema_hist / max_abs_range
    ) * 100 * 3

    return (
        macd_scaled,
        signal_scaled,
        hist_scaled,
        ema_scaled
    )


# ============================================================
# EMA Cross Up
#
# Pine Script:
# ema_cross_up = ta.crossover(ema_scaled, 0)
# ============================================================

def crossover_zero(series):

    if len(series) < 2:
        return False

    previous = series.iloc[-2]
    current = series.iloc[-1]

    return previous <= 0 and current > 0


# ============================================================
# Yahoo Finance 데이터 다운로드
# ============================================================

def get_data(symbol, interval, period):

    try:

        ticker = yf.Ticker(symbol)

        df = ticker.history(
            interval=interval,
            period=period,
            auto_adjust=True
        )

        if df is None or df.empty:
            print(f"{symbol} {interval} 데이터 없음")
            return None

        return df

    except Exception as e:

        print(
            f"{symbol} {interval} 데이터 오류: {e}"
        )

        return None


# ============================================================
# 텔레그램 메시지 보내기
# ============================================================

def send_telegram(message):

    if not TELEGRAM_BOT_TOKEN:
        print("TELEGRAM_BOT_TOKEN 없음")
        return

    if not TELEGRAM_CHAT_ID:
        print("TELEGRAM_CHAT_ID 없음")
        return

    url = (
        f"https://api.telegram.org/bot"
        f"{TELEGRAM_BOT_TOKEN}/sendMessage"
    )

    data = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message
    }

    try:

        response = requests.post(
            url,
            data=data,
            timeout=20
        )

        response.raise_for_status()

        print("텔레그램 전송 성공")

    except Exception as e:

        print(
            f"텔레그램 전송 오류: {e}"
        )


# ============================================================
# MTF MACD BUY 검사
#
# 원본 Pine Script:
#
# higher_bullish_count =
# (1H hist > 0)
# +
# (1D hist > 0)
# +
# (1W hist > 0)
#
# higher_bullish_majority >= 2
#
# short_confirm_recent =
# highest(hist_15m, 2) > 0
#
# entry_condition =
# ema_cross_up
# and ema_rising_recent
# and short_confirm_recent
# and higher_bullish_majority_recent
# ============================================================

def check_mtf_macd(symbol):

    print("")
    print("=" * 60)
    print(f"{symbol} 검사 시작")
    print("=" * 60)

    # --------------------------------------------------------
    # 일봉
    # 현재 차트 = TradingView 1D
    # --------------------------------------------------------

    df_1d = get_data(
        symbol,
        "1d",
        "2y"
    )

    if df_1d is None:
        return

    close_1d = df_1d["Close"].dropna()

    if len(close_1d) < 100:
        print("일봉 데이터 부족")
        return

    (
        macd_1d_scaled,
        signal_1d_scaled,
        hist_1d_scaled,
        ema_scaled
    ) = calculate_scaled_values(close_1d)

    # EMA Histogram 0선 상향 돌파
    ema_cross_up = crossover_zero(
        ema_scaled
    )

    # --------------------------------------------------------
    # Pine Script:
    #
    # ema_rising_recent =
    # ta.valuewhen(
    # ema_scaled > ema_scaled[1],
    # ema_scaled,
    # 0
    # ) > ema_scaled[1]
    #
    # 최신 EMA가 상승 중인지 확인
    # --------------------------------------------------------

    ema_rising_recent = (
        ema_scaled.iloc[-1]
        >
        ema_scaled.iloc[-2]
    )

    # --------------------------------------------------------
    # 15분봉
    # --------------------------------------------------------

    df_15m = get_data(
        symbol,
        "15m",
        "60d"
    )

    if df_15m is None:
        return

    close_15m = df_15m["Close"].dropna()

    if len(close_15m) < 50:
        print("15분봉 데이터 부족")
        return

    (
        _,
        _,
        hist_15m_scaled,
        _
    ) = calculate_scaled_values(
        close_15m
    )

    # Pine Script:
    # ta.highest(hist_15m, 2) > 0
    short_confirm_recent = (
        hist_15m_scaled
        .tail(2)
        .max()
        > 0
    )

    # --------------------------------------------------------
    # 1시간봉
    # --------------------------------------------------------

    df_1h = get_data(
        symbol,
        "1h",
        "730d"
    )

    if df_1h is None:
        return

    close_1h = df_1h["Close"].dropna()

    if len(close_1h) < 50:
        print("1시간봉 데이터 부족")
        return

    (
        _,
        _,
        hist_1h_scaled,
        _
    ) = calculate_scaled_values(
        close_1h
    )

    hist_1h_positive = (
        hist_1h_scaled.iloc[-1] > 0
    )

    # --------------------------------------------------------
    # 주봉
    # --------------------------------------------------------

    df_1w = get_data(
        symbol,
        "1wk",
        "5y"
    )

    if df_1w is None:
        return

    close_1w = df_1w["Close"].dropna()

    if len(close_1w) < 50:
        print("주봉 데이터 부족")
        return

    (
        _,
        _,
        hist_1w_scaled,
        _
    ) = calculate_scaled_values(
        close_1w
    )

    hist_1w_positive = (
        hist_1w_scaled.iloc[-1] > 0
    )

    # --------------------------------------------------------
    # 일봉 Histogram
    # --------------------------------------------------------

    hist_1d_positive = (
        hist_1d_scaled.iloc[-1] > 0
    )

    # --------------------------------------------------------
    # Pine Script:
    #
    # higher_bullish_count =
    # 1H +
    # 1D +
    # 1W
    #
    # >= 2
    # --------------------------------------------------------

    higher_bullish_count = sum([
        hist_1h_positive,
        hist_1d_positive,
        hist_1w_positive
    ])

    higher_bullish_majority_recent = (
        higher_bullish_count >= 2
    )

    # --------------------------------------------------------
    # 최종 MTF MACD BUY 조건
    # --------------------------------------------------------

    entry_condition = (
        ema_cross_up
        and ema_rising_recent
        and short_confirm_recent
        and higher_bullish_majority_recent
    )

    # --------------------------------------------------------
    # 현재 상태 출력
    # --------------------------------------------------------

    print("")
    print(f"EMA Cross Up: {ema_cross_up}")
    print(f"EMA Rising: {ema_rising_recent}")
    print(f"15m Confirm: {short_confirm_recent}")
    print(f"1H Bullish: {hist_1h_positive}")
    print(f"1D Bullish: {hist_1d_positive}")
    print(f"1W Bullish: {hist_1w_positive}")
    print(
        f"Bullish Count: "
        f"{higher_bullish_count}/3"
    )
    print(
        f"BUY SIGNAL: "
        f"{entry_condition}"
    )

    # --------------------------------------------------------
    # 매수 신호 발생
    # --------------------------------------------------------

    if entry_condition:

        current_price = close_1d.iloc[-1]

        current_date = (
            close_1d.index[-1]
            .strftime("%Y-%m-%d")
        )

        message = f"""
🟢 MTF MACD BUY 확정

📈 종목: {symbol}
💰 종가: ${current_price:.2f}
📅 일봉: {current_date}

━━━━━━━━━━━━━━

✅ EMA Histogram 0선 상향 돌파
✅ EMA 상승 확인
✅ 15분봉 상승 확인
✅ 1시간봉 상승
{"✅" if hist_1d_positive else "❌"} 일봉 상승
{"✅" if hist_1w_positive else "❌"} 주봉 상승

📊 MTF 상승:
{higher_bullish_count}/3

━━━━━━━━━━━━━━

🔒 확정된 데이터 기준
🚫 TradingView 알람 불필요
"""

        send_telegram(
            message
        )

    else:

        print(
            f"{symbol}: "
            f"매수 신호 없음"
        )


# ============================================================
# 메인 실행
# ============================================================

def main():

    print("")
    print("================================")
    print("NIKKO MTF MACD BOT")
    print("================================")

    print(
        datetime.now(
            timezone.utc
        ).strftime(
            "%Y-%m-%d %H:%M:%S UTC"
        )
    )

    for symbol in SYMBOLS:

        try:

            check_mtf_macd(symbol)

        except Exception as e:

            print("")
            print(
                f"{symbol} 실행 오류"
            )

            print(e)

    print("")
    print("검사 완료")


if __name__ == "__main__":

    main()
