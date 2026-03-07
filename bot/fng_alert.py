"""F&G 공포/탐욕 지수 알림 모듈.

30분 주기로 F&G 지수를 조회하여 DCA 매수 추천 알림을 텔레그램으로 발송.
- F&G ≤ 20: BTC/ETH/XRP 현물 DCA 추천 (구간 내 날짜에 따라 비중 제안)
- F&G 50~70: ETHU 매수 고려 알림
"""

import asyncio
import csv
import logging
import os
from datetime import datetime

import aiohttp

import config as cfg
from bot.report import send_telegram

logger = logging.getLogger(__name__)

FNG_API_URL = "https://api.alternative.me/fng/"
SENTIMENT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "sentiment")
FNG_CSV = os.path.join(SENTIMENT_DIR, "fng_daily.csv")

# 알림 주기 (초)
ALERT_INTERVAL = 1800  # 30분


def _fng_gauge(value: int) -> str:
    """F&G 값을 게이지 바 + 라벨로 변환."""
    if value <= 10:
        return "🟩⬜⬜⬜⬜ 극단적 공포"
    elif value <= 25:
        return "🟦🟩⬜⬜⬜ 극도의 공포"
    elif value <= 40:
        return "⬜🟩⬜⬜⬜ 공포"
    elif value <= 60:
        return "⬜⬜🟩⬜⬜ 중립"
    elif value <= 75:
        return "⬜⬜⬜🟩⬜ 탐욕"
    elif value <= 90:
        return "⬜⬜⬜🟩🟥 극도의 탐욕"
    else:
        return "⬜⬜⬜⬜🟥 극단적 탐욕"


async def fetch_current_fng() -> int | None:
    """alternative.me API에서 현재 F&G 값 조회."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                FNG_API_URL, params={"limit": "1"}, timeout=aiohttp.ClientTimeout(total=15)
            ) as resp:
                if resp.status != 200:
                    logger.warning("F&G API 응답 오류: %d", resp.status)
                    return None
                data = await resp.json(content_type=None)
                return int(data["data"][0]["value"])
    except Exception as e:
        logger.error("F&G API 조회 실패: %s", e)
        return None


def get_fear_streak() -> int:
    """fng_daily.csv에서 현재 F&G ≤ 20 연속일수 계산.

    오늘부터 과거로 거슬러 올라가며 F&G ≤ 20인 날을 세고,
    3일 이상 갭이 나면 중단.
    """
    if not os.path.exists(FNG_CSV):
        return 0

    rows = []
    with open(FNG_CSV) as f:
        for r in csv.DictReader(f):
            rows.append({"date": r["date"], "fng": int(r["fng"])})

    if not rows:
        return 0

    rows.sort(key=lambda x: x["date"], reverse=True)

    streak = 0
    prev_date = None
    for r in rows:
        if r["fng"] <= 20:
            if prev_date is not None:
                d1 = datetime.strptime(prev_date, "%Y-%m-%d")
                d2 = datetime.strptime(r["date"], "%Y-%m-%d")
                if (d1 - d2).days > 3:
                    break
            prev_date = r["date"]
            streak += 1
        else:
            if streak > 0:
                break
    return streak


def _buy_weight(streak_days: int) -> tuple[str, str]:
    """공포 구간 일수에 따른 DCA 비중 제안."""
    if streak_days <= 7:
        return "5~10%", "초반 구간, 보수적 매수"
    elif streak_days <= 15:
        return "10~15%", "중반 구간"
    elif streak_days <= 30:
        return "15~20%", "후반 구간, 적극 매수"
    else:
        return "20~25%", "장기 공포, 강력 매수"


def build_fng_alert(fng_value: int, streak_days: int) -> str:
    """F&G 알림 메시지 생성."""
    lines = [
        "📊 *F&G 공포/탐욕 지수*",
        f"현재: {fng_value} {_fng_gauge(fng_value)}",
    ]

    if fng_value <= 20 and streak_days > 0:
        lines.append(f"구간: F&G ≤ 20 연속 *{streak_days}일차*")
    elif fng_value <= 25:
        lines.append("⚠️ 공포 구간 근접 (F&G ≤ 25)")

    # DCA 매수 추천
    if fng_value <= 20:
        weight, comment = _buy_weight(streak_days)
        lines.append("")
        lines.append("💰 *현물 DCA 매수 추천*")
        lines.append(f"✅ BTC — 가용잔고의 {weight}")
        lines.append(f"✅ ETH — 가용잔고의 {weight}")
        lines.append(f"✅ XRP — 가용잔고의 {weight}")
        lines.append(f"📝 {comment}")
    elif fng_value <= 25:
        lines.append("")
        lines.append("💰 *현물 DCA 매수*")
        lines.append("⏳ F&G 20 이하 진입 시 매수 시작 추천")
    else:
        lines.append("")
        lines.append("💰 *현물 DCA*")
        lines.append("⛔ 공포 구간 아님 — DCA 대기")

    # ETHU 추천
    lines.append("")
    lines.append("📈 *ETHU (2x ETH ETF)*")
    if 50 <= fng_value <= 70:
        lines.append("✅ 매수 고려 — 상승 추세 진입 구간 (F&G 50~70)")
    elif fng_value > 70:
        lines.append("⚠️ 보유 중이면 익절 고려 (F&G 70+)")
    elif fng_value <= 20:
        lines.append("⛔ 매수 금지 — 레버리지 ETF는 하락장에서 복리 손실")
    else:
        lines.append("⏳ 대기 — F&G 50+ 상승 추세 진입 시 매수 고려")

    return "\n".join(lines)


async def send_fng_alert() -> None:
    """F&G 알림을 한 번 발송."""
    fng_value = await fetch_current_fng()
    if fng_value is None:
        logger.warning("F&G 값 조회 실패, 알림 건너뜀")
        return

    streak = get_fear_streak()
    msg = build_fng_alert(fng_value, streak)
    await send_telegram(msg)
    logger.info("F&G 알림 발송 (F&G=%d, streak=%d)", fng_value, streak)


async def fng_alert_loop() -> None:
    """30분 주기 F&G 알림 루프."""
    logger.info("F&G 알림 루프 시작 (주기: %ds)", ALERT_INTERVAL)

    # 시작 시 즉시 1회 발송
    await send_fng_alert()

    while True:
        try:
            await asyncio.sleep(ALERT_INTERVAL)
            await send_fng_alert()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error("F&G 알림 루프 오류: %s", e)
            await asyncio.sleep(60)
