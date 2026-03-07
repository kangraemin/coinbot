"""공통 코인 상태 포맷 — /status 및 4H 리포트에서 공유."""

import config as cfg


def rsi_bar(rsi_val: float) -> str:
    """RSI를 5칸 블록 바로 표현."""
    if rsi_val < 20:
        return "🟩⬜⬜⬜⬜ 극과매도"
    elif rsi_val < 40:
        return "🟦🟩⬜⬜⬜ 과매도"
    elif rsi_val < 60:
        return "⬜⬜🟩⬜⬜ 중립"
    elif rsi_val < 80:
        return "⬜⬜⬜🟩🟥 과매수"
    else:
        return "⬜⬜⬜⬜🟩 극과매수"


def action_hint(symbol: str, ind: dict, pos: dict | None) -> list[str]:
    """현재 상태 기반 진입 조건 — 롱/숏 분리, 조건별 체크.

    pos dict 인터페이스:
        side: "long" | "short"
        entry_price: float
        contracts: float | None
        unrealized_pnl: float | None
        tp_price: float | None
        sl_price: float | None
        entry_time: datetime | None
    pos=None -> 포지션 없음
    """
    p = cfg.SYMBOL_STRATEGY.get(symbol, {})
    rsi_long = p.get("rsi_long", 30)
    rsi_short = p.get("rsi_short", 65)

    close = ind["close"]
    bb_lower = ind["bb_lower"]
    bb_upper = ind["bb_upper"]
    ema200 = ind["ema200"]
    rsi = ind["rsi"]

    if pos:
        side = pos.get("side", "")
        arrow = "📈" if side == "long" else "📉"
        side_kr = "롱" if side == "long" else "숏"
        lines = [f"{arrow} *{side_kr} 보유 중*"]
        if pos.get("unrealized_pnl") is not None:
            lines.append(f"미실현 손익: {pos['unrealized_pnl']:+,.2f} USDT")
        return lines

    lines = []

    # ── 롱 조건 ──
    long_price_pct = (bb_lower - close) / close * 100
    long_rsi_gap = rsi - rsi_long
    ema_above = close > ema200

    long_ok = 0
    lines.append("")
    lines.append("🟢 *롱 진입 조건*")
    if close < bb_lower:
        lines.append(f"✅ 가격: BB하단({bb_lower:,.4f}) 이하")
        long_ok += 1
    else:
        lines.append(f"❌ 가격: {bb_lower:,.4f} 이하 필요 ({long_price_pct:+.1f}%)")
    if rsi < rsi_long:
        lines.append(f"✅ RSI: {rsi:.0f} (기준 {rsi_long} 이하)")
        long_ok += 1
    else:
        lines.append(f"❌ RSI: {rsi:.0f} → {rsi_long} 이하 필요 ({long_rsi_gap:.0f} 남음)")
    if ema_above:
        lines.append("✅ EMA200 위 (상승 추세)")
        long_ok += 1
    else:
        ema_gap_pct = (ema200 - close) / close * 100
        lines.append(f"❌ EMA200 위 필요 (+{ema_gap_pct:.1f}% 상승)")
    lines.append(f"→ {long_ok}/3 충족")

    # ── 숏 조건 ──
    short_price_pct = (bb_upper - close) / close * 100
    short_rsi_gap = rsi_short - rsi
    ema_below = close < ema200

    short_ok = 0
    lines.append("")
    lines.append("🔴 *숏 진입 조건*")
    if close > bb_upper:
        lines.append(f"✅ 가격: BB상단({bb_upper:,.4f}) 이상")
        short_ok += 1
    else:
        lines.append(f"❌ 가격: {bb_upper:,.4f} 이상 필요 ({short_price_pct:+.1f}%)")
    if rsi > rsi_short:
        lines.append(f"✅ RSI: {rsi:.0f} (기준 {rsi_short} 이상)")
        short_ok += 1
    else:
        lines.append(f"❌ RSI: {rsi:.0f} → {rsi_short} 이상 필요 ({short_rsi_gap:.0f} 남음)")
    if ema_below:
        lines.append("✅ EMA200 아래 (하락 추세)")
        short_ok += 1
    else:
        ema_gap_pct = (close - ema200) / close * 100
        lines.append(f"❌ EMA200 아래 필요 (-{ema_gap_pct:.1f}% 하락)")
    lines.append(f"→ {short_ok}/3 충족")

    return lines


def format_coin_status(ind: dict, symbol: str, pos: dict | None) -> str:
    """코인별 상태 포맷 — 한 줄 한 정보, 섹션 구분.

    ind: close, rsi, bb_lower, bb_upper, ema200, atr 등 인디케이터
    pos: action_hint 참조. None이면 포지션 없음.
    """
    coin = symbol.split("/")[0]
    p = cfg.SYMBOL_STRATEGY.get(symbol, {})

    close = ind["close"]
    rsi = ind["rsi"]
    bb_lower = ind["bb_lower"]
    bb_upper = ind["bb_upper"]
    ema200 = ind["ema200"]

    # BB 위치 %
    bb_range = bb_upper - bb_lower
    bb_pct = (close - bb_lower) / bb_range * 100 if bb_range > 0 else 50.0

    # EMA200 대비 %
    ema_diff_pct = (close - ema200) / ema200 * 100

    # ── 헤더 + 가격 ──
    lines = [
        f"\n🔸 *{coin}*",
        "",
        "💵 *현재 가격*",
        f"종가: {close:,.4f}",
    ]

    # ── RSI ──
    lines.append("")
    lines.append("📊 *RSI(14)*")
    lines.append(f"값: {rsi:.1f}")
    lines.append(rsi_bar(rsi))

    # ── 볼린저밴드 ──
    bb_bar_len = int(bb_pct / 5)
    bb_bar = "█" * bb_bar_len + "░" * (20 - bb_bar_len)
    lines.append("")
    lines.append("📉 *볼린저밴드*")
    lines.append(f"상단: {bb_upper:,.4f}")
    lines.append(f"하단: {bb_lower:,.4f}")
    lines.append(f"위치: {bb_pct:.0f}% `{bb_bar}`")

    # ── EMA200 ──
    lines.append("")
    lines.append("📈 *EMA200*")
    lines.append(f"값: {ema200:,.4f}")
    lines.append(f"괴리: {ema_diff_pct:+.1f}%")
    if ema_diff_pct > 0:
        lines.append("추세: 🟢 상승 (가격 > EMA200)")
    else:
        lines.append("추세: 🔴 하락 (가격 < EMA200)")

    # ── 포지션 ──
    lines.append("")
    if pos:
        side = pos.get("side", "")
        entry = pos.get("entry_price", 0)
        arrow = "📈" if side == "long" else "📉"
        side_kr = "롱" if side == "long" else "숏"

        lines.append(f"{arrow} *포지션: {side_kr}*")
        lines.append(f"진입가: {entry:,.4f}")

        contracts = pos.get("contracts")
        if contracts is not None:
            lines.append(f"수량: {contracts:.6f} ({contracts * entry:,.2f} USDT)")

        unrealized_pnl = pos.get("unrealized_pnl")
        if unrealized_pnl is not None:
            lines.append(f"미실현 손익: {unrealized_pnl:+,.2f} USDT")

        tp_price = pos.get("tp_price")
        sl_price = pos.get("sl_price")

        # TP/SL: 직접 전달된 값 우선, 없으면 ATR 기반 계산
        if tp_price is None or sl_price is None:
            atr = ind.get("atr", 0)
            if atr > 0 and entry > 0:
                tp_mult = p.get("tp_mult", 3.0)
                sl_mult = p.get("sl_mult", 2.0)
                if side == "long":
                    tp_price = tp_price or entry + atr * tp_mult
                    sl_price = sl_price or entry - atr * sl_mult
                else:
                    tp_price = tp_price or entry - atr * tp_mult
                    sl_price = sl_price or entry + atr * sl_mult

        if tp_price:
            lines.append(f"익절(TP): {tp_price:,.4f}")
        if sl_price:
            lines.append(f"손절(SL): {sl_price:,.4f}")

        # 경과 시간 (4H 리포트에서만 제공)
        entry_time = pos.get("entry_time")
        if entry_time:
            from datetime import datetime, timezone
            elapsed = datetime.now(timezone.utc) - entry_time
            elapsed_h = elapsed.total_seconds() / 3600
            timeout_h = 192  # SIGNAL_TIMEOUT_HOURS
            remain_h = max(0, timeout_h - elapsed_h)
            lines.append(f"경과: {elapsed_h:.0f}h / 타임아웃: {remain_h:.0f}h 남음")
    else:
        lines.append("⏳ *포지션: 대기 중*")

    # ── 진입 조건 ──
    lines.append("")
    lines.append("─────────────")
    hint_lines = action_hint(symbol, ind, pos)
    lines.extend(hint_lines)

    return "\n".join(lines)
