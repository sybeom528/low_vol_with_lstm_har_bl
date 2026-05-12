"""
lib/page_helpers.py - 페이지 헤더 + 서브헤더 + CSS 주입

모든 페이지의 상단부 일관성 보장.
참조: docs/plan/02_common.md 10절, 6.1절 (H-1 Pretendard 폰트)
"""

import streamlit as st


# === Pretendard 폰트 fallback chain (H-1) =============================
# CDN 차단 시 자동으로 다음 폰트로 fallback → 기능 정상 유지
_PRETENDARD_CSS = """
<style>
@import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');

html, body, [class*="css"] {
    font-family: "Pretendard", "Noto Sans KR", "Malgun Gothic",
                 -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* === st.metric KPI 라벨 폰트 확대 (2026-05-11) ===
   - 기본 라벨 ~14px → 17px (가독성 향상)
   - 두 selector 병기: Streamlit 버전 호환성 */
[data-testid="stMetricLabel"] > div {
    font-size: 17px !important;
    font-weight: 600 !important;
}
[data-testid="stMetricLabel"] p {
    font-size: 17px !important;
    font-weight: 600 !important;
}
</style>
"""


def inject_custom_css() -> None:
    """
    Pretendard 폰트 fallback chain 주입 (H-1).
    app.py 진입 시 1회 호출.
    """
    st.markdown(_PRETENDARD_CSS, unsafe_allow_html=True)


def render_page_header(page_name_en: str, page_name_ko: str) -> None:
    """
    모든 페이지 상단 Header (Overview 영역 1 일관 적용).

    Args:
        page_name_en: 영문 페이지명 (예: "Performance")
        page_name_ko: 한글 페이지명 (예: "성과 분석")

    페이지명 (영/한 병기, A-3 결정).
    우측 펀드 메타 (Active 표시 / 벤치마크 / 데이터 시점) 는 2026-05-11 제거됨:
      - Sub-header / Footer 와 정보 중복
      - 우상단 공간 활용도 낮음
    """
    st.title(page_name_en)
    st.caption(page_name_ko)

    # 사이드바 영역 검색 → 페이지 이동 시 highlight 안내 (2026-05-12 신규)
    if "_highlight_area" in st.session_state:
        st.info(
            f"🔍 검색하신 영역: **{st.session_state['_highlight_area']}** "
            f"— 페이지 내에서 동일한 이름의 섹션을 찾아 이동해 주십시오."
        )
        del st.session_state["_highlight_area"]


def render_subheader(title_en: str, title_ko: str, description: str) -> None:
    """
    페이지별 Sub-header 카드 (Performance 영역 2 패턴).

    Cobalt Blue 좌측 테두리 + 어두운 카드 배경.
    페이지의 핵심 메시지 / 영역 설명을 표시.

    description 에서 자주 쓰는 markdown 자동 변환:
      - **bold**     → <strong>bold</strong>
      - `code`       → <code>code</code>
    (HTML 박스 안에 markdown 직접 안 됨 → 변환 필요)
    """
    import re

    desc_html = description
    # **bold** → <strong>
    desc_html = re.sub(r"\*\*(.+?)\*\*", r"<strong style='color:#FAFAFA;'>\1</strong>", desc_html)
    # `code` → <code>
    desc_html = re.sub(
        r"`([^`]+?)`",
        r"<code style='background:#374151;padding:1px 5px;border-radius:3px;color:#FAFAFA;'>\1</code>",
        desc_html,
    )

    st.markdown(
        f"""
        <div style="
            background-color: #1F2937;
            border-left: 4px solid #3B82F6;
            padding: 16px;
            border-radius: 4px;
            margin-bottom: 16px;
        ">
            <div style="font-size: 18px; font-weight: bold; color: #FAFAFA;">
                ℹ️ {title_en} ({title_ko})
            </div>
            <div style="font-size: 14px; color: #9CA3AF; margin-top: 8px;">
                {desc_html}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> None:
    """
    모든 페이지에서 동일한 사이드바 렌더링 (C-4 6 그룹 + 2 토글).

    Streamlit multi-page 에서 각 페이지마다 자체 사이드바를 그려야 함
    (`config.toml: showSidebarNavigation = false` 설정 시 자동 nav 비활성).

    호출 위치: 각 페이지 진입 시 inject_custom_css() / init_session_state() 다음.
    """
    with st.sidebar:
        # ── 펀드명 + 메타 (C4-3) ──
        st.markdown("# Adaptive VolControl Fund")
        st.markdown("변동성 인지 적응 펀드")
        st.caption("Benchmark: SPY  |  Data: 2010-01 ~ 2025-12")
        st.divider()

        # ── 6 그룹 페이지 navigation (C4-1 c, C4-2 a) ──
        # 그룹 1: 개요
        st.markdown("##### ── 개요 ──")
        st.page_link("app.py", label="Overview", icon="📊")

        # 그룹 2: 체험 (★ Investment Simulator F-6)
        st.markdown("##### ── 체험 ──")
        st.page_link("pages/02_Investment_Simulator.py", label="Investment Simulator", icon="💵")

        # 그룹 3: 성과
        st.markdown("##### ── 성과 ──")
        st.page_link("pages/03_Performance.py", label="Performance", icon="📈")
        st.page_link("pages/04_Risk_Metrics.py", label="Risk Metrics", icon="⚠️")

        # 그룹 4: 보유
        st.markdown("##### ── 보유 ──")
        st.page_link("pages/05_Holdings.py", label="Holdings", icon="🏢")
        st.page_link("pages/06_Sector_Watch.py", label="Sector Watch", icon="🌐")

        # 그룹 5: 메타
        # 주의 (2026-05-11):
        #   - Methodology 페이지 통합 삭제 → BL+LSTM Sankey 는 Overview 영역 6.
        #   - Backtesting 페이지 통합 삭제 → Regime 메트릭 + Sub-events 4위기 는
        #     Risk Metrics 페이지 영역 5/6 으로 이전.
        st.markdown("##### ── 메타 ──")
        st.page_link("pages/09_About.py", label="About / FAQ", icon="ℹ️")

        st.divider()

        # ── 영역 검색 (2026-05-12 신규) ──────────────────────────────
        from lib.search_index import search_areas

        st.markdown("##### 🔍 영역 검색")
        query = st.text_input(
            "키워드로 영역 찾기",
            placeholder="예: var, 분포, sortino, regime, 시뮬레이션...",
            key="_sidebar_search_query",
            label_visibility="collapsed",
        )
        if query:
            results = search_areas(query, max_results=8)
            if results:
                st.caption(f"검색 결과 {len(results)}개")
                for i, r in enumerate(results):
                    btn_label = f"📍 {r['page']} → {r['area']}"
                    if st.button(
                        btn_label,
                        key=f"_sidebar_search_goto_{i}",
                        use_container_width=True,
                    ):
                        st.session_state["_highlight_area"] = r["area"]
                        st.switch_page(r["page_file"])
            else:
                st.info("일치하는 영역이 없습니다.")

        st.divider()

        # ── 토글 widget — 페이지 이동 시에도 상태 보존 ─────────────────
        # 문제 (Streamlit multipage 이슈, 2026-05-11):
        #   - widget key 와 session_state key 가 동일하면 페이지 이동 시
        #     widget 이 unmount 되었다가 re-mount 될 때 internal state 가
        #     default 로 reset → session_state 도 함께 reset.
        # 해결 (source-of-truth 패턴):
        #   - Source key: "period" / "show_spy" / "show_ew" / "show_ivw"
        #     (다른 페이지 코드는 이 key 를 read 함, init_session_state 보장)
        #   - Widget key: "_sidebar_<name>" (페이지 진입 시 재 mount 대비)
        #   - 페이지 진입 시: source → widget key 복원
        #   - Widget 변경 시: on_change callback 으로 source 업데이트
        # ────────────────────────────────────────────────────────────

        # 페이지 진입 시 widget key 를 source value 로 복원
        st.session_state["_sidebar_period"] = st.session_state.period
        st.session_state["_sidebar_show_spy"] = st.session_state.show_spy
        st.session_state["_sidebar_show_ew"] = st.session_state.show_ew
        st.session_state["_sidebar_show_ivw"] = st.session_state.show_ivw

        # ── 토글 1: 기간 (Period) — C4-4 ──
        st.subheader("📅 기간 (Period)")
        st.radio(
            "기간 선택",
            options=["TEST", "HO", "FULL"],
            format_func=lambda x: "Hold Out" if x == "HO" else x,
            key="_sidebar_period",
            on_change=_sync_period_to_source,
            label_visibility="collapsed",
        )

        # ── 토글 2: 비교 벤치마크 — C4-4 ──
        # 라벨: 직관적 한글 (내부 식별자 EW/IVW 는 caption / tooltip 에서만 유지).
        st.subheader("📊 비교 (Benchmark)")
        st.checkbox(
            "SPY (S&P 500 ETF)",
            key="_sidebar_show_spy",
            on_change=_sync_show_spy_to_source,
            help="시장 대표 벤치마크 — S&P 500 추종 ETF",
        )
        st.checkbox(
            "균등가중",
            key="_sidebar_show_ew",
            on_change=_sync_show_ew_to_source,
            help="펀드 universe (S&P 500 종목) 의 모든 종목을 동일 비중으로 보유한 가상 포트폴리오 (Equal-Weight, EW)",
        )
        st.checkbox(
            "역변동성 가중",
            key="_sidebar_show_ivw",
            on_change=_sync_show_ivw_to_source,
            help="변동성이 낮은 종목일수록 큰 비중을 주는 가상 포트폴리오 (Inverse Volatility Weight, IVW) — Low-Volatility Anomaly 활용",
        )


# === 사이드바 widget → source-of-truth sync callbacks =================
# (render_sidebar 위에 정의해도 되지만, helper 함수와 명확히 분리하기 위해
#  파일 하단에 배치. 단방향 sync: widget value → source key)
def _sync_period_to_source() -> None:
    st.session_state.period = st.session_state._sidebar_period


def _sync_show_spy_to_source() -> None:
    st.session_state.show_spy = st.session_state._sidebar_show_spy


def _sync_show_ew_to_source() -> None:
    st.session_state.show_ew = st.session_state._sidebar_show_ew


def _sync_show_ivw_to_source() -> None:
    st.session_state.show_ivw = st.session_state._sidebar_show_ivw
