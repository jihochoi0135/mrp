import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bat-Tracking Swing Quality Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.main { background-color: #0e1117; }
[data-testid="metric-container"] {
    background:#1a1d27; border:1px solid #2d3748;
    border-radius:10px; padding:14px 18px;
}
[data-testid="stMetricValue"]  { font-size:1.6rem; font-weight:700; color:#e2e8f0; }
[data-testid="stMetricLabel"]  { font-size:0.75rem; color:#94a3b8; text-transform:uppercase; letter-spacing:0.05em; }
[data-testid="stMetricDelta"]  { font-size:0.75rem; }
.section-header {
    font-size:1.0rem; font-weight:600; color:#94a3b8;
    text-transform:uppercase; letter-spacing:0.08em;
    padding:4px 0 12px 0; border-bottom:1px solid #2d3748; margin-bottom:20px;
}
[data-testid="stSidebar"] { background:#131722; border-right:1px solid #2d3748; }
.disclaimer {
    background:#1a2035; border-left:3px solid #3b82f6;
    border-radius:0 8px 8px 0; padding:12px 16px;
    margin:12px 0; font-size:0.82rem; color:#94a3b8; line-height:1.6;
}
.interpret-box {
    background:#111827; border:1px solid #1e3a5f;
    border-radius:10px; padding:16px 20px; margin:4px 0;
}
.interpret-box ul { margin:0; padding-left:18px; }
.interpret-box li { font-size:0.83rem; color:#94a3b8; line-height:1.8; margin-bottom:3px; }
.interpret-box .hl { color:#e2e8f0; font-weight:500; }
.formula-box {
    background:#0d1117; border:1px solid #30363d; border-radius:8px;
    padding:14px 18px; font-family:'Courier New',monospace;
    font-size:0.85rem; color:#58a6ff; margin:8px 0 16px 0; line-height:1.8;
}
.lang-tag {
    display:inline-block; font-size:0.7rem; padding:2px 9px;
    border-radius:8px; background:#1e3a5f; color:#60a5fa;
    border:1px solid #2d6aad; margin:4px 0 10px 0;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# TRANSLATIONS  (English / 한국어)
# ─────────────────────────────────────────────────────────────────
T = {
"English": {
    # sidebar
    "title": "Swing Quality",
    "filters": "Filters",
    "min_swings_lbl": "Min. Competitive Swings",
    "side_lbl": "Batter Side",
    "side_all": "All",
    "score_lbl": "Primary Score",
    "score_exp": "Expanded LSQ (default)",
    "score_mech": "Mechanical (shrunk)",
    "topn_lbl": "Top N (charts)",
    "footer": "Data: MLB Statcast · 2025\nExploratory estimates only.",
    # page names
    "pages": ["Overview","Player Ranking","Player Profile",
              "Power vs Efficiency","Contact & Expanded LSQ",
              "Reliability & Sample Size","Archetype Explorer","Methodology"],
    # ── Overview ──────────────────────────────────────────────────
    "ov_h": "Overview",
    "ov_disc": ("This overview shows the distribution of bat-tracking mechanical quality "
                "and how reliability varies with sample size. Reliability-adjusted scores "
                "are used for primary ranking."),
    "ov_k1":"Total Players","ov_k2":"Date Range","ov_k3":"Median Swings",
    "ov_k4":"Median Mech. Score","ov_k5":"Top Player","ov_k6":"≥ 90th Pctile",
    "ov_c1":"Distribution — Mechanical Score (Shrunk)",
    "ov_c2":"Distribution — Competitive Swings",
    "ov_c3":"Mechanical Score vs Reliability",
    "ov_c4":"Top 10 — Reliability-Adjusted Score",
    "ov_it": "How to Read: Overview",
    "ov_ip": [
        "<span class='hl'>Score Distribution (left)</span>: Right-skewed is normal. Most players cluster near the mean; elite hitters form a long right tail. Wider spread = more variance in swing mechanics across the league.",
        "<span class='hl'>Sample Size Distribution (right)</span>: The dashed line marks the current minimum-swing filter. Players left of this line are excluded. This prevents small-sample outliers from inflating ranks.",
        "<span class='hl'>Score vs Reliability Scatter</span>: Bottom-right players (high raw score, low reliability) are caution candidates — their estimates may stabilize differently with more swings. Ideal: top-right (high score + high reliability).",
        "<span class='hl'>Key rule</span>: Always compare Shrunk scores, not Raw scores. The Shrunk score accounts for sample size; Raw scores can be artificially inflated.",
    ],
    # ── Ranking ───────────────────────────────────────────────────
    "rk_h": "Player Ranking",
    "rk_bar": "Top {n} Players — {label}",
    "rk_tbl": "Full Ranking Table",
    "rk_dl": "Download Rankings CSV",
    "rk_it": "How to Read: Player Ranking",
    "rk_ip": [
        "<span class='hl'>Expanded LSQ (default)</span>: Blends bat-tracking mechanics with contact outcomes (xwOBA, Barrel%, Hard Hit%). Switch to Mechanical-only in the sidebar to isolate pure bat-tracking signal.",
        "<span class='hl'>Shrunk vs Raw</span>: Always rank by Shrunk scores. Players with fewer swings are pulled toward the mean — Raw scores can look misleadingly high for small samples.",
        "<span class='hl'>Reliability column</span>: Below 0.85 means the estimate may shift substantially as more swings are observed. Treat these ranks with caution.",
        "<span class='hl'>Label tiers</span>: Elite ≥ 90th pctile · Strong ≥ 70th · Above-avg ≥ 50th · Below-avg < 50th · Low < 25th.",
        "<span class='hl'>Side filter</span>: Percentiles are calculated across all hitters. Use the side filter when comparing LHH vs RHH separately.",
    ],
    # ── Profile ───────────────────────────────────────────────────
    "pf_h": "Player Profile",
    "pf_search": "Search Player",
    "pf_k1":"Avg Bat Speed","pf_k2":"Ideal Attack Angle Rate",
    "pf_k3":"Competitive Swings","pf_k4":"Mech Score Percentile",
    "pf_k5":"Reliability","pf_k6":"Exp LSQ Percentile",
    "pf_tbl":"Full Profile","pf_radar":"Percentile Radar","pf_bkdn":"Score Breakdown",
    "pf_median_lbl": "League Median",
    "pf_it": "How to Read: Player Profile",
    "pf_ip": [
        "<span class='hl'>Radar Chart</span>: Each axis = percentile (0–100) vs all qualified players. Larger shaded area = better overall profile. Dashed circle = league median (50th pctile).",
        "<span class='hl'>Raw Swing Power axis</span>: Directly tied to avg bat speed. Higher = more potential exit velocity, but power alone doesn't guarantee quality contact.",
        "<span class='hl'>Swing Path Efficiency axis</span>: How consistently the player swings in the ideal attack angle range (5°–20°). Higher = more favorable launch conditions produced repeatedly.",
        "<span class='hl'>Reliability axis</span>: Low reliability shrinks the radar inward intentionally — a player with few swings should show a smaller profile until estimates stabilize.",
        "<span class='hl'>Score Breakdown bars</span>: Shows each component's contribution. Imbalance (e.g., high power but low efficiency) reveals where improvement can target.",
        "<span class='hl'>xwOBA / Barrel / HardHit axes</span>: Outcome metrics. Strong mechanics diverging from weak outcomes may indicate pitch-selection or timing issues.",
    ],
    # ── Power vs Efficiency ───────────────────────────────────────
    "pe_h": "Raw Swing Power vs Swing Path Efficiency",
    "pe_disc": ("Each dot = one player. X-axis = power (bat speed z-score), "
                "Y-axis = efficiency (ideal attack angle rate z-score). "
                "Dot size = sample size. Color = Mechanical Percentile."),
    "pe_scatter": "Power vs Efficiency Quadrant",
    "pe_qsum": "Quadrant Summary",
    "pe_qtop": "Top 5 per Quadrant",
    "pe_q": ["High Power + High Efficiency","High Power + Low Efficiency",
             "Low Power + High Efficiency","Low Power + Low Efficiency"],
    "pe_it": "How to Read: Power vs Efficiency",
    "pe_ip": [
        "<span class='hl'>High Power + High Efficiency (top-right)</span>: Best mechanical profile. High bat speed AND consistent swing path. These hitters threaten for both average and power.",
        "<span class='hl'>High Power + Low Efficiency (bottom-right)</span>: Raw power hitters with inconsistent swing path. High K% risk, but when they connect, they hit it hard. Swing path adjustment has high upside.",
        "<span class='hl'>Low Power + High Efficiency (top-left)</span>: Contact-oriented hitters. Consistent path through the zone but limited bat speed caps the ceiling. Classic Tony Gwynn archetype.",
        "<span class='hl'>Low Power + Low Efficiency (bottom-left)</span>: Weaker profile on both dimensions. May still produce through pitch selection or BABIP, but bat-tracking signal is unfavorable.",
        "<span class='hl'>Dot size</span>: Larger = more competitive swings = more reliable estimate. Be cautious of small dots near the extremes — those estimates are more uncertain.",
    ],
    # ── Contact & LSQ ─────────────────────────────────────────────
    "cl_h": "Contact Efficiency & Expanded LSQ",
    "cl_disc": ("Evaluates whether bat-tracking mechanics align with expected offensive value "
                "(xwOBA, Barrel%, Hard Hit%). Alignment validates the mechanical score; "
                "divergence reveals analytically interesting outliers."),
    "cl_c1":"Mech Score vs xwOBA",
    "cl_c2":"Mech Score vs Barrel Rate",
    "cl_c3":"Mechanical vs Expanded LSQ — Alignment / Divergence",
    "cl_c4":"Expanded LSQ Ranking — Top {n}",
    "cl_it": "How to Read: Contact & Expanded LSQ",
    "cl_ip": [
        "<span class='hl'>Mech vs xwOBA</span>: Points above the trend line = better xwOBA than mechanics predict (pitch-selection edge). Points below = mechanics not translating to outcomes (timing, contact location issues).",
        "<span class='hl'>Mech vs Barrel Rate</span>: Barrel rate is a strong proxy for hard contact quality. High Mech + low barrel may signal swing-plane issues not fully captured by attack angle alone.",
        "<span class='hl'>Alignment / Divergence scatter</span>: Dashed diagonal = 1:1 line. Above = Expanded LSQ > Mech (outcome stats add value). Below = outcome stats drag profile down. Large gap either way is analytically interesting.",
        "<span class='hl'>Color gradient (Gap)</span>: Blue = outcome stats boost score above mechanics. Red = outcome stats pull score below mechanics. Gray = well-aligned.",
        "<span class='hl'>Key insight</span>: Mechanical score is a leading indicator. Outcome stats (xwOBA, Barrel%) are partially dependent on pitch mix and defense. Use both for the fullest picture.",
    ],
    # ── Reliability ───────────────────────────────────────────────
    "rl_h": "Reliability & Sample Size Analysis",
    "rl_disc": ("Players with small competitive-swing samples are shrunk toward the population mean. "
                "Formula: Reliability = 1 − exp(−swings / 100). "
                "This page shows how shrinkage works and flags high-raw / low-reliability outliers."),
    "rl_c1":"Reliability Curve — Swings vs Reliability",
    "rl_c2":"Shrinkage Effect — Raw vs Shrunk Score",
    "rl_c3":"Small Sample + High Raw Score — Caution Candidates",
    "rl_it": "How to Read: Reliability & Sample Size",
    "rl_ip": [
        "<span class='hl'>Reliability Curve</span>: Dotted line = theoretical formula 1−exp(−n/100). At n=100, reliability ≈ 0.63. At n=300, reliability ≈ 0.95. Points above curve = more consistent than expected; below = noisier.",
        "<span class='hl'>Shrinkage Scatter</span>: Points above the y=x diagonal were pulled down toward 0 (the mean). Further from diagonal = more shrinkage = less certain estimate. These players need more swings for a stable rank.",
        "<span class='hl'>Caution Table</span>: High Raw score but low reliability. Current high rank may be inflated. Watch for regression as sample size grows.",
        "<span class='hl'>Practical thresholds</span>: Reliability < 0.63 (< ~100 swings) = highly uncertain. 0.63–0.90 = moderate confidence. > 0.90 = stable estimate.",
        "<span class='hl'>Shrinkage Diff column</span>: Raw − Shrunk gap. Diff > 0.5 means the rank could shift substantially with more data.",
    ],
    # ── Archetype ─────────────────────────────────────────────────
    "ar_h": "Archetype Explorer",
    "ar_disc": ("K-Means clustering on standardized mechanical metrics + contact outcomes. "
                "PCA reduces to 2D for visualization. Archetypes are exploratory groupings, "
                "not definitive skill classifications."),
    "ar_k": "Number of Clusters (k)",
    "ar_c1":"PCA Cluster Visualization",
    "ar_c2":"Cluster Mean Profiles",
    "ar_c3":"Top Players per Cluster",
    "ar_arch": ["Power-Efficient","Raw Power","Contact-Efficient",
                "Low Mechanical","Mixed / Average"],
    "ar_it": "How to Read: Archetype Explorer",
    "ar_ip": [
        "<span class='hl'>PCA Plot</span>: PC1 and PC2 are linear combinations of all 6 input features. Higher % variance explained = better 2D representation of the original data. 70%+ is generally good.",
        "<span class='hl'>PC1 (horizontal)</span>: Typically represents overall swing quality — players further right tend to have better composite scores on both power and efficiency.",
        "<span class='hl'>PC2 (vertical)</span>: Often captures the power-vs-efficiency tradeoff — players at opposite ends of PC2 have different power/contact mixes.",
        "<span class='hl'>Cluster labels</span>: Auto-assigned based on mean feature values per cluster. Use as rough archetypes, not definitive skill grades.",
        "<span class='hl'>Adjust k</span>: k=3 for broad groups, k=5–7 for finer distinctions. If clusters heavily overlap in the PCA plot, fewer groups may be more appropriate.",
        "<span class='hl'>Symbol (circle/X)</span>: Circle = RHH, X = LHH. Archetype membership should be handedness-agnostic; systematic L/R differences are analytically interesting when observed.",
    ],
    # ── Methodology ───────────────────────────────────────────────
    "me_h": "Methodology & Interpretation",
    "me_disc": ("This score is an <strong>exploratory estimate</strong> of swing-quality profile based on "
                "publicly available bat-tracking and contact-quality metrics. It is <strong>not</strong> "
                "a direct measure of true hitting talent. Reliability-adjusted scores are used for ranking."),
    "me_s1":"Score Definitions","me_s2":"Data Source & Variables",
    "me_s3":"Interpretation Limits","me_s4":"Next Steps","me_s5":"Key References",
    "me_meta": {"Data Source":"MLB Statcast / Baseball Savant (2025)",
                "Minimum Qualified":"≥ 50 competitive swings (dashboard default: 100)"},
    "me_lim": [
        "Bat speed & attack angle are measured at contact — values vary with contact-point location (timing confounding; Powers & Yurko, 2026).",
        "Mechanical Score does not capture swing decision quality (Chase Rate, Zone Swing%).",
        "Expanded LSQ aggregates public stats; unmeasured confounders may exist.",
        "Players with < 100 competitive swings have meaningfully shrunk scores.",
        "Scores are a 2025 season snapshot; prior/future performance may differ.",
    ],
    "me_nxt": [
        "Bayesian uncertainty intervals around shrunk scores.",
        "State-space latent swing quality trend (monthly monitoring).",
        "Pitch-context adjusted score (fastball vs. offspeed split).",
        "Two-strike vs. hitter's count swing quality split.",
        "Team-level comparison dashboard.",
        "Full Swing Path Tilt integration as 3rd mechanical component.",
    ],
    "me_var_h": ["Variable","Unit","Description"],
    "me_var_r": [
        ("avg_bat_speed","mph","Avg bat head speed at 6\" reference point"),
        ("ideal_attack_angle_rate","%","Fraction of swings in 5°–20° attack angle range"),
        ("competitive_swings","count","Qualified swings used for reliability calc."),
        ("raw_swing_power_score","z","z(avg_bat_speed)"),
        ("swing_path_efficiency_score","z","z(ideal_attack_angle_rate)"),
        ("mechanical_swing_quality_score","composite","Weighted avg of power + efficiency"),
        ("mechanical_swing_quality_shrunk","composite","Reliability-adjusted mechanical score"),
        ("msq_reliability","0–1","1 − exp(−n/100)"),
        ("xwoba","rate","Expected weighted on-base average"),
        ("barrel_rate","%","Barrel contact rate"),
        ("hard_hit_rate","%","% batted balls ≥ 95 mph EV"),
        ("expanded_latent_swing_quality_shrunk","composite","Full Expanded LSQ (shrunk)"),
    ],
},

"한국어": {
    # sidebar
    "title": "스윙 퀄리티",
    "filters": "필터",
    "min_swings_lbl": "최소 경쟁 스윙 수",
    "side_lbl": "타격 방향",
    "side_all": "전체",
    "score_lbl": "기본 점수 기준",
    "score_exp": "Expanded LSQ (기본값)",
    "score_mech": "Mechanical (수축 조정)",
    "topn_lbl": "Top N (차트)",
    "footer": "데이터: MLB Statcast · 2025\n탐색적 추정치 전용.",
    # page names
    "pages": ["개요","선수 랭킹","선수 프로필",
              "파워 vs 효율","컨택 & Expanded LSQ",
              "신뢰도 & 샘플 크기","아키타입 탐색","방법론"],
    # ── Overview ──────────────────────────────────────────────────
    "ov_h": "개요",
    "ov_disc": ("배트 트래킹 메카니컬 퀄리티의 전체 분포와 샘플 크기에 따른 신뢰도 변화를 보여줍니다. "
                "경쟁 스윙이 적은 선수는 추정치가 불안정할 수 있으므로 신뢰도 조정 점수를 기본 랭킹에 사용합니다."),
    "ov_k1":"총 선수 수","ov_k2":"분석 기간","ov_k3":"중앙값 경쟁 스윙",
    "ov_k4":"중앙값 메카니컬 점수","ov_k5":"1위 선수","ov_k6":"90분위 이상",
    "ov_c1":"분포 — 메카니컬 점수 (수축 조정)",
    "ov_c2":"분포 — 경쟁 스윙 수 (샘플 크기)",
    "ov_c3":"메카니컬 점수 vs 신뢰도",
    "ov_c4":"Top 10 — 신뢰도 조정 점수",
    "ov_it": "개요 해석 방법",
    "ov_ip": [
        "<span class='hl'>점수 분포 (왼쪽)</span>: 오른쪽 꼬리가 긴 분포가 정상입니다. 대부분의 선수는 평균 근처에 모이고, 극소수의 엘리트 타자만 오른쪽 꼬리를 형성합니다. 분포가 넓을수록 리그 내 메카닉 편차가 크다는 의미입니다.",
        "<span class='hl'>경쟁 스윙 분포 (오른쪽)</span>: 점선은 현재 최소 스윙 필터입니다. 이 선 왼쪽 선수들은 랭킹에서 제외됩니다. 소수 스윙의 극단값이 랭킹을 왜곡하지 않도록 설계된 필터입니다.",
        "<span class='hl'>점수 vs 신뢰도 산점도</span>: 우하단 선수(Raw 점수 높음 + 신뢰도 낮음)는 주의 대상입니다. 스윙이 더 쌓이면 점수가 크게 변할 수 있습니다. 이상적인 위치는 우상단(점수 + 신뢰도 모두 높음)입니다.",
        "<span class='hl'>핵심 원칙</span>: 선수 비교 시 항상 Raw 점수가 아닌 Shrunk(수축 조정) 점수를 사용하세요. Shrunk 점수는 샘플 크기를 반영합니다.",
    ],
    # ── Ranking ───────────────────────────────────────────────────
    "rk_h": "선수 랭킹",
    "rk_bar": "Top {n} 선수 — {label}",
    "rk_tbl": "전체 랭킹 테이블",
    "rk_dl": "랭킹 CSV 다운로드",
    "rk_it": "랭킹 해석 방법",
    "rk_ip": [
        "<span class='hl'>Expanded LSQ (기본값)</span>: 배트 트래킹 메카닉과 컨택 결과물(xwOBA, Barrel%, Hard Hit%)을 결합합니다. 사이드바에서 Mechanical 전용으로 전환해 순수 배트 트래킹 신호만 볼 수도 있습니다.",
        "<span class='hl'>Shrunk vs Raw</span>: 항상 Shrunk 점수로 랭킹을 비교하세요. 스윙이 적은 선수는 평균 방향으로 당겨지므로 Raw 점수가 과장되어 보일 수 있습니다.",
        "<span class='hl'>신뢰도(Reliability) 컬럼</span>: 0.85 미만이면 추가 데이터에 따라 점수가 크게 바뀔 수 있습니다. 해당 선수의 순위는 참고용으로만 활용하세요.",
        "<span class='hl'>등급(Label) 기준</span>: Elite ≥ 90분위 · Strong ≥ 70분위 · Above-avg ≥ 50분위 · Below-avg < 50분위 · Low < 25분위.",
        "<span class='hl'>타격 방향 필터</span>: 백분위는 전체 타자 기준으로 계산됩니다. 좌/우타를 따로 분석할 때 Side 필터를 활용하세요.",
    ],
    # ── Profile ───────────────────────────────────────────────────
    "pf_h": "선수 프로필",
    "pf_search": "선수 검색",
    "pf_k1":"평균 배트 스피드","pf_k2":"이상적 어택 앵글 비율",
    "pf_k3":"경쟁 스윙 수","pf_k4":"메카니컬 점수 백분위",
    "pf_k5":"신뢰도","pf_k6":"Expanded LSQ 백분위",
    "pf_tbl":"전체 프로필","pf_radar":"백분위 레이더 차트","pf_bkdn":"점수 분해",
    "pf_median_lbl": "리그 중앙값",
    "pf_it": "선수 프로필 해석 방법",
    "pf_ip": [
        "<span class='hl'>레이더 차트</span>: 각 축은 전체 선수 대비 백분위(0–100)입니다. 색칠된 영역이 클수록 전반적으로 우수한 프로필입니다. 점선은 리그 중앙값(50분위)입니다.",
        "<span class='hl'>Raw Swing Power 축</span>: 평균 배트 스피드에 직접 연동됩니다. 높을수록 잠재적 Exit Velocity가 크지만, 파워만으로 품질 컨택이 보장되지는 않습니다.",
        "<span class='hl'>Swing Path Efficiency 축</span>: 이상적 어택 앵글 범위(5°–20°)에 얼마나 일관되게 스윙하는지를 반영합니다. 높을수록 유리한 발사각 조건이 반복적으로 형성됩니다.",
        "<span class='hl'>신뢰도(Reliability) 축</span>: 신뢰도가 낮으면 레이더 전체가 안쪽으로 수축됩니다. 스윙 샘플이 부족한 선수는 추정치가 안정될 때까지 레이더가 작게 보입니다.",
        "<span class='hl'>점수 분해 막대</span>: 파워와 효율의 기여 비율을 보여줍니다. 불균형(예: 파워 높고 효율 낮음)은 개선 방향을 시사합니다.",
        "<span class='hl'>xwOBA / Barrel / HardHit 축</span>: 결과물 기반 지표입니다. 메카닉이 좋은데 결과물이 낮으면 타이밍·선구안·운(BABIP)의 영향을 의심할 수 있습니다.",
    ],
    # ── Power vs Efficiency ───────────────────────────────────────
    "pe_h": "Raw 파워 vs 스윙 경로 효율",
    "pe_disc": ("각 점 = 선수 한 명. X축 = 파워(배트 스피드 z-점수), "
                "Y축 = 효율(이상적 어택 앵글 비율 z-점수). "
                "점 크기 = 경쟁 스윙 수. 색상 = 메카니컬 백분위."),
    "pe_scatter": "파워 vs 효율 사분면 산점도",
    "pe_qsum": "사분면 요약",
    "pe_qtop": "사분면별 Top 5",
    "pe_q": ["고파워 + 고효율","고파워 + 저효율",
             "저파워 + 고효율","저파워 + 저효율"],
    "pe_it": "파워 vs 효율 차트 해석 방법",
    "pe_ip": [
        "<span class='hl'>고파워 + 고효율 (우상단)</span>: 가장 이상적인 메카니컬 프로필. 배트 스피드가 높고 스윙 경로도 일관되어 타율과 장타력 모두 위협적입니다.",
        "<span class='hl'>고파워 + 저효율 (우하단)</span>: 파워는 있지만 스윙 경로가 불안정한 선수입니다. 삼진이 많지만 맞으면 강하게 치는 유형. 스윙 패스 조정 시 상승 여지가 큽니다.",
        "<span class='hl'>저파워 + 고효율 (좌상단)</span>: 컨택 중심 타자. 일관된 경로로 타율은 높지만 배트 스피드의 한계로 장타력이 제한됩니다. Tony Gwynn형 아키타입.",
        "<span class='hl'>저파워 + 저효율 (좌하단)</span>: 두 차원 모두 약한 선수. 선구안이나 운(BABIP)으로 성적을 낼 수 있지만 배트 트래킹 신호는 불리합니다.",
        "<span class='hl'>점 크기</span>: 클수록 경쟁 스윙이 많아 추정치가 더 신뢰할 수 있습니다. 극단부의 작은 점은 주의가 필요합니다.",
    ],
    # ── Contact & LSQ ─────────────────────────────────────────────
    "cl_h": "컨택 효율 & Expanded LSQ",
    "cl_disc": ("배트 트래킹 메카니컬 퀄리티가 기대 공격 가치(xwOBA, Barrel%, Hard Hit%)와 "
                "얼마나 일치하는지 평가합니다. 강한 일치는 메카니컬 점수의 유효성을 지지하며, "
                "괴리는 흥미로운 아웃라이어를 드러냅니다."),
    "cl_c1":"메카니컬 점수 vs xwOBA",
    "cl_c2":"메카니컬 점수 vs Barrel Rate",
    "cl_c3":"메카니컬 vs Expanded LSQ — 일치 / 괴리 분석",
    "cl_c4":"Expanded LSQ 랭킹 — Top {n}",
    "cl_it": "컨택 & Expanded LSQ 해석 방법",
    "cl_ip": [
        "<span class='hl'>메카니컬 점수 vs xwOBA</span>: 추세선 위 = 메카닉 예측보다 xwOBA가 높음(선구안 우위 가능성). 추세선 아래 = 좋은 메카닉이 결과물로 이어지지 않음(타이밍·구종 선택 문제 가능).",
        "<span class='hl'>메카니컬 점수 vs Barrel Rate</span>: Barrel Rate는 강한 컨택 품질의 대리 지표입니다. 메카니컬 점수 높고 Barrel Rate 낮다면 어택 앵글만으로 포착되지 않는 스윙 플레인 문제가 있을 수 있습니다.",
        "<span class='hl'>일치 / 괴리 산점도</span>: 점선 대각선이 1:1 기준선입니다. 위 = Expanded LSQ > Mech(결과물이 메카닉보다 좋음). 아래 = 결과물이 메카닉에 미치지 못함. 큰 괴리는 분석적으로 흥미로운 케이스입니다.",
        "<span class='hl'>색상 (Gap)</span>: 파란색 = 결과물이 메카닉 점수를 끌어올림. 빨간색 = 결과물이 메카닉보다 낮음. 회색 = 잘 정렬된 선수.",
        "<span class='hl'>핵심 시사점</span>: 메카니컬 점수는 선행 지표이고, 결과물 지표는 피칭 전략과 수비에 영향을 받습니다. 두 가지를 함께 보는 것이 가장 완전한 분석입니다.",
    ],
    # ── Reliability ───────────────────────────────────────────────
    "rl_h": "신뢰도 & 샘플 크기 분석",
    "rl_disc": ("경쟁 스윙 샘플이 적은 선수는 모집단 평균 방향으로 수축(shrink)됩니다. "
                "공식: Reliability = 1 − exp(−스윙 수 / 100). "
                "이 페이지는 수축 메커니즘과 주의 대상 선수를 보여줍니다."),
    "rl_c1":"신뢰도 곡선 — 경쟁 스윙 vs 신뢰도",
    "rl_c2":"수축 효과 — Raw vs Shrunk 점수",
    "rl_c3":"소샘플 + 높은 Raw 점수 — 주의 대상 선수",
    "rl_it": "신뢰도 차트 해석 방법",
    "rl_ip": [
        "<span class='hl'>신뢰도 곡선</span>: 점선이 이론적 공식(1−exp(−n/100))입니다. n=100이면 신뢰도 ≈ 0.63, n=300이면 ≈ 0.95. 곡선 위 점 = 예상보다 일관된 결과, 곡선 아래 = 더 많은 노이즈.",
        "<span class='hl'>수축 효과 산점도</span>: y=x 대각선 위의 점은 Raw 점수가 0(평균) 방향으로 당겨진 것입니다. 대각선에서 멀수록 더 많이 수축된 선수이며, 안정적인 순위를 위해 더 많은 스윙이 필요합니다.",
        "<span class='hl'>주의 대상 테이블</span>: Raw 점수는 높지만 신뢰도가 낮은 선수들입니다. 현재 높은 순위는 과장되어 있을 수 있으며, 스윙이 더 쌓이면 하락할 가능성이 있습니다.",
        "<span class='hl'>실용적 기준</span>: 신뢰도 < 0.63(약 100 스윙 미만) = 매우 불확실. 0.63–0.90 = 중간 신뢰. 0.90 이상 = 안정적 추정.",
        "<span class='hl'>Shrinkage Diff 컬럼</span>: Raw − Shrunk 차이. 0.5 이상이면 더 많은 데이터가 쌓였을 때 순위가 크게 변동할 수 있습니다.",
    ],
    # ── Archetype ─────────────────────────────────────────────────
    "ar_h": "아키타입 탐색기",
    "ar_disc": ("메카니컬 지표와 컨택 결과물을 표준화 후 K-Means 클러스터링 적용. "
                "PCA로 2차원 축약해 시각화합니다. 탐색적 그룹화이며 최종 분류가 아닙니다."),
    "ar_k": "클러스터 수 (k)",
    "ar_c1":"PCA 클러스터 시각화",
    "ar_c2":"클러스터별 평균 프로필",
    "ar_c3":"클러스터별 상위 선수",
    "ar_arch": ["파워-효율형","순수 파워형","컨택 효율형","낮은 메카닉형","혼합/평균형"],
    "ar_it": "아키타입 탐색기 해석 방법",
    "ar_ip": [
        "<span class='hl'>PCA 산점도</span>: PC1과 PC2는 6개 변수의 선형 결합입니다. 설명 분산 비율이 높을수록 이 2D 시각화가 원래 데이터를 더 잘 대표합니다. 일반적으로 70% 이상이면 양호합니다.",
        "<span class='hl'>PC1 (수평축)</span>: 보통 전반적인 스윙 퀄리티를 대표합니다. 오른쪽 선수일수록 파워와 효율 모두에서 높은 복합 점수를 가지는 경향이 있습니다.",
        "<span class='hl'>PC2 (수직축)</span>: 종종 파워-컨택 효율 트레이드오프를 포착합니다. PC2 상단과 하단 선수는 서로 다른 파워/컨택 조합을 가집니다.",
        "<span class='hl'>클러스터 레이블</span>: 각 클러스터 내 평균 지표 기준으로 자동 부여됩니다. 대략적인 아키타입으로만 활용하세요.",
        "<span class='hl'>k 조정</span>: k=3은 넓은 분류, k=5–7은 세밀한 구분에 적합합니다. PCA 차트에서 클러스터가 많이 겹치면 k를 줄이는 것이 좋습니다.",
        "<span class='hl'>기호</span>: 원=우타, X=좌타. 좌/우타 간 체계적 차이가 발견되면 분석적으로 흥미로운 패턴입니다.",
    ],
    # ── Methodology ───────────────────────────────────────────────
    "me_h": "방법론 & 해석 가이드",
    "me_disc": ("이 점수는 공개 배트 트래킹 및 컨택 품질 지표 기반 <strong>탐색적 추정치</strong>입니다. "
                "실제 타격 재능을 직접 측정한 값이 <strong>아닙니다</strong>. "
                "신뢰도 조정 점수를 기본 랭킹에 사용합니다."),
    "me_s1":"점수 정의","me_s2":"데이터 소스 & 변수",
    "me_s3":"해석 주의사항","me_s4":"향후 개선 방향","me_s5":"주요 참고문헌",
    "me_meta": {"데이터 소스":"MLB Statcast / Baseball Savant (2025)",
                "최소 기준":"경쟁 스윙 ≥ 50회 (대시보드 기본값: 100회)"},
    "me_lim": [
        "배트 스피드와 어택 앵글은 컨택 순간에 측정됩니다 — 컨택 포인트 위치에 따라 값이 달라집니다(Timing Confounding; Powers & Yurko 2026).",
        "메카니컬 점수는 스윙 결정 품질(Chase Rate, Zone Swing%)을 반영하지 않습니다.",
        "Expanded LSQ는 공개 스탯의 집계값이며 측정되지 않은 교란 변수가 존재할 수 있습니다.",
        "경쟁 스윙 100회 미만인 선수의 점수는 의미 있는 수준으로 수축 처리됩니다.",
        "2025 시즌 스냅샷이므로 이전/이후 시즌과 성과가 다를 수 있습니다.",
    ],
    "me_nxt": [
        "수축 점수의 베이지안 불확실성 구간 추가.",
        "상태 공간(State-Space) 기반 월별 모니터링 트렌드.",
        "구종별 조정 점수(패스트볼 vs 오프스피드 분리).",
        "투 스트라이크 vs 타자 유리 카운트별 스윙 퀄리티 분리.",
        "팀 단위 비교 대시보드.",
        "Swing Path Tilt를 세 번째 메카니컬 요소로 통합.",
    ],
    "me_var_h": ["변수명","단위","설명"],
    "me_var_r": [
        ("avg_bat_speed","mph","배트 헤드 6인치 기준 평균 스피드"),
        ("ideal_attack_angle_rate","%","어택 앵글 5°–20° 범위 스윙 비율"),
        ("competitive_swings","회","신뢰도 계산에 사용된 유효 스윙 수"),
        ("raw_swing_power_score","z","z(avg_bat_speed)"),
        ("swing_path_efficiency_score","z","z(ideal_attack_angle_rate)"),
        ("mechanical_swing_quality_score","복합","파워+효율 가중 평균"),
        ("mechanical_swing_quality_shrunk","복합","신뢰도 조정(수축) 메카니컬 점수"),
        ("msq_reliability","0–1","1 − exp(−n/100)"),
        ("xwoba","비율","기대 가중 출루율"),
        ("barrel_rate","%","배럴 타구 비율"),
        ("hard_hit_rate","%","EV 95 mph 이상 타구 비율"),
        ("expanded_latent_swing_quality_shrunk","복합","Expanded LSQ (신뢰도 조정)"),
    ],
},
}

# ─────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────
COLORS = {
    "primary":"#3b82f6","success":"#10b981","warning":"#f59e0b",
    "danger":"#ef4444","purple":"#8b5cf6","pink":"#ec4899",
}
LABEL_COLOR_MAP = {
    "Elite reliability-adjusted mechanical profile":"#3b82f6",
    "Strong reliability-adjusted mechanical profile":"#10b981",
    "Above-average reliability-adjusted mechanical profile":"#f59e0b",
    "Below-average reliability-adjusted mechanical profile":"#f87171",
    "Low reliability-adjusted mechanical profile":"#6b7280",
    "Elite bat-tracking mechanical profile":"#3b82f6",
    "Strong bat-tracking mechanical profile":"#10b981",
    "Elite reliability-adjusted expanded profile":"#8b5cf6",
    "Strong reliability-adjusted expanded profile":"#3b82f6",
    "Above-average reliability-adjusted expanded profile":"#10b981",
    "Below-average reliability-adjusted expanded profile":"#f59e0b",
    "Low reliability-adjusted expanded profile":"#ef4444",
    "Elite swing-quality profile":"#3b82f6",
    "Strong swing-quality profile":"#10b981",
}
PL = dict(
    paper_bgcolor="#161b27", plot_bgcolor="#161b27",
    font=dict(color="#e2e8f0", family="Inter, sans-serif"),
    margin=dict(l=20, r=20, t=50, b=20),
)

# ─────────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────────
# NOTE (Phase 4 bridge, MRP Section III.2/IV.3): `revised_mechanical_swing_quality_scores.csv`
# ("mech_df") was loaded here in the original prototype but never referenced anywhere
# in the app below - dead code, removed. Only `main_df` and `comp_df` are actually used.
#
# NOTE (path fix): Streamlit Community Cloud always runs `streamlit run` from the
# REPOSITORY ROOT, not from the folder containing this script. Since this script lives
# at `dashboard/streamlit_app.py` and the data lives at `dashboard/data/`, a plain
# relative path like "data/xxx.csv" resolves against the repo root and fails with
# FileNotFoundError. Resolving paths relative to this script's own file location
# (via __file__) makes the app work identically regardless of where it is launched from.
APP_DIR = Path(__file__).parent

@st.cache_data
def load_data():
    main = pd.read_csv(APP_DIR / "data" / "dashboard_ready_swing_quality_scores.csv")
    comp = pd.read_csv(APP_DIR / "data" / "composite_latent_swing_quality_scores.csv")
    return main, comp

main_df, comp_df = load_data()

# ─────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────
with st.sidebar:
    # ── language selector (top) ────────────────────────────────
    lang = st.radio("", ["English","한국어"], horizontal=True,
                    label_visibility="collapsed")
    badge = "English" if lang == "English" else "한국어"
    st.markdown(f"<div class='lang-tag'>{badge}</div>", unsafe_allow_html=True)
    tx = T[lang]
    st.markdown(f"## {tx['title']}")
    st.divider()

    page = st.radio("", tx["pages"], label_visibility="collapsed")
    st.divider()

    st.markdown(f"#### {tx['filters']}")
    min_swings  = st.slider(tx["min_swings_lbl"], 50, 500, 100, step=25)
    side_opts   = [tx["side_all"], "R", "L"]
    side_sel    = st.selectbox(tx["side_lbl"], side_opts)
    side_val    = "All" if side_sel == tx["side_all"] else side_sel
    score_opts  = [tx["score_exp"], tx["score_mech"]]
    score_sel   = st.selectbox(tx["score_lbl"], score_opts)
    top_n       = st.selectbox(tx["topn_lbl"], [10, 20, 50], index=1)
    st.divider()
    st.markdown(f"<div style='font-size:0.7rem;color:#64748b;line-height:1.7;'>"
                f"{tx['footer']}</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────
def apply_filters(df, min_sw, side):
    d = df[df["competitive_swings"] >= min_sw].copy()
    if side != "All":
        d = d[d["side"] == side]
    return d

def sec(text):
    st.markdown(f"<div class='section-header'>{text}</div>", unsafe_allow_html=True)

def disc(text):
    st.markdown(f"<div class='disclaimer'>{text}</div>", unsafe_allow_html=True)

def interpret(title_key, items_key):
    label = "How to Read" if lang=="English" else "해석 방법"
    with st.expander(f"{label}: {tx[title_key]}", expanded=False):
        rows = "".join(f"<li>{i}</li>" for i in tx[items_key])
        st.markdown(f"<div class='interpret-box'><ul>{rows}</ul></div>",
                    unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
# FILTER + RANK
# ─────────────────────────────────────────────────────────────────
df = apply_filters(main_df, min_swings, side_val)

rank_col   = ("expanded_latent_swing_quality_shrunk"
              if score_sel == tx["score_exp"]
              else "mechanical_swing_quality_shrunk")
rank_label = ("Expanded LSQ (Shrunk)"
              if score_sel == tx["score_exp"]
              else "Mechanical Score (Shrunk)")

df_ranked = df.sort_values(rank_col, ascending=False).reset_index(drop=True)
df_ranked.index += 1

# map displayed page name → canonical EN page name
page_idx = tx["pages"].index(page)
page_en  = T["English"]["pages"][page_idx]


# ═══════════════════════════════════════════════════════════════
# P1  OVERVIEW
# ═══════════════════════════════════════════════════════════════
if page_en == "Overview":
    st.markdown(f"## {tx['ov_h']}")
    disc(tx["ov_disc"])
    interpret("ov_it","ov_ip")

    top_p    = df_ranked.iloc[0]
    above_90 = (df_ranked["mechanical_swing_quality_shrunk_percentile"] >= 90).sum()

    k1,k2,k3,k4,k5,k6 = st.columns(6)
    k1.metric(tx["ov_k1"], f"{len(df):,}")
    k2.metric(tx["ov_k2"], f"{main_df['date_start'].iloc[0]} → {main_df['date_end'].iloc[0]}")
    k3.metric(tx["ov_k3"], f"{df['competitive_swings'].median():.0f}")
    k4.metric(tx["ov_k4"], f"{df['mechanical_swing_quality_shrunk'].median():.3f}")
    k5.metric(tx["ov_k5"], top_p["name"].split(",")[0])
    k6.metric(tx["ov_k6"], f"{above_90}")
    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        sec(tx["ov_c1"])
        fig = px.histogram(df, x="mechanical_swing_quality_shrunk", nbins=50,
                           color="mechanical_swing_quality_adjusted_label",
                           color_discrete_map=LABEL_COLOR_MAP, opacity=0.85)
        fig.update_layout(**PL, showlegend=True,
                          legend=dict(font=dict(size=9),bgcolor="#1a1d27",
                                      bordercolor="#2d3748",borderwidth=1))
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, width="stretch")

    with c2:
        sec(tx["ov_c2"])
        fig2 = px.histogram(df, x="competitive_swings", nbins=40,
                            color_discrete_sequence=[COLORS["purple"]], opacity=0.85)
        fig2.add_vline(x=min_swings, line_dash="dash", line_color=COLORS["warning"],
                       annotation_text=f"Min: {min_swings}",
                       annotation_position="top right",
                       annotation_font_color=COLORS["warning"])
        fig2.update_layout(**PL, showlegend=False)
        fig2.update_traces(marker_line_width=0)
        st.plotly_chart(fig2, width="stretch")

    sec(tx["ov_c3"])
    fig3 = px.scatter(df, x="competitive_swings", y="mechanical_swing_quality_score",
                      size="msq_reliability", color="msq_reliability",
                      color_continuous_scale=[[0,"#ef4444"],[0.5,"#f59e0b"],[1,"#3b82f6"]],
                      hover_name="name",
                      hover_data={"avg_bat_speed":":.2f","ideal_attack_angle_rate":":.3f",
                                  "competitive_swings":True,"msq_reliability":":.3f"},
                      size_max=14, opacity=0.8)
    fig3.update_layout(**PL, coloraxis_colorbar=dict(title="Reliability"))
    st.plotly_chart(fig3, width="stretch")

    sec(tx["ov_c4"])
    scols = ["name","side","avg_bat_speed","ideal_attack_angle_rate","competitive_swings",
             "mechanical_swing_quality_shrunk","mechanical_swing_quality_shrunk_percentile",
             "expanded_latent_swing_quality_shrunk"]
    st.dataframe(
        df_ranked[scols].head(10).rename(columns={
            "name":"Player","side":"B","avg_bat_speed":"Bat Speed",
            "ideal_attack_angle_rate":"Ideal AA%","competitive_swings":"Swings",
            "mechanical_swing_quality_shrunk":"Mech Score",
            "mechanical_swing_quality_shrunk_percentile":"Mech Pctile",
            "expanded_latent_swing_quality_shrunk":"Exp LSQ"})
        .style.format({"Bat Speed":"{:.1f}","Ideal AA%":"{:.3f}",
                       "Mech Score":"{:.3f}","Mech Pctile":"{:.1f}","Exp LSQ":"{:.3f}"}),
        width="stretch", height=380)


# ═══════════════════════════════════════════════════════════════
# P2  RANKING
# ═══════════════════════════════════════════════════════════════
elif page_en == "Player Ranking":
    st.markdown(f"## {tx['rk_h']}")
    interpret("rk_it","rk_ip")

    sec(tx["rk_bar"].format(n=top_n, label=rank_label))
    top_df = df_ranked.head(top_n).copy()
    top_df["lbl"] = [f"{i}. {n}" for i,n in zip(top_df.index, top_df["name"])]
    fig_b = px.bar(top_df.sort_values(rank_col), x=rank_col, y="lbl",
                   orientation="h", color=rank_col,
                   color_continuous_scale=[[0,"#2d3748"],[0.4,"#3b82f6"],[1,"#8b5cf6"]],
                   hover_name="name",
                   hover_data={"avg_bat_speed":":.1f","ideal_attack_angle_rate":":.3f",
                               "competitive_swings":True},
                   labels={rank_col:rank_label,"lbl":""})
    fig_b.update_layout(**PL, showlegend=False, coloraxis_showscale=False,
                        height=max(400,top_n*22), yaxis=dict(tickfont=dict(size=11)))
    fig_b.update_traces(marker_line_width=0)
    st.plotly_chart(fig_b, width="stretch")

    sec(tx["rk_tbl"])
    tbl = df_ranked.copy()
    tbl.insert(0,"Rank",range(1,len(tbl)+1))
    dcols = {"Rank":"Rank","name":"Player","side":"B",
             "avg_bat_speed":"Bat Speed","ideal_attack_angle_rate":"Ideal AA%",
             "competitive_swings":"Swings","msq_reliability":"Reliability",
             "mechanical_swing_quality_shrunk":"Mech Score",
             "mechanical_swing_quality_shrunk_percentile":"Mech Pctile",
             "mechanical_swing_quality_adjusted_label":"Mech Label",
             "xwoba":"xwOBA","barrel_rate":"Barrel%","hard_hit_rate":"HardHit%",
             "expanded_latent_swing_quality_shrunk":"Exp LSQ",
             "expanded_lsq_label":"Exp Label"}
    show = tbl[[c for c in dcols if c in tbl.columns]].rename(columns=dcols)
    st.dataframe(show.style.format({"Bat Speed":"{:.1f}","Ideal AA%":"{:.1%}",
                                    "Reliability":"{:.3f}","Mech Score":"{:.3f}",
                                    "Mech Pctile":"{:.1f}","xwOBA":"{:.3f}",
                                    "Barrel%":"{:.1f}","HardHit%":"{:.1f}","Exp LSQ":"{:.3f}"}),
                 width="stretch", height=550)
    csv = df_ranked.to_csv(index=False).encode("utf-8")
    st.download_button(tx["rk_dl"], csv, "swing_quality_rankings.csv","text/csv")


# ═══════════════════════════════════════════════════════════════
# P3  PLAYER PROFILE
# ═══════════════════════════════════════════════════════════════
elif page_en == "Player Profile":
    st.markdown(f"## {tx['pf_h']}")
    interpret("pf_it","pf_ip")

    all_names   = sorted(main_df["name"].tolist())
    default_idx = all_names.index("Stanton, Giancarlo") if "Stanton, Giancarlo" in all_names else 0
    search      = st.selectbox(tx["pf_search"], all_names, index=default_idx)
    player      = main_df[main_df["name"]==search].iloc[0]
    comp_player = comp_df[comp_df["name"]==search]

    mp  = player["mechanical_swing_quality_shrunk_percentile"]
    ep  = player["expanded_latent_swing_quality_shrunk_percentile"]
    tier = ("Elite" if mp>=90 else "Strong" if mp>=70 else "Above-avg" if mp>=50 else "Below-avg")

    k1,k2,k3,k4,k5,k6 = st.columns(6)
    k1.metric(tx["pf_k1"], f"{player['avg_bat_speed']:.1f} mph")
    k2.metric(tx["pf_k2"], f"{player['ideal_attack_angle_rate']:.1%}")
    k3.metric(tx["pf_k3"], f"{int(player['competitive_swings']):,}")
    k4.metric(tx["pf_k4"], f"{mp:.1f}", delta=tier)
    k5.metric(tx["pf_k5"], f"{player['msq_reliability']:.3f}")
    k6.metric(tx["pf_k6"], f"{ep:.1f}")
    st.divider()

    cl, cr = st.columns([1,1])
    with cl:
        sec(tx["pf_tbl"])
        pitems = {
            "Name":player["name"],"Side":player["side"],
            "Avg Bat Speed":f"{player['avg_bat_speed']:.2f} mph",
            "Ideal Attack Angle Rate":f"{player['ideal_attack_angle_rate']:.3f}",
            "Competitive Swings":int(player["competitive_swings"]),
            "Raw Swing Power Score":f"{player['raw_swing_power_score']:.3f}",
            "Swing Path Efficiency":f"{player['swing_path_efficiency_score']:.3f}",
            "Mechanical Score (Raw)":f"{player['mechanical_swing_quality_score']:.3f}",
            "Mechanical Score (Shrunk)":f"{player['mechanical_swing_quality_shrunk']:.3f}",
            "Mech Percentile":f"{player['mechanical_swing_quality_shrunk_percentile']:.1f}",
            "Mech Label":player["mechanical_swing_quality_adjusted_label"],
            "Reliability":f"{player['msq_reliability']:.3f}",
            "xwOBA":f"{player['xwoba']:.3f}","xBA":f"{player['xba']:.3f}",
            "xSLG":f"{player['xslg']:.3f}",
            "Avg Exit Velocity":f"{player['avg_exit_velocity']:.1f} mph",
            "Hard Hit Rate":f"{player['hard_hit_rate']:.1f}%",
            "Barrel Rate":f"{player['barrel_rate']:.1f}%",
            "Expanded LSQ (Shrunk)":f"{player['expanded_latent_swing_quality_shrunk']:.3f}",
            "Expanded LSQ Pctile":f"{player['expanded_latent_swing_quality_shrunk_percentile']:.1f}",
            "Expanded Label":player["expanded_lsq_label"],
        }
        if not comp_player.empty:
            cp = comp_player.iloc[0]
            pitems["Swing Tilt"] = f"{cp['swing_tilt']:.1f}°"
            pitems["Attack Angle"] = f"{cp['attack_angle']:.1f}°"
            pitems["Attack Direction"] = f"{cp['attack_direction']:.1f}°"
        pdf = pd.DataFrame([(k,str(v)) for k,v in pitems.items()],
                           columns=["Variable","Value"])
        st.dataframe(pdf, width="stretch", hide_index=True, height=580)

    with cr:
        sec(tx["pf_radar"])
        aqdf = main_df[main_df["competitive_swings"]>=50].copy()
        def pct(col, val):
            return float((aqdf[col]<val).mean()*100)
        axes = [("Raw Power","raw_swing_power_score"),
                ("Path Eff.","swing_path_efficiency_score"),
                ("Mech Quality","mechanical_swing_quality_shrunk"),
                ("Reliability","msq_reliability"),
                ("xwOBA","xwoba"),("Barrel%","barrel_rate"),
                ("HardHit%","hard_hit_rate"),
                ("Exp LSQ","expanded_latent_swing_quality_shrunk")]
        cats = [a[0] for a in axes]
        vals = [pct(a[1], player[a[1]]) for a in axes]
        vc = vals+[vals[0]]; cc = cats+[cats[0]]
        fig_r = go.Figure()
        fig_r.add_trace(go.Scatterpolar(r=vc, theta=cc, fill="toself",
                                         fillcolor="rgba(59,130,246,0.2)",
                                         line=dict(color="#3b82f6",width=2),
                                         name=player["name"]))
        fig_r.add_trace(go.Scatterpolar(r=[50]*len(cc), theta=cc,
                                         line=dict(color="#475569",width=1,dash="dash"),
                                         name=tx["pf_median_lbl"]))
        fig_r.update_layout(**PL,
                            polar=dict(bgcolor="#1a1d27",
                                       radialaxis=dict(visible=True,range=[0,100],
                                                       tickfont=dict(size=9,color="#64748b")),
                                       angularaxis=dict(tickfont=dict(size=10,color="#94a3b8"))),
                            showlegend=True,
                            legend=dict(x=0.85,y=1.1,bgcolor="#1a1d27",
                                        bordercolor="#2d3748",borderwidth=1),
                            height=450)
        st.plotly_chart(fig_r, width="stretch")

        sec(tx["pf_bkdn"])
        bdf = pd.DataFrame({
            "Component":["Power\n(0.50×z_bat_speed)","Path Eff.\n(0.50×z_aa)",
                         "Mech Shrunk","Exp LSQ"],
            "Value":[player["raw_swing_power_score"]*0.5,
                     player["swing_path_efficiency_score"]*0.5,
                     player["mechanical_swing_quality_shrunk"],
                     player["expanded_latent_swing_quality_shrunk"]],
            "Color":[COLORS["primary"],COLORS["success"],COLORS["purple"],COLORS["pink"]],
        })
        fig_bd = px.bar(bdf, x="Value", y="Component", orientation="h",
                        color="Component",
                        color_discrete_map={r["Component"]:r["Color"] for _,r in bdf.iterrows()})
        fig_bd.update_layout(**PL, showlegend=False, height=220)
        fig_bd.update_traces(marker_line_width=0)
        st.plotly_chart(fig_bd, width="stretch")


# ═══════════════════════════════════════════════════════════════
# P4  POWER vs EFFICIENCY
# ═══════════════════════════════════════════════════════════════
elif page_en == "Power vs Efficiency":
    st.markdown(f"## {tx['pe_h']}")
    disc(tx["pe_disc"])
    interpret("pe_it","pe_ip")

    QL = tx["pe_q"]
    sec(tx["pe_scatter"])
    fig_sc = px.scatter(df, x="raw_swing_power_score", y="swing_path_efficiency_score",
                        size="competitive_swings",
                        color="mechanical_swing_quality_shrunk_percentile",
                        color_continuous_scale=[[0,"#374151"],[0.4,"#3b82f6"],[0.7,"#8b5cf6"],[1,"#ec4899"]],
                        hover_name="name",
                        hover_data={"avg_bat_speed":":.2f","ideal_attack_angle_rate":":.3f",
                                    "competitive_swings":True,
                                    "mechanical_swing_quality_shrunk":":.3f",
                                    "mechanical_swing_quality_shrunk_percentile":":.1f"},
                        size_max=18, opacity=0.8)
    fig_sc.add_hline(y=0,line_dash="dash",line_color="#475569",line_width=1)
    fig_sc.add_vline(x=0,line_dash="dash",line_color="#475569",line_width=1)
    kw = dict(font=dict(size=9,color="#64748b"),showarrow=False)
    fig_sc.add_annotation(x=2.5, y=2.2,  text=QL[0],**kw)
    fig_sc.add_annotation(x=2.5, y=-2.4, text=QL[1],**kw)
    fig_sc.add_annotation(x=-2.5,y=2.2,  text=QL[2],**kw)
    fig_sc.add_annotation(x=-2.5,y=-2.4, text=QL[3],**kw)
    fig_sc.update_layout(**PL, height=600, coloraxis_colorbar=dict(title="Mech Pctile"))
    st.plotly_chart(fig_sc, width="stretch")

    dfq = df.copy()
    conds = [(dfq["raw_swing_power_score"]>=0)&(dfq["swing_path_efficiency_score"]>=0),
             (dfq["raw_swing_power_score"]>=0)&(dfq["swing_path_efficiency_score"]<0),
             (dfq["raw_swing_power_score"]<0)&(dfq["swing_path_efficiency_score"]>=0),
             (dfq["raw_swing_power_score"]<0)&(dfq["swing_path_efficiency_score"]<0)]
    dfq["Quadrant"] = np.select(conds, QL, default=QL[3])

    sec(tx["pe_qsum"])
    qs = (dfq.groupby("Quadrant")
          .agg(Players=("name","count"),
               Avg_Bat_Speed=("avg_bat_speed","mean"),
               Avg_Mech=("mechanical_swing_quality_shrunk","mean"),
               Avg_ExpLSQ=("expanded_latent_swing_quality_shrunk","mean"))
          .round(3).reset_index())
    st.dataframe(qs, width="stretch", hide_index=True)

    sec(tx["pe_qtop"])
    for ql in QL:
        qdf = dfq[dfq["Quadrant"]==ql].nlargest(5,"mechanical_swing_quality_shrunk")
        with st.expander(ql):
            st.dataframe(qdf[["name","side","avg_bat_speed","ideal_attack_angle_rate",
                               "competitive_swings","mechanical_swing_quality_shrunk",
                               "expanded_latent_swing_quality_shrunk"]].rename(columns={
                "name":"Player","side":"B","avg_bat_speed":"Bat Speed",
                "ideal_attack_angle_rate":"Ideal AA%","competitive_swings":"Swings",
                "mechanical_swing_quality_shrunk":"Mech Score",
                "expanded_latent_swing_quality_shrunk":"Exp LSQ"}),
                         width="stretch", hide_index=True)


# ═══════════════════════════════════════════════════════════════
# P5  CONTACT & EXPANDED LSQ
# ═══════════════════════════════════════════════════════════════
elif page_en == "Contact & Expanded LSQ":
    st.markdown(f"## {tx['cl_h']}")
    disc(tx["cl_disc"])
    interpret("cl_it","cl_ip")

    c1,c2 = st.columns(2)
    with c1:
        sec(tx["cl_c1"])
        f1 = px.scatter(df, x="mechanical_swing_quality_shrunk", y="xwoba",
                        color="mechanical_swing_quality_adjusted_label",
                        color_discrete_map=LABEL_COLOR_MAP,
                        size="competitive_swings", hover_name="name",
                        size_max=14, opacity=0.8, trendline="ols")
        f1.update_layout(**PL, showlegend=False, height=380)
        st.plotly_chart(f1, width="stretch")
    with c2:
        sec(tx["cl_c2"])
        f2 = px.scatter(df, x="mechanical_swing_quality_shrunk", y="barrel_rate",
                        color="expanded_lsq_label",
                        color_discrete_map=LABEL_COLOR_MAP,
                        size="competitive_swings", hover_name="name",
                        size_max=14, opacity=0.8, trendline="ols")
        f2.update_layout(**PL, showlegend=False, height=380)
        st.plotly_chart(f2, width="stretch")

    sec(tx["cl_c3"])
    df2 = df.copy()
    df2["gap"] = df2["expanded_latent_swing_quality_shrunk"]-df2["mechanical_swing_quality_shrunk"]
    f3 = px.scatter(df2, x="mechanical_swing_quality_shrunk",
                    y="expanded_latent_swing_quality_shrunk",
                    color="gap",
                    color_continuous_scale=[[0,"#ef4444"],[0.5,"#6b7280"],[1,"#3b82f6"]],
                    size="competitive_swings", hover_name="name",
                    hover_data={"xwoba":":.3f","barrel_rate":":.1f","gap":":.3f"},
                    size_max=14, opacity=0.8)
    mn,mx = df2["mechanical_swing_quality_shrunk"].min(), df2["mechanical_swing_quality_shrunk"].max()
    f3.add_shape(type="line",x0=mn,y0=mn,x1=mx,y1=mx,
                 line=dict(color="#475569",dash="dash",width=1))
    f3.update_layout(**PL, height=500, coloraxis_colorbar=dict(title="Exp−Mech"))
    st.plotly_chart(f3, width="stretch")

    sec(tx["cl_c4"].format(n=top_n))
    top_exp = df.nlargest(top_n,"expanded_latent_swing_quality_shrunk").sort_values("expanded_latent_swing_quality_shrunk")
    f4 = px.bar(top_exp, x="expanded_latent_swing_quality_shrunk", y="name",
                orientation="h", color="expanded_latent_swing_quality_shrunk",
                color_continuous_scale=[[0,"#2d3748"],[0.5,"#8b5cf6"],[1,"#ec4899"]],
                hover_name="name",
                hover_data={"xwoba":":.3f","barrel_rate":":.1f"})
    f4.update_layout(**PL, showlegend=False, coloraxis_showscale=False,
                     height=max(400,top_n*22))
    f4.update_traces(marker_line_width=0)
    st.plotly_chart(f4, width="stretch")


# ═══════════════════════════════════════════════════════════════
# P6  RELIABILITY
# ═══════════════════════════════════════════════════════════════
elif page_en == "Reliability & Sample Size":
    st.markdown(f"## {tx['rl_h']}")
    disc(tx["rl_disc"])
    interpret("rl_it","rl_ip")

    all_r = main_df.copy()
    c1,c2 = st.columns(2)

    with c1:
        sec(tx["rl_c1"])
        f1 = px.scatter(all_r, x="competitive_swings", y="msq_reliability",
                        color="msq_reliability",
                        color_continuous_scale=[[0,"#ef4444"],[0.5,"#f59e0b"],[1,"#10b981"]],
                        hover_name="name", opacity=0.7)
        xc = np.linspace(0, all_r["competitive_swings"].max(), 300)
        yc = 1-np.exp(-xc/100)
        f1.add_trace(go.Scatter(x=xc,y=yc,mode="lines",
                                line=dict(color="#3b82f6",width=2.5,dash="dot"),
                                name="1−exp(−n/100)"))
        f1.add_vline(x=100,line_dash="dash",line_color="#f59e0b",
                     annotation_text="n=100 (Rel≈0.63)",
                     annotation_font_color="#f59e0b",annotation_position="top right")
        f1.update_layout(**PL, height=380, coloraxis_colorbar=dict(title="Reliability"))
        st.plotly_chart(f1, width="stretch")

    with c2:
        sec(tx["rl_c2"])
        f2 = px.scatter(all_r, x="mechanical_swing_quality_score",
                        y="mechanical_swing_quality_shrunk",
                        color="competitive_swings",
                        color_continuous_scale=[[0,"#ef4444"],[0.5,"#f59e0b"],[1,"#3b82f6"]],
                        hover_name="name",
                        hover_data={"competitive_swings":True,"msq_reliability":":.3f"},
                        opacity=0.75)
        m1 = all_r["mechanical_swing_quality_score"].min()
        m2 = all_r["mechanical_swing_quality_score"].max()
        f2.add_shape(type="line",x0=m1,y0=m1,x1=m2,y1=m2,
                     line=dict(color="#475569",dash="dash",width=1))
        f2.update_layout(**PL, height=380, coloraxis_colorbar=dict(title="Swings"))
        st.plotly_chart(f2, width="stretch")

    sec(tx["rl_c3"])
    out = (all_r[all_r["msq_reliability"]<0.85]
           .nlargest(20,"mechanical_swing_quality_score").copy())
    out["diff"] = out["mechanical_swing_quality_score"]-out["mechanical_swing_quality_shrunk"]
    st.dataframe(
        out[["name","side","competitive_swings","msq_reliability",
             "mechanical_swing_quality_score","mechanical_swing_quality_shrunk","diff"]]
        .rename(columns={"name":"Player","side":"B","competitive_swings":"Swings",
                         "msq_reliability":"Reliability",
                         "mechanical_swing_quality_score":"Raw Score",
                         "mechanical_swing_quality_shrunk":"Shrunk Score",
                         "diff":"Shrinkage Diff"})
        .style.format({"Reliability":"{:.3f}","Raw Score":"{:.3f}",
                       "Shrunk Score":"{:.3f}","Shrinkage Diff":"{:.3f}"}),
        width="stretch", hide_index=True, height=420)


# ═══════════════════════════════════════════════════════════════
# P7  ARCHETYPE
# ═══════════════════════════════════════════════════════════════
elif page_en == "Archetype Explorer":
    st.markdown(f"## {tx['ar_h']}")
    disc(tx["ar_disc"])
    interpret("ar_it","ar_ip")

    k = st.slider(tx["ar_k"], 3, 7, 5)
    feats = ["raw_swing_power_score","swing_path_efficiency_score",
             "mechanical_swing_quality_shrunk","xwoba","barrel_rate","hard_hit_rate"]

    dcl = df[feats+["name","side","competitive_swings","avg_bat_speed",
                    "expanded_latent_swing_quality_shrunk"]].dropna().copy()
    X = StandardScaler().fit_transform(dcl[feats])
    dcl["Cluster"] = KMeans(n_clusters=k,random_state=42,n_init=10).fit_predict(X).astype(str)
    pca = PCA(n_components=2,random_state=42); pcs = pca.fit_transform(X)
    dcl["PC1"],dcl["PC2"] = pcs[:,0], pcs[:,1]
    ev = pca.explained_variance_ratio_
    st.info(f"PCA: PC1 {ev[0]*100:.1f}% + PC2 {ev[1]*100:.1f}% = {sum(ev)*100:.1f}%")

    sec(tx["ar_c1"])
    fp = px.scatter(dcl, x="PC1", y="PC2", color="Cluster", symbol="side",
                    hover_name="name",
                    hover_data={"raw_swing_power_score":":.3f",
                                "swing_path_efficiency_score":":.3f",
                                "xwoba":":.3f","avg_bat_speed":":.1f"},
                    color_discrete_sequence=px.colors.qualitative.Bold,
                    opacity=0.85,
                    labels={"PC1":f"PC1 ({ev[0]*100:.1f}%)","PC2":f"PC2 ({ev[1]*100:.1f}%)"})
    fp.update_layout(**PL, height=520)
    st.plotly_chart(fp, width="stretch")

    sec(tx["ar_c2"])
    cm = dcl.groupby("Cluster")[feats].mean().round(3)
    cm["n"] = dcl.groupby("Cluster")["name"].count()
    AL = tx["ar_arch"]
    arch_map = {}
    for cl, row in cm.iterrows():
        if row["raw_swing_power_score"]>0.5 and row["swing_path_efficiency_score"]>0.5:
            arch_map[cl]=AL[0]
        elif row["raw_swing_power_score"]>0.5 and row["swing_path_efficiency_score"]<=0:
            arch_map[cl]=AL[1]
        elif row["raw_swing_power_score"]<=0 and row["swing_path_efficiency_score"]>0.3:
            arch_map[cl]=AL[2]
        elif row["mechanical_swing_quality_shrunk"]<-0.5:
            arch_map[cl]=AL[3]
        else:
            arch_map[cl]=AL[4]
    cm["Archetype"] = cm.index.map(arch_map)
    st.dataframe(cm.style.format({c:"{:.3f}" for c in feats}), width="stretch")

    sec(tx["ar_c3"])
    dcl["Archetype"] = dcl["Cluster"].map(arch_map)
    for cl in sorted(dcl["Cluster"].unique()):
        arch = arch_map.get(cl,"—")
        n_cl = len(dcl[dcl["Cluster"]==cl])
        top5 = dcl[dcl["Cluster"]==cl].nlargest(5,"mechanical_swing_quality_shrunk")
        with st.expander(f"Cluster {cl} — {arch}  ({n_cl} players)"):
            st.dataframe(
                top5[["name","side","avg_bat_speed","raw_swing_power_score",
                       "swing_path_efficiency_score","xwoba","barrel_rate",
                       "expanded_latent_swing_quality_shrunk"]].rename(columns={
                    "name":"Player","side":"B","avg_bat_speed":"Bat Speed",
                    "raw_swing_power_score":"Power","swing_path_efficiency_score":"Efficiency",
                    "xwoba":"xwOBA","barrel_rate":"Barrel%",
                    "expanded_latent_swing_quality_shrunk":"Exp LSQ"}),
                width="stretch", hide_index=True)


# ═══════════════════════════════════════════════════════════════
# P8  METHODOLOGY
# ═══════════════════════════════════════════════════════════════
elif page_en == "Methodology":
    st.markdown(f"## {tx['me_h']}")
    st.markdown(f"<div class='disclaimer'>{tx['me_disc']}</div>",
                unsafe_allow_html=True)
    st.divider()

    c1,c2 = st.columns(2)
    with c1:
        st.markdown(f"### {tx['me_s1']}")
        st.markdown("**Mechanical Swing Quality Score**")
        st.markdown(
            "<div class='formula-box'>"
            "Mechanical_SQ = 0.50 × z(avg_bat_speed)<br>"
            "               + 0.50 × z(ideal_attack_angle_rate)"
            "</div>", unsafe_allow_html=True)
        if lang=="한국어":
            st.markdown("배트 스피드(파워)와 이상적 어택 앵글 비율(경로 효율)을 결합합니다. 두 지표 모두 전체 선수 기준 z-표준화됩니다.")
        else:
            st.markdown("Combines raw swing power (avg bat speed) and swing path efficiency (ideal attack angle rate). Both z-standardized across qualified players.")

        st.markdown("**Reliability Adjustment (Shrinkage)**")
        st.markdown(
            "<div class='formula-box'>"
            "Reliability = 1 − exp(−competitive_swings / 100)<br><br>"
            "Mech_SQ_Shrunk = Mech_SQ × Reliability"
            "</div>", unsafe_allow_html=True)
        if lang=="한국어":
            st.markdown("스윙이 적은 선수의 점수를 0(평균) 방향으로 수축합니다. n=100일 때 신뢰도 ≈ 0.63, n=300일 때 ≈ 0.95.")
        else:
            st.markdown("Players with fewer swings are shrunk toward 0. At n=100, reliability ≈ 0.63. At n=300, reliability ≈ 0.95.")

        st.markdown("**Expanded Latent Swing Quality Score**")
        st.markdown(
            "<div class='formula-box'>"
            "Exp_LSQ = 0.40 × z(Mech_SQ_Shrunk)<br>"
            "         + 0.25 × z(xwOBA)<br>"
            "         + 0.15 × z(barrel_rate)<br>"
            "         + 0.10 × z(hard_hit_rate)<br>"
            "         + 0.10 × z(avg_exit_velocity)"
            "</div>", unsafe_allow_html=True)
        if lang=="한국어":
            st.markdown("배트 트래킹 메카닉과 기대 공격 결과물을 통합합니다. 특정 변수가 없으면 가중치를 재정규화합니다.")
        else:
            st.markdown("Integrates bat-tracking mechanics with expected offensive outcomes. Weights renormalized if any component is unavailable.")

    with c2:
        st.markdown(f"### {tx['me_s2']}")
        date_label = "Date Range" if lang=="English" else "분석 기간"
        players_label = "Total Players" if lang=="English" else "전체 선수"
        meta = dict(tx["me_meta"])
        meta[date_label] = f"{main_df['date_start'].iloc[0]} → {main_df['date_end'].iloc[0]}"
        meta[players_label] = f"{len(main_df):,} (all) / {len(df):,} (filtered)"
        for k,v in meta.items():
            st.markdown(f"**{k}:** {v}")
        var_df = pd.DataFrame(tx["me_var_r"], columns=tx["me_var_h"])
        st.dataframe(var_df, width="stretch", hide_index=True, height=400)

    st.divider()
    c3,c4 = st.columns(2)
    with c3:
        st.markdown(f"### {tx['me_s3']}")
        for i,item in enumerate(tx["me_lim"],1):
            st.markdown(f"**{i}.** {item}")
    with c4:
        st.markdown(f"### {tx['me_s4']}")
        for i,item in enumerate(tx["me_nxt"],1):
            st.markdown(f"**{i}.** {item}")

    st.divider()
    st.markdown(f"### {tx['me_s5']}")
    # NOTE (Phase 4 bridge, MRP Section III.2/IV.3): replaced the original reference list,
    # which did not match the MRP's confirmed 11-source Literature Review, with the
    # subset that directly justifies this dashboard's own formulas and design choices.
    refs = [("Powers & Yurko (2026)","Swinging, Fast and Slow: Bat-Tracking Data and the Evaluation of Hitters","arXiv:2507.01238"),
            ("Efron & Morris (1973)","Stein's Estimation Rule and Its Competitors","J. Amer. Statist. Assoc."),
            ("Hopkins (2000)","Measures of Reliability in Sports Medicine and Science","Sports Medicine"),
            ("OECD & JRC-European Commission (2008)","Handbook on Constructing Composite Indicators","OECD Publishing"),
            ("Du & Yuan (2021)","A Survey of Competitive Sports Data Visualization and Visual Analysis","J. Visualization")]
    st.dataframe(pd.DataFrame(refs, columns=["Citation","Title","Source"]),
                 width="stretch", hide_index=True)
    note = ("Full reference list (11 sources) available in the accompanying MRP Literature Review, Chapter 2."
            if lang == "English" else
            "전체 참고문헌 목록(11편)은 본 MRP 문헌 고찰(Chapter 2)에서 확인 가능합니다.")
    st.caption(note)
    st.markdown("---")
    st.markdown("<div style='font-size:0.75rem;color:#475569;text-align:center;'>"
                "Bat-Tracking Swing Quality Dashboard · 2025 · Exploratory research tool"
                "</div>", unsafe_allow_html=True)