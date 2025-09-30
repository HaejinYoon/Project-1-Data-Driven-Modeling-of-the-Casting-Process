import pandas as pd
import joblib
import matplotlib.pyplot as plt
from shiny import App, ui, render, reactive
from shiny.ui import update_slider, update_numeric, update_select, update_navs
import seaborn as sns
import pathlib
import plotly.express as px
from shinywidgets import render_plotly, output_widget
import plotly.express as px
import numpy as np
import matplotlib

matplotlib.use("Agg")   # Tkinter 대신 Agg backend 사용 (GUI 필요 없음)

app_dir = pathlib.Path(__file__).parent
# ===== 한글 깨짐 방지 설정 =====
plt.rcParams["font.family"] = "Malgun Gothic"   # 윈도우: 맑은 고딕
plt.rcParams["axes.unicode_minus"] = False      # 마이너스 기호 깨짐 방지

# ===== 모델 불러오기 =====
MODEL_PATH = "./models/model_2.pkl"
model = joblib.load(MODEL_PATH)

# ===== 데이터 불러오기 =====
df_raw = pd.read_csv("./data/train_raw.csv")

# ★ 특정 이상치 행 제거
df_raw = df_raw[
    (df_raw["low_section_speed"] != 65535) &
    (df_raw["lower_mold_temp3"] != 65503) &
    (df_raw["physical_strength"] != 65535)
]

# 예측용 데이터도 동일 처리
df_predict = pd.read_csv("./data/train.csv")
df_predict["pressure_speed_ratio"] = df_predict["pressure_speed_ratio"].replace([np.inf, -np.inf], np.nan)


# 예측 탭용 (모델 input 그대로)
df_predict = pd.read_csv("./data/train.csv")
df_predict["pressure_speed_ratio"] = df_predict["pressure_speed_ratio"].replace([np.inf, -np.inf], np.nan)

df_predict = df_predict[
    (df_predict["low_section_speed"] != 65535) &
    (df_predict["lower_mold_temp3"] != 65503) &
    (df_predict["physical_strength"] != 65535)
]

# 탐색 탭용 (필터링/EDA)
drop_cols_explore = ["id","line","name","mold_name","date","time", "registration_time", "passorfail"]
df_explore = df_raw.drop(columns=drop_cols_explore)

# 예측에서 제외할 컬럼
drop_cols = [
    "real_time",   # registration_time → real_time
    "passorfail",
    # "count",
    # "global_count",
    # "monthly_count",
    # "speed_ratio",
	# "pressure_speed_ratio",
    # "shift",
]
used_columns = df_predict.drop(columns=drop_cols).columns

# 그룹 분류
cat_cols = ["mold_code","working","emergency_stop","heating_furnace", "shift", "tryshot_signal"]
num_cols = [c for c in used_columns if c not in cat_cols]

# ===== 라벨 맵 =====
label_map = {
    "count": "일조 생산 수",
    "monthly_count": "월 생산 수",
    "global_count": "총 누적 생산 수",
    "working": "작동 여부",
    "emergency_stop": "비상 정지",
    "molten_temp": "용탕 온도",
    "facility_operation_cycleTime": "설비 작동 사이클타임",
    "production_cycletime": "생산 사이클타임",
    "low_section_speed": "하위 구간 주입 속도",
    "high_section_speed": "상위 구간 주입 속도",
    "molten_volume": "주입한 금속 양",
    "cast_pressure": "주입 압력",
    "biscuit_thickness": "비스킷 두께",
    "upper_mold_temp1": "상부금형1 온도",
    "upper_mold_temp2": "상부금형2 온도",
    "upper_mold_temp3": "상부금형3 온도",
    "lower_mold_temp1": "하부금형1 온도",
    "lower_mold_temp2": "하부금형2 온도",
    "lower_mold_temp3": "하부금형3 온도",
    "sleeve_temperature": "주입 관 온도",
    "physical_strength": "물리적 강도",
    "Coolant_temperature": "냉각수 온도",
    "EMS_operation_time": "EMS 작동 시간",
    "mold_code": "금형 코드",
    "heating_furnace": "가열로",
    "shift": "주, 야간 조",
    "tryshot_signal": "시험 가동 여부"
}

# ===== 라벨 정의 (표시 텍스트 = 한글, 실제 var = 변수명) =====
labels = [
    {"id": "label1", "text": label_map["upper_mold_temp1"], "var": "upper_mold_temp1",
     "x": 200, "y": 50, "w": 120, "h": 30,
     "arrow_from": (200+60, 80), "arrow_to": (400, 160)},  # 아랫변 중앙

    {"id": "label2", "text": label_map["lower_mold_temp1"], "var": "lower_mold_temp1",
     "x": 650, "y": 50, "w": 120, "h": 30,
     "arrow_from": (650+60, 80), "arrow_to": (580, 160)},  # 아랫변 중앙

    {"id": "label3", "text": label_map["cast_pressure"], "var": "cast_pressure",
     "x": 900, "y": 250, "w": 100, "h": 30,
     "arrow_from": (900+50, 280), "arrow_to": (780, 360)},  # 아랫변 중앙

    {"id": "label4", "text": label_map["molten_volume"], "var": "molten_volume",
     "x": 700, "y": 150, "w": 120, "h": 30,
     "arrow_from": (700+60, 180), "arrow_to": (780, 280)},  # 아랫변 중앙

    {"id": "label5", "text": label_map["sleeve_temperature"], "var": "sleeve_temperature",
     "x": 670, "y": 400, "w": 120, "h": 30,
     "arrow_from": (670+60, 400), "arrow_to": (600, 360)},  # 윗변 중앙

    {"id": "label6", "text": label_map["high_section_speed"], "var": "high_section_speed",
     "x": 400, "y": 70, "w": 160, "h": 30,
     "arrow_from": (400+80, 100), "arrow_to": (510, 180)},  # 아랫변 중앙

    {"id": "label7", "text": label_map["low_section_speed"], "var": "low_section_speed",
     "x": 400, "y": 420, "w": 160, "h": 30,
     "arrow_from": (400+80, 420), "arrow_to": (510, 320)},  # 윗변 중앙
]

def get_label(col): return label_map.get(col, col)

# ===== Helper: 슬라이더 + 인풋 =====
def make_num_slider(col):
    return ui.div(
        ui.input_slider(
            f"{col}_slider", get_label(col),
            min=int(df_predict[col].min()), max=int(df_predict[col].max()),
            value=int(df_predict[col].mean()), width="100%"
        ),
        ui.input_numeric(col, "", value=int(df_predict[col].mean()), width="110px"),
        style="display: flex; align-items: center; gap: 8px; justify-content: space-between;"
    )

# ===== 범주형 없음도 추가 ========
def make_select(col, label=None, width="100%"):
    label = label if label else get_label(col)
    if(col == "tryshot_signal"):
        choices = ["없음"] + sorted(df_predict[col].dropna().unique().astype(str))
    else:
        choices = sorted(df_predict[col].dropna().unique().astype(str)) + ["없음"]
    return ui.input_select(col, label, choices=choices, width=width)

def make_svg(labels):
    parts = []
    for lbl in labels:
        # 화살표 시작점: arrow_from 있으면 사용, 없으면 중앙
        if "arrow_from" in lbl:
            cx, cy = lbl["arrow_from"]
        else:
            cx = lbl["x"] + lbl["w"]/2
            cy = lbl["y"] + lbl["h"]/2

        x2, y2 = lbl["arrow_to"]
        text = label_map.get(lbl["var"], lbl["var"])

        parts.append(f"""
        <g>
        <rect x="{lbl['x']}" y="{lbl['y']}" width="{lbl['w']}" height="{lbl['h']}" 
                fill="#e0e6ef" stroke="black"/>
        <text x="{lbl['x'] + lbl['w']/2}" y="{lbl['y'] + lbl['h']/2}" 
                fill="black" font-size="14" font-weight="bold"
                text-anchor="middle" dominant-baseline="middle">{text}</text>
        <line x1="{cx}" y1="{cy}" x2="{x2}" y2="{y2}" 
                stroke="red" marker-end="url(#arrow)"/>
        </g>
        """)
    return "\n".join(parts)

svg_code = f"""
<svg width="1000" height="500" xmlns="http://www.w3.org/2000/svg">
  <image href="die-castings.gif" width="1000" height="500"/>
  <defs>
    <marker id="arrow" markerWidth="10" markerHeight="10" refX="6" refY="3" orient="auto">
      <path d="M0,0 L0,6 L6,3 z" fill="red"/>
    </marker>
  </defs>
  {make_svg(labels)}
</svg>
"""

# ===== CSS (카드 전체 클릭영역) =====
card_click_css = """
<style>
/* 개요 전용 카드만 hover 효과 */
.overview-card {
    transition: transform 0.2s ease, box-shadow 0.2s ease, background-color 0.2s ease;
    position: relative;
}

.overview-card:hover {
    background-color: #f8f9fa !important;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    transform: translateY(-2px);
}

/* 카드 전체를 클릭 가능하게 하는 투명 버튼 */
.card-link {
    position: absolute;
    inset: 0;
    z-index: 10;
    cursor: pointer;
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
}
.card-link:hover,
.card-link:focus,
.card-link:active {
    background: transparent !important;
    box-shadow: none !important;
}
</style>
"""

# ===== UI =====
app_ui = ui.page_fluid(
    ui.head_content(
        ui.tags.link(rel="icon", type="image/x-icon", href="favicon.ico"),
        ui.tags.link(rel="icon", type="image/png", sizes="32x32", href="favicon-32.png"),
        ui.tags.link(rel="apple-touch-icon", sizes="180x180", href="apple-touch-icon.png"),
        ui.tags.link(rel="icon", type="image/png", sizes="192x192", href="icon-192.png"),
        ui.tags.link(rel="icon", type="image/png", sizes="512x512", href="icon-512.png"),
        # Font Awesome 아이콘 불러오기
        ui.tags.link(
            rel="stylesheet",
            href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css"
        ),
    ),
    ui.h2("주조 공정 불량 예측 대시보드", style="text-align:center;"),

    ui.navset_tab(
        # 1. Overview
        # ===== 네비게이션 탭 =====
        ui.nav_panel("개요",
            ui.HTML(card_click_css),
            ui.layout_columns(
                ui.card(
                    {"class": "overview-card"},
                    ui.card_header("데이터 탐색"),
                    "📊 데이터 확인",
                    ui.input_action_button("go_explore", "", class_="card-link")
                ),
                ui.card(
                    {"class": "overview-card"},
                    ui.card_header("예측"),
                    "🤖 모델 예측",
                    ui.input_action_button("go_predict", "", class_="card-link")
                ),
                ui.card(
                    {"class": "overview-card"},
                    ui.card_header("모델링"),
                    "⚙️ 모델 학습",
                    ui.input_action_button("go_model", "", class_="card-link")
                ),
            )
        ),

        # 2. 데이터 탐색 (EDA)
        ui.nav_panel(
            "데이터 탐색",
            ui.navset_tab(
                ui.nav_panel(
                    "개요",
                        # -------------------- 상단 SVG + 버튼 --------------------
                        ui.div(
                            {"style": "position: relative; display:flex; justify-content:center;"},
                            ui.HTML(svg_code),
                            *[
                                ui.input_action_button(
                                    f"btn_{lbl['id']}", "",
                                    style=f"""
                                        position:absolute;
                                        top:{lbl['y']}px; left:calc(50% - 500px + {lbl['x']}px);
                                        width:{lbl['w']}px; height:{lbl['h']}px;
                                        opacity:0; cursor:pointer;
                                    """
                                )
                                for lbl in labels
                            ]
                        ),

                        # -------------------- JS 코드 삽입 --------------------
                        ui.tags.script("""
                            Shiny.addCustomMessageHandler("switch_tab_with_label", function(msg) {
                                let tabs = document.querySelectorAll('.nav-link');
                                tabs.forEach(function(tab) {
                                    if (tab.textContent.trim() === msg.tab) {
                                        tab.click();
                                    }
                                });
                            });
                        """),

                        # -------------------- 설명 영역 --------------------
                        ui.div(
                            {
                                "style": """
                                    margin-top:30px; 
                                    padding:20px; 
                                    border:1px solid #ddd; 
                                    border-radius:10px; 
                                    background:#fafafa;
                                """
                            },
                            ui.markdown("""

                                ### [주조 공정]

                                주조(Casting)는 금속을 녹여 원하는 형상을 만드는 제조 공정입니다.
                                고체 상태의 금속을 고온에서 녹여 액체 상태로 만든 뒤, 미리 준비된 금형에 부어 응고시키면 제품 형태가 완성됩니다.

                                주조 공정은 복잡한 형상, 대량 생산, 재료 절감이 가능하여 자동차, 기계 부품 등 다양한 산업 분야에서 널리 활용됩니다.


                                주요 목적:
                                - 금속을 원하는 형상과 치수로 성형
                                - 기계적 강도와 품질 확보
                                - 공정 효율 및 생산성 향상

                                ---

                                ### [다이캐스팅 공정]

                                다이캐스팅(Die Casting)은 고압을 이용해 용융 금속을 금형 내로 빠르게 주입하여 복잡한 형상을 가진 금속 부품을 고속으로 생산하는 공정입니다.

                                정밀한 치수, 매끄러운 표면, 높은 생산성을 달성할 수 있는 것이 특징입니다.

                                ---

                                ### [주조 공정 단계]

                                1. 용융 단계 (Melting)
                                - 금속을 고온에서 녹이는 과정입니다.
                                - 용해로를 통해 일정 온도로 금속을 유지하며, 주입 가능한 액체 상태를 만듭니다.
                                - 이 단계에서 금속의 균질성과 온도 관리가 매우 중요합니다.
                                                    

                                2. 충진 단계 (Filling)
                                - 녹인 금속을 금형 내부로 주입하는 단계입니다.
                                - 주입 속도, 주입 압력, 금형 설계에 따라 내부 충진 상태가 달라지고,
                                    제품 내부 결함(공극, 불균질 등)에 영향을 줍니다.
                                - 일부 공정에서는 전자 교반(EMS)을 통해 금속 혼합을 개선하기도 합니다.
                                                    

                                3. 냉각 단계 (Cooling)
                                - 주입된 금속이 금형 내에서 응고되는 단계입니다.
                                - 금속의 냉각 속도와 금형 온도를 적절히 제어해야 수축, 변형, 내부 응력 등을 최소화할 수 있습니다.
                                - 냉각수와 금형 온도 관리가 주요 역할을 합니다.
                                                    

                                4. 공정 속도 및 장비 운전
                                - 장비 사이클 시간과 실제 생산 속도는 공정 효율과 품질 안정성에 직결됩니다.
                                - 장비 가동 상태, 비상 정지 여부 등을 관리하며 생산 계획에 따라 운용됩니다.
                                                    

                                5. 품질 평가 (Inspection)
                                - 최종 주조물은 두께, 강도 등 물리적 특성을 평가합니다.
                                - 합격/불합격(pass/fail) 여부를 결정하며, 이를 기반으로 공정 최적화와 품질 개선을 수행합니다.
                                ---

                                ### [요약]

                                본 데이터 분석에서는 위와 같은 공정 단계별 데이터를 활용하여, 주조 조건(온도, 속도, 금형, 가열로 등)이 최종 양품/불량(passorfail)에
                                어떤 영향을 주는지 탐색하고 시각화하였습니다.
                                이를 통해 주조 공정의 주요 인자들을 이해하고, 품질 개선 및 불량 감소에 기여할 수 있는 근거를 마련할 수 있습니다.


                                """)
                                )    
                ),
                ui.nav_panel("그래프",
                    ui.layout_sidebar(
                        ui.sidebar(
                            #분포 필터
                            ui.div(
                                f"데이터 분포 그래프 필터",
                                style="background-color:#e9ecef; padding:8px 12px; border-radius:6px; text-align:center; font-weight:bold;"
                            ),
                            ui.input_selectize(
                                "mold_code2",
                                "Mold Code 선택",
                                choices=list(map(str, sorted(df_explore["mold_code"].dropna().unique())))
                            ),
                            ui.input_select(
                                "var",
                                "분석 변수 선택",
                                choices={c: get_label(c) for c in df_explore.columns if c not in ["mold_code", "passorfail"]}
                            ),
                            ui.output_ui("filter_ui"),   # ★ 선택된 변수에 맞는 필터 UI
                        ),
                        ui.card(
                            ui.card_header("데이터 분포"),
                                ui.output_plot("dist_plot"),
                        ),
                    ),
                    ui.layout_sidebar(
                        ui.sidebar(
                            # 시계열 필터
                            ui.div(
                                f"시계열 데이터 필터",
                                style="background-color:#e9ecef; padding:8px 12px; border-radius:6px; text-align:center; font-weight:bold;"
                            ),
                            ui.input_select(
                                "ts_var", "Y축 변수 선택",
                                choices={c: get_label(c) for c in df_explore.columns if c not in ["id","line","name","mold_name","date","time", "registration_time", "passorfail"]}
                                # choices=[c for c in df_raw.columns if c not in ["id","line","name","mold_name","date","time", "registration_time"]]
                            ),
                            ui.output_ui("ts_filter_ui")   # 시계열 전용 시간 필터
                        ),
                        ui.card(
                            ui.card_header("시계열 데이터"),
                                output_widget("timeseries_plot")
                        ),
                    ),
                    ui.layout_columns(
                        # 1행
                        ui.card(
                            ui.card_header("데이터 요약"),
                            ui.output_table("df_summary"),
                        ),
                        ui.card(
                            ui.card_header("컬럼별 결측치 비율"),
                            ui.output_plot("missing_plot"),
                        ),
                        # 2행
                        ui.card(
                            ui.card_header("변수 타입 분포"),
                            ui.output_plot("dtype_pie"),
                        ),
                        ui.card(
                            ui.card_header("수치형 변수 상관관계"),
                            ui.output_plot("corr_heatmap_overview"),
                        ),
                        col_widths=[6, 6],  # 2열 레이아웃
                    ),
                )
            )
        ),

        # # 3. 전처리 과정
        # ui.nav_panel(
        #     "전처리",
        #     # ui.card(ui.card_header("결측치 처리 전/후 비교"), ui.output_plot("preprocess_plot")),
        #     # ui.card(ui.card_header("이상치 처리 결과"), ui.output_plot("outlier_plot"))
        # ),

        # 4. 모델 학습
        ui.nav_panel(
            "모델 학습",
            # ui.card(ui.card_header("변수 중요도"), ui.output_plot("feature_importance_plot")),
            # ui.card(ui.card_header("모델 성능"), ui.output_plot("model_eval_plot"))
        ),

        # 5. 예측
        ui.nav_panel(
            "예측",
            ui.navset_tab(
                ui.nav_panel("예측",
                    # 입력 변수 카드
                    ui.div(
                        ui.card(
                            ui.card_header("입력 변수", style="background-color:#f8f9fa; text-align:center;"),
                            ui.card_body(
                                # 생산 환경 정보 카드 (최상단)
                                ui.card(
                                    ui.card_header("생산 환경 정보", style="background-color:#f8f9fa; text-align:center;"),
                                    ui.card_body(
                                        ui.layout_columns(
                                            ui.div(
                                                f"생산 라인: {df_raw['line'].iloc[0]}",
                                                style="background-color:#e9ecef; padding:8px 12px; border-radius:6px; text-align:center; font-weight:bold;"
                                            ),
                                            ui.div(
                                                f"장비 이름: {df_raw['name'].iloc[0]}",
                                                style="background-color:#e9ecef; padding:8px 12px; border-radius:6px; text-align:center; font-weight:bold;"
                                            ),
                                            ui.div(
                                                f"금형 이름: {df_raw['mold_name'].iloc[0]}",
                                                style="background-color:#e9ecef; padding:8px 12px; border-radius:6px; text-align:center; font-weight:bold;"
                                            ),
                                            col_widths=[4,4,4]
                                        )
                                    )
                                ),

                                # === 공정 상태 관련 (4열) ===
                                ui.card(
                                    ui.card_header("공정 상태 관련", style="background-color:#f8f9fa;"),
                                    ui.card_body(
                                        ui.layout_columns(
                                            ui.input_numeric("count", "일조 누적 제품 개수", value=int(df_predict["count"].mean())),
                                            ui.input_numeric("monthly_count", "월간 누적 제품 개수", value=int(df_predict["monthly_count"].mean())),
                                            ui.input_numeric("global_count", "전체 누적 제품 개수", value=int(df_predict["global_count"].mean())),
                                            ui.input_numeric("speed_ratio", "상하 구역 속도 비율", value=int(df_predict["speed_ratio"].mean())),
                                            ui.input_numeric("pressure_speed_ratio", "주조 압력 속도 비율", value=int(df_predict["pressure_speed_ratio"].mean())),
                                            make_select("working", "장비 가동 여부"),
                                            make_select("emergency_stop", "비상 정지 여부"),
                                            make_select("tryshot_signal", "측정 딜레이 여부"),
                                            make_select("shift", "주, 야간 조"),
                                            col_widths=[3,3,3,3]
                                        )
                                    )
                                ),

                                # === 용융 단계 (n행 4열) ===
                                ui.card(
                                    ui.card_header("용융 단계", style="background-color:#f8f9fa;"),
                                    ui.card_body(
                                        ui.layout_columns(
                                            make_num_slider("molten_temp"),
                                            make_select("heating_furnace", "용해로"),
                                            col_widths=[6,6]
                                        )
                                    )
                                ),

                                # === 충진 단계 (n행 4열) ===
                                ui.card(
                                    ui.card_header("충진 단계", style="background-color:#f8f9fa;"),
                                    ui.card_body(
                                        ui.layout_columns(
                                            make_num_slider("sleeve_temperature"),
                                            make_num_slider("EMS_operation_time"),
                                            make_num_slider("low_section_speed"),
                                            make_num_slider("high_section_speed"),
                                            make_num_slider("molten_volume"),
                                            make_num_slider("cast_pressure"),
                                            ui.input_select("mold_code", "금형 코드", choices=sorted(df_predict["mold_code"].dropna().unique().astype(str))),
                                            col_widths=[3,3,3,3]
                                        )
                                    )
                                ),

                                # === 냉각 단계 (n행 4열) ===
                                ui.card(
                                    ui.card_header("냉각 단계", style="background-color:#f8f9fa;"),
                                    ui.card_body(
                                        ui.layout_columns(
                                            make_num_slider("upper_mold_temp1"),
                                            make_num_slider("upper_mold_temp2"),
                                            make_num_slider("upper_mold_temp3"),
                                            make_num_slider("lower_mold_temp1"),
                                            make_num_slider("lower_mold_temp2"),
                                            make_num_slider("lower_mold_temp3"),
                                            make_num_slider("Coolant_temperature"),
                                            col_widths=[3,3,3,3]
                                        )
                                    )
                                ),

                                # === 공정 속도 관련 (n행 4열) ===
                                ui.card(
                                    ui.card_header("공정 속도 관련", style="background-color:#f8f9fa;"),
                                    ui.card_body(
                                        ui.layout_columns(
                                            make_num_slider("facility_operation_cycleTime"),
                                            make_num_slider("production_cycletime"),
                                            col_widths=[6,6]
                                        )
                                    )
                                ),

                                # === 품질 및 성능 (n행 4열) ===
                                ui.card(
                                    ui.card_header("품질 및 성능", style="background-color:#f8f9fa;"),
                                    ui.card_body(
                                        ui.layout_columns(
                                            make_num_slider("biscuit_thickness"),
                                            make_num_slider("physical_strength"),
                                            col_widths=[6,6]
                                        )
                                    )
                                )
                            )
                        ),
                        style="max-width: 1200px; margin: 0 auto;"
                    ),

                    ui.br(),

                    # 예측 실행 + 결과 카드 (sticky)
                    ui.div(
                        ui.card(
                            ui.card_header(
                                ui.div(
                                    [
                                        ui.input_action_button(
                                            "predict_btn", "예측 실행",
                                            class_="btn btn-primary btn-lg",
                                            style="flex:1;"
                                        ),
                                        ui.input_action_button(
                                            "reset_btn", ui.HTML('<i class="fa-solid fa-rotate-left"></i>'),
                                            class_="btn btn-secondary btn-lg",
                                            style="margin-left:10px; width:60px;"
                                        )
                                    ],
                                    style="display:flex; align-items:center; width:100%;"
                                ),
                                style="text-align:center; background-color:#f8f9fa;"
                            ),
                            ui.card_body(
                                ui.output_ui("prediction_result")
                            )
                        ),
                        style="""
                            position: -webkit-sticky;
                            position: sticky;
                            bottom: 1px;
                            z-index: 1000;
                            max-width: 1200px;
                            margin: 0 auto;
                        """
                    ),

                    ui.hr(),

                    # 분석 시각화 카드
                    ui.card(
                        ui.card_header("분석 시각화", style="background-color:#f8f9fa; text-align:center;"),
                        ui.card_body(
                            ui.navset_tab(
                                ui.nav_panel("변수 중요도", ui.output_plot("feature_importance_plot")),
                                ui.nav_panel("분포 비교", ui.output_plot("distribution_plot")),
                                ui.nav_panel("공정별 불량률", ui.output_plot("process_trend_plot"))
                            )
                        )
                    )
                                        ),
                ui.nav_panel("개선",
                ),
            )
        ),
        id="main_nav",   # ⭐ 탭 컨트롤을 위한 id
    )
)


# ===== SERVER (변경 없음) =====
def server(input, output, session):
    #====== 개요에서 카드 클릭 시 탭이동 =================================
    @reactive.Effect
    @reactive.event(input.go_explore)
    def _():
        update_navs("main_nav", selected="데이터 탐색")

    @reactive.Effect
    @reactive.event(input.go_predict)
    def _():
        update_navs("main_nav", selected="예측")

    @reactive.Effect
    @reactive.event(input.go_model)
    def _():
        update_navs("main_nav", selected="모델 학습")
    #=================================================================

    # 서버 함수 안에
    @reactive.effect
    @reactive.event(input.goto_explore)
    def _():
        ui.update_navs("main_tabs", selected="데이터 탐색")

    @reactive.effect
    @reactive.event(input.goto_preprocess)
    def _():
        ui.update_navs("main_tabs", selected="전처리")

    @reactive.effect
    @reactive.event(input.goto_train)
    def _():
        ui.update_navs("main_tabs", selected="모델 학습")

    @reactive.effect
    @reactive.event(input.goto_predict)
    def _():
        ui.update_navs("main_tabs", selected="예측")

    # 버튼 클릭 시 탭 전환
    for lbl in labels:
        @reactive.Effect
        @reactive.event(getattr(input, f"btn_{lbl['id']}"))
        async def _(lbl=lbl):
            # 1️⃣ 탭 전환 (비동기 → await 필요)
            await session.send_custom_message("switch_tab_with_label", {
                "tab": "그래프",
                "label": lbl["var"]
            })
            # 2️⃣ 드롭다운 선택값 업데이트 (동기 → await 쓰면 안됨)
            session.send_input_message(
                "var",
                update_select("var", selected=lbl["var"])
            )
            session.send_input_message(
                "ts_var",
                update_select("ts_var", selected=lbl["var"])
            )

    @output
    @render.text
    def selected_var():
        return f"현재 선택된 변수: {input.var() or '없음'}"

    last_proba = reactive.value(None)
    loading = reactive.value(False)

    def get_input_data():
        data = {}
        for col in cat_cols + num_cols:
            data[col] = [input[col]()]

        return pd.DataFrame(data)

    for col in num_cols:
        @reactive.effect
        @reactive.event(input[col])
        def _(col=col):
            update_slider(f"{col}_slider", value=input[col]())
        @reactive.effect
        @reactive.event(input[f"{col}_slider"])
        def _(col=col):
            update_numeric(col, value=input[f"{col}_slider"]())

    @reactive.effect
    @reactive.event(input.reset_btn)
    def _():
        # 범주형 변수: 첫 번째 값으로 초기화
        for col in cat_cols:
            first_val = str(sorted(df_predict[col].dropna().unique())[0])
            if(col == "tryshot_signal"):
                first_val = "없음"
            ui.update_select(col, selected=first_val)

        # 수치형 변수: 안전하게 숫자 변환 후 평균값으로 초기화
        for col in num_cols:
            series = pd.to_numeric(df_predict[col], errors="coerce")       # 문자열 → 숫자 (에러시 NaN)
            series = series.replace([np.inf, -np.inf], np.nan)             # inf → NaN
            mean_val = series.dropna().mean()                              # NaN 제거 후 평균
            default_val = int(mean_val) if pd.notna(mean_val) else 0       # fallback: 0
            update_slider(f"{col}_slider", value=default_val)
            update_numeric(col, value=default_val)

        # 예측 결과 초기화
        last_proba.set(None)

    @reactive.effect
    @reactive.event(input.predict_btn)
    def _():
        loading.set(True)
        try:
            X = get_input_data()
            proba = model.predict_proba(X)[0,1]
            last_proba.set(proba)
        except Exception as e:
            last_proba.set(f"error:{e}")
        finally:
            loading.set(False)

    @render.ui
    def prediction_result():
        if loading():
            return ui.div(
                ui.div(class_="spinner-border text-primary", role="status"),
                ui.HTML("<div style='margin-top:10px;'>예측 실행 중...</div>"),
                style="text-align:center; padding:20px;"
            )

        proba = last_proba()
        if proba is None:
            return ui.div(
                ui.HTML("<span style='color:gray; font-size:18px;'>아직 예측을 실행하지 않았습니다.</span>"),
                style="text-align:center; padding:20px;"
            )

        if isinstance(proba, str) and proba.startswith("error:"):
            return ui.div(
                ui.HTML(f"<span style='color:red;'>예측 중 오류 발생: {proba[6:]}</span>")
            )

        if proba < 0.02:
            style = "background-color:#d4edda; color:#155724; font-size:18px; font-weight:bold; padding:15px; text-align:center; border-radius:8px;"
        elif proba < 0.04:
            style = "background-color:#fff3cd; color:#856404; font-size:18px; font-weight:bold; padding:15px; text-align:center; border-radius:8px;"
        else:
            style = "background-color:#f8d7da; color:#721c24; font-size:18px; font-weight:bold; padding:15px; text-align:center; border-radius:8px;"

        judgment = "불량품" if proba >= 0.2 else "양품"

        return ui.div(
            [
                ui.HTML(f"예상 불량률: {proba*100:.2f}%"),
                ui.br(),
                ui.HTML(f"최종 판정: <span style='font-size:22px;'>{judgment}</span>")
            ],
            style=style
        )

    @render.plot
    def feature_importance_plot():
        try:
            importances = model.named_steps["model"].feature_importances_
            feat_names = model.named_steps["preprocessor"].get_feature_names_out()
            imp_df = pd.DataFrame({"feature": feat_names, "importance": importances})
            imp_df = imp_df.sort_values("importance", ascending=False).head(10)

            plt.figure(figsize=(8,5))
            plt.barh(imp_df["feature"], imp_df["importance"])
            plt.gca().invert_yaxis()
            plt.title("변수 중요도 Top 10")
            plt.tight_layout()
        except Exception:
            plt.figure()
            plt.text(0.5,0.5,"변수 중요도 계산 불가",ha="center",va="center")

    @render.plot
    def distribution_plot():
        try:
            plt.figure(figsize=(8,5))
            df_good = df_predict[df_predict["passorfail"]==0]["biscuit_thickness"]
            df_bad = df_predict[df_predict["passorfail"]==1]["biscuit_thickness"]

            plt.hist(df_good, bins=30, alpha=0.6, label="양품")
            plt.hist(df_bad, bins=30, alpha=0.6, label="불량품")

            plt.axvline(df_predict["biscuit_thickness"].mean(), color="red", linestyle="--", label="평균")
            plt.legend()
            plt.title("비스킷 두께 분포 (양품 vs 불량)")
            plt.tight_layout()
        except Exception:
            plt.figure()
            plt.text(0.5,0.5,"분포 그래프 생성 불가",ha="center",va="center")

    @render.plot
    def process_trend_plot():
        try:
            mold_trend = df_predict.groupby("mold_code")["passorfail"].mean().sort_values(ascending=False)
            plt.figure(figsize=(8,5))
            mold_trend.plot(kind="bar")
            plt.ylabel("불량률")
            plt.title("금형 코드별 불량률")
            plt.tight_layout()
        except Exception:
            plt.figure()
            plt.text(0.5,0.5,"공정별 그래프 생성 불가",ha="center",va="center")

    # ===== 데이터 요약 카드 =====
    @output
    @render.table
    def df_summary():
        return pd.DataFrame({
            "항목": ["행 개수", "열 개수", "총 결측치", "결측치 비율(%)",
                "수치형 변수 개수", "범주형 변수 개수"],
            "값": [
                f"{df_raw.shape[0]:,}",
                f"{df_raw.shape[1]:,}",
                f"{df_raw.isna().sum().sum():,}",
                round(df_raw.isna().sum().sum() / (df_raw.shape[0]*df_raw.shape[1]) * 100, 2),
                df_raw.select_dtypes(include="number").shape[1],
                df_raw.select_dtypes(exclude="number").shape[1],
            ]
        })

    # --- 결측치 비율 ---
    @output
    @render.plot
    def missing_plot():
        # 결측치 비율 계산
        na_ratio = (df_explore.isna().mean() * 100)
        na_ratio = na_ratio[na_ratio > 0].sort_values(ascending=False).head(6)  # 상위 6개만

        if na_ratio.empty:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "결측치가 있는 컬럼이 없습니다.", ha="center", va="center")
            ax.axis("off")
            return fig

        fig, ax = plt.subplots(figsize=(6, 3))
        sns.barplot(x=na_ratio.index, y=na_ratio.values, ax=ax, color="tomato")

        # 막대 위에 라벨 표시
        for i, v in enumerate(na_ratio.values):
            ax.text(i, v + 0.5, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)

        ax.set_ylabel("결측치 비율 (%)")
        ax.set_xlabel("컬럼명")
        ax.set_title("결측치 비율 상위 6개 컬럼")
        ax.tick_params(axis="x", rotation=45)
        ax.set_ylim(0, max(na_ratio.values) * 1.2)  # 여백 확보

        return fig

    # --- 변수 타입 분포 ---
    @output
    @render.plot
    def dtype_pie():
        num = df_explore.select_dtypes(include="number").shape[1]
        cat = df_explore.select_dtypes(exclude="number").shape[1]
        fig, ax = plt.subplots()
        ax.pie([num, cat], labels=["수치형", "범주형"], autopct="%1.1f%%", colors=["skyblue", "orange"])
        ax.set_title("변수 타입 비율")
        return fig

    # --- 수치형 변수 상관 관계 히트맵 ---
    @output
    @render.plot
    def corr_heatmap_overview():
        num_df = df_explore.select_dtypes(include="number")
        if num_df.shape[1] < 2:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, "수치형 변수가 부족합니다.", ha="center", va="center")
            ax.axis("off")
            return fig

        corr = num_df.corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax, cbar=True)
        ax.set_title("수치형 변수 상관관계 히트맵")
        return fig

    # --- 동적 필터 UI ---
    @output
    @render.ui
    def filter_ui():
        var = input.var()
        if var not in df_explore.columns:
            return None

        # registration_time → datetime slider (10분 단위)
        if var == "registration_time":
            times = pd.to_datetime(df_explore["registration_time"], errors="coerce")
            times = times.dropna()
            if times.empty:
                return ui.markdown("⚠️ registration_time 컬럼에 유효한 datetime 값이 없습니다.")
            min_t, max_t = times.min(), times.max()

            # 초기 범위: 최대값 - 10분 ~ 최대값
            min_t, max_t = times.min(), times.max()
            # init_end = min_t + pd.Timedelta(minutes=10)
            # if init_end > max_t:
            #     init_end = max_t

            return ui.input_slider(
                "ts_range",
                "시간 범위 선택",
                min=min_t, max=max_t,
                value=[min_t, max_t],
                step=600,
                time_format="%Y-%m-%d %H:%M"
            )

        # 범주형 변수
        if not pd.api.types.is_numeric_dtype(df_explore[var]):
            categories = df_explore[var].dropna().astype(str).unique().tolist()
            categories = sorted(categories) + ["없음"]
            return ui.input_checkbox_group(
                "filter_val",
                f"{label_map.get(var, var)} 선택",
                choices=categories,
                selected=categories
            )

        # 수치형 변수
        min_val, max_val = df_explore[var].min(), df_explore[var].max()
        return ui.input_slider(
            "filter_val",
            f"{label_map.get(var, var)} 범위",
            min=min_val, max=max_val,
            value=[min_val, max_val]
        )
    
    # --- 데이터 필터링 ---
    @reactive.calc
    def filtered_df():
        dff = df_explore.copy()
        var = input.var()

        if var in dff.columns and "filter_val" in input:
            rng = input.filter_val()
            if rng is None:
                return dff

            # registration_time 필터
            if var == "registration_time":
                dff["registration_time"] = pd.to_datetime(dff["registration_time"], errors="coerce")
                dff = dff.dropna(subset=["registration_time"])
                start, end = pd.to_datetime(rng[0]), pd.to_datetime(rng[1])
                dff = dff[(dff["registration_time"] >= start) & (dff["registration_time"] <= end)]

            # 범주형 필터
            elif not pd.api.types.is_numeric_dtype(dff[var]):
                selected = rng
                if "없음" in selected:
                    dff = dff[(dff[var].isin([x for x in selected if x != "없음"])) | (dff[var].isna()) | (dff[var]=="")]
                else:
                    dff = dff[dff[var].isin(selected)]

            # 수치형 필터
            else:
                start, end = float(rng[0]), float(rng[1])
                dff = dff[(dff[var] >= start) & (dff[var] <= end)]

        return dff


    # --- 그래프 출력 ---
    @output
    @render.plot
    def dist_plot():
        dff = filtered_df()
        var = input.var()
        fig, ax = plt.subplots()

        if pd.api.types.is_numeric_dtype(dff[var]):
            sns.histplot(dff[var].dropna(), kde=True, ax=ax)
            ax.set_title(f"[{input.mold_code2()}] {label_map.get(var, var)}  분포 (히스토그램)")
        else:
            sns.countplot(x=dff[var], ax=ax, order=dff[var].value_counts().index)
            ax.set_title(f"[{input.mold_code2()}] {label_map.get(var, var)}  분포 (막대그래프)")
            ax.tick_params(axis="x", rotation=45)

        return fig
    

    @output
    @render.ui
    def ts_filter_ui():
        if "registration_time" not in df_raw.columns:
            return ui.markdown("⚠️ registration_time 없음")

        times = pd.to_datetime(df_raw["registration_time"], errors="coerce").dropna()
        if times.empty:
            return ui.markdown("⚠️ 유효한 datetime 값 없음")

        min_t, max_t = times.min(), times.max()
        # init_end = min_t + pd.Timedelta(minutes=10)
        # if init_end > max_t:
        #     init_end = max_t

        return ui.input_slider(
            "ts_range",
            "시간 범위 선택",
            min=min_t, max=max_t,
            value=[min_t, max_t],
            step=600,
            time_format="%Y-%m-%d %H:%M"
        )

    @output
    @render_plotly
    def timeseries_plot():
        if "registration_time" not in df_raw.columns:
            return px.scatter(title="⚠️ registration_time 없음")

        var = input.ts_var()
        rng = input.ts_range()

        dff = df_raw.copy()
        dff["registration_time"] = pd.to_datetime(dff["registration_time"], errors="coerce")
        dff = dff.dropna(subset=["registration_time", var, "passorfail"])
        dff["registration_time_str"] = dff["registration_time"].dt.strftime("%Y-%m-%d %H:%M:%S")

        if rng is not None:
            start, end = pd.to_datetime(rng[0]), pd.to_datetime(rng[1])
            dff = dff[(dff["registration_time"] >= start) & (dff["registration_time"] <= end)]

        if dff.empty:
            return px.scatter(title="⚠️ 선택한 구간에 데이터 없음")

        # pass/fail을 범주형으로 변환 → 색상 강제
        dff["불량여부"] = dff["passorfail"].map({0: "Pass", 1: "Fail"})

        fig = px.scatter(
            dff,
            x="registration_time_str",
            y=var,
            color="불량여부",
            color_discrete_map={"Pass": "green", "Fail": "red"},
            title=f"{label_map.get(var, var)} 시계열 값",
            labels={
                "registration_time_str": "등록 시간",
                var: label_map.get(var, var)   # ← y축 라벨 한글 표시
            },
        )

        # 배경 흰색 + 눈금선은 그대로 유지
        fig.update_layout(
            plot_bgcolor="white",   # 그래프 영역 배경
            paper_bgcolor="white",  # 전체 영역 배경
            xaxis=dict(
                showline=True,       # x축 라인 보이기
                linecolor="black",   # x축 라인 색
                showgrid=True,       # x축 그리드 보이기
                gridcolor="lightgray"
            ),
            yaxis=dict(
                showline=True,       # y축 라인 보이기
                linecolor="black",   # y축 라인 색
                showgrid=True,       # y축 그리드 보이기
                gridcolor="lightgray"
            )
        )

        # 배경 흰색, 보조선 점선
        fig.update_layout(
            plot_bgcolor="white",
            xaxis=dict(showgrid=True, gridcolor="lightgray", griddash="dot"),
            yaxis=dict(showgrid=True, gridcolor="lightgray", griddash="dot"),
            hovermode="x unified",
            margin=dict(l=40, r=20, t=40, b=40),
            legend_title_text=""  # ← 범례 제목 제거
        )

        fig.update_traces(marker=dict(size=5, opacity=0.5))

        return fig




app = App(app_ui, server, static_assets=app_dir / "www")
