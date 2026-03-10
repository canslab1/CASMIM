"""Data structures for the SARS epidemic simulation (NumPy SoA layout)."""

from dataclasses import dataclass
import numpy as np


# ===========================================================================
# StateEnum — 疾病狀態列舉
# ---------------------------------------------------------------------------
# 資料結構設計說明（繁體中文）：
#   本類別定義 SARS 模擬中每一個體（people）所處的疾病狀態。狀態轉移的
#   基本流程為：
#       易感 (SUSCEPTIBLE) → 潛伏 (EXPOSED) → 感染/傳染 (INFECTIVE)
#           → 康復 (RECOVERED) → 免疫 (IMMUNE) → 回到易感 (SUSCEPTIBLE)
#   在任何帶有傳染力的階段，個體也可能死亡 (DIED)。
#   ISOLATED（隔離治療）與 QUARANTINED（居家檢疫）為政策介入後產生的
#   附加狀態，原始 C++ 版本並未將其列入 enum，而是以 #define 常數另外
#   定義，但 Python 版為方便統一管理，將其一併納入本類別。
#   SIZE = 8 表示所有狀態的總數，用於統計陣列的維度配置。
#
# 對應 C++ 定義（Unit1.h）：
#   enum UState {
#       SUSCEPTIBLE = 0, EXPOSED = 1, INFECTIVE = 2,
#       RECOVERED = 3, IMMUNE = 4, DIED = 5
#   };
#   // 以下兩個狀態以 #define 常數定義，不在 enum 中：
#   #define ISOLATED    6
#   #define QUARANTINED 7
# ===========================================================================
class StateEnum:
    SUSCEPTIBLE = 0
    EXPOSED     = 1
    INFECTIVE   = 2
    RECOVERED   = 3
    IMMUNE      = 4
    DIED        = 5
    ISOLATED    = 6
    QUARANTINED = 7
    SIZE        = 8


# ===========================================================================
# AgeEnum — 年齡層列舉
# ---------------------------------------------------------------------------
# 資料結構設計說明（繁體中文）：
#   將人口依年齡分為三個層級：青年 (YOUNG)、壯年 (PRIME)、老年 (OLD)。
#   年齡層會影響兩項關鍵模擬參數：
#     1. 死亡率（mortality）：老年 > 壯年 > 青年，依據 SARS 實際臨床統計。
#     2. 初始人口分配比例：由 SimulationParams 中的 old_man_rate 與
#        young_man_rate 控制，剩餘比例歸入壯年。
#   模擬初始化時，每個個體會依上述比例隨機指定一個年齡層，並在整個模擬
#   過程中保持不變。
#
# 對應 C++ 定義（Unit1.h）：
#   enum UAge { YOUNG_MAN, PRIME_MAN, OLD_MAN };
# ===========================================================================
class AgeEnum:
    YOUNG = 0
    PRIME = 1
    OLD   = 2


# ===========================================================================
# PolicyIndex — 防疫政策索引常數
# ---------------------------------------------------------------------------
# 資料結構設計說明（繁體中文）：
#   本類別定義各項防疫介入措施在布林策略陣列 people_policy[N, MAX_POLICY]
#   中對應的索引位置。每個個體擁有一組長度為 MAX_POLICY 的布林旗標，
#   用以記錄該個體目前是否受到某項政策的保護或約束。
#
#   索引意義如下：
#     0 (FACE_MASK)        — 是否配戴口罩，降低傳播機率。
#     1 (TAKE_TEMPERATURE) — 是否接受體溫量測，提升偵測率以提早隔離。
#     2 (STOP_VISITANT)    — 是否禁止探病，減少醫院內交叉感染。
#     3 (VACCINE)          — 是否已接種疫苗，可直接獲得免疫。
#     4 (STOP_CONTACT)     — 是否減少社交接觸（如停班停課）。
#     5 (MEDICAL_POLICY)   — 是否適用醫療措施（如抗病毒藥物）。
#     6, 7                 — 保留未使用。
#     8 (HOME)             — 標記個體是否處於居家隔離狀態。
#     9 (HOSPITAL)         — 標記個體是否處於醫院隔離狀態。
#   MAX_POLICY = 10 為陣列總長度，涵蓋所有可能的政策欄位。
#
# 對應 C++ 定義（Unit1.h）：
#   #define FACE_MASK          0
#   #define TAKE_TEMPERATURE   1
#   #define STOP_VISITANT      2
#   #define VACCINE            3
#   #define STOP_CONTACT       4
#   #define MEDICAL_POLICY     5
#   #define HOME               8
#   #define HOSPITAL           9
#   #define MAX_POLICY        10
# ===========================================================================
class PolicyIndex:
    FACE_MASK        =  0
    TAKE_TEMPERATURE =  1
    STOP_VISITANT    =  2
    VACCINE          =  3
    STOP_CONTACT     =  4
    MEDICAL_POLICY   =  5
    HOME             =  8
    HOSPITAL         =  9
    MAX_POLICY       = 10


# ===========================================================================
# Colors — 狀態顏色常數（ARGB 格式）
# ---------------------------------------------------------------------------
# 資料結構設計說明（繁體中文）：
#   定義每個疾病狀態在視覺化格網（world lattice）上對應的顯示顏色。
#   Python 版使用 32 位元 ARGB 格式（0xAARRGGBB），以配合 Qt 的
#   QImage Format_ARGB32 繪圖介面。
#
#   顏色對應邏輯：
#     - SUSCEPTIBLE（易感）、EXPOSED（潛伏）、IMMUNE（免疫）
#       均使用天藍色 (SKY_BLUE)，表示外觀無異狀。
#     - INFECTIVE（傳染期）使用紅色 (RED)，醒目標示具傳染力個體。
#     - RECOVERED（康復期）使用銀灰色 (SILVER)，表示已不具傳染力。
#     - DIED（死亡）使用黑色 (BLACK)。
#   STATE_COLORS 陣列以 StateEnum 值為索引，方便以向量化操作批次查表
#   將所有個體的狀態一次轉換為對應顏色。
#
# 對應 C++ 定義（Unit1.h）：
#   #define DEFAULT_COLOR      (clSkyBlue)
#   #define SUSCEPTIBLE_COLOR  (DEFAULT_COLOR)
#   #define EXPOSED_COLOR      (DEFAULT_COLOR)
#   #define INFECTIVE_COLOR    (clRed)
#   #define RECOVERED_COLOR    (clSilver)
#   #define IMMUNE_COLOR       (DEFAULT_COLOR)
#   #define DIED_COLOR         (clBlack)
#   #define INVISIBLE_COLOR    (clGray)
# ===========================================================================
class Colors:
    """ARGB color constants for QImage Format_ARGB32."""
    SKY_BLUE = 0xFF87CEEB
    RED      = 0xFFFF0000
    SILVER   = 0xFFC0C0C0
    BLACK    = 0xFF000000
    GRAY     = 0xFF808080

    DEFAULT     = SKY_BLUE
    SUSCEPTIBLE = SKY_BLUE
    EXPOSED     = SKY_BLUE
    INFECTIVE   = RED
    RECOVERED   = SILVER
    IMMUNE      = SKY_BLUE
    DIED        = BLACK

    STATE_COLORS = np.array(
        [SUSCEPTIBLE, EXPOSED, INFECTIVE, RECOVERED, IMMUNE, DIED],
        dtype=np.uint32,
    )


# ===========================================================================
# SimulationParams — 模擬參數組態
# ---------------------------------------------------------------------------
# 資料結構設計說明（繁體中文）：
#   本 dataclass 集中管理所有可由使用者透過 GUI 調整的模擬參數。
#   在原始 C++ 版本中，這些參數分散於各個 TEdit / TCheckBox 控制項，
#   在模擬啟動時透過 ->Text.ToInt() 或 ->Text.ToDouble() 即時讀取。
#   Python 版將其統一為一個不可變的參數物件，以利單元測試與批次實驗。
#
#   參數分類：
#     ● 規模參數
#       - max_population:       最大人口數（對應 MAX_PEOPLE）
#       - max_agent:            每人可擁有的分身數（對應 MAX_AGENT）
#       - max_height/max_width: 格網世界的高度與寬度（對應 MAX_CELL_HEIGHT/WIDTH）
#
#     ● 疾病自然史參數（單位：天）
#       - exposed_period:     平均潛伏期（接觸→出現傳染力）
#       - symptomatic_period: 平均症狀期（出現症狀的持續天數）
#       - infective_period:   平均傳染期（具傳染力的天數）
#       - recovered_period:   平均康復期（症狀消失→產生抗體）
#       - immune_period:      平均免疫期（抗體存續天數，過後回到易感）
#       - quarantine_period:  隔離/檢疫天數
#
#     ● 傳播與偵測參數
#       - transmission_prob: 每次接觸的感染機率
#       - immune_prob:       自然免疫機率
#       - detect_rate:       症狀偵測率（將感染者送入隔離的機率）
#       - super_rate:        超級傳播者出現機率
#
#     ● 年齡別死亡率
#       - mortality_old / mortality_prime / mortality_young
#
#     ● 防疫政策效果與覆蓋率
#       - mask_effect/mask_available:         口罩防護效果/口罩供應率
#       - temp_effect/temp_available:         體溫篩檢效果/篩檢覆蓋率
#       - hospital_effect/hospital_available: 醫院隔離效果/收治率
#       - home_available:                     居家隔離配合率
#       - visit_available:                    禁止探病執行率
#       - contact_available:                  減少接觸執行率
#       - medical_policy_effect/available:    醫療措施效果/覆蓋率
#
#     ● 移動參數
#       - gossip_steps: 分身每日移動步數
#       - gossip_fixed: 是否固定步長（True=固定, False=隨機）
#
#     ● 隔離分級
#       - isolated_level_b: 是否啟用 B 級隔離（較嚴格）
#       - trace_on:         是否啟用接觸者追蹤
#
# 對應 C++ 定義（Unit1.h — 巨集讀取 GUI 控制項）：
#   #define EXPOSED_PERIOD     (Edit_AvgExposedPeriod->Text.ToInt())
#   #define SYMPTOMATIC_PERIOD (Edit_AvgSymptomaticPeriod->Text.ToInt())
#   #define INFECTIVE_PERIOD   (Edit_AvgInfectiousPeriod->Text.ToInt())
#   #define RECOVERED_PERIOD   (Edit_AvgRecoveredPeriod->Text.ToInt())
#   #define IMMUNE_PERIOD      (Edit_AvgAntibodyPeriod->Text.ToInt())
#   #define QUARANTINED_PERIOD (Edit_QuarantinedPeriod->Text.ToInt())
# ===========================================================================
@dataclass
class SimulationParams:
    """All configurable simulation parameters (from GUI controls)."""
    max_population: int = 100000
    max_agent:      int =      5
    max_height:     int =    500
    max_width:      int =    500

    # Disease parameters
    exposed_period:     int =  5
    symptomatic_period: int = 23
    infective_period:   int =  3
    recovered_period:   int =  7
    immune_period:      int = 60
    quarantine_period:  int = 10

    transmission_prob: float = 0.05
    immune_prob:       float = 0.02
    detect_rate:       float = 0.9
    super_rate:        float = 0.0001

    mortality_old:   float = 0.52
    mortality_prime: float = 0.17
    mortality_young: float = 0.05

    old_man_rate:   float = 0.2
    young_man_rate: float = 0.3

    # Policy parameters
    mask_effect:              float = 0.9
    mask_available:           float = 0.9
    temp_effect:              float = 0.9
    temp_available:           float = 0.9
    hospital_effect:          float = 0.5
    hospital_available:       float = 0.95
    home_available:           float = 0.81
    visit_available:          float = 0.9
    contact_available:        float = 0.9
    medical_policy_effect:    float = 0.9
    medical_policy_available: float = 0.9

    # Movement
    gossip_steps: int  = 1
    gossip_fixed: bool = True

    # Quarantine class
    isolated_level_b: bool = False
    trace_on:         bool = True


# ===========================================================================
# SimulationData — 模擬狀態資料（NumPy 結構陣列佈局）
# ---------------------------------------------------------------------------
# 資料結構設計說明（繁體中文）：
#   本類別持有整個 SARS 模擬的完整執行期狀態。原始 C++ 版本採用
#   「陣列中的結構」（Array-of-Structures, AoS）佈局——即宣告一個
#   UPeople 結構體陣列 society.people[MAX_PEOPLE]，每個結構體內含
#   該個體的所有欄位。Python 版改為「結構中的陣列」（Structure-of-
#   Arrays, SoA）佈局——每個欄位各自成為一條 NumPy 陣列，以個體 ID
#   為索引。此轉換的目的在於：
#     1. 利用 NumPy 向量化運算大幅加速批量狀態更新與統計彙整。
#     2. 改善記憶體存取的空間局部性（spatial locality），因為同一
#        欄位的資料在記憶體中連續排列，對快取更友善。
#     3. 方便以布林遮罩（boolean mask）篩選特定狀態的個體子集。
#
#   主要資料區塊：
#     ● People 陣列群（索引維度：N = max_population）
#       - people_state[N]:     int8, 疾病狀態（StateEnum 值）
#       - people_count[N]:     int32, 目前狀態已持續的天數計數器
#       - people_limit[N]:     int32, 目前狀態應持續的天數上限
#       - people_timer[N]:     int32, 通用計時器（對應 UAttribute.timer）
#       - people_immunity[N]:  bool, 是否具先天免疫力
#       - people_super[N]:     bool, 是否為超級傳播者
#       - people_age[N]:       int8, 年齡層（AgeEnum 值）
#       - people_isolated[N]:  bool, 是否處於隔離治療狀態
#       - people_quarantined[N]: bool, 是否處於居家檢疫狀態
#       - people_quarantined_count[N]: int32, 居家檢疫累計次數
#       - people_quarantined_level[N]: int32, 居家檢疫等級
#       - people_policy[N, MAX_POLICY]: bool, 各項防疫政策旗標
#
#     ● Agent（分身）陣列群（索引維度：N x M，M = max_agent）
#       - agent_visible[N, M]: bool, 分身是否可見（存活/活動中）
#       - agent_home[N, M]:    bool, 分身是否在家
#       - agent_loc_x[N, M]:   int32, 分身在格網上的 X 座標
#       - agent_loc_y[N, M]:   int32, 分身在格網上的 Y 座標
#       每個個體最多可擁有 M 個分身，分布在格網世界的不同位置，
#       代表「社會分身」（social agent）的概念：一個人在社會中可能
#       同時存在於多個場域（如家庭、工作場所、醫院）。
#
#     ● World 格網陣列群（索引維度：H x W）
#       - world_people_id[H, W]: int32, 該格位所屬的個體 ID（-1=空格）
#       - world_agent_no[H, W]:  int32, 該格位所屬的分身編號（-1=空格）
#       - world_color[H, W]:     uint32, 該格位的顯示顏色（ARGB）
#
#     ● 統計資料
#       - day:                   int, 目前模擬天數
#       - statistic[SIZE]:       int64, 各狀態的目前人口統計
#       - old_statistic[SIZE]:   int64, 前一天的人口統計（用於計算差異）
#       - infected_by_hospital:  int, 院內感染累計人數
#       - infected_by_normal:    int, 社區感染累計人數
#
# 對應 C++ 定義（Unit1.h）：
#   struct UAttribute {
#       bool immunity;
#       bool super;
#       long timer;
#       UAge age;
#       bool policy[MAX_POLICY];
#       bool isolated;
#       bool quarantined;
#       long quarantinedCount;
#       long quarantinedLevel;
#   };
#
#   struct UAgent {
#       long no;
#       bool visible;
#       bool home;
#       POINT location;
#   };
#
#   struct UPeople {
#       long ID;
#       UState state;
#       long count;
#       long limit;
#       UAttribute attr;
#       UAgent agent[MAX_AGENT];
#   };
#
#   struct USociety {
#       long day;
#       UPeople people[MAX_PEOPLE];
#   };
#
#   struct UCell {
#       long peopleID;
#       long agentNo;
#       TColor color;
#   };
#
#   // 實體變數宣告：
#   USociety society;
#   UCell world[MAX_CELL_HEIGHT][MAX_CELL_WIDTH];
#   long oldStatistic[STATE_SIZE], statistic[STATE_SIZE];
#   long infectedByHospital, infectedByNormal;
#
# AoS → SoA 轉換對照：
#   C++ AoS: society.people[i].state       → Python SoA: people_state[i]
#   C++ AoS: society.people[i].attr.age    → Python SoA: people_age[i]
#   C++ AoS: society.people[i].agent[j].location.x
#                                          → Python SoA: agent_loc_x[i, j]
#   C++ AoS: world[y][x].peopleID         → Python SoA: world_people_id[y, x]
# ===========================================================================
class SimulationData:
    """Holds all simulation state as NumPy arrays (Structure-of-Arrays)."""

    def __init__(self, params: SimulationParams):
        N = params.max_population
        M = params.max_agent
        H = params.max_height
        W = params.max_width

        self.N = N
        self.M = M
        self.H = H
        self.W = W

        # People arrays
        self.people_state = np.zeros(N, dtype=np.int8)
        self.people_count = np.zeros(N, dtype=np.int32)
        self.people_limit = np.zeros(N, dtype=np.int32)
        self.people_timer = np.zeros(N, dtype=np.int32)
        self.people_immunity = np.zeros(N, dtype=bool)
        self.people_super = np.zeros(N, dtype=bool)
        self.people_age = np.zeros(N, dtype=np.int8)
        self.people_isolated = np.zeros(N, dtype=bool)
        self.people_quarantined = np.zeros(N, dtype=bool)
        self.people_quarantined_count = np.zeros(N, dtype=np.int32)
        self.people_quarantined_level = np.zeros(N, dtype=np.int32)
        self.people_policy = np.zeros((N, PolicyIndex.MAX_POLICY), dtype=bool)

        # Agent arrays
        self.agent_visible = np.zeros((N, M), dtype=bool)
        self.agent_home = np.zeros((N, M), dtype=bool)
        self.agent_loc_x = np.full((N, M), -1, dtype=np.int32)
        self.agent_loc_y = np.full((N, M), -1, dtype=np.int32)

        # World/lattice arrays
        self.world_people_id = np.full((H, W), -1, dtype=np.int32)
        self.world_agent_no = np.full((H, W), -1, dtype=np.int32)
        self.world_color = np.full((H, W), Colors.DEFAULT, dtype=np.uint32)

        # Statistics
        self.day = 0
        self.statistic = np.zeros(StateEnum.SIZE, dtype=np.int64)
        self.old_statistic = np.zeros(StateEnum.SIZE, dtype=np.int64)
        self.infected_by_hospital = 0
        self.infected_by_normal = 0

    def reset(self):
        """Reset all arrays for a new simulation run."""
        self.people_state[:] = 0
        self.people_count[:] = 0
        self.people_limit[:] = 0
        self.people_timer[:] = 0
        self.people_immunity[:] = False
        self.people_super[:] = False
        self.people_age[:] = 0
        self.people_isolated[:] = False
        self.people_quarantined[:] = False
        self.people_quarantined_count[:] = 0
        self.people_quarantined_level[:] = 0
        self.people_policy[:] = False

        self.agent_visible[:] = False
        self.agent_home[:] = False
        self.agent_loc_x[:] = -1
        self.agent_loc_y[:] = -1

        self.world_people_id[:] = -1
        self.world_agent_no[:] = -1
        self.world_color[:] = Colors.DEFAULT

        self.day = 0
        self.statistic[:] = 0
        self.old_statistic[:] = 0
        self.infected_by_hospital = 0
        self.infected_by_normal = 0


# ===========================================================================
# MeasureData — 多次模擬量測資料彙整
# ---------------------------------------------------------------------------
# 資料結構設計說明（繁體中文）：
#   本類別用於跨越多次模擬執行（multiple runs）的統計量測數據蒐集。
#   在進行蒙地卡羅式重複實驗時，每次模擬結束後會將該次的每日各狀態
#   人數累加進 value1，並在 value2 中計算平均值，以消除隨機因素的影響。
#
#   資料欄位：
#     - size:   已累計的模擬執行次數。
#     - value1[MAX_DAYS, STATE_SIZE]: int64 二維陣列，
#       第一維為天數索引（最多 MAX_DAYS=365 天），第二維為狀態索引
#       （對應 StateEnum），儲存各天各狀態人數的「累加值」。
#     - value2[MAX_DAYS, STATE_SIZE]: float64 二維陣列，
#       與 value1 結構相同，儲存各天各狀態人數的「平均值」。
#       平均值 = value1 / size，於每次執行結束後更新。
#
#   使用流程：
#     1. 建立 MeasureData 實例（size=0, 陣列全零）。
#     2. 每次模擬結束後，將當次的每日統計累加入 value1，size += 1。
#     3. 重新計算 value2 = value1 / size。
#     4. 繪圖時以 value2 的資料呈現多次實驗的平均疫情曲線。
#
# 對應 C++ 定義（Unit1.h）：
#   struct UMeasure {
#       long size;
#       long value1[MAX_VALUE_SIZE][STATE_SIZE];
#       double value2[MAX_VALUE_SIZE][STATE_SIZE];
#   };
# ===========================================================================
class MeasureData:
    """Stores accumulative and average statistics across runs."""

    MAX_DAYS = 365

    def __init__(self):
        self.size = 0
        self.value1 = np.zeros((self.MAX_DAYS, StateEnum.SIZE), dtype=np.int64)
        self.value2 = np.zeros((self.MAX_DAYS, StateEnum.SIZE), dtype=np.float64)
