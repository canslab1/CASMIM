"""Core simulation engine for the SARS epidemic simulation.

Faithful port of the C++ simulation logic to Python, operating on the
NumPy Structure-of-Arrays layout defined in ``sars_sim.models``.

When Numba is available, the hot per-day loop (``change_society``) is
delegated to a JIT-compiled kernel in ``engine_numba.py`` for 50-100x
speedup.  Set the environment variable ``CASMIM_NO_NUMBA=1`` to force
the pure-Python fallback path.
"""

import os
import random
from collections import deque

import numpy as np

from .models import (
    AgeEnum,
    Colors,
    PolicyIndex as PI,
    SimulationData,
    SimulationParams,
    StateEnum as S,
)

# ------------------------------------------------------------------
# Optional Numba acceleration
# ------------------------------------------------------------------
_USE_NUMBA = False
if not os.environ.get("CASMIM_NO_NUMBA"):
    try:
        from .engine_numba import change_society_kernel as _numba_kernel
        _USE_NUMBA = True
    except ImportError:
        pass


class SimulationEngine:
    """Runs one simulation step (day) on the shared *SimulationData*."""

    # Neighbourhood offsets for a 3x3 Moore neighbourhood encoded as a flat
    # index 0..8 where index 4 is the cell itself.
    _AGENT_SELF = 4

    def __init__(self, params: SimulationParams, data: SimulationData):
        self.params = params
        self.data = data

        # Flag that the GUI can toggle to indicate the medical-policy
        # checkbox is checked.
        self.medical_policy_enabled: bool = False

        # People whose state changed during the current day-step.
        # Used for incremental world-color updates instead of full repaint.
        self.dirty_pids: set = set()

        # Pre-allocated buffers for the Numba kernel
        N = data.N
        self._dirty_flags = np.zeros(N, dtype=np.bool_)
        self._infected_counts = np.zeros(2, dtype=np.int64)
        self._bfs_queue_pid = np.empty(N, dtype=np.int32)
        self._bfs_queue_level = np.empty(N, dtype=np.int32)
        self._bfs_visited = np.zeros(N, dtype=np.bool_)
        self._bfs_visited_stack = np.empty(N, dtype=np.int32)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    # ================================================================
    # changeSociety — 推進社會模擬一天
    # ================================================================
    # 演算法說明：
    #   每次呼叫代表模擬時間推進一天。首先以 50% 機率隨機決定
    #   遍歷人口的方向（正序或逆序），藉此消除固定遍歷順序帶來
    #   的偏差效應（fairness）。接著依序對每個個體執行 changePeople
    #   以更新其疾病狀態與社會接觸行為。
    #   C++ 版本中另會更新 ProgressBar 以顯示進度，Python 版省略此 GUI 操作。
    #
    # C++ 原始碼 (Unit1.cpp line 630):
    # void TSARS_Form::changeSociety(void)
    # {
    #     bool dir = FLIP(0.5);
    #     ProgressBar->Position = 0;
    #     MemoStatus->Lines->Append(AnsiString(++society.day));
    #     for (long MaxPopulation = Edit_MaxPopulation->Text.ToInt(),
    #          i = (dir ? 0 : MaxPopulation - 1);
    #          (dir ? i < MaxPopulation : i >= 0);
    #          (dir ? i++ : i--)) {
    #         changePeople(i);
    #         if ((i % STEP_SIZE) == 0) ProgressBar->Position = i;
    #     }
    # }
    # ================================================================
    def change_society(self) -> None:
        """Advance the simulation by one day.

        When Numba is available, the entire per-person loop is executed
        by a JIT-compiled kernel for dramatically better performance.
        The pure-Python fallback is used otherwise.
        """
        d = self.data
        self.dirty_pids.clear()
        d.day += 1

        if _USE_NUMBA:
            self._change_society_numba()
        else:
            self._change_society_python()

    # ------------------------------------------------------------------
    # Numba-accelerated day-step
    # ------------------------------------------------------------------

    def _change_society_numba(self) -> None:
        """Delegate the full day-step to the JIT-compiled kernel."""
        d = self.data
        p = self.params

        # Pack the two scalar infection counters into a mutable array
        self._infected_counts[0] = d.infected_by_hospital
        self._infected_counts[1] = d.infected_by_normal

        _numba_kernel(
            # People arrays
            d.people_state, d.people_count, d.people_timer, d.people_policy,
            d.people_isolated, d.people_quarantined,
            d.people_quarantined_count, d.people_quarantined_level,
            d.people_immunity, d.people_age, d.people_super,
            # Agent arrays
            d.agent_visible, d.agent_home, d.agent_loc_x, d.agent_loc_y,
            # World arrays
            d.world_people_id, d.world_agent_no,
            # Statistics
            d.statistic, self._infected_counts,
            # Pre-allocated buffers
            self._dirty_flags, self._bfs_queue_pid, self._bfs_queue_level,
            self._bfs_visited, self._bfs_visited_stack,
            # Scalar parameters
            d.N, d.W, d.H,
            p.exposed_period, p.symptomatic_period, p.infective_period,
            p.recovered_period, p.immune_period, p.quarantine_period,
            p.transmission_prob, p.medical_policy_effect,
            p.mask_effect, p.temp_effect, p.detect_rate,
            p.mortality_young, p.mortality_prime, p.mortality_old,
            p.hospital_effect,
            p.gossip_steps, p.gossip_fixed,
            p.trace_on, p.isolated_level_b,
            self.medical_policy_enabled, p.medical_policy_available,
        )

        # Unpack scalar counters back
        d.infected_by_hospital = int(self._infected_counts[0])
        d.infected_by_normal = int(self._infected_counts[1])

        # Convert dirty boolean flags to the set expected by the rest of the code
        self.dirty_pids = set(np.flatnonzero(self._dirty_flags).tolist())

    # ------------------------------------------------------------------
    # Pure-Python fallback day-step (original logic, unchanged)
    # ------------------------------------------------------------------

    def _change_society_python(self) -> None:
        """Pure-Python fallback for ``change_society``."""
        d = self.data
        direction = random.random() < 0.5

        if direction:
            for pid in range(d.N):
                self.change_people(pid)
        else:
            for pid in range(d.N - 1, -1, -1):
                self.change_people(pid)

    # ================================================================
    # import_cases — 匯入初始病例（對應 C++ Button_NewCaseClick）
    # ================================================================
    # 演算法說明：
    #   此方法用於在模擬中注入新的感染病例。重複 num_cases 次，
    #   每次隨機挑選一位狀態為 SUSCEPTIBLE（易感）、無免疫力、
    #   且擁有至少一個分身（agent）的個體。找到後根據參數設定其
    #   初始狀態為 EXPOSED（潛伏期）或 INFECTIVE（傳染期），並
    #   隨機設定已經過的時間（timer），使不同病例的發病時間錯開。
    #   同時設定是否為超級傳播者（super），更新統計計數，並將該
    #   個體所有分身在世界地圖上的顏色更新為對應狀態的顏色。
    #
    # C++ 原始碼 (Unit1.cpp line 459, Button_NewCaseClick):
    # void __fastcall TSARS_Form::Button_NewCaseClick(TObject *Sender)
    # {
    #     long ID;
    #     UState state;
    #     if (!CheckBox_InputAsWellAsRun->Checked) society.day++;
    #     backupStatistic();
    #     for (long i = Edit_NewCase->Text.ToInt(); i > 0; i--) {
    #         do {} while (society.people[(ID = (long)_lrand() %
    #             Edit_MaxPopulation->Text.ToInt())].state != SUSCEPTIBLE
    #             || society.people[ID].attr.immunity
    #             || society.people[ID].count == 0);
    #         if ((state = society.people[ID].state =
    #             (RadioButton_Exposed->Checked ? EXPOSED : INFECTIVE))
    #             == EXPOSED)
    #             society.people[ID].attr.timer =
    #                 random(Edit_AvgExposedPeriod->Text.ToInt());
    #         else
    #             society.people[ID].attr.timer =
    #                 random(Edit_AvgSymptomaticPeriod->Text.ToInt());
    #         society.people[ID].attr.super = CheckBox_Super->Checked;
    #         ++statistic[(int)state];
    #         for (long count = society.people[ID].count, j = 0;
    #              j < count; j++)
    #             world[society.people[ID].agent[j].location.y]
    #                  [society.people[ID].agent[j].location.x].color
    #                  = getColor(state);
    #     }
    #     ...
    # }
    # ================================================================
    def import_cases(
        self,
        num_cases: int,
        as_exposed: bool,
        is_super: bool,
    ) -> None:
        """Seed *num_cases* infections among susceptible people."""
        d = self.data
        p = self.params

        for _ in range(num_cases):
            # Pick a random susceptible person who has agents and no immunity.
            attempts = 0
            while True:
                pid = random.randrange(d.N)
                if (
                    d.people_state[pid] == S.SUSCEPTIBLE
                    and not d.people_immunity[pid]
                    and d.people_count[pid] > 0
                ):
                    break
                attempts += 1
                if attempts > d.N * 10:
                    return  # safety valve

            if as_exposed:
                new_state = S.EXPOSED
                d.people_timer[pid] = random.randint(0, max(p.exposed_period - 1, 0))
            else:
                new_state = S.INFECTIVE
                d.people_timer[pid] = random.randint(
                    0, max(p.symptomatic_period - 1, 0)
                )

            d.people_state[pid] = new_state
            self.dirty_pids.add(pid)
            d.people_super[pid] = is_super
            d.statistic[new_state] += 1

            # Update world colours for every agent of this person.
            color = Colors.STATE_COLORS[new_state]
            for i in range(d.people_count[pid]):
                y = d.agent_loc_y[pid, i]
                x = d.agent_loc_x[pid, i]
                if 0 <= y < d.H and 0 <= x < d.W:
                    d.world_color[y, x] = color

    # ------------------------------------------------------------------
    # Per-person update
    # ------------------------------------------------------------------

    # ================================================================
    # changePeople — 更新單一個體的狀態（每日主迴圈的核心）
    # ================================================================
    # 演算法說明：
    #   對指定個體 (pid) 執行三階段更新：
    #   1. 居家隔離計時：若個體正處於居家隔離 (quarantined) 且狀態為
    #      SUSCEPTIBLE / EXPOSED / INFECTIVE，則推進隔離天數計數器。
    #   2. 疾病狀態轉換：若個體處於 EXPOSED / INFECTIVE / RECOVERED /
    #      IMMUNE（無永久免疫）狀態，執行 changePeopleByOngoing 以處理
    #      狀態轉移（如潛伏→發病、發病→死亡或康復、康復→免疫等）。
    #   3. 接觸傳播（gossip）：若個體為 SUSCEPTIBLE，或雖為 EXPOSED /
    #      INFECTIVE 但尚未被居家隔離，則執行 changePeopleByGossip 模擬
    #      其分身在鄰近格子與他人接觸的行為。
    #   每一階段結束後重新讀取狀態，因為前一階段可能已改變狀態。
    #
    # C++ 原始碼 (Unit1.cpp line 642):
    # void TSARS_Form::changePeople(long ID)
    # {
    #     if (society.people[ID].count > 0) {
    #         UState state = society.people[ID].state;
    #         if (state == SUSCEPTIBLE || state == EXPOSED
    #             || state == INFECTIVE)
    #             if (society.people[ID].attr.policy[HOME]
    #                 && society.people[ID].attr.quarantined)
    #                 quarantinePeopleByOngoing(ID, state);
    #         state = society.people[ID].state;
    #         if (state == EXPOSED || state == INFECTIVE
    #             || state == RECOVERED
    #             || (state == IMMUNE
    #                 && !society.people[ID].attr.immunity))
    #             changePeopleByOngoing(ID, state);
    #         state = society.people[ID].state;
    #         if (state == SUSCEPTIBLE
    #             || ((state == EXPOSED || state == INFECTIVE)
    #                 && society.people[ID].attr.policy[HOME]
    #                 && !society.people[ID].attr.quarantined))
    #             changePeopleByGossip(ID);
    #     }
    # }
    # ================================================================
    def change_people(self, pid: int) -> None:
        d = self.data

        if d.people_count[pid] == 0:
            return

        state = int(d.people_state[pid])

        # Quarantine ongoing tick
        if state in (S.SUSCEPTIBLE, S.EXPOSED, S.INFECTIVE):
            if d.people_policy[pid, PI.HOME] and d.people_quarantined[pid]:
                self.quarantine_people_by_ongoing(pid, state)

        # Re-read state (may have changed)
        state = int(d.people_state[pid])

        # Disease progression
        if state in (S.EXPOSED, S.INFECTIVE, S.RECOVERED) or (
            state == S.IMMUNE and not d.people_immunity[pid]
        ):
            self.change_people_by_ongoing(pid, state)

        # Re-read state
        state = int(d.people_state[pid])

        # Gossip / movement
        if state == S.SUSCEPTIBLE or (
            state in (S.EXPOSED, S.INFECTIVE)
            and d.people_policy[pid, PI.HOME]
            and not d.people_quarantined[pid]
        ):
            self.change_people_by_gossip(pid)

    # ------------------------------------------------------------------
    # Quarantine tick
    # ------------------------------------------------------------------

    # ================================================================
    # quarantinePeopleByOngoing — 居家隔離天數遞增與期滿解除
    # ================================================================
    # 演算法說明：
    #   每日呼叫一次，將該個體的居家隔離天數計數器 (quarantinedCount)
    #   加 1。若計數器超過設定的隔離期間 (QUARANTINED_PERIOD)，則：
    #   - 解除居家隔離狀態 (quarantined = false)
    #   - 重置計數器與隔離等級
    #   - 若該個體並非同時被醫院隔離 (isolated)，則將其所有分身
    #     恢復為可見 (visible = true)，代表回歸正常社會活動。
    #
    # C++ 原始碼 (Unit1.cpp line 660):
    # void TSARS_Form::quarantinePeopleByOngoing(long ID, UState state)
    # {
    #     if (++society.people[ID].attr.quarantinedCount
    #         > QUARANTINED_PERIOD) {
    #         society.people[ID].attr.quarantined      = false;
    #         society.people[ID].attr.quarantinedCount = 0;
    #         society.people[ID].attr.quarantinedLevel = 0;
    #         if (!society.people[ID].attr.policy[HOSPITAL]
    #             || !society.people[ID].attr.isolated)
    #             for (long count = society.people[ID].count, i = 0;
    #                  i < count;
    #                  society.people[ID].agent[i++].visible = true);
    #     }
    # }
    # ================================================================
    def quarantine_people_by_ongoing(self, pid: int, state: int) -> None:
        d = self.data
        p = self.params

        d.people_quarantined_count[pid] += 1

        if d.people_quarantined_count[pid] > p.quarantine_period:
            d.people_quarantined[pid] = False
            d.people_quarantined_count[pid] = 0
            d.people_quarantined_level[pid] = 0
            # Restore visibility unless isolated in hospital.
            if not (d.people_policy[pid, PI.HOSPITAL] and d.people_isolated[pid]):
                for i in range(d.people_count[pid]):
                    d.agent_visible[pid, i] = True

    # ------------------------------------------------------------------
    # Disease-state progression
    # ------------------------------------------------------------------

    # ================================================================
    # changePeopleByOngoing — 疾病狀態隨時間推進的轉移邏輯
    # ================================================================
    # 演算法說明：
    #   此方法實作 SEIR 模型的狀態轉移（含死亡與免疫衰退）：
    #   (1) 每次呼叫先將 timer + 1，代表在目前狀態又經過一天。
    #   (2) EXPOSED → INFECTIVE：潛伏期結束 (timer > EXPOSED_PERIOD)
    #       後轉為傳染期，重置 timer。
    #   (3) INFECTIVE 且尚未被醫院隔離：嘗試透過 isolatePeopleByHospital
    #       偵測是否應送醫隔離。
    #   (4) INFECTIVE → DIED / RECOVERED：症狀期結束
    #       (timer > SYMPTOMATIC_PERIOD) 後，依死亡率 getDiedRate 以
    #       隨機數決定死亡或康復。死亡者的分身僅在「家」位置可見
    #       （代表遺體在家），康復者恢復所有分身可見。同時清除隔離
    #       與居家隔離旗標。
    #   (5) RECOVERED → IMMUNE：康復期結束後轉為免疫狀態。若免疫
    #       期為 0 則設為永久免疫。
    #   (6) IMMUNE → SUSCEPTIBLE：免疫期結束後免疫消退，回到易感
    #       狀態，並取消疫苗政策標記。
    #
    # C++ 原始碼 (Unit1.cpp line 671):
    # void TSARS_Form::changePeopleByOngoing(long ID, UState state)
    # {
    #     long timer = ++society.people[ID].attr.timer;
    #     if (state == EXPOSED && timer > EXPOSED_PERIOD) {
    #         ++statistic[(int)(society.people[ID].state = INFECTIVE)];
    #         society.people[ID].attr.timer = 0;
    #     }
    #     if (state == INFECTIVE
    #         && society.people[ID].attr.policy[HOSPITAL]
    #         && !society.people[ID].attr.isolated)
    #         isolatePeopleByHospital(ID, state, timer);
    #     if (state == INFECTIVE && timer > SYMPTOMATIC_PERIOD) {
    #         ++statistic[(int)(society.people[ID].state =
    #             (FLIP(getDiedRate(ID)) ? DIED : RECOVERED))];
    #         society.people[ID].attr.timer = 0;
    #         if (society.people[ID].state == DIED)
    #             for (long count = society.people[ID].count, i = 0;
    #                  i < count; i++)
    #                 society.people[ID].agent[i].visible =
    #                     society.people[ID].agent[i].home;
    #         else
    #             for (long count = society.people[ID].count, i = 0;
    #                  i < count; i++)
    #                 society.people[ID].agent[i].visible = true;
    #         if (society.people[ID].attr.isolated)
    #             finishMedicalPolicy(ID);
    #         society.people[ID].attr.isolated         = false;
    #         society.people[ID].attr.quarantined      = false;
    #         society.people[ID].attr.quarantinedCount = 0;
    #         society.people[ID].attr.quarantinedLevel = 0;
    #     }
    #     if (state == RECOVERED && timer > RECOVERED_PERIOD) {
    #         ++statistic[(int)(society.people[ID].state = IMMUNE)];
    #         society.people[ID].attr.immunity =
    #             (IMMUNE_PERIOD == 0);
    #         society.people[ID].attr.timer = 0;
    #     }
    #     if (state == IMMUNE && timer > IMMUNE_PERIOD) {
    #         ++statistic[(int)(society.people[ID].state = SUSCEPTIBLE)];
    #         society.people[ID].attr.timer = 0;
    #         society.people[ID].attr.policy[VACCINE] = false;
    #     }
    # }
    # ================================================================
    def change_people_by_ongoing(self, pid: int, state: int) -> None:
        d = self.data
        p = self.params

        d.people_timer[pid] += 1
        t = int(d.people_timer[pid])

        # EXPOSED -> INFECTIVE
        if state == S.EXPOSED and t > p.exposed_period:
            d.people_state[pid] = S.INFECTIVE
            self.dirty_pids.add(pid)
            d.statistic[S.INFECTIVE] += 1
            d.people_timer[pid] = 0

        # Hospital detection while INFECTIVE
        if (
            state == S.INFECTIVE
            and d.people_policy[pid, PI.HOSPITAL]
            and not d.people_isolated[pid]
        ):
            self.isolate_by_hospital(pid, state, t)

        # INFECTIVE -> DIED / RECOVERED
        if state == S.INFECTIVE and t > p.symptomatic_period:
            died = random.random() < self.get_died_rate(pid)
            new_state = S.DIED if died else S.RECOVERED
            d.people_state[pid] = new_state
            self.dirty_pids.add(pid)
            d.statistic[new_state] += 1
            d.people_timer[pid] = 0

            if new_state == S.DIED:
                for i in range(d.people_count[pid]):
                    d.agent_visible[pid, i] = d.agent_home[pid, i]
            else:
                for i in range(d.people_count[pid]):
                    d.agent_visible[pid, i] = True

            if d.people_isolated[pid]:
                self.finish_medical_policy(pid)
            d.people_isolated[pid] = False
            d.people_quarantined[pid] = False
            d.people_quarantined_count[pid] = 0
            d.people_quarantined_level[pid] = 0

        # RECOVERED -> IMMUNE
        if state == S.RECOVERED and t > p.recovered_period:
            d.people_state[pid] = S.IMMUNE
            self.dirty_pids.add(pid)
            d.statistic[S.IMMUNE] += 1
            d.people_immunity[pid] = (p.immune_period == 0)
            d.people_timer[pid] = 0

        # IMMUNE -> SUSCEPTIBLE
        if state == S.IMMUNE and t > p.immune_period:
            d.people_state[pid] = S.SUSCEPTIBLE
            self.dirty_pids.add(pid)
            d.statistic[S.SUSCEPTIBLE] += 1
            d.people_timer[pid] = 0
            d.people_policy[pid, PI.VACCINE] = False

    # ------------------------------------------------------------------
    # Hospital isolation
    # ------------------------------------------------------------------

    # ================================================================
    # isolatePeopleByHospital — 醫院偵測與隔離傳染期個體
    # ================================================================
    # 演算法說明：
    #   當個體處於 INFECTIVE 且有醫院政策 (HOSPITAL) 但尚未被隔離時，
    #   檢查是否會被偵測到：
    #   (1) 若該個體有量體溫政策 (TAKE_TEMPERATURE)，以體溫偵測效果
    #       (temp_effect) 的機率被發現。
    #   (2) 若上述未偵測到，但已經過傳染前期 (timer > INFECTIVE_PERIOD)，
    #       則以門診偵測率 (detect_rate) 的機率被發現。
    #   一旦偵測到：
    #   - 設定 isolated = true，統計隔離人數 +1
    #   - 若該個體同時處於居家隔離，則取消居家隔離（因已升級為醫院隔離）
    #   - 將所有分身設為僅家位置可見（模擬住院，不再外出）
    #   - 若啟用接觸追蹤 (trace_on)，以 level=1 啟動追蹤
    #   - 若啟用醫療政策 (CheckBox_MedicalPolicy)，對周圍鄰居散發防疫物資
    #
    # C++ 原始碼 (Unit1.cpp line 712):
    # void TSARS_Form::isolatePeopleByHospital(long ID,
    #     UState state, long timer)
    # {
    #     if ((society.people[ID].attr.policy[TAKE_TEMPERATURE]
    #          && FLIP(Edit_TempEffect->Text.ToDouble()))
    #         || (timer > INFECTIVE_PERIOD
    #             && FLIP(Edit_DetectRate->Text.ToDouble()))) {
    #         society.people[ID].attr.isolated = true;
    #         ++statistic[ISOLATED];
    #         if (society.people[ID].attr.policy[HOME]
    #             && society.people[ID].attr.quarantined) {
    #             society.people[ID].attr.quarantined      = false;
    #             society.people[ID].attr.quarantinedCount = 0;
    #             society.people[ID].attr.quarantinedLevel = 0;
    #         }
    #         for (long count = society.people[ID].count, i = 0;
    #              i < count; i++)
    #             society.people[ID].agent[i].visible =
    #                 society.people[ID].agent[i].home;
    #         if (CheckBox_TraceOn->Checked)
    #             traceContactPeople(ID, 1);
    #         if (CheckBox_MedicalPolicy->Checked)
    #             startMedicalPolicy(ID);
    #     }
    # }
    # ================================================================
    def isolate_by_hospital(self, pid: int, state: int, timer: int) -> None:
        d = self.data
        p = self.params

        detected = False
        if d.people_policy[pid, PI.TAKE_TEMPERATURE] and random.random() < p.temp_effect:
            detected = True
        if not detected and timer > p.infective_period and random.random() < p.detect_rate:
            detected = True

        if not detected:
            return

        d.people_isolated[pid] = True
        d.statistic[S.ISOLATED] += 1

        if d.people_policy[pid, PI.HOME] and d.people_quarantined[pid]:
            d.people_quarantined[pid] = False
            d.people_quarantined_count[pid] = 0
            d.people_quarantined_level[pid] = 0

        for i in range(d.people_count[pid]):
            d.agent_visible[pid, i] = d.agent_home[pid, i]

        if p.trace_on:
            self.trace_contact_people(pid, 1)

        if self.medical_policy_enabled:
            self.start_medical_policy(pid)

    # ------------------------------------------------------------------
    # Gossip / agent movement
    # ------------------------------------------------------------------

    # ================================================================
    # changePeopleByGossip — 個體的分身進行鄰近接觸（gossip）
    # ================================================================
    # 演算法說明：
    #   模擬個體透過其分身 (agent) 與世界地圖上鄰近格子的其他個體
    #   接觸。隨機決定遍歷分身的方向（正序或逆序）。對每個可見的
    #   分身，檢查以下條件決定是否進行接觸：
    #   - 若分身位於「家」位置 (home)，一定接觸
    #   - 若無停止接觸政策 (STOP_CONTACT)，一定接觸
    #   - 若有 STOP_CONTACT 政策，以 1/count 的機率接觸（降低社交活動量）
    #   一旦某個分身成功造成傳染 (changeAgentByGossip 回傳 true)，
    #   即停止該個體本日的接觸（break），代表「每天最多被感染一次」。
    #
    # C++ 原始碼 (Unit1.cpp line 766):
    # void TSARS_Form::changePeopleByGossip(long ID)
    # {
    #     bool dir = FLIP(0.5);
    #     for (long count = society.people[ID].count,
    #          i = (dir ? 0 : count - 1);
    #          (dir ? i < count : i >= 0);
    #          (dir ? i++ : i--))
    #         if (society.people[ID].agent[i].visible
    #             && (society.people[ID].agent[i].home
    #                 || !society.people[ID].attr.policy[STOP_CONTACT]
    #                 || FLIP(1. / society.people[ID].count))
    #             && changeAgentByGossip(ID, i))
    #             break;
    # }
    # ================================================================
    def change_people_by_gossip(self, pid: int) -> None:
        d = self.data
        direction = random.random() < 0.5
        count = int(d.people_count[pid])

        if direction:
            agent_range = range(count)
        else:
            agent_range = range(count - 1, -1, -1)

        for i in agent_range:
            if d.agent_visible[pid, i]:
                if (
                    d.agent_home[pid, i]
                    or not d.people_policy[pid, PI.STOP_CONTACT]
                    or random.random() < (1.0 / count if count > 0 else 1.0)
                ):
                    if self.change_agent_by_gossip(pid, i):
                        break

    # ================================================================
    # changeAgentByGossip — 單一分身執行多次接觸嘗試
    # ================================================================
    # 演算法說明：
    #   決定此分身本次接觸的嘗試次數 (MaxGossip)，規則如下：
    #   - 若有 STOP_CONTACT 政策且分身不在家位置：
    #     MaxGossip = random(0, gossip_steps)，即大幅減少接觸次數
    #   - 若使用固定接觸模式 (gossip_fixed)：
    #     MaxGossip = gossip_steps（固定值）
    #   - 否則：MaxGossip = random(1, gossip_steps)（隨機但至少 1 次）
    #   在最多 MaxGossip 次迴圈中，若分身仍可見且尚未發生傳染，
    #   則呼叫 touchOtherAgent 嘗試與鄰近格子的個體接觸。
    #   一旦發生傳染即停止並回傳 true。
    #
    # C++ 原始碼 (Unit1.cpp line 775):
    # bool TSARS_Form::changeAgentByGossip(long ID, long no)
    # {
    #     bool changed = false;
    #     long MaxGossip =
    #         ((society.people[ID].attr.policy[STOP_CONTACT]
    #           && !society.people[ID].agent[no].home)
    #         ? random(Edit_Gossip->Text.ToInt() + 1)
    #         : (RadioButton_Fixed->Checked
    #            ? Edit_Gossip->Text.ToInt()
    #            : (random(Edit_Gossip->Text.ToInt()) + 1)));
    #     for (long i = 0; !changed && i < MaxGossip; i++)
    #         if (society.people[ID].agent[no].visible)
    #             changed = touchOtherAgent(ID, no);
    #     return changed;
    # }
    # ================================================================
    def change_agent_by_gossip(self, pid: int, no: int) -> bool:
        d = self.data
        p = self.params

        if d.people_policy[pid, PI.STOP_CONTACT] and not d.agent_home[pid, no]:
            max_gossip = random.randint(0, p.gossip_steps)
        elif p.gossip_fixed:
            max_gossip = p.gossip_steps
        else:
            max_gossip = random.randint(1, p.gossip_steps)

        changed = False
        for _ in range(max_gossip):
            if not changed and d.agent_visible[pid, no]:
                changed = self.touch_other_agent(pid, no)
        return changed

    # ------------------------------------------------------------------
    # Agent interaction with a neighbour cell
    # ------------------------------------------------------------------

    # ================================================================
    # touchOtherAgent — 分身與鄰近格子個體的實際接觸與傳染判定
    # ================================================================
    # 演算法說明：
    #   此為接觸傳染的核心方法。流程如下：
    #   (1) 隨機選取目標方位 target (0-8，其中 4 為自身位置)。
    #   (2) 掃描 Moore 鄰域（3x3 格，排除自身）：若發現超級傳播者
    #       (super && INFECTIVE)，則強制將 target 指向該格。世界地圖
    #       使用環形邊界 (torus topology)。
    #   (3) 若 target 為自身 (AGENT_SELF=4)，不接觸，回傳 false。
    #   (4) 根據 target 的 flat index 計算 (row_offset, col_offset)，
    #       找出目標格的世界座標，取得目標個體 (gID) 及其分身編號 (gNo)。
    #   (5) 若目標分身不可見，不接觸。
    #   (6) 傳染判定：若自身為 SUSCEPTIBLE 且目標為 INFECTIVE：
    #       - 訪客限制檢查：若目標已住院隔離，且自身有 STOP_VISITANT
    #         政策且不在家位置，則不接觸（禁止探病）。
    #       - 以 getTransmissionRate 計算傳染機率，隨機判定是否感染。
    #       - 若感染：狀態轉為 EXPOSED，記錄感染來源（院內或社區）。
    #   (7) 居家隔離觸發：若自身有 HOSPITAL 及 HOME 政策但尚未被隔離/
    #       居家隔離：
    #       - 接觸到醫院隔離者 → 被居家隔離 (level=1)，啟動追蹤
    #       - 接觸到 level=1 居家隔離者且啟用 LevelB → 被居家隔離 (level=2)
    #       - 居家隔離後將所有分身設為僅家位置可見
    #
    # C++ 原始碼 (Unit1.cpp line 786):
    # bool TSARS_Form::touchOtherAgent(long ID, long no)
    # {
    #     bool changed   = false;
    #     long target    = random(NEIGHBORS_SIZE) % NEIGHBORS_SIZE;
    #     long MaxHeight = Edit_MaxWorldHeight->Text.ToInt();
    #     long MaxWidth  = Edit_MaxWorldWidth->Text.ToInt();
    #     long baseY     = society.people[ID].agent[no].location.y;
    #     long baseX     = society.people[ID].agent[no].location.x;
    #
    #     for (long count = 0, row = -1; row <= 1; row++)
    #         for (long column = -1; column <= 1; column++, count++)
    #             if (row != 0 || column != 0) {
    #                 long x = (baseX + column + MaxWidth)  % MaxWidth;
    #                 long y = (baseY + row    + MaxHeight) % MaxHeight;
    #                 long gID = world[y][x].peopleID;
    #                 if (society.people[gID].attr.super
    #                     && society.people[gID].state == INFECTIVE) {
    #                     target = count;
    #                     exit;
    #                 }
    #             }
    #
    #     if (target != AGENT_SELF) {
    #         long x = (baseX + ((target==0||target==3||target==6)
    #                   ? -1 : ((target==2||target==5||target==8)
    #                   ? 1 : 0)) + MaxWidth)  % MaxWidth;
    #         long y = (baseY + ((0<=target&&target<=2)
    #                   ? -1 : ((6<=target&&target<=8)
    #                   ? 1 : 0)) + MaxHeight) % MaxHeight;
    #         long gID = world[y][x].peopleID;
    #         long gNo = world[y][x].agentNo;
    #
    #         if (!society.people[gID].agent[gNo].visible)
    #             return changed;
    #
    #         UState gState = society.people[gID].state;
    #
    #         if (society.people[ID].state == SUSCEPTIBLE
    #             && gState == INFECTIVE) {
    #             if (society.people[gID].attr.policy[HOSPITAL]
    #                 && society.people[gID].attr.isolated)
    #                 if (society.people[ID].attr.policy[STOP_VISITANT]
    #                     && !society.people[ID].agent[no].home)
    #                     return changed;
    #             if (FLIP(getTransmissionRate(ID, no, gID, gNo))) {
    #                 ++statistic[(int)(society.people[ID].state
    #                     = EXPOSED)];
    #                 society.people[ID].attr.timer = 0;
    #                 changed = true;
    #                 if (society.people[gID].attr.isolated)
    #                     ++infectedByHospital;
    #                 else
    #                     ++infectedByNormal;
    #             }
    #         }
    #         if (society.people[ID].attr.policy[HOSPITAL]
    #             && !society.people[ID].attr.isolated
    #             && society.people[ID].attr.policy[HOME]
    #             && !society.people[ID].attr.quarantined) {
    #             if (society.people[gID].attr.policy[HOSPITAL]
    #                 && society.people[gID].attr.isolated) {
    #                 society.people[ID].attr.quarantined      = true;
    #                 society.people[ID].attr.quarantinedLevel = 1;
    #                 if (CheckBox_TraceOn->Checked)
    #                     traceContactPeople(ID, 2);
    #             }
    #             else if (society.people[gID].attr.policy[HOME]
    #                 && society.people[gID].attr.quarantined
    #                 && society.people[gID].attr.quarantinedLevel == 1
    #                 && RadioButton_IsolatedLevelB->Checked) {
    #                 society.people[ID].attr.quarantined      = true;
    #                 society.people[ID].attr.quarantinedLevel = 2;
    #             }
    #             if (society.people[ID].attr.policy[HOME]
    #                 && society.people[ID].attr.quarantined) {
    #                 ++statistic[QUARANTINED];
    #                 society.people[ID].attr.quarantinedCount = 0;
    #                 for (long count = society.people[ID].count,
    #                      i = 0; i < count; i++)
    #                     society.people[ID].agent[i].visible =
    #                         society.people[ID].agent[i].home;
    #             }
    #         }
    #     }
    #     return changed;
    # }
    # ================================================================
    def touch_other_agent(self, pid: int, no: int) -> bool:
        d = self.data
        p = self.params
        W = d.W
        H = d.H

        target = random.randint(0, 8)
        baseX = int(d.agent_loc_x[pid, no])
        baseY = int(d.agent_loc_y[pid, no])

        # Scan Moore neighbourhood for a super-spreader.
        count = 0
        for row in (-1, 0, 1):
            for col in (-1, 0, 1):
                if row != 0 or col != 0:
                    x = (baseX + col + W) % W
                    y = (baseY + row + H) % H
                    gID = int(d.world_people_id[y, x])
                    if (
                        gID >= 0
                        and d.people_super[gID]
                        and d.people_state[gID] == S.INFECTIVE
                    ):
                        target = count
                count += 1

        if target == self._AGENT_SELF:
            return False

        # Convert flat target index to (row_offset, col_offset).
        if target in (0, 3, 6):
            col_offset = -1
        elif target in (2, 5, 8):
            col_offset = 1
        else:
            col_offset = 0

        if target in (0, 1, 2):
            row_offset = -1
        elif target in (6, 7, 8):
            row_offset = 1
        else:
            row_offset = 0

        x = (baseX + col_offset + W) % W
        y = (baseY + row_offset + H) % H
        gID = int(d.world_people_id[y, x])
        gNo = int(d.world_agent_no[y, x])

        if gID < 0 or gNo < 0:
            return False

        if not d.agent_visible[gID, gNo]:
            return False

        gState = int(d.people_state[gID])
        changed = False

        # ---- Transmission: susceptible pid meets infective gID ----
        if d.people_state[pid] == S.SUSCEPTIBLE and gState == S.INFECTIVE:
            # Visitant check
            if d.people_policy[gID, PI.HOSPITAL] and d.people_isolated[gID]:
                if d.people_policy[pid, PI.STOP_VISITANT] and not d.agent_home[pid, no]:
                    return False

            if random.random() < self.get_transmission_rate(pid, no, gID, gNo):
                d.people_state[pid] = S.EXPOSED
                self.dirty_pids.add(pid)
                d.statistic[S.EXPOSED] += 1
                d.people_timer[pid] = 0
                changed = True

                if d.people_isolated[gID]:
                    d.infected_by_hospital += 1
                else:
                    d.infected_by_normal += 1

        # ---- Quarantine contact ----
        if (
            d.people_policy[pid, PI.HOSPITAL]
            and not d.people_isolated[pid]
            and d.people_policy[pid, PI.HOME]
            and not d.people_quarantined[pid]
        ):
            triggered = False
            if d.people_policy[gID, PI.HOSPITAL] and d.people_isolated[gID]:
                d.people_quarantined[pid] = True
                d.people_quarantined_level[pid] = 1
                triggered = True
                if p.trace_on:
                    self.trace_contact_people(pid, 2)
            elif (
                d.people_policy[gID, PI.HOME]
                and d.people_quarantined[gID]
                and d.people_quarantined_level[gID] == 1
                and p.isolated_level_b
            ):
                d.people_quarantined[pid] = True
                d.people_quarantined_level[pid] = 2
                triggered = True

            if d.people_policy[pid, PI.HOME] and d.people_quarantined[pid] and triggered:
                d.statistic[S.QUARANTINED] += 1
                d.people_quarantined_count[pid] = 0
                for i in range(d.people_count[pid]):
                    d.agent_visible[pid, i] = d.agent_home[pid, i]

        return changed

    # ------------------------------------------------------------------
    # Transmission & mortality helpers
    # ------------------------------------------------------------------

    # ================================================================
    # getTransmissionRate — 計算傳染機率（考慮各項防疫措施）
    # ================================================================
    # 演算法說明：
    #   以基礎傳染機率 (transmission_prob) 為起點，依據雙方的防疫
    #   政策逐步調整：
    #   (1) 若感染源 (gID) 已被醫院隔離或居家隔離，傳染機率乘以
    #       其分身數量 (count)。原因是隔離者的活動集中在家，接觸密度
    #       提高（模擬探病或家庭內傳播效應）。
    #   (2) 被接觸者 (pid) 的防護：
    #       - 若有醫療政策 (MEDICAL_POLICY)：機率乘以 (1 - 醫療政策效果)
    #       - 否則若有口罩政策 (FACE_MASK)：機率乘以 (1 - 口罩效果)
    #       （醫療政策優先於口罩，不疊加）
    #   (3) 感染源 (gID) 若有口罩政策：機率再乘以 (1 - 口罩效果)
    #       （雙方口罩效果可疊加）
    #   (4) 若雙方都有量體溫政策 (TAKE_TEMPERATURE)：機率再乘以
    #       (1 - 體溫偵測效果)
    #
    # C++ 原始碼 (Unit1.cpp line 1069):
    # double TSARS_Form::getTransmissionRate(long ID, long no,
    #     long gID, long gNo)
    # {
    #     double transmissionProb =
    #         Edit_TransmissionProb->Text.ToDouble();
    #     if ((society.people[gID].attr.policy[HOSPITAL]
    #          && society.people[gID].attr.isolated)
    #         || (society.people[gID].attr.policy[HOME]
    #             && society.people[gID].attr.quarantined))
    #         transmissionProb *= society.people[gID].count;
    #     if (society.people[ID].attr.policy[MEDICAL_POLICY]
    #         && !Edit_MedicalPolicyEffect->Text.IsEmpty())
    #         transmissionProb *=
    #             (1. - Edit_MedicalPolicyEffect->Text.ToDouble());
    #     else if (society.people[ID].attr.policy[FACE_MASK]
    #              && !Edit_MaskEffect->Text.IsEmpty())
    #         transmissionProb *=
    #             (1. - Edit_MaskEffect->Text.ToDouble());
    #     if (society.people[gID].attr.policy[FACE_MASK]
    #         && !Edit_MaskEffect->Text.IsEmpty())
    #         transmissionProb *=
    #             (1. - Edit_MaskEffect->Text.ToDouble());
    #     if (society.people[ID].attr.policy[TAKE_TEMPERATURE]
    #         && society.people[gID].attr.policy[TAKE_TEMPERATURE])
    #         transmissionProb *=
    #             (1. - Edit_TempEffect->Text.ToDouble());
    #     return transmissionProb;
    # }
    # ================================================================
    def get_transmission_rate(
        self, pid: int, no: int, gID: int, gNo: int
    ) -> float:
        d = self.data
        p = self.params

        prob = p.transmission_prob

        if (
            (d.people_policy[gID, PI.HOSPITAL] and d.people_isolated[gID])
            or (d.people_policy[gID, PI.HOME] and d.people_quarantined[gID])
        ):
            prob *= d.people_count[gID]

        if d.people_policy[pid, PI.MEDICAL_POLICY]:
            prob *= 1.0 - p.medical_policy_effect
        elif d.people_policy[pid, PI.FACE_MASK]:
            prob *= 1.0 - p.mask_effect

        if d.people_policy[gID, PI.FACE_MASK]:
            prob *= 1.0 - p.mask_effect

        if (
            d.people_policy[pid, PI.TAKE_TEMPERATURE]
            and d.people_policy[gID, PI.TAKE_TEMPERATURE]
        ):
            prob *= 1.0 - p.temp_effect

        return prob

    # ================================================================
    # getDiedRate — 計算個體的死亡率（依年齡與醫療介入）
    # ================================================================
    # 演算法說明：
    #   根據個體的年齡層 (age) 選取對應的基礎死亡率：
    #   - YOUNG_MAN（青年）：mortality_young
    #   - PRIME_MAN（壯年）：mortality_prime
    #   - OLD_MAN  （老年）：mortality_old
    #   若個體已被醫院隔離 (isolated)，則死亡率乘以
    #   (1 - hospital_effect)，代表醫療照護降低死亡風險。
    #
    # C++ 原始碼 (Unit1.cpp line 1112):
    # double TSARS_Form::getDiedRate(long ID)
    # {
    #     double mortality;
    #     UAge age = society.people[ID].attr.age;
    #     if (age == YOUNG_MAN)
    #         mortality = Edit_MortalityYoungMan->Text.ToDouble();
    #     if (age == PRIME_MAN)
    #         mortality = Edit_MortalityPrimeMan->Text.ToDouble();
    #     if (age == OLD_MAN)
    #         mortality = Edit_MortalityOldMan->Text.ToDouble();
    #     if (society.people[ID].attr.isolated
    #         && !Edit_HospitalEffect->Text.IsEmpty())
    #         mortality *=
    #             (1. - Edit_HospitalEffect->Text.ToDouble());
    #     return mortality;
    # }
    # ================================================================
    def get_died_rate(self, pid: int) -> float:
        d = self.data
        p = self.params
        age = int(d.people_age[pid])

        if age == AgeEnum.YOUNG:
            mortality = p.mortality_young
        elif age == AgeEnum.PRIME:
            mortality = p.mortality_prime
        else:
            mortality = p.mortality_old

        if d.people_isolated[pid]:
            mortality *= 1.0 - p.hospital_effect

        return mortality

    # ------------------------------------------------------------------
    # Contact tracing (recursive)
    # ------------------------------------------------------------------

    # ================================================================
    # traceContactPeople — 接觸者追蹤（遞迴掃描鄰域）
    # ================================================================
    # 演算法說明：
    #   從指定個體 (pid) 出發，掃描其所有分身的 Moore 鄰域（3x3）。
    #   對鄰域中每個格子的個體 (tempID)，若其：
    #   - 狀態為 SUSCEPTIBLE / EXPOSED / INFECTIVE
    #   - 有居家隔離政策 (HOME) 但尚未被居家隔離
    #   則將其強制居家隔離：設定 quarantined=true、計數器歸零、
    #   隔離等級設為傳入的 level 值，並將其所有分身設為僅家位置可見。
    #   若啟用 IsolatedLevelB 且目前 level=1，則對被隔離者遞迴追蹤
    #   其鄰域（level=2），實現二階接觸者追蹤。
    #
    #   Python 版本改為使用 BFS (廣度優先搜尋) 搭配 visited 集合，
    #   避免原始 C++ 遞迴版可能造成的堆疊溢位，同時避免重複處理
    #   已被隔離的個體。
    #
    # C++ 原始碼 (Unit1.cpp line 856):
    # void TSARS_Form::traceContactPeople(long ID, long level)
    # {
    #     long MaxWorldWidth  = Edit_MaxWorldWidth->Text.ToInt();
    #     long MaxWorldHeight = Edit_MaxWorldHeight->Text.ToInt();
    #     for (long count = society.people[ID].count, i = 0;
    #          i < count; i++)
    #         for (long middleX =
    #              society.people[ID].agent[i].location.x,
    #              currX = middleX - 1;
    #              currX <= middleX + 1; currX++) {
    #             long locX = (currX + MaxWorldWidth) % MaxWorldWidth;
    #             for (long middleY =
    #                  society.people[ID].agent[i].location.y,
    #                  currY = middleY - 1;
    #                  currY <= middleY + 1; currY++) {
    #                 long locY   = (currY + MaxWorldHeight)
    #                               % MaxWorldHeight;
    #                 long tempID = world[locY][locX].peopleID;
    #                 long tempState =
    #                     society.people[tempID].state;
    #                 if (tempState == SUSCEPTIBLE
    #                     || tempState == EXPOSED
    #                     || tempState == INFECTIVE)
    #                     if (society.people[tempID].attr.policy[HOME]
    #                         && !society.people[tempID]
    #                              .attr.quarantined) {
    #                         society.people[tempID]
    #                             .attr.quarantined      = true;
    #                         society.people[tempID]
    #                             .attr.quarantinedCount = 0;
    #                         society.people[tempID]
    #                             .attr.quarantinedLevel = level;
    #                         for (long count =
    #                              society.people[tempID].count,
    #                              k = 0; k < count; k++)
    #                             society.people[tempID].agent[k]
    #                                 .visible =
    #                                 society.people[tempID].agent[k]
    #                                     .home;
    #                         ++statistic[QUARANTINED];
    #                         if (RadioButton_IsolatedLevelB->Checked
    #                             && level == 1)
    #                             traceContactPeople(tempID, 2);
    #                     }
    #             }
    #         }
    # }
    # ================================================================
    def trace_contact_people(self, pid: int, level: int) -> None:
        """Iterative BFS contact tracing with visited set.

        Replaces the original recursive version to avoid stack overflow
        and redundant processing of already-quarantined people.
        """
        d = self.data
        p = self.params
        W = d.W
        H = d.H

        queue = deque()
        queue.append((pid, level))
        visited = {pid}

        while queue:
            cur_pid, cur_level = queue.popleft()

            for agent_idx in range(d.people_count[cur_pid]):
                locX = int(d.agent_loc_x[cur_pid, agent_idx])
                locY = int(d.agent_loc_y[cur_pid, agent_idx])

                for row in (-1, 0, 1):
                    for col in (-1, 0, 1):
                        nx = (locX + col + W) % W
                        ny = (locY + row + H) % H
                        tempID = int(d.world_people_id[ny, nx])

                        if tempID < 0 or tempID in visited:
                            continue
                        visited.add(tempID)

                        tempState = int(d.people_state[tempID])
                        if (
                            tempState in (S.SUSCEPTIBLE, S.EXPOSED, S.INFECTIVE)
                            and d.people_policy[tempID, PI.HOME]
                            and not d.people_quarantined[tempID]
                        ):
                            d.people_quarantined[tempID] = True
                            d.people_quarantined_count[tempID] = 0
                            d.people_quarantined_level[tempID] = cur_level

                            for k in range(d.people_count[tempID]):
                                d.agent_visible[tempID, k] = d.agent_home[tempID, k]

                            d.statistic[S.QUARANTINED] += 1

                            if p.isolated_level_b and cur_level == 1:
                                queue.append((tempID, 2))

    # ------------------------------------------------------------------
    # Medical-policy spread / removal
    # ------------------------------------------------------------------

    # ================================================================
    # startMedicalPolicy — 對被隔離者的家周圍鄰居發放醫療防護
    # ================================================================
    # 演算法說明：
    #   當個體被送醫隔離後，針對其「家」位置的分身，掃描 Moore
    #   鄰域（3x3）。對鄰域中尚未擁有醫療政策 (MEDICAL_POLICY)
    #   的個體，以 medical_policy_available 的機率賦予該政策。
    #   這模擬了衛生單位在病患住所附近發放口罩、消毒液等防疫物資
    #   的行為。僅針對「家」位置的分身執行，因為只有居住地的鄰居
    #   才會被納入防疫範圍。
    #
    # C++ 原始碼 (Unit1.cpp line 732):
    # void TSARS_Form::startMedicalPolicy(long ID)
    # {
    #     long MaxWorldWidth  = Edit_MaxWorldWidth->Text.ToInt();
    #     long MaxWorldHeight = Edit_MaxWorldHeight->Text.ToInt();
    #     for (long count = society.people[ID].count, i = 0;
    #          i < count; i++)
    #         if (society.people[ID].agent[i].home)
    #             for (long middleX =
    #                  society.people[ID].agent[i].location.x,
    #                  currX = middleX - 1;
    #                  currX <= middleX + 1; currX++) {
    #                 long locX = (currX + MaxWorldWidth)
    #                             % MaxWorldWidth;
    #                 for (long middleY =
    #                      society.people[ID].agent[i].location.y,
    #                      currY = middleY - 1;
    #                      currY <= middleY + 1; currY++) {
    #                     long locY = (currY + MaxWorldHeight)
    #                                 % MaxWorldHeight;
    #                     long tempID =
    #                         world[locY][locX].peopleID;
    #                     if (!society.people[tempID]
    #                          .attr.policy[MEDICAL_POLICY])
    #                         society.people[tempID]
    #                             .attr.policy[MEDICAL_POLICY] =
    #                             FLIP(Edit_MedicalPolicyAvailable
    #                                 ->Text.ToDouble());
    #                 }
    #             }
    # }
    # ================================================================
    def start_medical_policy(self, pid: int) -> None:
        d = self.data
        p = self.params
        W = d.W
        H = d.H

        for agent_idx in range(d.people_count[pid]):
            if not d.agent_home[pid, agent_idx]:
                continue

            locX = int(d.agent_loc_x[pid, agent_idx])
            locY = int(d.agent_loc_y[pid, agent_idx])

            for row in (-1, 0, 1):
                for col in (-1, 0, 1):
                    nx = (locX + col + W) % W
                    ny = (locY + row + H) % H
                    tempID = int(d.world_people_id[ny, nx])

                    if tempID < 0:
                        continue

                    if not d.people_policy[tempID, PI.MEDICAL_POLICY]:
                        if random.random() < p.medical_policy_available:
                            d.people_policy[tempID, PI.MEDICAL_POLICY] = True

    # ================================================================
    # finishMedicalPolicy — 移除被隔離者家周圍鄰居的醫療防護
    # ================================================================
    # 演算法說明：
    #   當被隔離的個體康復或死亡（離開 INFECTIVE 狀態）時呼叫。
    #   掃描其「家」位置分身的 Moore 鄰域（3x3），將鄰域中所有
    #   個體的醫療政策 (MEDICAL_POLICY) 設為 false。
    #   這代表防疫物資的效期結束或衛生單位撤除防疫措施。
    #   注意：此操作與 startMedicalPolicy 對稱，但移除時不做機率
    #   判定，而是無條件全部移除。
    #
    # C++ 原始碼 (Unit1.cpp line 750):
    # void TSARS_Form::finishMedicalPolicy(long ID)
    # {
    #     long MaxWorldWidth  = Edit_MaxWorldWidth->Text.ToInt();
    #     long MaxWorldHeight = Edit_MaxWorldHeight->Text.ToInt();
    #     for (long count = society.people[ID].count, i = 0;
    #          i < count; i++)
    #         if (society.people[ID].agent[i].home)
    #             for (long middleX =
    #                  society.people[ID].agent[i].location.x,
    #                  currX = middleX - 1;
    #                  currX <= middleX + 1; currX++) {
    #                 long locX = (currX + MaxWorldWidth)
    #                             % MaxWorldWidth;
    #                 for (long middleY =
    #                      society.people[ID].agent[i].location.y,
    #                      currY = middleY - 1;
    #                      currY <= middleY + 1; currY++) {
    #                     long locY = (currY + MaxWorldHeight)
    #                                 % MaxWorldHeight;
    #                     society.people[
    #                         world[locY][locX].peopleID
    #                     ].attr.policy[MEDICAL_POLICY] = false;
    #                 }
    #             }
    # }
    # ================================================================
    def finish_medical_policy(self, pid: int) -> None:
        d = self.data
        W = d.W
        H = d.H

        for agent_idx in range(d.people_count[pid]):
            if not d.agent_home[pid, agent_idx]:
                continue

            locX = int(d.agent_loc_x[pid, agent_idx])
            locY = int(d.agent_loc_y[pid, agent_idx])

            for row in (-1, 0, 1):
                for col in (-1, 0, 1):
                    nx = (locX + col + W) % W
                    ny = (locY + row + H) % H
                    tempID = int(d.world_people_id[ny, nx])

                    if tempID < 0:
                        continue

                    d.people_policy[tempID, PI.MEDICAL_POLICY] = False
