"""Policy management for the SARS epidemic simulation (ported from C++)."""

import random

import numpy as np

from .models import PolicyIndex, StateEnum


class PolicyManager:
    """Applies public-health policy actions to the simulated population.

    Each ``apply_*`` method mirrors the corresponding checkbox / slider
    interaction in the original C++ GUI.  Bulk operations are vectorized
    with NumPy; only :meth:`apply_vaccine` uses a scalar loop because it
    must pick *distinct* un-vaccinated individuals.
    """

    def __init__(self, params, data):
        """
        Parameters
        ----------
        params : SimulationParams
            Read-only reference to current simulation parameters.
        data : SimulationData
            Mutable simulation state (people arrays, statistics, etc.).
        """
        self.params = params
        self.data = data

    # ------------------------------------------------------------------
    # Mask
    # ------------------------------------------------------------------
    #
    # 演算法說明（口罩政策）：
    #   對應 C++ 原始碼中的 CheckBox_MaskClick 事件處理函式。
    #   當使用者勾選「口罩」核取方塊且口罩供給率欄位非空時，
    #   遍歷所有人口，對每個人以伯努利試驗（Bernoulli trial）決定
    #   是否配戴口罩。機率由 available（口罩可得率）決定。
    #   原始 C++ 使用巨集 FLIP(p) 產生均勻亂數並與 p 比較，
    #   等價於 random_value < p 的伯努利試驗。
    #   Python 版本使用 NumPy 向量化運算：
    #     np.random.random(N) < available
    #   一次為所有人口產生布林遮罩，效能遠優於逐人迴圈。
    #   若政策未啟用（enabled=False），則將所有人的口罩旗標清除。
    #
    # 對應 C++ 原始碼 (Unit1.cpp, line 1051):
    # ---------------------------------------------------------------
    # void __fastcall TSARS_Form::CheckBox_MaskClick(TObject *Sender)
    # {
    #     if (!Edit_MaskAvailable->Text.IsEmpty()) {
    #         for (long MaxPopulation = Edit_MaxPopulation->Text.ToInt(),
    #              i = 0;
    #              i < MaxPopulation;
    #              society.people[i++].attr.policy[FACE_MASK] =
    #                  CheckBox_Mask->Checked &&
    #                  FLIP(Edit_MaskAvailable->Text.ToDouble()));
    #         ...
    #     }
    # }
    # ---------------------------------------------------------------
    # 其中 #define FLIP(x)
    #         ((double)random(1000000L) < ((double)(x) * (1000000.)))
    # FLIP(p) 等價於以機率 p 回傳 true 的伯努利試驗。
    #
    def apply_mask_policy(self, enabled, available):
        """Vectorized: each person gets FACE_MASK if *enabled* and FLIP(*available*).

        Parameters
        ----------
        enabled : bool
            Whether the mask policy checkbox is ticked.
        available : float
            Probability (0-1) that a given person actually wears a mask.
        """
        # 注意：此處使用 params.max_population（= actual_pop）而非
        # data.N（原始最大人口數）。這是刻意設計：初始化後
        # max_population 被縮減為實際擁有分身的人數，因此模擬中途
        # 切換政策時，僅對前 actual_pop 個索引的個體生效，索引
        # >= actual_pop 的個體維持初始化時的政策狀態不變。
        # 其餘政策方法（apply_hospital_policy 等）皆同此設計。
        N = self.params.max_population
        if enabled:
            self.data.people_policy[:N, PolicyIndex.FACE_MASK] = (
                np.random.random(N) < available
            )
        else:
            self.data.people_policy[:N, PolicyIndex.FACE_MASK] = False

    # ------------------------------------------------------------------
    # Hospital isolation
    # ------------------------------------------------------------------
    #
    # 演算法說明（醫院隔離政策）：
    #   對應 C++ 原始碼中的 CheckBox_HospitalClick 事件處理函式。
    #   當使用者勾選「醫院隔離」核取方塊且醫院可用率欄位非空時，
    #   遍歷所有人口，對每個人以伯努利試驗決定是否可送醫隔離。
    #   機率由 available（醫院可用率）決定。
    #   原始 C++ 同樣使用 FLIP(p) 巨集進行隨機判定。
    #   Python 版本以 np.random.random(N) < available 向量化實作。
    #
    #   特殊邏輯：當政策被關閉（enabled=False）時，除了清除
    #   HOSPITAL 旗標外，還須將所有人的 isolated（隔離中）屬性
    #   重設為 False，以確保無人殘留在隔離狀態中。
    #   這對應 C++ 中 CheckBox_Hospital->Checked 為 false 時的
    #   第二個迴圈，將 society.people[i].attr.isolated 設為 false。
    #
    # 對應 C++ 原始碼 (Unit1.cpp, line 1091):
    # ---------------------------------------------------------------
    # void __fastcall TSARS_Form::CheckBox_HospitalClick(TObject *Sender)
    # {
    #     if (!Edit_HospitalAvailable->Text.IsEmpty()) {
    #         for (long MaxPopulation = Edit_MaxPopulation->Text.ToInt(),
    #              i = 0;
    #              i < MaxPopulation;
    #              i++)
    #             society.people[i].attr.policy[HOSPITAL] =
    #                 CheckBox_Hospital->Checked &&
    #                 FLIP(Edit_HospitalAvailable->Text.ToDouble());
    #         ...
    #     }
    #     if (!CheckBox_Hospital->Checked)
    #         for (long MaxPopulation = Edit_MaxPopulation->Text.ToInt(),
    #              i = 0;
    #              i < MaxPopulation;
    #              society.people[i++].attr.isolated = false);
    # }
    # ---------------------------------------------------------------
    # 其中 FLIP(p) 等價於以機率 p 回傳 true 的伯努利試驗。
    # 第二個迴圈確保政策關閉時解除所有人的隔離狀態。
    #
    def apply_hospital_policy(self, enabled, available):
        """Vectorized: each person gets HOSPITAL if *enabled* and FLIP(*available*).

        When the hospital policy is disabled, existing isolation flags are
        also cleared so that no one remains stuck in an isolation state.

        Parameters
        ----------
        enabled : bool
            Whether the hospital isolation checkbox is ticked.
        available : float
            Probability (0-1) that a given person is admitted to hospital.
        """
        N = self.params.max_population
        if enabled:
            self.data.people_policy[:N, PolicyIndex.HOSPITAL] = (
                np.random.random(N) < available
            )
        else:
            self.data.people_policy[:N, PolicyIndex.HOSPITAL] = False
            self.data.people_isolated[:N] = False  # clear isolation when hospital disabled

    # ------------------------------------------------------------------
    # Temperature screening
    # ------------------------------------------------------------------
    #
    # 演算法說明（量體溫政策）：
    #   對應 C++ 原始碼中的 CheckBox_TempClick 事件處理函式。
    #   當使用者勾選「量體溫」核取方塊且量測可用率欄位非空時，
    #   遍歷所有人口，對每個人以伯努利試驗決定是否接受體溫篩檢。
    #   機率由 available（體溫量測可用率）決定。
    #   此政策為純旗標設定，無額外的狀態清除邏輯。
    #   當 enabled=False 時，僅將 TAKE_TEMPERATURE 旗標全部清除。
    #   Python 版本以 np.random.random(N) < available 向量化實作。
    #
    # 對應 C++ 原始碼 (Unit1.cpp, line 1167):
    # ---------------------------------------------------------------
    # void __fastcall TSARS_Form::CheckBox_TempClick(TObject *Sender)
    # {
    #     if (!Edit_TempAvailable->Text.IsEmpty()) {
    #         for (long MaxPopulation = Edit_MaxPopulation->Text.ToInt(),
    #              i = 0;
    #              i < MaxPopulation;
    #              society.people[i++].attr.policy[TAKE_TEMPERATURE] =
    #                  CheckBox_Temp->Checked &&
    #                  FLIP(Edit_TempAvailable->Text.ToDouble()));
    #         ...
    #     }
    # }
    # ---------------------------------------------------------------
    # 其中 FLIP(p) 等價於以機率 p 回傳 true 的伯努利試驗。
    #
    def apply_temperature_policy(self, enabled, available):
        """Vectorized: TAKE_TEMPERATURE policy.

        Parameters
        ----------
        enabled : bool
            Whether the temperature-check checkbox is ticked.
        available : float
            Probability (0-1) that a given person is screened.
        """
        N = self.params.max_population
        if enabled:
            self.data.people_policy[:N, PolicyIndex.TAKE_TEMPERATURE] = (
                np.random.random(N) < available
            )
        else:
            self.data.people_policy[:N, PolicyIndex.TAKE_TEMPERATURE] = False

    # ------------------------------------------------------------------
    # Home quarantine
    # ------------------------------------------------------------------
    #
    # 演算法說明（居家隔離政策）：
    #   對應 C++ 原始碼中的 CheckBox_HomeClick 事件處理函式。
    #   當使用者勾選「居家隔離」核取方塊且居家隔離可用率欄位非空時，
    #   遍歷所有人口，對每個人以伯努利試驗決定是否執行居家隔離。
    #   機率由 available（居家隔離可用率）決定。
    #
    #   特殊邏輯：當政策被關閉（enabled=False）時，除了清除
    #   HOME 旗標外，還須重設三個居家隔離相關屬性：
    #     - quarantined（是否正在居家隔離中）→ False
    #     - quarantinedCount（居家隔離天數計數器）→ 0
    #     - quarantinedLevel（居家隔離層級）→ 0
    #   這確保政策關閉後，無人殘留在居家隔離狀態中。
    #   Python 版本以 np.random.random(N) < available 向量化實作。
    #
    # 對應 C++ 原始碼 (Unit1.cpp, line 1232):
    # ---------------------------------------------------------------
    # void __fastcall TSARS_Form::CheckBox_HomeClick(TObject *Sender)
    # {
    #     if (!Edit_HomeAvailable->Text.IsEmpty()) {
    #         for (long MaxPopulation = Edit_MaxPopulation->Text.ToInt(),
    #              i = 0;
    #              i < MaxPopulation;
    #              society.people[i++].attr.policy[HOME] =
    #                  CheckBox_Home->Checked &&
    #                  FLIP(Edit_HomeAvailable->Text.ToDouble()));
    #         ...
    #     }
    #     if (!CheckBox_Home->Checked)
    #         for (long MaxPopulation = Edit_MaxPopulation->Text.ToInt(),
    #              i = 0;
    #              i < MaxPopulation;
    #              i++) {
    #             society.people[i].attr.quarantined      = false;
    #             society.people[i].attr.quarantinedCount = 0;
    #             society.people[i].attr.quarantinedLevel = 0;
    #         }
    # }
    # ---------------------------------------------------------------
    # 其中 FLIP(p) 等價於以機率 p 回傳 true 的伯努利試驗。
    # 第二個迴圈確保政策關閉時解除所有隔離狀態及歸零計數器。
    #
    def apply_home_quarantine_policy(self, enabled, available):
        """Vectorized: HOME quarantine policy.

        Disabling the policy also resets all quarantine-related counters
        and levels so that no one remains in quarantine.

        Parameters
        ----------
        enabled : bool
            Whether the home quarantine checkbox is ticked.
        available : float
            Probability (0-1) that a given person is quarantined at home.
        """
        N = self.params.max_population
        if enabled:
            self.data.people_policy[:N, PolicyIndex.HOME] = (
                np.random.random(N) < available
            )
        else:
            self.data.people_policy[:N, PolicyIndex.HOME] = False
            self.data.people_quarantined[:N] = False
            self.data.people_quarantined_count[:N] = 0
            self.data.people_quarantined_level[:N] = 0

    # ------------------------------------------------------------------
    # Visit restriction
    # ------------------------------------------------------------------
    #
    # 演算法說明（禁止探病政策）：
    #   對應 C++ 原始碼中的 CheckBox_VisitClick 事件處理函式。
    #   當使用者勾選「禁止探病」核取方塊且探病限制可用率欄位非空時，
    #   遍歷所有人口，對每個人以伯努利試驗決定是否限制訪客探視。
    #   機率由 available（探病限制可用率）決定。
    #   此政策為純旗標設定，無額外的狀態清除邏輯。
    #   當 enabled=False 時，僅將 STOP_VISITANT 旗標全部清除。
    #   Python 版本以 np.random.random(N) < available 向量化實作。
    #
    # 對應 C++ 原始碼 (Unit1.cpp, line 1185):
    # ---------------------------------------------------------------
    # void __fastcall TSARS_Form::CheckBox_VisitClick(TObject *Sender)
    # {
    #     if (!Edit_VisitAvailable->Text.IsEmpty()) {
    #         for (long MaxPopulation = Edit_MaxPopulation->Text.ToInt(),
    #              i = 0;
    #              i < MaxPopulation;
    #              society.people[i++].attr.policy[STOP_VISITANT] =
    #                  CheckBox_Visit->Checked &&
    #                  FLIP(Edit_VisitAvailable->Text.ToDouble()));
    #         ...
    #     }
    # }
    # ---------------------------------------------------------------
    # 其中 FLIP(p) 等價於以機率 p 回傳 true 的伯努利試驗。
    #
    def apply_visit_restriction(self, enabled, available):
        """Vectorized: STOP_VISITANT policy.

        Parameters
        ----------
        enabled : bool
            Whether the visitor-restriction checkbox is ticked.
        available : float
            Probability (0-1) that a given person stops receiving visitors.
        """
        N = self.params.max_population
        if enabled:
            self.data.people_policy[:N, PolicyIndex.STOP_VISITANT] = (
                np.random.random(N) < available
            )
        else:
            self.data.people_policy[:N, PolicyIndex.STOP_VISITANT] = False

    # ------------------------------------------------------------------
    # Contact reduction
    # ------------------------------------------------------------------
    #
    # 演算法說明（減少接觸政策）：
    #   對應 C++ 原始碼中的 CheckBox_ContactClick 事件處理函式。
    #   當使用者勾選「減少接觸」核取方塊且接觸限制可用率欄位非空時，
    #   遍歷所有人口，對每個人以伯努利試驗決定是否減少社交接觸。
    #   機率由 available（接觸限制可用率）決定。
    #   此政策為純旗標設定，無額外的狀態清除邏輯。
    #   當 enabled=False 時，僅將 STOP_CONTACT 旗標全部清除。
    #   Python 版本以 np.random.random(N) < available 向量化實作。
    #
    # 對應 C++ 原始碼 (Unit1.cpp, line 1283):
    # ---------------------------------------------------------------
    # void __fastcall TSARS_Form::CheckBox_ContactClick(TObject *Sender)
    # {
    #     if (!Edit_ContactAvailable->Text.IsEmpty()) {
    #         for (long MaxPopulation = Edit_MaxPopulation->Text.ToInt(),
    #              i = 0;
    #              i < MaxPopulation;
    #              society.people[i++].attr.policy[STOP_CONTACT] =
    #                  CheckBox_Contact->Checked &&
    #                  FLIP(Edit_ContactAvailable->Text.ToDouble()));
    #         ...
    #     }
    # }
    # ---------------------------------------------------------------
    # 其中 FLIP(p) 等價於以機率 p 回傳 true 的伯努利試驗。
    #
    def apply_contact_reduction(self, enabled, available):
        """Vectorized: STOP_CONTACT policy.

        Parameters
        ----------
        enabled : bool
            Whether the contact-reduction checkbox is ticked.
        available : float
            Probability (0-1) that a given person reduces contacts.
        """
        N = self.params.max_population
        if enabled:
            self.data.people_policy[:N, PolicyIndex.STOP_CONTACT] = (
                np.random.random(N) < available
            )
        else:
            self.data.people_policy[:N, PolicyIndex.STOP_CONTACT] = False

    # ------------------------------------------------------------------
    # Vaccine
    # ------------------------------------------------------------------
    #
    # 演算法說明（疫苗接種政策）：
    #   對應 C++ 原始碼中的 Button_VaccineClick 事件處理函式。
    #   與其他政策不同，疫苗接種不是使用伯努利試驗對全體人口設定旗標，
    #   而是從尚未接種疫苗的人口中，隨機挑選指定數量（count）的「不重複」
    #   個體進行疫苗接種。
    #
    #   演算法步驟：
    #     1. 計算尚未接種疫苗的人數（unvaccinated）。
    #        若無人可接種，直接返回。
    #     2. 將接種數量上限設為 min(count, unvaccinated)，避免無窮迴圈。
    #     3. 對每一劑疫苗，以拒絕取樣法（rejection sampling）隨機選取
    #        一位尚未接種的個體：
    #        - 不斷產生隨機 ID，直到找到 policy[VACCINE] == False 的人。
    #        - 原始 C++ 使用 do-while 迴圈：
    #          do {} while (society.people[(ID = _lrand() % Max)].attr.policy[VACCINE]);
    #     4. 將該個體的 VACCINE 旗標設為 True。
    #     5. 若該個體尚未具有免疫力（immunity == False）：
    #        - 將其狀態轉為 IMMUNE（免疫）。
    #        - 統計計數器 statistic[IMMUNE] 加一。
    #        - 根據 forever 參數決定是否永久免疫。
    #        - 重設計時器 timer = 0。
    #
    #   此方法使用純量迴圈而非向量化運算，因為每次接種必須挑選
    #   「不重複」的個體，且接種數量通常較少。
    #   Python 版本加入安全閥（safety valve）：若嘗試次數超過 N*10
    #   仍未找到可接種者，則提前返回，避免極端情況下的無窮迴圈。
    #
    # 對應 C++ 原始碼 (Unit1.cpp, line 1212):
    # ---------------------------------------------------------------
    # void __fastcall TSARS_Form::Button_VaccineClick(TObject *Sender)
    # {
    #     long ID;
    #     if (!Edit_VaccineAvailable->Text.IsEmpty()) {
    #         for (long vaccine = Edit_VaccineAvailable->Text.ToInt(),
    #              i = 0;
    #              i < vaccine;
    #              i++) {
    #             do {} while (
    #                 society.people[
    #                     (ID = (long)_lrand() %
    #                      Edit_MaxPopulation->Text.ToInt())
    #                 ].attr.policy[VACCINE]);
    #             society.people[ID].attr.policy[VACCINE] = true;
    #             if (!society.people[ID].attr.immunity) {
    #                 ++statistic[(int)(
    #                     society.people[ID].state = IMMUNE)];
    #                 society.people[ID].attr.immunity =
    #                     CheckBox_VaccineForever->Checked;
    #                 society.people[ID].attr.timer = 0;
    #             }
    #         }
    #         ...
    #         showResult();
    #     }
    # }
    # ---------------------------------------------------------------
    # 注意：C++ 版本的 do-while 迴圈在人口幾乎全部接種完畢時
    # 可能需要大量迭代才能找到未接種者。Python 版本透過預先計算
    # 未接種人數並設置安全閥來緩解此問題。
    #
    def apply_vaccine(self, count, forever):
        """Apply vaccine to *count* random people who do not already have VACCINE.

        This is intentionally a scalar loop: each vaccinated person must be
        a *distinct* individual chosen uniformly at random from the
        un-vaccinated sub-population, and the count is typically small.

        Parameters
        ----------
        count : int
            Number of people to vaccinate in this batch.
        forever : bool
            If ``True`` the person gains permanent immunity; otherwise the
            immunity duration is governed by the normal immune-period timer.
        """
        N = self.params.max_population
        d = self.data

        # Safety valve: cap count to number of un-vaccinated people.
        unvaccinated = int(np.sum(~d.people_policy[:N, PolicyIndex.VACCINE]))
        if unvaccinated == 0:
            return
        count = min(count, unvaccinated)

        for _ in range(count):
            # Pick a random person who has not already been vaccinated.
            attempts = 0
            while True:
                pid = random.randrange(N)
                if not d.people_policy[pid, PolicyIndex.VACCINE]:
                    break
                attempts += 1
                if attempts > N * 10:
                    return  # safety valve
            d.people_policy[pid, PolicyIndex.VACCINE] = True
            if not d.people_immunity[pid]:
                d.statistic[StateEnum.IMMUNE] += 1
                d.people_state[pid] = StateEnum.IMMUNE
                d.people_immunity[pid] = forever
                d.people_timer[pid] = 0

    # ------------------------------------------------------------------
    # Medical policy toggle
    # ------------------------------------------------------------------
    #
    # 演算法說明（醫療政策總開關）：
    #   對應 C++ 原始碼中的 CheckBox_MedicalPolicyClick 事件處理函式。
    #   此方法為醫療政策的總開關。當使用者取消勾選「醫療政策」核取方塊時，
    #   遍歷所有人口，將每個人的 MEDICAL_POLICY 旗標清除為 False。
    #   與其他政策不同，此方法僅處理「關閉」的情況——
    #   啟用醫療政策的具體效果由其他子政策方法分別處理。
    #   此方法的作用是確保總開關關閉時，所有人的醫療政策旗標
    #   被一致性地清除。
    #   Python 版本以 NumPy 陣列切片直接賦值 False，為向量化操作。
    #
    # 對應 C++ 原始碼 (Unit1.cpp, line 1428):
    # ---------------------------------------------------------------
    # void __fastcall TSARS_Form::CheckBox_MedicalPolicyClick(
    #     TObject *Sender)
    # {
    #     ...
    #     if (!CheckBox_MedicalPolicy->Checked)
    #         for (long MaxPopulation = Edit_MaxPopulation->Text.ToInt(),
    #              i = 0;
    #              i < MaxPopulation;
    #              society.people[i++].attr.policy[MEDICAL_POLICY] =
    #                  false);
    # }
    # ---------------------------------------------------------------
    # 此處僅處理取消勾選的情況，將所有人口的 MEDICAL_POLICY
    # 旗標重設為 false。
    #
    def apply_medical_policy_toggle(self, enabled):
        """When the medical-policy checkbox is unchecked, clear all flags.

        Parameters
        ----------
        enabled : bool
            Current state of the medical-policy checkbox.
        """
        if not enabled:
            N = self.params.max_population
            self.data.people_policy[:N, PolicyIndex.MEDICAL_POLICY] = False
