"""World/Lattice management for the SARS epidemic simulation.

Handles initialization of the lattice grid, population setup, and
distributed agent placement. Ported from the original C++ implementation.
"""

import numpy as np

from .models import (
    AgeEnum,
    Colors,
    SimulationData,
    SimulationParams,
    StateEnum,
)


class WorldManager:
    """Manages the simulation lattice and population initialization.

    Parameters
    ----------
    params : SimulationParams
        Configuration parameters for the simulation.
    data : SimulationData
        Shared simulation state arrays (modified in-place).
    """

    def __init__(self, params: SimulationParams, data: SimulationData) -> None:
        self.params = params
        self.data = data

    # ------------------------------------------------------------------
    # Lattice reset
    # ------------------------------------------------------------------

    # ================================================================
    # 【init_world / initWorld】
    # ----------------------------------------------------------------
    # 演算法說明（繁體中文）：
    #   初始化世界格子（Lattice）。將每一個格子的 peopleID 和 agentNo
    #   設為 EMPTY（-1），代表該格子尚未被任何人的分身點佔用。
    #   同時將所有格子的顏色重設為預設色（天藍色 DEFAULT_COLOR）。
    #   此方法在模擬開始前呼叫，確保格子處於乾淨的初始狀態。
    #
    # 對應 C++ 原始碼（Unit1.cpp, initWorld, line 202）：
    #   void TSARS_Form::initWorld(void)
    #   {
    #       for (long MaxHeight = Edit_MaxWorldHeight->Text.ToInt(),
    #            i = 0; i < MaxHeight; i++)
    #           for (long MaxWidth = Edit_MaxWorldWidth->Text.ToInt(),
    #                j = 0; j < MaxWidth; j++) {
    #               world[i][j].peopleID = world[i][j].agentNo = EMPTY;
    #               world[i][j].color    = DEFAULT_COLOR;
    #           }
    #   }
    #
    # Python 版本差異：
    #   使用 NumPy 陣列的切片賦值一次完成，等價於 C++ 的雙重迴圈。
    # ================================================================
    def init_world(self) -> None:
        """Reset the lattice to its default empty state.

        Sets every cell to unoccupied (people_id = -1, agent_no = -1)
        and paints the entire grid with the default sky-blue color.
        """
        self.data.world_people_id[:] = -1
        self.data.world_agent_no[:] = -1
        self.data.world_color[:] = Colors.DEFAULT

    # ------------------------------------------------------------------
    # Population initialization (vectorized)
    # ------------------------------------------------------------------

    # ================================================================
    # 【init_society / initSociety + initPeople + initAttribute + initAgents】
    # ----------------------------------------------------------------
    # 演算法說明（繁體中文）：
    #   初始化整個社會的人口。對應 C++ 中四個函式的合併：
    #     1. initSociety：迴圈呼叫 initPeople(i) 初始化每一個人。
    #     2. initPeople(ID)：設定每人的 ID、健康狀態、計數器，然後
    #        呼叫 initAttribute 和 initAgents。
    #     3. initAttribute(ID, state)：
    #        - immunity 根據狀態是否為 IMMUNE 設定。
    #        - super（超級傳播者）設為 false。
    #        - timer 計時器歸零。
    #        - age 年齡分類：先擲骰子判斷是否為老年人（OLD_MAN），
    #          若否再擲一次判斷是否為年輕人（YOUNG_MAN），
    #          否則為壯年人（PRIME_MAN）。
    #        - isolated、quarantined 等隔離旗標全部歸零。
    #        - policy 陣列全部設為 false。
    #     4. initAgents(ID) / initAgent(ID, no)：
    #        - 將該人所有的分身點插槽（agent slot）設為不可見、
    #          非根據地、位置為 EMPTY。
    #   Python 版本使用向量化 NumPy 運算一次完成全部初始化。
    #
    # 對應 C++ 原始碼（Unit1.cpp, lines 213-264）：
    #   void TSARS_Form::initSociety(TObject *Sender)
    #   {
    #       for (long MaxPopulation = Edit_MaxPopulation->Text.ToInt(),
    #            i = society.day = 0; i < MaxPopulation; initPeople(i++));
    #       initPolicies(Sender);
    #   }
    #
    #   void TSARS_Form::initPeople(long ID)
    #   {
    #       society.people[ID].ID    = ID;
    #       society.people[ID].state = (FLIP(Edit_ImmuneProb->Text.ToDouble())
    #                                   ? IMMUNE : SUSCEPTIBLE);
    #       society.people[ID].count = 0;
    #       society.people[ID].limit = 0;
    #       initAttribute(ID, society.people[ID].state);
    #       initAgents(ID);
    #   }
    #
    #   void TSARS_Form::initAttribute(long ID, UState state)
    #   {
    #       society.people[ID].attr.immunity         = (state == IMMUNE);
    #       society.people[ID].attr.super            = false;
    #       society.people[ID].attr.timer            = 0;
    #       society.people[ID].attr.age              =
    #           (FLIP(Edit_PopulationRateOldMan->Text.ToDouble())
    #            ? OLD_MAN
    #            : (FLIP(Edit_PopulationRateYoungMan->Text.ToDouble())
    #               ? YOUNG_MAN : PRIME_MAN));
    #       society.people[ID].attr.isolated         = false;
    #       society.people[ID].attr.quarantined      = false;
    #       society.people[ID].attr.quarantinedCount = 0;
    #       society.people[ID].attr.quarantinedLevel = 0;
    #       for (long i = 0; i < MAX_POLICY;
    #            society.people[ID].attr.policy[i++] = false);
    #   }
    #
    #   void TSARS_Form::initAgents(long ID)
    #   {
    #       for (long MaxAgent = Edit_MaxAgent->Text.ToInt(),
    #            i = 0; i < MaxAgent; initAgent(ID, i++));
    #   }
    #
    #   void TSARS_Form::initAgent(long ID, long no)
    #   {
    #       society.people[ID].agent[no].no         = no;
    #       society.people[ID].agent[no].visible    = false;
    #       society.people[ID].agent[no].home       = false;
    #       society.people[ID].agent[no].location.x = EMPTY;
    #       society.people[ID].agent[no].location.y = EMPTY;
    #   }
    #
    # Python 版本差異：
    #   使用 NumPy 向量化操作取代逐人迴圈，效能大幅提升。
    #   年齡分配的兩階段 FLIP 邏輯以兩個獨立的 rng.random(N) 實現。
    # ================================================================
    def init_society(self) -> None:
        """Initialize all people with random attributes (vectorized).

        Each person is assigned an initial health state (IMMUNE with
        probability ``immune_prob``, otherwise SUSCEPTIBLE), an age
        category, and all tracking fields are zeroed out.  Agent slots
        are reset to invisible / unplaced.
        """
        p = self.params
        d = self.data
        N = d.N
        rng = np.random.default_rng()

        # --- State: IMMUNE with probability immune_prob, else SUSCEPTIBLE ---
        rolls = rng.random(N)
        d.people_state[:] = np.where(
            rolls < p.immune_prob, StateEnum.IMMUNE, StateEnum.SUSCEPTIBLE
        ).astype(np.int8)

        # --- Counts / limits ---
        d.people_count[:] = 0
        d.people_limit[:] = 0

        # --- Immunity mirrors the IMMUNE state ---
        d.people_immunity[:] = d.people_state == StateEnum.IMMUNE

        # --- Super-spreader (assigned later during agent distribution) ---
        d.people_super[:] = False

        # --- Timer ---
        d.people_timer[:] = 0

        # --- Age assignment using two independent random draws ---
        # C++ logic: FLIP(old_rate) ? OLD : (FLIP(young_rate) ? YOUNG : PRIME)
        # This uses two independent coin flips, giving:
        #   P(OLD)   = old_rate
        #   P(YOUNG) = (1 - old_rate) * young_rate
        #   P(PRIME) = (1 - old_rate) * (1 - young_rate)
        roll1 = rng.random(N)
        roll2 = rng.random(N)
        ages = np.full(N, AgeEnum.PRIME, dtype=np.int8)
        ages[roll1 < p.old_man_rate] = AgeEnum.OLD
        ages[(roll1 >= p.old_man_rate) & (roll2 < p.young_man_rate)] = AgeEnum.YOUNG

        d.people_age[:] = ages

        # --- Isolation / quarantine ---
        d.people_isolated[:] = False
        d.people_quarantined[:] = False
        d.people_quarantined_count[:] = 0
        d.people_quarantined_level[:] = 0

        # --- Policy flags ---
        d.people_policy[:] = False

        # --- Agent slots: invisible, no home, unplaced ---
        d.agent_visible[:] = False
        d.agent_home[:] = False
        d.agent_loc_x[:] = -1
        d.agent_loc_y[:] = -1

        # --- Day counter ---
        d.day = 0

    # ------------------------------------------------------------------
    # Distributed agent placement (sequential)
    # ------------------------------------------------------------------

    # ================================================================
    # 【generate_distributed_agents / generateDistributedAgents】
    # ----------------------------------------------------------------
    # 演算法說明（繁體中文）：
    #   將每個人的「分身點」（agent）分配到世界格子上。核心步驟如下：
    #
    #   步驟1 — 決定每人的分身點上限（limit）：
    #     C++ 版：對每人取 4 個均勻隨機數（1～MaxAgent）的平均值，
    #     利用中央極限定理使分布趨近常態。
    #     Python 版：直接使用高斯分布 N(μ, σ²)，其中 μ=(1+M)/2，
    #     σ=(μ-1)/2，並微調總和使其精確等於格子總數（H×W）。
    #     這是 Python 版本的刻意改良，效果等價但更精確。
    #
    #   步驟2 — 從右下到左上逐格放置分身點：
    #     遍歷每個格子 (i, j)，隨機挑選一個尚未用完配額的人，
    #     將其下一個可用的 agent slot 放置於該格子上。
    #     同時更新格子的 peopleID、agentNo 和顏色。
    #
    #   步驟3 — 設定根據地與超級傳播者：
    #     對每個至少有一個分身點的人，隨機選取其中一個分身點
    #     作為「根據地」（home agent）。
    #     然後以 superRate 的機率將該人設為超級傳播者（super）。
    #
    #   步驟4 — 統計分身點分布：
    #     計算 distribution[k] = 恰好擁有 k+1 個分身點的人數。
    #
    # 對應 C++ 原始碼（Unit1.cpp, generateDistributedAgents, line 268）：
    #   void TSARS_Form::generateDistributedAgents(void)
    #   {
    #       long n1, n2, n3, n4;
    #       long ID, no, count[MAX_AGENT];
    #       long MaxPopulation = Edit_MaxPopulation->Text.ToInt();
    #       long MaxAgent = Edit_MaxAgent->Text.ToInt();
    #       double superRate = Edit_SuperRate->Text.ToDouble();
    #
    #       // 步驟1: 以四個均勻隨機數的平均值來決定每人的分身點上限
    #       for (long i = 0; i < MaxPopulation; i++) {
    #           n1 = (long)_lrand() % MaxAgent + 1;
    #           n2 = (long)_lrand() % MaxAgent + 1;
    #           n3 = (long)_lrand() % MaxAgent + 1;
    #           n4 = (long)_lrand() % MaxAgent + 1;
    #           society.people[i].limit = (n1 + n2 + n3 + n4) / 4;
    #       }
    #       // 步驟2: 從右下到左上遍歷每個格子，隨機選人放置分身點
    #       for (long i = Edit_MaxWorldHeight->Text.ToInt() - 1;
    #            i >= 0; i--)
    #           for (long j = Edit_MaxWorldWidth->Text.ToInt() - 1;
    #                j >= 0; j--) {
    #               do {} while (
    #                   society.people[
    #                       (ID = (long)_lrand() % MaxPopulation)
    #                   ].count >= society.people[ID].limit);
    #               society.people[ID].agent[
    #                   (no = society.people[ID].count++)
    #               ].visible = true;
    #               society.people[ID].agent[no].location.x = j;
    #               society.people[ID].agent[no].location.y = i;
    #               world[i][j].peopleID = ID;
    #               world[i][j].agentNo  = no;
    #               world[i][j].color = getColor(society.people[ID].state);
    #           }
    #       for (long i = 0; i < MAX_AGENT; count[i++] = 0);
    #       // 步驟3: 為每個人設定根據地，並依超級傳播者比率設定
    #       for (long i = 0; i < MaxPopulation; i++) {
    #           if (society.people[i].count > 0) {
    #               society.people[i].agent[
    #                   rand() % society.people[i].count
    #               ].home = true;
    #               society.people[i].attr.super = FLIP(superRate);
    #           }
    #           ++count[society.people[i].count];
    #       }
    #       // 步驟4: 顯示分身點統計
    #       Memo_Population->Lines->Clear();
    #       for (long MaxAgent = Edit_MaxAgent->Text.ToInt(),
    #            i = 1; i <= MaxAgent; i++)
    #           Memo_Population->Lines->Append(
    #               "count[" + AnsiString(i) + "] = "
    #               + AnsiString(count[i]));
    #   }
    #
    # Python 版本差異：
    #   - 步驟1 改用高斯分布取代四個均勻隨機數的平均，並精確調整
    #     總和以匹配格子總數，避免 C++ 版本中可能出現的配額不足。
    #   - 步驟2 的放置迴圈邏輯與 C++ 完全一致（sequential）。
    #   - 步驟3、4 使用 NumPy 向量化加速。
    # ================================================================
    def generate_distributed_agents(self) -> tuple:
        """Place agents on the lattice with a random-quota distribution.

        For every cell in the grid (iterated from bottom-right to
        top-left, matching the original C++ order), a random person
        whose current agent count has not yet reached their limit is
        chosen and one of their agent slots is placed at that cell.

        After all cells are filled, one agent per person is randomly
        designated as the *home* agent, and super-spreader status is
        assigned with probability ``super_rate``.

        Returns
        -------
        (distribution, actual_population) : tuple
            distribution : np.ndarray, shape (max_agent,)
                ``distribution[k]`` is the number of people whose final
                agent count equals ``k + 1``.
            actual_population : int
                Number of people who were assigned at least one agent.
        """
        p = self.params
        d = self.data
        N = d.N
        M = d.M
        H = p.max_height
        W = p.max_width
        rng = np.random.default_rng()

        # --- Assign per-person agent counts (normal distribution) ---
        # μ = (1 + MaxAgent) / 2,  σ = (μ - 1) / 2
        # Actual population ≈ total_cells / μ, capped at N.
        # The sum of all counts is adjusted to equal total_cells exactly
        # so that every person's limit is fully used during placement.
        total_cells = H * W
        mu = (1 + M) / 2.0
        sigma = (mu - 1.0) / 2.0
        actual_n = min(round(total_cells / mu), N)

        # Generate normally distributed counts for actual_n people
        raw = rng.normal(loc=mu, scale=sigma, size=actual_n)
        agent_counts = np.clip(np.rint(raw).astype(np.int32), 1, M)

        # Fine-tune sum to match total_cells exactly
        while (diff := int(agent_counts.sum()) - total_cells) != 0:
            if diff > 0:
                candidates = np.where(agent_counts > 1)[0]
                n_adj = min(diff, len(candidates))
                chosen = rng.choice(candidates, size=n_adj, replace=False)
                agent_counts[chosen] -= 1
            else:
                candidates = np.where(agent_counts < M)[0]
                n_adj = min(-diff, len(candidates))
                chosen = rng.choice(candidates, size=n_adj, replace=False)
                agent_counts[chosen] += 1

        # Randomly select which N people receive agents
        chosen_pids = rng.choice(N, size=actual_n, replace=False)
        d.people_limit[:] = 0
        d.people_limit[chosen_pids] = agent_counts

        # Local references for tight loop performance
        people_count = d.people_count
        people_limit = d.people_limit
        people_state = d.people_state
        agent_visible = d.agent_visible
        agent_loc_x = d.agent_loc_x
        agent_loc_y = d.agent_loc_y
        world_people_id = d.world_people_id
        world_agent_no = d.world_agent_no
        world_color = d.world_color

        # --- Place one agent per grid cell (sequential) ---
        for i in range(H - 1, -1, -1):
            for j in range(W - 1, -1, -1):
                # Pick a random person whose count < limit
                while True:
                    pid = rng.integers(0, N)
                    if people_count[pid] < people_limit[pid]:
                        break

                no = people_count[pid]
                people_count[pid] = no + 1

                agent_visible[pid, no] = True
                agent_loc_x[pid, no] = j
                agent_loc_y[pid, no] = i

                world_people_id[i, j] = pid
                world_agent_no[i, j] = no
                world_color[i, j] = self.get_color(people_state[pid])

        # --- Designate one random home agent per active person ---
        active_mask = people_count > 0
        active_indices = np.nonzero(active_mask)[0]

        for pid in active_indices:
            cnt = people_count[pid]
            home_idx = rng.integers(0, cnt)
            d.agent_home[pid, home_idx] = True

            # Super-spreader assignment
            if rng.random() < p.super_rate:
                d.people_super[pid] = True

        # --- Build distribution histogram ---
        # distribution[k] = number of people with exactly (k+1) agents
        actual_population = int(active_indices.shape[0])
        distribution = np.zeros(M, dtype=np.int64)
        counts = people_count[active_mask]
        for k in range(1, M + 1):
            distribution[k - 1] = np.sum(counts == k)

        return distribution, actual_population

    # ------------------------------------------------------------------
    # Color update (vectorized)
    # ------------------------------------------------------------------

    # ================================================================
    # 【reset_world_colors / resetWorld】
    # ----------------------------------------------------------------
    # 演算法說明（繁體中文）：
    #   重新繪製整個世界格子的顏色。遍歷所有格子，對於已被佔用的
    #   格子（peopleID >= 0），根據該人目前的健康狀態查詢對應顏色
    #   並更新格子顏色。未被佔用的格子則設為預設色。
    #   此方法在每日模擬步驟後呼叫，以反映狀態變化。
    #
    # 對應 C++ 原始碼（Unit1.cpp, resetWorld, line 488）：
    #   void TSARS_Form::resetWorld(void)
    #   {
    #       for (long MaxHeight = Edit_MaxWorldHeight->Text.ToInt(),
    #            i = 0; i < MaxHeight; i++)
    #           for (long MaxWidth = Edit_MaxWorldWidth->Text.ToInt(),
    #                j = 0; j < MaxWidth; j++)
    #               world[i][j].color =
    #                   getColor(society.people[world[i][j].peopleID].state);
    #   }
    #
    # Python 版本差異：
    #   使用 NumPy 向量化操作取代雙重迴圈，利用布林遮罩區分
    #   已佔用與未佔用的格子，一次更新所有顏色。
    #   C++ 版本假設所有格子皆已佔用，Python 版本額外處理空格子。
    # ================================================================
    def reset_world_colors(self) -> None:
        """Repaint every occupied cell according to its owner's state.

        Unoccupied cells (people_id == -1) are left at the default
        color.  This is a fully vectorized operation over the entire
        H x W grid.
        """
        ids = self.data.world_people_id              # (H, W) int32
        occupied = ids >= 0

        # Only index into people_state for occupied cells to avoid
        # out-of-bounds on the -1 sentinel values.
        states = np.empty_like(ids, dtype=np.int8)
        states[occupied] = self.data.people_state[ids[occupied]]

        # Map states to ARGB colors
        colors = self.data.world_color
        colors[occupied] = Colors.STATE_COLORS[states[occupied]]
        colors[~occupied] = Colors.DEFAULT

    # ================================================================
    # 【update_dirty_colors】— Python 專有最佳化，無 C++ 對應
    # ----------------------------------------------------------------
    # 演算法說明（繁體中文）：
    #   僅重新繪製「狀態已變更」之人所擁有的格子顏色，而非整個格子。
    #   在 C++ 原版中，drawLattice() 每次都逐像素重繪整個格子，
    #   效率不佳。Python 版本引入「髒標記集合」（dirty_pids），
    #   僅更新狀態實際發生變化的人所佔用的格子。
    #
    #   演算法邏輯：
    #     1. 若 dirty_pids 為空，直接返回（無需更新）。
    #     2. 若變更人數超過總人口的 10%，改為呼叫 reset_world_colors()
    #        做全量重繪（因為 NumPy 向量化全量更新比 Python 迴圈
    #        逐人更新更快）。
    #     3. 否則，對每個髒標記的人 pid，查詢其新狀態對應的顏色，
    #        遍歷該人所有已放置的分身點，更新對應格子的顏色。
    #
    # 無對應 C++ 原始碼。
    # C++ 版本的 drawLattice() 每次完整重繪整個格子：
    #   for (i = 0; i < MaxHeight; i++)
    #       for (j = 0; j < MaxWidth; j++)
    #           Canvas->Pixels[...] = world[i][j].color;
    # Python 版本以此方法取代，大幅提升每日模擬步驟的繪圖效率。
    # ================================================================
    def update_dirty_colors(self, dirty_pids) -> None:
        """Repaint only the cells owned by people in *dirty_pids*.

        Much faster than :meth:`reset_world_colors` when only a small
        fraction of the population changed state during a day-step.
        Falls back to a full repaint when the dirty set is large.
        """
        d = self.data
        n_dirty = len(dirty_pids)
        if n_dirty == 0:
            return
        # If more than 10 % of population changed, full repaint is cheaper
        # than per-person iteration (NumPy vectorised vs Python loop).
        if n_dirty > d.N // 10:
            self.reset_world_colors()
            return

        for pid in dirty_pids:
            color = Colors.STATE_COLORS[int(d.people_state[pid])]
            count = int(d.people_count[pid])
            for i in range(count):
                y = int(d.agent_loc_y[pid, i])
                x = int(d.agent_loc_x[pid, i])
                d.world_color[y, x] = color

    # ------------------------------------------------------------------
    # State-to-color mapping
    # ------------------------------------------------------------------

    # ================================================================
    # 【get_color / getColor】
    # ----------------------------------------------------------------
    # 演算法說明（繁體中文）：
    #   根據健康狀態（state）回傳對應的 ARGB 顏色值。
    #   狀態與顏色的對應關係：
    #     SUSCEPTIBLE（易感者）→ 綠色
    #     EXPOSED（已暴露）    → 黃色
    #     INFECTIVE（感染者）  → 紅色
    #     RECOVERED（已康復）  → 灰色
    #     IMMUNE（免疫者）     → 白色
    #     DIED（死亡）         → 黑色
    #   若狀態不在上述範圍內，回傳預設色。
    #
    # 對應 C++ 原始碼（Unit1.cpp, getColor, line 313）：
    #   TColor TSARS_Form::getColor(UState state)
    #   {
    #       TColor color = DEFAULT_COLOR;
    #       switch (state) {
    #           case SUSCEPTIBLE: color = SUSCEPTIBLE_COLOR; break;
    #           case EXPOSED:     color = EXPOSED_COLOR;     break;
    #           case INFECTIVE:   color = INFECTIVE_COLOR;   break;
    #           case RECOVERED:   color = RECOVERED_COLOR;   break;
    #           case IMMUNE:      color = IMMUNE_COLOR;      break;
    #           case DIED:        color = DIED_COLOR;         break;
    #       }
    #       return color;
    #   }
    #
    # Python 版本差異：
    #   使用預先建立的 STATE_COLORS 查詢表（NumPy 陣列）直接以
    #   state 值作為索引取得顏色，取代 switch-case 語句。
    # ================================================================
    @staticmethod
    def get_color(state: int) -> int:
        """Return the ARGB color for the given health state.

        Parameters
        ----------
        state : int
            One of the ``StateEnum`` values (0..5).

        Returns
        -------
        int
            32-bit ARGB color value.
        """
        return int(Colors.STATE_COLORS[state])
