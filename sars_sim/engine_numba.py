"""Numba JIT-compiled kernels for the SARS epidemic simulation engine.

This module provides a high-performance, JIT-compiled version of the core
simulation loop (``change_society``).  Every function is a **faithful,
line-by-line translation** of the corresponding Python method in
``engine.py``; the control flow, branching conditions, and state-machine
transitions are identical.  The only behavioural difference is the random
number generator: Numba uses its own internal RNG (seeded separately)
instead of Python's ``random`` module, so exact random trajectories will
differ while the statistical distribution of outcomes is preserved.

All functions receive flat NumPy arrays and scalar parameters — no Python
objects are accessed — so Numba can compile them to native machine code and
bypass the Python interpreter overhead entirely.
"""

import numpy as np
import numba as nb

# ======================================================================
# Constants (mirrors models.py enums — duplicated to avoid importing
# Python objects into Numba's nopython compilation context)
# ======================================================================

# StateEnum
_S_SUSCEPTIBLE = np.int8(0)
_S_EXPOSED     = np.int8(1)
_S_INFECTIVE   = np.int8(2)
_S_RECOVERED   = np.int8(3)
_S_IMMUNE      = np.int8(4)
_S_DIED        = np.int8(5)
_S_ISOLATED    = 6   # statistic index only
_S_QUARANTINED = 7   # statistic index only

# PolicyIndex
_PI_FACE_MASK        = 0
_PI_TAKE_TEMPERATURE = 1
_PI_STOP_VISITANT    = 2
_PI_VACCINE          = 3
_PI_STOP_CONTACT     = 4
_PI_MEDICAL_POLICY   = 5
_PI_HOME             = 8
_PI_HOSPITAL         = 9

# AgeEnum
_AGE_YOUNG = np.int8(0)
_AGE_PRIME = np.int8(1)
_AGE_OLD   = np.int8(2)

# Moore neighbourhood — flat-index 4 is self
_AGENT_SELF = 4

# Pre-computed offset tables: flat index 0..8 → (row, col) offset
_COL_OFFSETS = np.array([-1,  0,  1, -1, 0, 1, -1, 0, 1], dtype=np.int32)
_ROW_OFFSETS = np.array([-1, -1, -1,  0, 0, 0,  1, 1, 1], dtype=np.int32)


# ======================================================================
# Helper: mortality rate
# ======================================================================
@nb.njit(cache=True)
def _get_died_rate(people_age, people_isolated, pid,
                   mortality_young, mortality_prime, mortality_old,
                   hospital_effect):
    """Mirrors ``SimulationEngine.get_died_rate``."""
    age = people_age[pid]
    if age == _AGE_YOUNG:
        mortality = mortality_young
    elif age == _AGE_PRIME:
        mortality = mortality_prime
    else:
        mortality = mortality_old
    if people_isolated[pid]:
        mortality *= 1.0 - hospital_effect
    return mortality


# ======================================================================
# Helper: transmission rate
# ======================================================================
@nb.njit(cache=True)
def _get_transmission_rate(people_policy, people_isolated, people_quarantined,
                           people_count, pid, gID,
                           transmission_prob, medical_policy_effect,
                           mask_effect, temp_effect):
    """Mirrors ``SimulationEngine.get_transmission_rate``."""
    prob = transmission_prob

    if ((people_policy[gID, _PI_HOSPITAL] and people_isolated[gID])
            or (people_policy[gID, _PI_HOME] and people_quarantined[gID])):
        prob *= people_count[gID]

    if people_policy[pid, _PI_MEDICAL_POLICY]:
        prob *= 1.0 - medical_policy_effect
    elif people_policy[pid, _PI_FACE_MASK]:
        prob *= 1.0 - mask_effect

    if people_policy[gID, _PI_FACE_MASK]:
        prob *= 1.0 - mask_effect

    if (people_policy[pid, _PI_TAKE_TEMPERATURE]
            and people_policy[gID, _PI_TAKE_TEMPERATURE]):
        prob *= 1.0 - temp_effect

    return prob


# ======================================================================
# Contact tracing — iterative BFS (Phase 2)
# ======================================================================
@nb.njit(cache=True)
def _trace_bfs(start_pid, start_level,
               people_count, people_state, people_policy,
               people_quarantined, people_quarantined_count,
               people_quarantined_level,
               agent_visible, agent_home, agent_loc_x, agent_loc_y,
               world_people_id, statistic,
               W, H, isolated_level_b,
               bfs_queue_pid, bfs_queue_level, bfs_visited,
               bfs_visited_stack):
    """Mirrors ``SimulationEngine.trace_contact_people`` (BFS version).

    Uses pre-allocated arrays for the BFS queue and visited set.
    Only the entries actually touched in ``bfs_visited`` are cleared
    afterwards (tracked via ``bfs_visited_stack``).
    """
    head = 0
    tail = 0
    visited_top = 0

    # Seed the BFS
    bfs_queue_pid[tail] = start_pid
    bfs_queue_level[tail] = start_level
    tail += 1
    bfs_visited[start_pid] = True
    bfs_visited_stack[visited_top] = start_pid
    visited_top += 1

    while head < tail:
        cur_pid = bfs_queue_pid[head]
        cur_level = bfs_queue_level[head]
        head += 1

        for agent_idx in range(people_count[cur_pid]):
            locX = agent_loc_x[cur_pid, agent_idx]
            locY = agent_loc_y[cur_pid, agent_idx]

            for row in range(-1, 2):
                for col in range(-1, 2):
                    nx = (locX + col + W) % W
                    ny = (locY + row + H) % H
                    tempID = world_people_id[ny, nx]

                    if tempID < 0 or bfs_visited[tempID]:
                        continue
                    bfs_visited[tempID] = True
                    bfs_visited_stack[visited_top] = tempID
                    visited_top += 1

                    tempState = people_state[tempID]
                    if ((tempState == _S_SUSCEPTIBLE
                         or tempState == _S_EXPOSED
                         or tempState == _S_INFECTIVE)
                            and people_policy[tempID, _PI_HOME]
                            and not people_quarantined[tempID]):
                        people_quarantined[tempID] = True
                        people_quarantined_count[tempID] = 0
                        people_quarantined_level[tempID] = cur_level

                        for k in range(people_count[tempID]):
                            agent_visible[tempID, k] = agent_home[tempID, k]

                        statistic[_S_QUARANTINED] += 1

                        if isolated_level_b and cur_level == 1:
                            bfs_queue_pid[tail] = tempID
                            bfs_queue_level[tail] = 2
                            tail += 1

    # Clean up visited flags (only the entries we touched)
    for i in range(visited_top):
        bfs_visited[bfs_visited_stack[i]] = False


# ======================================================================
# Medical policy spread / removal
# ======================================================================
@nb.njit(cache=True)
def _start_medical_policy(pid,
                          people_count, people_policy,
                          agent_home, agent_loc_x, agent_loc_y,
                          world_people_id,
                          W, H, medical_policy_available):
    """Mirrors ``SimulationEngine.start_medical_policy``."""
    for agent_idx in range(people_count[pid]):
        if not agent_home[pid, agent_idx]:
            continue

        locX = agent_loc_x[pid, agent_idx]
        locY = agent_loc_y[pid, agent_idx]

        for row in range(-1, 2):
            for col in range(-1, 2):
                nx = (locX + col + W) % W
                ny = (locY + row + H) % H
                tempID = world_people_id[ny, nx]
                if tempID < 0:
                    continue
                if not people_policy[tempID, _PI_MEDICAL_POLICY]:
                    if np.random.random() < medical_policy_available:
                        people_policy[tempID, _PI_MEDICAL_POLICY] = True


@nb.njit(cache=True)
def _finish_medical_policy(pid,
                           people_count, people_policy,
                           agent_home, agent_loc_x, agent_loc_y,
                           world_people_id,
                           W, H):
    """Mirrors ``SimulationEngine.finish_medical_policy``."""
    for agent_idx in range(people_count[pid]):
        if not agent_home[pid, agent_idx]:
            continue

        locX = agent_loc_x[pid, agent_idx]
        locY = agent_loc_y[pid, agent_idx]

        for row in range(-1, 2):
            for col in range(-1, 2):
                nx = (locX + col + W) % W
                ny = (locY + row + H) % H
                tempID = world_people_id[ny, nx]
                if tempID >= 0:
                    people_policy[tempID, _PI_MEDICAL_POLICY] = False


# ======================================================================
# Hospital isolation detection
# ======================================================================
@nb.njit(cache=True)
def _isolate_by_hospital(pid, state, timer,
                         people_state, people_count, people_policy,
                         people_isolated, people_quarantined,
                         people_quarantined_count, people_quarantined_level,
                         agent_visible, agent_home, agent_loc_x, agent_loc_y,
                         world_people_id, statistic,
                         W, H, N,
                         temp_effect, infective_period, detect_rate,
                         trace_on, isolated_level_b,
                         medical_policy_enabled, medical_policy_available,
                         bfs_queue_pid, bfs_queue_level,
                         bfs_visited, bfs_visited_stack):
    """Mirrors ``SimulationEngine.isolate_by_hospital``."""
    detected = False
    if people_policy[pid, _PI_TAKE_TEMPERATURE] and np.random.random() < temp_effect:
        detected = True
    if not detected and timer > infective_period and np.random.random() < detect_rate:
        detected = True

    if not detected:
        return

    people_isolated[pid] = True
    statistic[_S_ISOLATED] += 1

    if people_policy[pid, _PI_HOME] and people_quarantined[pid]:
        people_quarantined[pid] = False
        people_quarantined_count[pid] = 0
        people_quarantined_level[pid] = 0

    for i in range(people_count[pid]):
        agent_visible[pid, i] = agent_home[pid, i]

    if trace_on:
        _trace_bfs(pid, 1,
                   people_count, people_state, people_policy,
                   people_quarantined, people_quarantined_count,
                   people_quarantined_level,
                   agent_visible, agent_home, agent_loc_x, agent_loc_y,
                   world_people_id, statistic,
                   W, H, isolated_level_b,
                   bfs_queue_pid, bfs_queue_level,
                   bfs_visited, bfs_visited_stack)

    if medical_policy_enabled:
        _start_medical_policy(pid,
                              people_count, people_policy,
                              agent_home, agent_loc_x, agent_loc_y,
                              world_people_id,
                              W, H, medical_policy_available)


# ======================================================================
# Quarantine tick
# ======================================================================
@nb.njit(cache=True)
def _quarantine_tick(pid,
                     people_count, people_policy,
                     people_isolated, people_quarantined,
                     people_quarantined_count, people_quarantined_level,
                     agent_visible,
                     quarantine_period):
    """Mirrors ``SimulationEngine.quarantine_people_by_ongoing``."""
    people_quarantined_count[pid] += 1

    if people_quarantined_count[pid] > quarantine_period:
        people_quarantined[pid] = False
        people_quarantined_count[pid] = 0
        people_quarantined_level[pid] = 0
        # Restore visibility unless isolated in hospital
        if not (people_policy[pid, _PI_HOSPITAL] and people_isolated[pid]):
            for i in range(people_count[pid]):
                agent_visible[pid, i] = True


# ======================================================================
# Disease-state progression
# ======================================================================
@nb.njit(cache=True)
def _change_ongoing(pid, state, dirty_flags,
                    people_state, people_count, people_timer, people_policy,
                    people_isolated, people_quarantined,
                    people_quarantined_count, people_quarantined_level,
                    people_immunity, people_age,
                    agent_visible, agent_home, agent_loc_x, agent_loc_y,
                    world_people_id, statistic,
                    W, H, N,
                    exposed_period, symptomatic_period, infective_period,
                    recovered_period, immune_period,
                    mortality_young, mortality_prime, mortality_old,
                    hospital_effect, temp_effect, detect_rate,
                    trace_on, isolated_level_b,
                    medical_policy_enabled, medical_policy_available,
                    bfs_queue_pid, bfs_queue_level,
                    bfs_visited, bfs_visited_stack):
    """Mirrors ``SimulationEngine.change_people_by_ongoing``."""
    people_timer[pid] += 1
    t = people_timer[pid]

    # EXPOSED → INFECTIVE
    if state == _S_EXPOSED and t > exposed_period:
        people_state[pid] = _S_INFECTIVE
        dirty_flags[pid] = True
        statistic[_S_INFECTIVE] += 1
        people_timer[pid] = 0

    # Hospital detection while INFECTIVE
    if (state == _S_INFECTIVE
            and people_policy[pid, _PI_HOSPITAL]
            and not people_isolated[pid]):
        _isolate_by_hospital(pid, state, t,
                             people_state, people_count, people_policy,
                             people_isolated, people_quarantined,
                             people_quarantined_count, people_quarantined_level,
                             agent_visible, agent_home, agent_loc_x, agent_loc_y,
                             world_people_id, statistic,
                             W, H, N,
                             temp_effect, infective_period, detect_rate,
                             trace_on, isolated_level_b,
                             medical_policy_enabled, medical_policy_available,
                             bfs_queue_pid, bfs_queue_level,
                             bfs_visited, bfs_visited_stack)

    # INFECTIVE → DIED / RECOVERED
    if state == _S_INFECTIVE and t > symptomatic_period:
        died_rate = _get_died_rate(people_age, people_isolated, pid,
                                  mortality_young, mortality_prime, mortality_old,
                                  hospital_effect)
        if np.random.random() < died_rate:
            new_state = _S_DIED
        else:
            new_state = _S_RECOVERED

        people_state[pid] = new_state
        dirty_flags[pid] = True
        statistic[new_state] += 1
        people_timer[pid] = 0

        if new_state == _S_DIED:
            for i in range(people_count[pid]):
                agent_visible[pid, i] = agent_home[pid, i]
        else:
            for i in range(people_count[pid]):
                agent_visible[pid, i] = True

        if people_isolated[pid]:
            _finish_medical_policy(pid,
                                  people_count, people_policy,
                                  agent_home, agent_loc_x, agent_loc_y,
                                  world_people_id,
                                  W, H)
        people_isolated[pid] = False
        people_quarantined[pid] = False
        people_quarantined_count[pid] = 0
        people_quarantined_level[pid] = 0

    # RECOVERED → IMMUNE
    if state == _S_RECOVERED and t > recovered_period:
        people_state[pid] = _S_IMMUNE
        dirty_flags[pid] = True
        statistic[_S_IMMUNE] += 1
        people_immunity[pid] = (immune_period == 0)
        people_timer[pid] = 0

    # IMMUNE → SUSCEPTIBLE
    if state == _S_IMMUNE and t > immune_period:
        people_state[pid] = _S_SUSCEPTIBLE
        dirty_flags[pid] = True
        statistic[_S_SUSCEPTIBLE] += 1
        people_timer[pid] = 0
        people_policy[pid, _PI_VACCINE] = False


# ======================================================================
# Agent interaction (touch neighbour)
# ======================================================================
@nb.njit(cache=True)
def _touch_other_agent(pid, no, dirty_flags, infected_counts,
                       people_state, people_count, people_policy,
                       people_timer, people_isolated, people_quarantined,
                       people_quarantined_count, people_quarantined_level,
                       people_super,
                       agent_visible, agent_home, agent_loc_x, agent_loc_y,
                       world_people_id, world_agent_no, statistic,
                       W, H, N,
                       transmission_prob, medical_policy_effect,
                       mask_effect, temp_effect,
                       trace_on, isolated_level_b,
                       bfs_queue_pid, bfs_queue_level,
                       bfs_visited, bfs_visited_stack):
    """Mirrors ``SimulationEngine.touch_other_agent``."""
    target = np.random.randint(0, 9)
    baseX = agent_loc_x[pid, no]
    baseY = agent_loc_y[pid, no]

    # Scan Moore neighbourhood for a super-spreader
    count = 0
    for row in range(-1, 2):
        for col in range(-1, 2):
            if row != 0 or col != 0:
                x = (baseX + col + W) % W
                y = (baseY + row + H) % H
                gID = world_people_id[y, x]
                if (gID >= 0
                        and people_super[gID]
                        and people_state[gID] == _S_INFECTIVE):
                    target = count
            count += 1

    if target == _AGENT_SELF:
        return False

    # Convert flat target index to grid coordinates using lookup tables
    col_offset = _COL_OFFSETS[target]
    row_offset = _ROW_OFFSETS[target]

    x = (baseX + col_offset + W) % W
    y = (baseY + row_offset + H) % H
    gID = world_people_id[y, x]
    gNo = world_agent_no[y, x]

    if gID < 0 or gNo < 0:
        return False

    if not agent_visible[gID, gNo]:
        return False

    gState = people_state[gID]
    changed = False

    # ---- Transmission: susceptible pid meets infective gID ----
    if people_state[pid] == _S_SUSCEPTIBLE and gState == _S_INFECTIVE:
        # Visitant check
        if people_policy[gID, _PI_HOSPITAL] and people_isolated[gID]:
            if people_policy[pid, _PI_STOP_VISITANT] and not agent_home[pid, no]:
                return False

        rate = _get_transmission_rate(
            people_policy, people_isolated, people_quarantined,
            people_count, pid, gID,
            transmission_prob, medical_policy_effect,
            mask_effect, temp_effect)

        if np.random.random() < rate:
            people_state[pid] = _S_EXPOSED
            dirty_flags[pid] = True
            statistic[_S_EXPOSED] += 1
            people_timer[pid] = 0
            changed = True

            if people_isolated[gID]:
                infected_counts[0] += 1   # infected_by_hospital
            else:
                infected_counts[1] += 1   # infected_by_normal

    # ---- Quarantine contact ----
    if (people_policy[pid, _PI_HOSPITAL]
            and not people_isolated[pid]
            and people_policy[pid, _PI_HOME]
            and not people_quarantined[pid]):
        triggered = False
        if people_policy[gID, _PI_HOSPITAL] and people_isolated[gID]:
            people_quarantined[pid] = True
            people_quarantined_level[pid] = 1
            triggered = True
            if trace_on:
                _trace_bfs(pid, 2,
                           people_count, people_state, people_policy,
                           people_quarantined, people_quarantined_count,
                           people_quarantined_level,
                           agent_visible, agent_home, agent_loc_x, agent_loc_y,
                           world_people_id, statistic,
                           W, H, isolated_level_b,
                           bfs_queue_pid, bfs_queue_level,
                           bfs_visited, bfs_visited_stack)
        elif (people_policy[gID, _PI_HOME]
              and people_quarantined[gID]
              and people_quarantined_level[gID] == 1
              and isolated_level_b):
            people_quarantined[pid] = True
            people_quarantined_level[pid] = 2
            triggered = True

        if (people_policy[pid, _PI_HOME]
                and people_quarantined[pid]
                and triggered):
            statistic[_S_QUARANTINED] += 1
            people_quarantined_count[pid] = 0
            for i in range(people_count[pid]):
                agent_visible[pid, i] = agent_home[pid, i]

    return changed


# ======================================================================
# Single-agent gossip attempts
# ======================================================================
@nb.njit(cache=True)
def _change_agent_by_gossip(pid, no, dirty_flags, infected_counts,
                            people_state, people_count, people_policy,
                            people_timer, people_isolated, people_quarantined,
                            people_quarantined_count, people_quarantined_level,
                            people_super,
                            agent_visible, agent_home, agent_loc_x, agent_loc_y,
                            world_people_id, world_agent_no, statistic,
                            W, H, N,
                            gossip_steps, gossip_fixed,
                            transmission_prob, medical_policy_effect,
                            mask_effect, temp_effect,
                            trace_on, isolated_level_b,
                            bfs_queue_pid, bfs_queue_level,
                            bfs_visited, bfs_visited_stack):
    """Mirrors ``SimulationEngine.change_agent_by_gossip``."""
    if people_policy[pid, _PI_STOP_CONTACT] and not agent_home[pid, no]:
        max_gossip = np.random.randint(0, gossip_steps + 1)
    elif gossip_fixed:
        max_gossip = gossip_steps
    else:
        if gossip_steps >= 1:
            max_gossip = np.random.randint(1, gossip_steps + 1)
        else:
            max_gossip = 1

    changed = False
    for _ in range(max_gossip):
        if not changed and agent_visible[pid, no]:
            changed = _touch_other_agent(
                pid, no, dirty_flags, infected_counts,
                people_state, people_count, people_policy,
                people_timer, people_isolated, people_quarantined,
                people_quarantined_count, people_quarantined_level,
                people_super,
                agent_visible, agent_home, agent_loc_x, agent_loc_y,
                world_people_id, world_agent_no, statistic,
                W, H, N,
                transmission_prob, medical_policy_effect,
                mask_effect, temp_effect,
                trace_on, isolated_level_b,
                bfs_queue_pid, bfs_queue_level,
                bfs_visited, bfs_visited_stack)
    return changed


# ======================================================================
# Per-person gossip dispatch
# ======================================================================
@nb.njit(cache=True)
def _change_people_by_gossip(pid, dirty_flags, infected_counts,
                             people_state, people_count, people_policy,
                             people_timer, people_isolated, people_quarantined,
                             people_quarantined_count, people_quarantined_level,
                             people_super,
                             agent_visible, agent_home, agent_loc_x, agent_loc_y,
                             world_people_id, world_agent_no, statistic,
                             W, H, N,
                             gossip_steps, gossip_fixed,
                             transmission_prob, medical_policy_effect,
                             mask_effect, temp_effect,
                             trace_on, isolated_level_b,
                             bfs_queue_pid, bfs_queue_level,
                             bfs_visited, bfs_visited_stack):
    """Mirrors ``SimulationEngine.change_people_by_gossip``."""
    direction = np.random.random() < 0.5
    cnt = people_count[pid]

    if direction:
        start, stop, step = 0, cnt, 1
    else:
        start, stop, step = cnt - 1, -1, -1

    for i in range(start, stop, step):
        if agent_visible[pid, i]:
            if (agent_home[pid, i]
                    or not people_policy[pid, _PI_STOP_CONTACT]
                    or np.random.random() < (1.0 / cnt if cnt > 0 else 1.0)):
                if _change_agent_by_gossip(
                        pid, i, dirty_flags, infected_counts,
                        people_state, people_count, people_policy,
                        people_timer, people_isolated, people_quarantined,
                        people_quarantined_count, people_quarantined_level,
                        people_super,
                        agent_visible, agent_home, agent_loc_x, agent_loc_y,
                        world_people_id, world_agent_no, statistic,
                        W, H, N,
                        gossip_steps, gossip_fixed,
                        transmission_prob, medical_policy_effect,
                        mask_effect, temp_effect,
                        trace_on, isolated_level_b,
                        bfs_queue_pid, bfs_queue_level,
                        bfs_visited, bfs_visited_stack):
                    break


# ======================================================================
# Per-person dispatch (change_people)
# ======================================================================
@nb.njit(cache=True)
def _change_people(pid, dirty_flags, infected_counts,
                   people_state, people_count, people_timer, people_policy,
                   people_isolated, people_quarantined,
                   people_quarantined_count, people_quarantined_level,
                   people_immunity, people_age, people_super,
                   agent_visible, agent_home, agent_loc_x, agent_loc_y,
                   world_people_id, world_agent_no, statistic,
                   W, H, N,
                   exposed_period, symptomatic_period, infective_period,
                   recovered_period, immune_period, quarantine_period,
                   transmission_prob, medical_policy_effect,
                   mask_effect, temp_effect, detect_rate,
                   mortality_young, mortality_prime, mortality_old,
                   hospital_effect,
                   gossip_steps, gossip_fixed,
                   trace_on, isolated_level_b,
                   medical_policy_enabled, medical_policy_available,
                   bfs_queue_pid, bfs_queue_level,
                   bfs_visited, bfs_visited_stack):
    """Mirrors ``SimulationEngine.change_people``."""
    if people_count[pid] == 0:
        return

    state = people_state[pid]

    # Quarantine ongoing tick
    if (state == _S_SUSCEPTIBLE or state == _S_EXPOSED or state == _S_INFECTIVE):
        if people_policy[pid, _PI_HOME] and people_quarantined[pid]:
            _quarantine_tick(pid,
                             people_count, people_policy,
                             people_isolated, people_quarantined,
                             people_quarantined_count, people_quarantined_level,
                             agent_visible,
                             quarantine_period)

    # Re-read state (may have changed)
    state = people_state[pid]

    # Disease progression
    if (state == _S_EXPOSED or state == _S_INFECTIVE or state == _S_RECOVERED
            or (state == _S_IMMUNE and not people_immunity[pid])):
        _change_ongoing(pid, state, dirty_flags,
                        people_state, people_count, people_timer, people_policy,
                        people_isolated, people_quarantined,
                        people_quarantined_count, people_quarantined_level,
                        people_immunity, people_age,
                        agent_visible, agent_home, agent_loc_x, agent_loc_y,
                        world_people_id, statistic,
                        W, H, N,
                        exposed_period, symptomatic_period, infective_period,
                        recovered_period, immune_period,
                        mortality_young, mortality_prime, mortality_old,
                        hospital_effect, temp_effect, detect_rate,
                        trace_on, isolated_level_b,
                        medical_policy_enabled, medical_policy_available,
                        bfs_queue_pid, bfs_queue_level,
                        bfs_visited, bfs_visited_stack)

    # Re-read state
    state = people_state[pid]

    # Gossip / movement
    if (state == _S_SUSCEPTIBLE
            or ((state == _S_EXPOSED or state == _S_INFECTIVE)
                and people_policy[pid, _PI_HOME]
                and not people_quarantined[pid])):
        _change_people_by_gossip(
            pid, dirty_flags, infected_counts,
            people_state, people_count, people_policy,
            people_timer, people_isolated, people_quarantined,
            people_quarantined_count, people_quarantined_level,
            people_super,
            agent_visible, agent_home, agent_loc_x, agent_loc_y,
            world_people_id, world_agent_no, statistic,
            W, H, N,
            gossip_steps, gossip_fixed,
            transmission_prob, medical_policy_effect,
            mask_effect, temp_effect,
            trace_on, isolated_level_b,
            bfs_queue_pid, bfs_queue_level,
            bfs_visited, bfs_visited_stack)


# ======================================================================
# Main kernel — replaces the Python change_society loop
# ======================================================================
@nb.njit(cache=True)
def change_society_kernel(
        # People arrays
        people_state, people_count, people_timer, people_policy,
        people_isolated, people_quarantined,
        people_quarantined_count, people_quarantined_level,
        people_immunity, people_age, people_super,
        # Agent arrays
        agent_visible, agent_home, agent_loc_x, agent_loc_y,
        # World arrays
        world_people_id, world_agent_no,
        # Statistics
        statistic, infected_counts,
        # Pre-allocated buffers
        dirty_flags, bfs_queue_pid, bfs_queue_level,
        bfs_visited, bfs_visited_stack,
        # Scalar parameters
        N, W, H,
        exposed_period, symptomatic_period, infective_period,
        recovered_period, immune_period, quarantine_period,
        transmission_prob, medical_policy_effect,
        mask_effect, temp_effect, detect_rate,
        mortality_young, mortality_prime, mortality_old,
        hospital_effect,
        gossip_steps, gossip_fixed,
        trace_on, isolated_level_b,
        medical_policy_enabled, medical_policy_available,
):
    """JIT-compiled replacement for ``SimulationEngine.change_society``.

    Iterates over all people in a randomised direction and applies the
    full per-person update logic (quarantine tick, disease progression,
    gossip / transmission).  Operates on the same NumPy arrays as the
    Python engine, so all mutations are visible to the caller.
    """
    # Clear dirty flags
    for i in range(N):
        dirty_flags[i] = False

    # Randomise traversal direction (50% forward, 50% reverse)
    direction = np.random.random() < 0.5

    if direction:
        start, stop, step = 0, N, 1
    else:
        start, stop, step = N - 1, -1, -1

    for pid in range(start, stop, step):
        _change_people(pid, dirty_flags, infected_counts,
                       people_state, people_count, people_timer, people_policy,
                       people_isolated, people_quarantined,
                       people_quarantined_count, people_quarantined_level,
                       people_immunity, people_age, people_super,
                       agent_visible, agent_home, agent_loc_x, agent_loc_y,
                       world_people_id, world_agent_no, statistic,
                       W, H, N,
                       exposed_period, symptomatic_period, infective_period,
                       recovered_period, immune_period, quarantine_period,
                       transmission_prob, medical_policy_effect,
                       mask_effect, temp_effect, detect_rate,
                       mortality_young, mortality_prime, mortality_old,
                       hospital_effect,
                       gossip_steps, gossip_fixed,
                       trace_on, isolated_level_b,
                       medical_policy_enabled, medical_policy_available,
                       bfs_queue_pid, bfs_queue_level,
                       bfs_visited, bfs_visited_stack)
