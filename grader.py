# grader.py
# Dual scoring: Navigation Score + Privacy Efficiency Score
# Final score is ALWAYS in range [0.0, 1.0]

def grade_episode(
    episode_reward: float,
    reached_goal: bool,
    privacy_violations: int,
    total_steps: int
) -> dict:

    # ── Score 1: Navigation Quality ──────────────────
    # Formula: normalize episode_reward into [0.0, 1.0]
    raw_nav = (episode_reward + 0.5) / 2.5
    nav_score = round(max(0.0, min(1.0, raw_nav)), 4)

    # ── Score 2: Privacy Efficiency ──────────────────
    # 0 violations = 1.0 (perfect), more violations = lower score
    priv_score = round(
        max(0.0, 1.0 - (privacy_violations / max(total_steps, 1))),
        4
    )

    # ── Final Combined Score ─────────────────────────
    # Weighted: 60% navigation + 40% privacy
    final = round(0.6 * nav_score + 0.4 * priv_score, 4)

    return {
        "navigation_score": nav_score,
        "privacy_efficiency_score": priv_score,
        "final_score": final,
        "reached_goal": reached_goal
    }