# inference.py — FoveaEnv Baseline AI Agent
import requests
import json

BASE_URL = "http://127.0.0.1:8000"

def get_move_toward_goal(agent_pos, look_center):
    """Simple greedy agent: move toward goal (bottom-right corner)."""
    row, col = agent_pos
    # Goal is always at [grid_size//2, grid_size-1] = [3, 6]
    goal_row, goal_col = 3, 6

    # Avoid privacy zones by checking last_event
    if col < goal_col:
        return "right"
    elif row < goal_row:
        return "down"
    elif row > goal_row:
        return "up"
    else:
        return "stay"

def get_look_toward_goal(agent_pos, look_center):
    """Look ahead toward goal."""
    row, col = agent_pos
    look_row, look_col = look_center
    goal_col = 6

    if look_col < goal_col:
        return "right"
    return "stay"

def run_episode(task_id="easy", verbose=True):
    """Run one full episode and return the final score."""

    # 1. Reset
    reset_resp = requests.post(f"{BASE_URL}/reset", json={"task_id": task_id})
    obs = reset_resp.json()

    if verbose:
        print(f"\n{'='*50}")
        print(f"Task: {task_id.upper()} | Max Steps: {obs['max_steps']}")
        print(f"Start: {obs['agent_pos']} | Event: {obs['last_event']}")
        print(f"{'='*50}")

    total_reward = 0.0
    final_score = None

    # 2. Step loop
    while True:
        agent_pos   = obs["agent_pos"]
        look_center = obs["look_center"]
        last_event  = obs["last_event"]
        step_count  = obs["step_count"]

        # Smart move — avoid re-entering hazard
        if last_event == "hazard_hit":
            move = "up"   # retreat upward if hit hazard
        else:
            move = get_move_toward_goal(agent_pos, look_center)

        look    = get_look_toward_goal(agent_pos, look_center)
        inspect = (last_event == "privacy_violation")  # inspect only if needed

        step_resp = requests.post(f"{BASE_URL}/step", json={
            "move": move,
            "look": look,
            "inspect": inspect
        })
        if step_resp.status_code != 200:
            print(f"⚠️  Server error {step_resp.status_code} at step {step_count}")
            break

        obs = step_resp.json()
        if obs is None:
            print("⚠️  Empty response from server")
            break

        reward = obs.get("reward", 0.0)
        done   = obs.get("done", False)
        total_reward += reward

        if verbose:
            print(f"Step {obs['step_count']:02d} | pos={obs['agent_pos']} | "
                  f"move={move} | event={obs['last_event']} | reward={reward:+.2f}")

        if done:
            final_score = obs.get("score", {})
            break

    if verbose:
        print(f"\n{'='*50}")
        print(f"Episode Done! Total reward: {total_reward:.2f}")
        if final_score:
            print(f"Navigation Score:       {final_score.get('navigation_score', 0):.3f}")
            print(f"Privacy Score:          {final_score.get('privacy_score', 0):.3f}")
            print(f"🏆 FINAL SCORE:         {final_score.get('final_score', 0):.3f} / 1.0")
        print(f"{'='*50}\n")

    return final_score

def run_all_tasks():
    """Run all 3 tasks and print summary."""
    print("\n🤖 FoveaEnv Baseline Agent — Running All Tasks\n")
    results = {}

    for task in ["easy", "medium", "hard"]:
        score = run_episode(task_id=task, verbose=True)
        results[task] = score

    print("\n📊 SUMMARY")
    print(f"{'Task':<10} {'Nav Score':<15} {'Privacy Score':<18} {'Final Score':<12}")
    print("-" * 55)
    for task, score in results.items():
        if score:
            nav   = score.get("navigation_score", 0)
            priv  = score.get("privacy_score", 0)
            final = score.get("final_score", 0)
            print(f"{task:<10} {nav:<15.3f} {priv:<18.3f} {final:<12.3f}")

if __name__ == "__main__":
    run_all_tasks()