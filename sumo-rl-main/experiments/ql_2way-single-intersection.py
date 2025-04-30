import os
import sys
import glob
import pandas as pd
from sumo_rl import SumoEnvironment
from sumo_rl.agents import QLAgent
from sumo_rl.exploration import EpsilonGreedy

if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

EPISODES = 5
TIMESTEPS_PER_EPISODE = 20000

# Ensure output directory exists
output_dir = "outputs/2way-single-intersection"
os.makedirs(output_dir, exist_ok=True)

for ep in range(1, EPISODES + 1):
    out_csv_base = f"{output_dir}/ql_ep{ep}"
    out_csv = f"{out_csv_base}.csv"
    env = SumoEnvironment(
        net_file="sumo_rl/nets/2way-single-intersection/single-intersection.net.xml",
        route_file="sumo_rl/nets/2way-single-intersection/single-intersection-vhvh.rou.xml",
        out_csv_name=out_csv_base,  # SumoEnvironment appends .csv if not present, but we handle both
        use_gui=True,
        num_seconds=TIMESTEPS_PER_EPISODE,
        min_green=10,
        max_green=30,
        sumo_warnings=False,
    )
    initial_states = env.reset()
    ql_agents = {
        ts: QLAgent(
            starting_state=env.encode(initial_states[ts], ts),
            state_space=env.observation_space,
            action_space=env.action_space,
            alpha=0.1,
            gamma=0.99,
            exploration_strategy=EpsilonGreedy(
                initial_epsilon=0.05, min_epsilon=0.005, decay=1.0
            ),
        )
        for ts in env.ts_ids
    }
    done = {"__all__": False}
    while not done["__all__"]:
        actions = {ts: ql_agents[ts].act() for ts in ql_agents.keys()}
        s, r, done, _ = env.step(action=actions)
        for agent_id in ql_agents.keys():
            ql_agents[agent_id].learn(next_state=env.encode(s[agent_id], agent_id), reward=r[agent_id])
    env.save_csv(out_csv_base, ep)  # This should create out_csv_base + '.csv'
    env.close()
    # Print episode results
    csv_pattern = f"{output_dir}/ql_ep{ep}_*.csv"
    csv_files = glob.glob(csv_pattern)
    if csv_files:
        result_csv = csv_files[0]  # Take the first match (should be only one)
        try:
            df = pd.read_csv(result_csv)
            mean_reward = df['reward'].mean() if 'reward' in df.columns else None
            episode_length = len(df)
            print(f"Episode {ep} results:")
            print(f"  Mean reward: {mean_reward}")
            print(f"  Episode length (steps): {episode_length}")
            print(f"  CSV saved to: {result_csv}")
        except Exception as e:
            print(f"Could not read results for episode {ep}: {e}")
    else:
        print(f"No CSV file found for episode {ep} (looked for: {csv_pattern})")