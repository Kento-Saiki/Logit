import numpy as np
import pandas as pd

# ハイパーパラメータ
alpha = 0.1  # 学習率
gamma = 0.9  # 割引率
epsilon = 0.1  # 探索率（ε-グリーディ）
num_episodes = 10000  # 学習エピソード数
num_simulations = 100  # 収益シミュレーションの試行回数

# 状態空間
days = 10  # 1～10日
seats_A = np.arange(0, 101, 10)  # 0, 10, ..., 100
seats_B = np.arange(0, 151, 10)  # 0, 10, ..., 150
customers = np.arange(0, 601, 50)  # 0, 50, ..., 600（潜在顧客数、50人刻み）
state_space = [(t, a, b, c) for t in range(1, days + 1) for a in seats_A for b in seats_B for c in customers]
num_states = len(state_space)

# 行動空間
prices = np.arange(1000, 2001, 100)  # 1000, 1100, ..., 2000
actions = [(pA, pB) for pA in prices for pB in prices]
num_actions = len(actions)

# Qテーブルの初期化
Q = np.zeros((num_states, num_actions))

# 多項ロジットモデルのパラメータ
beta_0_A = 2.0  # 席種Aの基本効用
beta_0_B = 1.5  # 席種Bの基本効用
beta_p = -0.002  # 価格感度（負）
beta_t = 0.1  # 残り日数の影響（正）
initial_customers = 600  # 初期潜在顧客数

# 多項ロジットモデルによる購入確率
def choice_probabilities(pA, pB, t):
    v_A = beta_0_A + beta_p * pA + beta_t * (11 - t)
    v_B = beta_0_B + beta_p * pB + beta_t * (11 - t)
    v_0 = 0.0  # 購入しない選択肢
    denom = np.exp(v_A) + np.exp(v_B) + np.exp(v_0)
    P_A = np.exp(v_A) / denom
    P_B = np.exp(v_B) / denom
    P_0 = np.exp(v_0) / denom
    return P_A, P_B, P_0

# シミュレーション環境（顧客数減少を考慮）
def simulate_sales(pA, pB, t, seats_A_left, seats_B_left, customers_left):
    P_A, P_B, _ = choice_probabilities(pA, pB, t)
    # 購入枚数は残在庫と残顧客数で制限
    n_A = min(np.random.binomial(customers_left, P_A), seats_A_left)
    n_B = min(np.random.binomial(customers_left, P_B), seats_B_left)
    reward = pA * n_A + pB * n_B
    next_seats_A = max(seats_A_left - n_A, 0)
    next_seats_B = max(seats_B_left - n_B, 0)
    next_customers = max(customers_left - n_A - n_B, 0)
    return reward, next_seats_A, next_seats_B, next_customers

# 状態インデックス取得
state_to_idx = {s: i for i, s in enumerate(state_space)}

# Q学習
for episode in range(num_episodes):
    t, seats_A_left, seats_B_left, customers_left = 1, 100, 150, initial_customers  # 初期状態
    done = False
    
    while not done:
        state = (t, seats_A_left, seats_B_left, customers_left)
        if state not in state_to_idx:
            # 離散化外の状態を近似（最も近い顧客数を選択）
            closest_c = customers[np.argmin(np.abs(customers - customers_left))]
            state = (t, seats_A_left, seats_B_left, closest_c)
            if state not in state_to_idx:
                break
        
        state_idx = state_to_idx[state]
        
        # ε-グリーディで行動選択
        if np.random.rand() < epsilon:
            action_idx = np.random.randint(num_actions)
        else:
            action_idx = np.argmax(Q[state_idx])
        
        pA, pB = actions[action_idx]
        
        # シミュレーション実行
        reward, next_seats_A, next_seats_B, next_customers = simulate_sales(
            pA, pB, t, seats_A_left, seats_B_left, customers_left
        )
        next_t = t + 1
        next_state = (next_t, next_seats_A, next_seats_B, next_customers)
        
        # 終了条件
        if next_t > days or (next_seats_A == 0 and next_seats_B == 0) or next_customers == 0:
            done = True
            next_Q = 0
        else:
            # 次の状態を離散化
            closest_c = customers[np.argmin(np.abs(customers - next_customers))]
            next_state = (next_t, next_seats_A, next_seats_B, closest_c)
            next_state_idx = state_to_idx.get(next_state, state_idx)
            next_Q = np.max(Q[next_state_idx])
        
        # Q値更新
        Q[state_idx, action_idx] += alpha * (reward + gamma * next_Q - Q[state_idx, action_idx])
        
        # 状態更新
        t, seats_A_left, seats_B_left, customers_left = next_t, next_seats_A, next_seats_B, next_customers
    
    epsilon = max(0.01, epsilon * 0.995)  # 探索率の減衰

# 全日程の最適価格と期待収益を計算
results = []
for t in range(1, days + 1):
    # 代表的な状態（初期在庫と中間在庫、顧客数は初期600と300）
    for seats_A_left, seats_B_left, customers_left in [(100, 150, 600), (50, 75, 300)]:
        state = (t, seats_A_left, seats_B_left, customers_left)
        if state not in state_to_idx:
            # 最も近い顧客数を選択
            closest_c = customers[np.argmin(np.abs(customers - customers_left))]
            state = (t, seats_A_left, seats_B_left, closest_c)
            if state not in state_to_idx:
                continue
        
        state_idx = state_to_idx[state]
        optimal_action_idx = np.argmax(Q[state_idx])
        optimal_pA, optimal_pB = actions[optimal_action_idx]
        
        # 期待収益の計算（モンテカルロシミュレーション）
        total_rewards = []
        for _ in range(num_simulations):
            reward, _, _, _ = simulate_sales(
                optimal_pA, optimal_pB, t, seats_A_left, seats_B_left, customers_left
            )
            total_rewards.append(reward)
        expected_reward = np.mean(total_rewards)
        
        results.append({
            'Day': t,
            'Seats_A': seats_A_left,
            'Seats_B': seats_B_left,
            'Customers_Left': customers_left,
            'Price_A': optimal_pA,
            'Price_B': optimal_pB,
            'Expected_Revenue': expected_reward
        })

# 結果をデータフレームで表示
df_results = pd.DataFrame(results)
print("\n全日程の最適価格と期待収益:")
print(df_results)

# 総収益の推定（初期状態から最適価格でシミュレーション）
total_rewards = []
for _ in range(num_simulations):
    t, seats_A_left, seats_B_left, customers_left = 1, 100, 150, initial_customers
    total_reward = 0
    while t <= days and (seats_A_left > 0 or seats_B_left > 0) and customers_left > 0:
        state = (t, seats_A_left, seats_B_left, customers_left)
        closest_c = customers[np.argmin(np.abs(customers - customers_left))]
        state = (t, seats_A_left, seats_B_left, closest_c)
        state_idx = state_to_idx.get(state, None)
        if state_idx is None:
            break
        action_idx = np.argmax(Q[state_idx])
        pA, pB = actions[action_idx]
        reward, next_seats_A, next_seats_B, next_customers = simulate_sales(
            pA, pB, t, seats_A_left, seats_B_left, customers_left
        )
        total_reward += reward
        t, seats_A_left, seats_B_left, customers_left = t + 1, next_seats_A, next_seats_B, next_customers
    total_rewards.append(total_reward)
print(f"\n全日程の総期待収益（平均）：{np.mean(total_rewards):.2f}円")
print(f"総期待収益の標準偏差：{np.std(total_rewards):.2f}円")