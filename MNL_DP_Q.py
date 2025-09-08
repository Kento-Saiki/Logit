import numpy as np
import pandas as pd

# ハイパーパラメータ
alpha = 0.1  # 学習率
gamma = 0.9  # 割引率
epsilon = 0.3  # 初期探索率
num_episodes = 100000  # 学習エピソード数
num_simulations = 200  # 収益シミュレーションの試行回数
initial_customers = 600  # 初期潜在顧客数

# 状態空間
days = 10
seats_A = np.arange(0, 101, 10)
seats_B = np.arange(0, 151, 10)
customers = np.arange(0, initial_customers + 1, 50)
state_space = [(t, a, b, c) for t in range(1, days + 1) for a in seats_A for b in seats_B for c in customers]
num_states = len(state_space)

# 行動空間
prices = np.arange(1000, 2001, 100)
actions = [(pA, pB) for pA in prices for pB in prices]
num_actions = len(actions)

# Qテーブルの初期化（ランダム初期化で探索促進）
Q = np.random.normal(0, 0.01, (num_states, num_actions))

# 多項ロジットモデルのパラメータ
beta_0_A = 2.0
beta_0_B = 1.5
beta_p = -0.004  # 購入確率を上げる
beta_t = 0.1

# 購入確率
def choice_probabilities(pA, pB, t):
    v_A = beta_0_A + beta_p * pA + beta_t * (11 - t)
    v_B = beta_0_B + beta_p * pB + beta_t * (11 - t)
    v_0 = 0.0
    denom = np.exp(v_A) + np.exp(v_B) + np.exp(v_0)
    P_A = np.exp(v_A) / denom
    P_B = np.exp(v_B) / denom
    return P_A, P_B

# シミュレーション
def simulate_sales(pA, pB, t, seats_A_left, seats_B_left, customers_left):
    P_A, P_B = choice_probabilities(pA, pB, t)
    max_n_A = min(seats_A_left, customers_left)
    n_A = np.random.binomial(max_n_A, P_A)
    max_n_B = min(seats_B_left, customers_left - n_A)
    n_B = np.random.binomial(max_n_B, P_B)
    reward = pA * n_A + pB * n_B
    next_seats_A = max(seats_A_left - n_A, 0)
    next_seats_B = max(seats_B_left - n_B, 0)
    next_customers = max(customers_left - n_A - n_B, 0)
    return reward, next_seats_A, next_seats_B, next_customers, n_A, n_B

# 状態インデックス
state_to_idx = {s: i for i, s in enumerate(state_space)}

# Q学習
for episode in range(num_episodes):
    t, seats_A_left, seats_B_left, customers_left = 1, 100, 150, initial_customers
    done = False
    episode_reward = 0
    total_sold = 0
    
    while not done:
        state = (t, seats_A_left, seats_B_left, customers_left)
        closest_c = customers[np.argmin(np.abs(customers - customers_left))]
        closest_a = seats_A[np.argmin(np.abs(seats_A - seats_A_left))]
        closest_b = seats_B[np.argmin(np.abs(seats_B - seats_B_left))]
        state = (t, closest_a, closest_b, closest_c)
        state_idx = state_to_idx[state]
        
        # ε-グリーディ
        if np.random.rand() < epsilon:
            action_idx = np.random.randint(num_actions)
        else:
            action_idx = np.argmax(Q[state_idx])
        
        pA, pB = actions[action_idx]
        
        # シミュレーション
        reward, next_seats_A, next_seats_B, next_customers, n_A, n_B = simulate_sales(
            pA, pB, t, seats_A_left, seats_B_left, customers_left
        )
        episode_reward += reward
        total_sold += n_A + n_B
        
        # 次の状態
        next_t = t + 1
        closest_c_next = customers[np.argmin(np.abs(customers - next_customers))]
        next_state = (next_t, next_seats_A, next_seats_B, closest_c_next)
        
        # 終了条件
        if next_t > days or (next_seats_A == 0 and next_seats_B == 0) or next_customers == 0:
            done = True
            next_Q = 0
        else:
            next_state_idx = state_to_idx.get(next_state, state_idx)
            next_Q = np.max(Q[next_state_idx])
        
        # Q値更新
        Q[state_idx, action_idx] += alpha * (reward + gamma * next_Q - Q[state_idx, action_idx])
        
        # 状態更新
        t, seats_A_left, seats_B_left, customers_left = next_t, next_seats_A, next_seats_B, next_customers
    
    if episode % 10000 == 0:
        print(f"Episode {episode}: Total Reward = {episode_reward:.2f}, Total Sold = {total_sold}")
    
    epsilon = max(0.01, epsilon * 0.995)

# 全日程の最適価格と期待収益
results = []
total_sold_A = 0
total_sold_B = 0
t, seats_A_left, seats_B_left, customers_left = 1, 100, 150, initial_customers
total_expected_reward = 0

while t <= days and (seats_A_left > 0 or seats_B_left > 0) and customers_left > 0:
    # 状態を離散化
    closest_c = customers[np.argmin(np.abs(customers - customers_left))]
    closest_a = seats_A[np.argmin(np.abs(seats_A - seats_A_left))]
    closest_b = seats_B[np.argmin(np.abs(seats_B - seats_B_left))]
    state = (t, closest_a, closest_b, closest_c)
    state_idx = state_to_idx.get(state)
    
    if state_idx is None:
        # 最も近い有効な状態を検索
        min_dist = float('inf')
        best_state = state
        for s in state_space:
            dist = abs(s[0] - t) + abs(s[1] - seats_A_left) + abs(s[2] - seats_B_left) + abs(s[3] - customers_left)
            if dist < min_dist:
                min_dist = dist
                best_state = s
        state = best_state
        state_idx = state_to_idx[state]
    
    optimal_action_idx = np.argmax(Q[state_idx])
    optimal_pA, optimal_pB = actions[optimal_action_idx]
    
    # 期待値計算
    total_rewards = []
    total_n_A = []
    total_n_B = []
    for _ in range(num_simulations):
        reward, _, _, _, n_A, n_B = simulate_sales(
            optimal_pA, optimal_pB, t, seats_A_left, seats_B_left, customers_left
        )
        total_rewards.append(reward)
        total_n_A.append(n_A)
        total_n_B.append(n_B)
    expected_reward = np.mean(total_rewards)
    expected_n_A = np.mean(total_n_A)
    expected_n_B = np.mean(total_n_B)
    total_expected_reward += expected_reward
    
    # デバッグ：購入確率
    P_A, P_B = choice_probabilities(optimal_pA, optimal_pB, t)
    print(f"Day {t}: P_A = {P_A:.3f}, P_B = {P_B:.3f}, Expected Sold = {expected_n_A + expected_n_B:.2f}")
    
    results.append({
        'Day': t,
        'Seats_A': seats_A_left,
        'Seats_B': seats_B_left,
        'Customers_Left': customers_left,
        'Price_A': optimal_pA,
        'Price_B': optimal_pB,
        'Expected_Seats_A_Sold': expected_n_A,
        'Expected_Seats_B_Sold': expected_n_B,
        'Expected_Revenue': expected_reward
    })
    
    # 状態更新（整数に丸める）
    seats_A_left = max(round(seats_A_left - expected_n_A), 0)
    seats_B_left = max(round(seats_B_left - expected_n_B), 0)
    customers_left = max(round(customers_left - expected_n_A - expected_n_B), 0)
    total_sold_A += expected_n_A
    total_sold_B += expected_n_B
    t += 1

# 結果表示
df_results = pd.DataFrame(results)
print("\n全日程の最適価格と期待収益:")
print(df_results)
print(f"\n総販売枚数：席種A = {total_sold_A:.2f}, 席種B = {total_sold_B:.2f}")
print(f"総在庫との比較：席種A残 = {100 - total_sold_A:.2f}, 席種B残 = {150 - total_sold_B:.2f}")
print(f"総顧客数との比較：残顧客 = {initial_customers - total_sold_A - total_sold_B:.2f}")
print(f"総期待収益（日ごとの合計）：{total_expected_reward:.2f}円")

# 総収益の推定
total_rewards = []
for _ in range(num_simulations):
    t, seats_A_left, seats_B_left, customers_left = 1, 100, 150, initial_customers
    total_reward = 0
    while t <= days and (seats_A_left > 0 or seats_B_left > 0) and customers_left > 0:
        closest_c = customers[np.argmin(np.abs(customers - customers_left))]
        closest_a = seats_A[np.argmin(np.abs(seats_A - seats_A_left))]
        closest_b = seats_B[np.argmin(np.abs(seats_B - seats_B_left))]
        state = (t, closest_a, closest_b, closest_c)
        state_idx = state_to_idx.get(state)
        if state_idx is None:
            t += 1
            continue
        action_idx = np.argmax(Q[state_idx])
        pA, pB = actions[action_idx]
        reward, next_seats_A, next_seats_B, next_customers, _, _ = simulate_sales(
            pA, pB, t, seats_A_left, seats_B_left, customers_left
        )
        total_reward += reward
        t, seats_A_left, seats_B_left, customers_left = t + 1, next_seats_A, next_seats_B, next_customers
    total_rewards.append(total_reward)
print(f"\n全日程の総期待収益（平均）：{np.mean(total_rewards):.2f}円")
print(f"総期待収益の標準偏差：{np.std(total_rewards):.2f}円")