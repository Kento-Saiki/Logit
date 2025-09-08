import numpy as np
from scipy.special import gammaln
import time
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 日本語フォントの設定
plt.rcParams['font.family'] = "Yu Gothic" # Windowsの場合
plt.rcParams['axes.unicode_minus'] = False

start = time.time()

# --- 1. 状況の設定 ---
# ★警告: C_A_init, C_B_init, N_init, T の値を大きくすると計算が終わりません。
# ★テストには C_A_init=10, C_B_init=10, N_init=30, T=3 程度を推奨します。

# 席種A, B および「購入しない」選択肢
# 顧客・在庫・期間のパラメータ
N_init = 40      # 販売開始時点の潜在顧客数
T = 3            # 全販売期間（日数）
C_A_init = 10    # 席種Aの初期在庫
C_B_init = 10    # 席種Bの初期在庫

# 多項ロジットモデルのパラメータ（仮定）
alpha_A = 4.4919 #ホームバック自由席
alpha_B = 4.4541 #ミックスバック自由席
beta_price = -0.0005

# 最適価格の探索範囲を離散値で定義
price_candidates = np.arange(2000, 4001, 200)
print(f"探索する価格候補: {price_candidates}")


# --- 2. 需要関数（多項ロジット）の定義 ---
def get_probabilities(r_A, r_B):
    """価格ペア(r_A, r_B)から、各選択肢の購入確率を計算する"""
    V_A = alpha_A + beta_price * r_A
    V_B = alpha_B + beta_price * r_B
    exp_V_A = np.exp(V_A)
    exp_V_B = np.exp(V_B)
    denominator = 1 + exp_V_A + exp_V_B
    
    p_A = exp_V_A / denominator
    p_B = exp_V_B / denominator
    p_no_purchase = 1 / denominator
    
    return p_A, p_B, p_no_purchase

# --- 3. 価値関数の定義（多項分布・個別在庫） ---
# DPの状態は (C_A, C_B) のタプルで管理
def V_1_multi(r_A, r_B, C_tuple, N_t):
    C_A, C_B = C_tuple
    p_A, p_B, p_0 = get_probabilities(r_A, r_B)
    expected_revenue = 0
    
    # sA:席種Aの希望者, sB:席種Bの希望者
    for s_A in range(N_t + 1):
        for s_B in range(N_t - s_A + 1):
            s_0 = N_t - s_A - s_B
            
            log_multinomial_coeff = gammaln(N_t + 1) - gammaln(s_A + 1) - gammaln(s_B + 1) - gammaln(s_0 + 1)
            log_prob_term = s_A * np.log(p_A) + s_B * np.log(p_B) + s_0 * np.log(p_0)
            prob = np.exp(log_multinomial_coeff + log_prob_term)
            
            # ★在庫制約を席種ごとに適用
            sold_A = min(s_A, C_A)
            sold_B = min(s_B, C_B)
            
            revenue = sold_A * r_A + sold_B * r_B
            expected_revenue += prob * revenue
            
    return expected_revenue

def V_t_multi(r_A, r_B, C_tuple, V_next, N_t):
    C_A, C_B = C_tuple
    p_A, p_B, p_0 = get_probabilities(r_A, r_B)
    expected_total_value = 0
    
    for s_A in range(N_t + 1):
        for s_B in range(N_t - s_A + 1):
            s_0 = N_t - s_A - s_B
            
            log_multinomial_coeff = gammaln(N_t + 1) - gammaln(s_A + 1) - gammaln(s_B + 1) - gammaln(s_0 + 1)
            log_prob_term = s_A * np.log(p_A) + s_B * np.log(p_B) + s_0 * np.log(p_0)
            prob = np.exp(log_multinomial_coeff + log_prob_term)
            
            sold_A = min(s_A, C_A)
            sold_B = min(s_B, C_B)
            
            revenue = sold_A * r_A + sold_B * r_B
            
            # 次の時点の在庫タプルを計算
            next_C_tuple = (C_A - sold_A, C_B - sold_B)
            future_value = V_next(next_C_tuple)
            
            expected_total_value += prob * (revenue + future_value)

    return expected_total_value

# --- 4. 最適価格ペア探索関数の定義 ---
def opt_r_multi(C_tuple, V_next, N_t, is_final_day):
    best_value = -1
    best_prices = (price_candidates[0], price_candidates[0])

    for r_A in price_candidates:
        for r_B in price_candidates:
            if is_final_day:
                value = V_1_multi(r_A, r_B, C_tuple, N_t)
            else:
                value = V_t_multi(r_A, r_B, C_tuple, V_next, N_t)
            
            if value > best_value:
                best_value = value
                best_prices = (r_A, r_B)
    
    return best_prices, best_value

# --- 5. DPモデルの求解（後ろ向き計算） ---
print("--- 後ろ向き計算開始 ---")
N_dict = {t: max(0, N_init - int(t * (N_init / (T+1)))) for t in range(T + 1)}

V_dict = {}
policy = {} # 最適価格ペアを保存する方針辞書

for t in range(T, -1, -1):
    print(f"DP計算中... t={t}")
    V_dict[t] = {}
    policy[t] = {}
    
    if t == T: # 販売終了後は価値0
        continue

    # ★在庫ペア(C_A, C_B)の全組み合わせをループ
    for C_A in range(C_A_init + 1):
        for C_B in range(C_B_init + 1):
            C_tuple = (C_A, C_B)
            
            is_final_day = (t == T - 1)
            V_next_func = lambda next_C_tuple: V_dict[t+1].get(next_C_tuple, 0)
            
            prices, V_max = opt_r_multi(C_tuple, V_next_func, N_dict[t], is_final_day)
            
            V_dict[t][C_tuple] = V_max
            policy[t][C_tuple] = prices


# --- 6. 最適価格の導出（前向き計算） ---
print("\n--- 前向き計算開始 ---")
r_A_values = []
r_B_values = []
C_A_path = [C_A_init]
C_B_path = [C_B_init]
sales_A_path = []
sales_B_path = []

current_C_A = C_A_init
current_C_B = C_B_init

for t in range(T):
    current_C_tuple = (current_C_A, current_C_B)
    
    # ★方針(policy)辞書から現在の在庫状態に最適な価格ペアを取得
    r_max_A, r_max_B = policy[t][current_C_tuple]
    
    r_A_values.append(r_max_A)
    r_B_values.append(r_max_B)
    
    # 期待販売数で在庫を更新
    p_A, p_B, _ = get_probabilities(r_max_A, r_max_B)
    
    sold_A = min(current_C_A, int(round(N_dict[t] * p_A)))
    sold_B = min(current_C_B, int(round(N_dict[t] * p_B)))
    
    sales_A_path.append(sold_A)
    sales_B_path.append(sold_B)
    
    current_C_A -= sold_A
    current_C_B -= sold_B
    C_A_path.append(current_C_A)
    C_B_path.append(current_C_B)

    print(f"t={t}: 在庫(A,B)=({current_C_tuple}), 最適価格(A,B)=({r_max_A}, {r_max_B}), 販売数(A,B)=({sold_A}, {sold_B})")

# --- 7. 結果の可視化 ---
fig, axs = plt.subplots(1, 2, figsize=(16, 6))

# 最適価格の推移
axs[0].plot(range(T), r_A_values, marker='o', label='席種A 最適価格')
axs[0].plot(range(T), r_B_values, marker='s', label='席種B 最適価格')
axs[0].set_title('最適価格の推移')
axs[0].set_xlabel('販売期間（日）')
axs[0].set_ylabel('価格（円）')
axs[0].grid(True)
axs[0].legend()

# 在庫の推移
axs[1].plot(range(T+1), C_A_path, marker='o', label='席種A 在庫')
axs[1].plot(range(T+1), C_B_path, marker='s', label='席種B 在庫')
axs[1].set_title('在庫数の推移')
axs[1].set_xlabel('販売期間（日）')
axs[1].set_ylabel('在庫数')
axs[1].grid(True)
axs[1].legend()

plt.tight_layout()
plt.show()

end = time.time()
print(f"\nTotal time: {(end - start):.2f} seconds")