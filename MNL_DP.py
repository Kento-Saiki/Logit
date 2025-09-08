import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import gammaln
import time
start = time.time()

#状況の設定
#席種；ホームサポーター自由席，ミックスバック自由席，ホームバック自由席
N_init = 2000 # 販売開始時点の潜在顧客数
C_init = 300  # 販売開始時点のチケット在庫数（キャパシティ）
T = 40        # 全販売期間（日数）

rmin=1700 #最低価格
rmax=17000 #最高価格
dr=1 #価格変動幅

r=np.arange(rmin, rmax+1, dr) #価格変動範囲
alpha=[0, 0.2596, 0.0142, 0] #固定効果 ASC 前から，購入しないという選択肢(k=0)，ホームサポーター自由席(k=1)，ミックスバック自由席(k=2)，ホームバック自由席(k=3)
beta=[-1.5410, 2.0064, 2.3843, -3.1859] #パラメータ（価格r，相手ランクoprank，順位rank，締め切り効果deadline）

def V_sum(r,oprank,rank,deadline): #総効用
    V_sum = 0
    for k in range(4): #総効用
        V_sum = V_sum + alpha[k] + beta[k] * r + alpha[0] + beta[0] * r + beta[1] *  oprank + beta[2] * rank + beta[3] * deadline
    return V_sum

# 確率Pの定義（ロジスティック関数の計算）選択肢kを選択
def P(k,r,oprank,rank,deadline): #例えば，席種k，価格r=100,残り日数t=3,相手ランクoprank=10,順位rank=4,締め切り効果deadline=1/(6.5+t)
    if k == 0 : #購入しないという選択をする場合の確率
        return 1 / (1 + np.exp(V_sum(r,oprank,rank,deadline)))
    else:
        return np.exp(alpha[k] + beta[0] * r + beta[1] * oprank + beta[2] * rank + beta[3] * deadline) / (1 + np.exp(V_sum(r,oprank,rank,deadline)))

def V_1(r,oprank,rank,deadline,C, N_t): #sk：席種kの購入希望者数
    s1 = np.arange(N_t + 1) # 購入希望者数s1が0人からN_t人までの全パターン
    s2 = np.arange(N_t+1 - s1) # 購入しない人の数s2
    s3=np.arange(N_t+1- s1 - s2) # 在庫数Cが0からC人までの全パターン
    # 以下は (N_t C s) * P(r)^s * (1-P(r))^(N_t-s) の計算
    log_comb1 = gammaln(N_t + 1)- gammaln(s1 + 1)- gammaln(N_t- s1 + 1) #ガンマ関数の自然対数=ln(N_t C s)
    log_comb2 = gammaln(N_t+1 - s1)- gammaln(s2 + 1)- gammaln(N_t+1 - s1- s2 + 1) #ガンマ関数の自然対数=ln(N_t C s)
    log_comb3 = gammaln(N_t+1- s1 - s2)- gammaln(s3 + 1)- gammaln(N_t+1- s1 - s2- s3 + 1) #ガンマ関数の自然対数=ln(N_t C s)
    log_p_r1 = s1 * np.log(P(1,r,oprank,rank,deadline)) #席種k=1を購入する
    log_p_r2 = s2 * np.log(P(2,r,oprank,rank,deadline)) #席種k=2を購入する
    log_p_r3 = s3 * np.log(P(3,r,oprank,rank,deadline)) #席種k=3を購入する
    log_p_not_r = (N_t- s1- s2- s3) * np.log(P(0,r,oprank,rank,deadline)) #購入しない
    log_probs = log_comb1 + log_comb2 + log_comb3 + log_p_r1 + log_p_r2 + log_p_r3 + log_p_not_r
    probs = np.exp(log_probs- np.max(log_probs)) #数値安定性のためのシフト
    probs /= probs.sum() #合計が1になるように正規化
    return np.sum(probs * np.minimum(s1, C) * r) #収益の計算

def V_t(r, C, V_next, N_t):
    #--ここから，上と同様の確率の計算
    s = np.arange(N_t + 1)
    log_comb_vals = gammaln(N_t + 1)- gammaln(s + 1)- gammaln(N_t- s + 1)
    log_p_r = s * np.log(P(r))
    log_p_not_r = (N_t- s) * np.log(1- P(r))
    log_probs = log_comb_vals + log_p_r + log_p_not_r
    probs = np.exp(log_probs- np.max(log_probs))
    probs /= probs.sum()
    #--ここまで
    V_next_values = np.array([V_next(max(C- sold, 0)) for sold in s]) #次の時点の最大価値
    return np.sum(probs * (np.minimum(s, C) * r + V_next_values))

#最適価格を求める関数
def opt_r_t(C, V_next, N_t): #価値関数V_tが最大（-V_t が最小）になる価格 r を探
    result = minimize_scalar(lambda r: -V_t(r, C, V_next, N_t), bounds=(0, 12), method='bounded')
    return result.x, V_t(result.x, C, V_next, N_t)

def opt_r_1(C, N_t):
    result = minimize_scalar(lambda r: -V_1(r, C, N_t), bounds=(0, 12), method='bounded')
    return result.x, V_1(result.x, C, N_t)


# 顧客数の減少パターンを定義
N_dict = {0: N_init}
for t in range(1, T):
    if t < T // 2: #小数切り捨てで整数を返す
    #販売期間の前半では、前日の顧客数（N_dict[t-1]）から毎日5人ずつ市場から離脱すると仮定 
        N_dict[t] = max(N_dict[t- 1]- 5, 0)
    else:
        N_dict[t] = max(N_dict[t- 1]- 10, 0)


# V_dict の計算
V_dict = {}
# t = T-1（最終日） の場合
V_dict[T-1] = {}
for C in range(1, C_init + 1):
    r_max, V_max = opt_r_1(C, N_dict[T-1])
    V_dict[T-1][C] = V_max
# 0 <= t < T-1（最終日より前） の場合
for t in range(T-2,-1,-1): #時間を遡って計算
    V_dict[t] = {}
    for C in range(1, C_init + 1):
        # 次の時点(t+1)の価値関数をラムダ式で渡す
        r_max, V_max = opt_r_t(C, lambda C: V_dict[t+1].get(C, 0), N_dict[t])
        V_dict[t][C] = V_max


# 各期間の最適価格を計算
t_values = range(T)
r_values = []
C_values = [C_init]

for t in t_values:
    # 現在の在庫 C_values[-1] と潜在顧客数 N_dict[t] の下で最適価格を計算
    if t < T-1:
        r_max, _ = opt_r_t(C_values[-1], lambda C: V_dict[t+1].get(C, 0),N_dict[t])
    else:
        r_max, _ = opt_r_1(C_values[-1], N_dict[t])
    r_values.append(r_max)
    if t < T-1:
        if t < T // 2:
            C_values.append(max(C_values[-1]- 5, 0))
        else:
            C_values.append(max(C_values[-1]- 10, 0))

print(r_values)

end = time.time()
print(f"Total time: {(end - start) / 60:.2f} minutes")