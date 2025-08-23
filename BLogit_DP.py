import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import gammaln
import time
start = time.time()


#状況の設定
#席種；ホームサポーター自由席，ミックスバック自由席，ホームバック自由席
mu=-1  #価格感応度
N_init = 2000 # 販売開始時点の潜在顧客数
C_init = 300  # 販売開始時点のチケット在庫数（キャパシティ）
T = 10        # 全販売期間（日数）

# 任意の1人の顧客が価格rで購入する確率P(r)の定義
def P(r): #exp(Vi)/{exp(0)+exp(Vi)} 何も購入しない場合の効用V0=0
    return np.exp(mu * r) / (1 + np.exp(mu * r))

def V_1(r, C, N_t): #最終期間前日の価値関数⇒これ以降，在庫の価値無し
    s = np.arange(N_t + 1) #購入希望者数sが0人からN_t人までの全パターン
                          #⇒Σを計算するための配列
    log_comb_vals = gammaln(N_t + 1)- gammaln(s + 1)- gammaln(N_t- s + 1) #全組み合わせの計算
    log_p_r = s * np.log(P(r)) #購入する確率部分の計算
    log_p_not_r = (N_t- s) * np.log(1- P(r)) #購入しない確率部分の計算
    log_probs = log_comb_vals + log_p_r + log_p_not_r #ln(組み合わせ×確率)
    probs = np.exp(log_probs- np.max(log_probs)) #exp{ln(組み合わせ×確率)-max(ln(組み合わせ×確率))}
                                                #⇒確率の相対的な大きさに変換
    probs /= probs.sum()#相対的な大きさから，合計が1になるように正規化
    return np.sum(probs * np.minimum(s, C) * r) #時刻T-1の価値関数を返す

def V_t(r, C, V_next, N_t): #最終期間前日以前の価値関数⇒これ以降も在庫の価値あり
    s = np.arange(N_t + 1) #購入希望者数sが0人からN_t人までの全パターン
                          #⇒Σを計算するための配列
    log_comb_vals = gammaln(N_t + 1)- gammaln(s + 1)- gammaln(N_t- s + 1) #全組み合わせの計算
    log_p_r = s * np.log(P(r)) #購入する確率部分の計算
    log_p_not_r = (N_t- s) * np.log(1- P(r)) #購入しない確率部分の計算
    log_probs = log_comb_vals + log_p_r + log_p_not_r #ln(組み合わせ×確率)
    probs = np.exp(log_probs- np.max(log_probs)) #exp{ln(組み合わせ×確率)-max(ln(組み合わせ×確率))}
                                                #⇒確率の相対的な大きさに変換
    probs /= probs.sum()#相対的な大きさから，合計が1になるように正規化
    V_next_values = np.array([V_next(max(C- sold, 0)) for sold in s])
   #⇒次の時点の価値関数 V_nextを在庫Cから売れた数soldを引いた値で計算
   #⇒V_next(C- sold)は関数である
    return np.sum(probs * (np.minimum(s, C) * r + V_next_values)) #時刻t (0=<t<T-1)の価値関数を返す

#最適価格 r*を求める関数（V_nextは関数）価値関数V_tが最大（-V_t が最小）になる価格 rをt探索
def opt_r_t(C, V_next, N_t): #引数は，在庫C，次の時点の価値関数V_next，潜在顧客数N_t
    result = minimize_scalar(lambda r:-V_t(r, C, V_next, N_t), bounds=(0, 12),#最適価格範囲：0~12 (連続値)
                              method='bounded') #minimize_scalar():最小化
    return result.x, V_t(result.x, C, V_next, N_t) #最適価格,その時の価値関数

def opt_r_1(C, N_t): #最終期間前日T-1の最適価格を求める関数
                    #⇒引数は，在庫C，潜在顧客数N_t
    result = minimize_scalar(lambda r:-V_1(r, C, N_t), bounds=(0, 12),
                              method='bounded')
    return result.x, V_1(result.x, C, N_t)

# 顧客数の減少パターンを定義
N_dict = {0: N_init} #時刻0において初期顧客数がN_init人存在していることを定義
for t in range(1, T):
    if t < T // 2:  # 小数切り捨てで整数を返す
    # 販売期間の前半では、前日の顧客数（N_dict[t-1]）から毎日5人ずつ市場から離脱すると仮定
        N_dict[t] = max(N_dict[t- 1]- 5, 0)
    else:
        N_dict[t] = max(N_dict[t- 1]- 10, 0)

print(f"各期間の潜在顧客数: {N_dict}")

#以下，後ろ向き帰納法で価値関数を計算（ここがV_maxに該当）
#全てのあり得る状況（時点tと在庫Cの組み合わせ）について、
#「最大の期待収益（価値）」を計算し、V_dictに保存「戦略マップ」
V_dict = {}
# t = T-1 の場合（逆向きなので時間が戻るように計算）
V_dict[T-1] = {}
for C in range(1, C_init + 1): #C = 1～C_init
    r_max, V_max = opt_r_1(C, N_dict[T-1])
    V_dict[T-1][C] = V_max
# 0 <= t < T-1 の場合
for t in range(T-2,-1,-1): #時間を遡って計算 T-2⇒T-1⇒...⇒0
    V_dict[t] = {}
    for C in range(1, C_init + 1): #各時点であり得る全ての在庫数Cについて計算
        # 次の時点(t+1)の最適価格と価値関数をラムダ関数で渡す
        # lambda C: V_dict[t+1].get(C, 0)は．t+1の時点で在庫がCだった場合の最大価値をV_dict[t+1]から取り出す
        #仮にC=0の場合（購入者数が在庫を上回り0になった場合）は，価値関数は0とする
        r_max, V_max = opt_r_t(C, lambda C: V_dict[t+1].get(C, 0), N_dict[t]) #T-1より前の時点なので在庫を考慮
        V_dict[t][C] = V_max



# 各期間の最適価格を計算（前向き計算の実施）「具体的な駒の動き」
t_values = range(T) # 各期間の時刻t 0,1,2,...,T-1
r_values = [] # 各期間の最適価格を格納するリスト
C_values = [C_init] #在庫の初期値を設定してスタート

for t in t_values: #時間を前向きに進めながら計算
    if t < T-1: #最終販売日前日以前 T-1の場合
        r_max, _ = opt_r_t(C_values[-1], lambda C: V_dict[t+1].get(C, 0), N_dict[t])
    else:
        r_max, _ = opt_r_1(C_values[-1], N_dict[t])
    r_values.append(r_max)
    if t < T-1: #その日の販売枚数を仮定して在庫を減らし、次の日の在庫状況を更新
        if t < T // 2:
            C_values.append(max(C_values[-1]- 5, 0)) 
        else:
            C_values.append(max(C_values[-1]- 10, 0))

print(f"各期間の最適価格: {r_values}")
print(f"各期間の在庫数: {C_values}")
end = time.time()
print(f"Total time: {(end - start) / 60:.2f} minutes")
