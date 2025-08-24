import pandas as pd
import numpy as np
from scipy.optimize import minimize
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os
import time
start = time.time()


# 日本語フォントの設定例 (Windowsの場合)
try:
    plt.rcParams['font.family'] = "Yu Gothic" # 游ゴシック (Windowsの場合)
    plt.rcParams['axes.unicode_minus'] = False # 負の符号を正しく表示するために設定
except Exception:
    # フォントが見つからない場合や他のOSの場合のフォールバック
    print("指定された日本語フォントが見つからないか、他のOS環境です。")

class DemandEstimator: 
    """
    Cho et al. (2024) の論文に基づく需要推定モデル

    product_ids=product_ids, #'ホームサポーター自由席', 'ホームバック自由席'
    feature_cols=feature_cols, #'販売価格', '残り日数', "相手ランク", "順位"
    regularization_strength=REGULARIZATION=0.01

    """
    def __init__(self, product_ids, feature_cols, regularization_strength=0.1):
        if not product_ids:
            raise ValueError("product_idsは空にできません。")
        
        self.base_feature_cols = [col for col in feature_cols if col != '残り日数'] # [残り日数，相手ランク，順位]
        self.extended_feature_cols = self.base_feature_cols + ['残り日数'] +  ['締め切り効果']#  [残り日数，相手ランク，順位]+[締め切り効果] 
        
        self.product_ids = product_ids
        self.regularization_strength = regularization_strength
        self.alpha_hat = None
        self.beta_hat = None
        self.scaler = MinMaxScaler()
        self.arrival_distribution = None

    def _prepare_data(self, df_raw):
        df = df_raw.copy()
        df['残り日数'] = df['残り日数']
        df['締め切り効果'] = 1.0 / (df['残り日数']**2 + 10)
        df['obs_id'] = df.groupby(['試合名', '残り日数']).ngroup()
        return df
    
    def _run_mle_step(self, baseline_product, df_processed):
        other_products = [p for p in self.product_ids if p != baseline_product]
        df_purchases = df_processed[df_processed['販売数量'] > 0].copy()
        obs_groups = df_purchases.groupby('obs_id')
        initial_params = np.zeros(len(other_products) + len(self.extended_feature_cols))

        def _negative_log_likelihood(params, df, obs_groups):
            alpha_star_params = params[:len(other_products)]
            beta_params = params[len(other_products):]
            alpha_star_map = dict(zip(other_products, alpha_star_params))
            beta_map = dict(zip(self.extended_feature_cols, beta_params))
            total_log_likelihood = 0
            for obs_id, group in obs_groups:
                sum_exp_utility = 0
                available_products_in_obs = df[df['obs_id'] == obs_id]
                for row in available_products_in_obs.itertuples(index=False):
                    utility = 0
                    if getattr(row, '席種') != baseline_product:
                        utility += alpha_star_map.get(getattr(row, '席種'), 0)
                    for feature in self.extended_feature_cols:
                        utility += beta_map.get(feature, 0) * getattr(row, feature)
                    sum_exp_utility += np.exp(np.clip(utility, -700, 700))
                if sum_exp_utility > 1e-9:
                    for _, row in group.iterrows():
                        utility = 0
                        if row['席種'] != baseline_product:
                            utility += alpha_star_map.get(row['席種'], 0)
                        for feature in self.extended_feature_cols:
                            utility += beta_map.get(feature, 0) * row[feature]
                        log_prob = utility - np.log(sum_exp_utility)
                        total_log_likelihood += log_prob * row['販売数量']
            l2_penalty = self.regularization_strength * np.sum(beta_params**2)
            return -total_log_likelihood + l2_penalty

        result = minimize(fun=_negative_log_likelihood, x0=initial_params, args=(df_purchases, obs_groups), method='Nelder-Mead', options={'disp': True, 'maxiter': 10000})
        
        if not result.success:
            return None, None
            
        estimated_params = result.x
        alpha_star_hat = dict(zip(other_products, estimated_params[:len(other_products)]))
        beta_hat = dict(zip(self.extended_feature_cols, estimated_params[len(other_products):]))
        
        return alpha_star_hat, beta_hat

    def _fit_arrival_distribution(self, df_raw):
        print("\n--- 到着分布（購入顧客のうち何%がその日到着するか）の学習 ---")
        df = df_raw.copy()
        daily_sales = df.groupby(['試合名', '残り日数'])['販売数量'].sum().reset_index()
        total_sales = df.groupby('試合名')['販売数量'].sum().reset_index().rename(columns={'販売数量': 'total_sales'})
        
        merged = pd.merge(daily_sales, total_sales, on='試合名')
        merged['arrival_prob'] = merged.apply(lambda row: row['販売数量'] / row['total_sales'] if row['total_sales'] > 0 else 0, axis=1)
        
        self.arrival_distribution = merged.groupby('残り日数')['arrival_prob'].mean().sort_index(ascending=False)
        self.arrival_distribution /= self.arrival_distribution.sum()
        print("来場者到着分布の学習完了。")

    def fit_case2(self, df_raw):
        print("--- Case 2: 市場シェア未知モデルの推定開始 ---")
        
        df_processed = self._prepare_data(df_raw)
        self.scaler.fit(df_processed[self.extended_feature_cols])
        df_processed[self.extended_feature_cols] = self.scaler.transform(df_processed[self.extended_feature_cols])

        initial_baseline = self.product_ids[0]
        print(f"\nStep 1: 仮のベースライン製品 '{initial_baseline}' でMLEを実行...")
        
        alpha_star_1, beta_1 = self._run_mle_step(initial_baseline, df_processed)
        if alpha_star_1 is None:
            print("Error: 1回目のMLEが収束しませんでした。")
            return self
            
        print("1回目のMLE完了。")
        print("推定されたβ:", beta_1)

        final_baseline = initial_baseline
        if all(alpha_star >= 0 for alpha_star in alpha_star_1.values()):
            print("\nStep 3: 全てのα*が非負です。仮のベースラインが最終ベースラインです。")
            self.beta_hat = beta_1
            self.alpha_hat = alpha_star_1
            self.alpha_hat[final_baseline] = 0.0
        else:
            print("\nStep 3: 負のα*が検出されました。ベースラインを再探索します。")
            min_alpha_star_prod = min(alpha_star_1, key=alpha_star_1.get)
            final_baseline = min_alpha_star_prod
            print(f"Step 5: 新しいベースライン製品は '{final_baseline}' です。再度MLEを実行...")
            
            alpha_star_2, beta_2 = self._run_mle_step(final_baseline, df_processed)
            if alpha_star_2 is None:
                print("Error: 2回目のMLEが収束しませんでした。")
                return self
            
            self.beta_hat = beta_2
            self.alpha_hat = alpha_star_2
            self.alpha_hat[final_baseline] = 0.0
            
        print("\n--- Case 2推定完了 ---")
        print(f"最終的なベースライン製品: {final_baseline}")
        print("最終的に推定された β (beta_hat):")
        for k, v in self.beta_hat.items(): print(f"  {k}: {v:.4f}")
        print("最終的に推定された α (alpha_hat):")
        for k, v in self.alpha_hat.items(): print(f"  {k}: {v:.4f}")

        self._fit_arrival_distribution(df_raw)
        
        return self

    def predict_probas(self, df_features_for_day):
        df_scaled = df_features_for_day.copy()
        df_scaled['残り日数'] = df_scaled['残り日数']
        df_scaled['締め切り効果'] = 1 / (df_scaled['残り日数']**2 + 10)

        df_scaled[self.extended_feature_cols] = self.scaler.transform(df_scaled[self.extended_feature_cols])
        utilities = {}
        for _, row in df_scaled.iterrows():
            prod_id = row['席種']
            utility = self.alpha_hat.get(prod_id, 0)
            for feature in self.extended_feature_cols:
                utility += self.beta_hat.get(feature, 0) * row[feature]
            utilities[prod_id] = utility
        exp_utilities = {k: np.exp(np.clip(v, -700, 700)) for k, v in utilities.items()}
        denominator = 1 + sum(exp_utilities.values())
        probas = {prod_id: exp_util / denominator for prod_id, exp_util in exp_utilities.items() if denominator > 1e-9}
        probas['no_purchase'] = 1 / denominator if denominator > 1e-9 else 1.0
        return probas

    # DemandEstimatorクラスの内部に追加
    def get_original_scale_params(self):
        """
        正規化されたデータで推定されたパラメータを、元のスケールのパラメータに逆変換する。
        """
        # スケーラーからmin, max, scaleの値を取得
        mins = self.scaler.data_min_
        maxs = self.scaler.data_max_
        scales = maxs - mins
        feature_names = self.scaler.get_feature_names_out()

        beta_original = {}
        constant_shift = 0

        # beta_hatを逆変換し、定数項のシフト分を計算
        for i, feature in enumerate(feature_names):
            if scales[i] > 1e-9: # 0除算を避ける
                # 1. 元のスケールのbetaを計算
                beta_original[feature] = self.beta_hat[feature] / scales[i]
                
                # 2. alphaに影響する定数項シフトを計算
                constant_shift -= (self.beta_hat[feature] * mins[i]) / scales[i]
            else:
                beta_original[feature] = self.beta_hat[feature]

        # alpha_hatを定数項シフトで調整
        alpha_original = {}
        for prod_id, alpha_val in self.alpha_hat.items():
            # ベースラインのalphaは0のまま
            if alpha_val == 0.0:
                alpha_original[prod_id] = 0.0
            else:
                alpha_original[prod_id] = alpha_val + constant_shift
        
        return alpha_original, beta_original

# --- 1. ファイルとモデル設定の定義 ---
all_csv_files = [
    'C:\\Users\\saiki\\OneDrive - 東京理科大学\\This4年前期\\卒業研究1\\卒業研究使用データ\\販売価格・販売量\\川崎フロンターレ戦.csv',
    'C:\\Users\\saiki\\OneDrive - 東京理科大学\\This4年前期\\卒業研究1\\卒業研究使用データ\\販売価格・販売量\\京都サンガ戦.csv',
    'C:\\Users\\saiki\\OneDrive - 東京理科大学\\This4年前期\\卒業研究1\\卒業研究使用データ\\販売価格・販売量\\ジュビロ磐田戦.csv',
    'C:\\Users\\saiki\\OneDrive - 東京理科大学\\This4年前期\\卒業研究1\\卒業研究使用データ\\販売価格・販売量\\浦和レッズ戦.csv',
    'C:\\Users\\saiki\\OneDrive - 東京理科大学\\This4年前期\\卒業研究1\\卒業研究使用データ\\販売価格・販売量\\名古屋グランパス戦.csv',
    'C:\\Users\\saiki\\OneDrive - 東京理科大学\\This4年前期\\卒業研究1\\卒業研究使用データ\\販売価格・販売量\\柏レイソル戦.csv',
    'C:\\Users\\saiki\\OneDrive - 東京理科大学\\This4年前期\\卒業研究1\\卒業研究使用データ\\販売価格・販売量\\東京ヴェルディ戦.csv',
    'C:\\Users\\saiki\\OneDrive - 東京理科大学\\This4年前期\\卒業研究1\\卒業研究使用データ\\販売価格・販売量\\町田ゼルビア戦.csv',
    'C:\\Users\\saiki\\OneDrive - 東京理科大学\\This4年前期\\卒業研究1\\卒業研究使用データ\\販売価格・販売量\\湘南ベルマーレ戦.csv',
    'C:\\Users\\saiki\\OneDrive - 東京理科大学\\This4年前期\\卒業研究1\\卒業研究使用データ\\販売価格・販売量\\横浜FM戦.csv',
    'C:\\Users\\saiki\\OneDrive - 東京理科大学\\This4年前期\\卒業研究1\\卒業研究使用データ\\販売価格・販売量\\サンフレッチェ広島戦.csv',
    'C:\\Users\\saiki\\OneDrive - 東京理科大学\\This4年前期\\卒業研究1\\卒業研究使用データ\\販売価格・販売量\\サガン鳥栖戦.csv',
    'C:\\Users\\saiki\\OneDrive - 東京理科大学\\This4年前期\\卒業研究1\\卒業研究使用データ\\販売価格・販売量\\コンサドーレ札幌戦.csv',
    'C:\\Users\\saiki\\OneDrive - 東京理科大学\\This4年前期\\卒業研究1\\卒業研究使用データ\\販売価格・販売量\\ヴィッセル神戸戦.csv',
    'C:\\Users\\saiki\\OneDrive - 東京理科大学\\This4年前期\\卒業研究1\\卒業研究使用データ\\販売価格・販売量\\アルビレックス新潟戦.csv',
    'C:\\Users\\saiki\\OneDrive - 東京理科大学\\This4年前期\\卒業研究1\\卒業研究使用データ\\販売価格・販売量\\アビスパ福岡戦.csv',
]


#2: 学習用ファイルリストからシミュレーション対象を選択 ###
train_files = [f for f in all_csv_files] # if f != simulation_target_file

print(f"学習データ数: {len(train_files)}試合")
#1: シミュレーション対象をファイルパスで指定 ###

product_ids = ['ミックスバック自由席','ホームサポーター自由席'] #,"ホームバック自由席"
feature_cols = ['販売価格', '残り日数', "相手ランク"] #, "相手ランク", "順位"
REGULARIZATION = 0.0001

# --- 2. データ読み込みと前処理（訓練データとテストデータを別々に作成） ---
print("\n--- 学習データの読み込みと前処理 ---")
train_df_list = []
for file in train_files:
    if os.path.exists(file):
        df = pd.read_csv(file, encoding='utf-8-sig')
        df['試合名'] = os.path.basename(file).replace('.csv', '')
        df['席種'] = df['席種'].str.replace(r'^\d+:\s*', '', regex=True)
        df = df[df['席種'].str.contains("ミックスバック自由席|ホームサポーター自由席", na=False)].copy() #席種の限定|ホームバック自由席
        df = df.iloc[6:].copy() # インデックス6から最後までを使用
        train_df_list.append(df)
df_train = pd.concat(train_df_list, ignore_index=True)
df_train = df_train.dropna(subset=feature_cols + ['販売数量'])
print(f"学習データの総件数: {len(df_train)}件")


# --- 3. モデルの推定（学習データのみを使用） ---
estimator = DemandEstimator(
    product_ids=product_ids, #'ホームサポーター自由席', 'ホームバック自由席','ミックスバック自由席'
    feature_cols=feature_cols, #'販売価格', '残り日数', "相手ランク", "順位"
    regularization_strength=REGULARIZATION
)
### 変更点 3: 学習データ(df_train)でモデルをfitする ###
estimator.fit_case2(df_train)

# 元のスケールに戻したパラメータを取得
alpha_orig, beta_orig = estimator.get_original_scale_params()

if alpha_orig is not None:
    print("\n--- 元のスケールに変換したパラメータ ---")
    print("変換後の β (beta_original):")
    for k, v in beta_orig.items():
        print(f"  {k}: {v:.4f}")
    
    print("変換後の α (alpha_original):")
    for k, v in alpha_orig.items():
        print(f"  {k}: {v:.4f}")


# --- 4. シミュレーションの実行（テストデータを使用） ---
for simulation_target_file in all_csv_files:
    testmatch_name = os.path.basename(simulation_target_file).replace('.csv', '')
    print(f"シミュレーション対象: {testmatch_name}")
    if not os.path.exists(simulation_target_file):
        raise FileNotFoundError(f"シミュレーション対象ファイルが見つかりません: {simulation_target_file}")
    df_test = pd.read_csv(simulation_target_file, encoding='utf-8-sig')
    df_test['試合名'] = testmatch_name
    df_test['席種'] = df_test['席種'].str.replace(r'^\d+:\s*', '', regex=True)
    df_test = df_test[df_test['席種'].str.contains("ミックスバック自由席|ホームサポーター自由席", na=False)].copy() #席種の限定|ホームバック自由席
    df_test = df_test.dropna(subset=feature_cols + ['販売数量']+['経過日数'])
    print(f"シミュレーションデータの総件数: {len(df_test)}件")
    print(f"\n--- 『{testmatch_name}』の販売シミュレーション開始 ---")
    # --- ここから、クラス外でのn0とLambdaの計算部分 --------------------------------------------------------------
    print("\n--- クラス外でのn0とLambdaの推定開始 ---")

    # df_testに前処理を適用（fit_case2の_prepare_dataと同様の処理）
    df_test_processed_for_n0_lambda = df_test.copy()
    df_test_processed_for_n0_lambda['残り日数'] = df_test_processed_for_n0_lambda['残り日数']
    df_test_processed_for_n0_lambda['締め切り効果'] = 1.0 / (df_test_processed_for_n0_lambda['残り日数']**2 + 10)
    df_test_processed_for_n0_lambda['開始効果'] = -np.log(df_test['経過日数'] + 0.1) / ((df_test['経過日数'] + 0.1) ** 4)  # ゼロ除算を避けるために小さな値を加える
    df_test_processed_for_n0_lambda['obs_id'] = df_test_processed_for_n0_lambda.groupby(['試合名', '残り日数']).ngroup()

    # 学習済みスケーラーでテストデータの特徴量を変換
    sum_term_for_n0_calc_external = 0
    # これは学習データでfitされたscalerをそのまま使用することが重要
    df_test_processed_scaled_for_n0_lambda = df_test_processed_for_n0_lambda.copy()
    df_test_processed_scaled_for_n0_lambda[estimator.extended_feature_cols] = estimator.scaler.transform(df_test_processed_scaled_for_n0_lambda[estimator.extended_feature_cols])
    unique_obs_ids_external = df_test_processed_scaled_for_n0_lambda[['試合名', '残り日数', 'obs_id']].drop_duplicates()

    for _, obs_row in unique_obs_ids_external.iterrows(): #全残り日数ごとに計算
        obs_id = obs_row['obs_id']# 固有idの取得
        daily_sales_prod = 0 #日にちごとの販売数量
        
        day_data_scaled_current_obs_external = df_test_processed_scaled_for_n0_lambda[df_test_processed_scaled_for_n0_lambda['obs_id'] == obs_id].copy()

        sum_exp_v_ij_for_day_external = 0 #ある残り日数の全席種での総効用
        for prod_id in estimator.product_ids:  #全席種ごとに計算
            seat_data_scaled_external = day_data_scaled_current_obs_external[day_data_scaled_current_obs_external['席種'] == prod_id]
            if seat_data_scaled_external.empty:
                continue
            #効用計算に必要な特徴量（販売価格, 残り日数, 相手ランク, 順位, 締め切り効果）の値を辞書として取得
            row_features_scaled_external = {feature: seat_data_scaled_external[feature].iloc[0] for feature in estimator.extended_feature_cols}
            
            v_ij_external = estimator.alpha_hat.get(prod_id, 0)# v=alpa
            for feature in estimator.extended_feature_cols:
                v_ij_external += estimator.beta_hat.get(feature, 0) * row_features_scaled_external[feature] #v=alpah + beta * x
            
            # obs_idがobs_row['obs_id']で、席種がprod_idの販売数量（当日の売上）を取得
            filtered = df_test_processed_scaled_for_n0_lambda[
                (df_test_processed_scaled_for_n0_lambda['obs_id'] == obs_id) &
                (df_test_processed_scaled_for_n0_lambda['席種'] == prod_id)
            ]
            if not filtered.empty:
                daily_sales_prod += filtered['販売数量'].iloc[0]

            MAX_EXP_ARG = 700
            v_ij_clipped_external = np.clip(v_ij_external, None, MAX_EXP_ARG) 
            sum_exp_v_ij_for_day_external += np.exp(v_ij_clipped_external)

        if sum_exp_v_ij_for_day_external > 0:
            sum_term_for_n0_calc_external += daily_sales_prod*(1 / sum_exp_v_ij_for_day_external)

    # df_testから観測された総購入者数を計算
    n_R_observed_total_external = df_test[df_test['席種'].isin(estimator.product_ids)]['販売数量'].sum()

    estimated_n0_total_external = sum_term_for_n0_calc_external 
    estimated_lambda_total_external = n_R_observed_total_external + estimated_n0_total_external 

    print(f"\n推定された総購入者数 (n_R_observed_total_external): {round(n_R_observed_total_external)}")
    print(f"推定された総非購入者数 (estimated_n0_total_external): {round(estimated_n0_total_external)}")
    print(f"推定された総来場者数 (estimated_lambda_total_external): {round(estimated_lambda_total_external)}")

    ### 変更点 4: シミュレーション対象としてdf_testを直接使用 ###
    sim_match_df = df_test.copy() 

    initial_arrivals_for_sim = estimated_lambda_total_external

    simulated_results = []
    for day in sorted(sim_match_df['残り日数'].unique(), reverse=True):
        day_data = sim_match_df[sim_match_df['残り日数'] == day]
        if day_data.empty: continue
        
        # 到着分布は学習データから作られたものを使用
        daily_arrival_prob = estimator.arrival_distribution.get(day, 0)
        daily_arrivals = initial_arrivals_for_sim * daily_arrival_prob
        
        df_predict_day = day_data[['席種'] + feature_cols]
        predicted_probas = estimator.predict_probas(df_predict_day)

        total_purchase_prob = sum(prob for prod, prob in predicted_probas.items() if prod != 'no_purchase')
        daily_total_predicted_sales = daily_arrivals * total_purchase_prob
        
        daily_predicted_sales = {}
        if total_purchase_prob > 1e-9:
            for prod, prob in predicted_probas.items():
                if prod != 'no_purchase':
                    daily_predicted_sales[prod] = daily_total_predicted_sales * (prob / total_purchase_prob)
        
        actual_sales = day_data.set_index('席種')['販売数量'].to_dict()
        result_row = {
            '残り日数': day,
            'シミュレーション_総購入数量_当日': sum(daily_predicted_sales.values()),
            '実際の総購入数量_当日': sum(actual_sales.values()),
        }
        for pid in product_ids:
            result_row[f'期待{pid}_購入数量'] = daily_predicted_sales.get(pid, 0)
            result_row[f'実際の_{pid}_購入数量'] = actual_sales.get(pid, 0)
        simulated_results.append(result_row)

    simulated_results_df = pd.DataFrame(simulated_results).sort_values('残り日数', ascending=False)
    #print("\n--- シミュレーション結果の詳細 ---")
    #print(simulated_results_df.round(2))

end = time.time()
print((end-start)/60)

