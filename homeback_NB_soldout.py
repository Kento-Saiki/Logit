import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler # StandardScalerをインポート
from statsmodels.discrete.discrete_model import NegativeBinomial # NegativeBinomialをインポート
import os # osモジュールをインポート

#状況：締め切り効果削除，

# --- 1. データの読み込みと前処理 ---おそらく期日別で分析すべき
learn_files = [
    # 絶対パスの記載方法が問題を引き起こす可能性があるので、相対パスに変更することを検討してください。
    # 例: 'data/京都サンガ戦.csv' のように、スクリプトと同じディレクトリにdataフォルダを作成し、その中にCSVを入れる
    'C:\\Users\\saiki\\OneDrive - 東京理科大学\\This4年前期\\卒業研究1\\卒業研究使用データ\\販売価格・販売量\\川崎フロンターレ戦.csv', #予定販売数量：1880
    'C:\\Users\\saiki\\OneDrive - 東京理科大学\\This4年前期\\卒業研究1\\卒業研究使用データ\\販売価格・販売量\\京都サンガ戦.csv', #予定販売数量：1554
    'C:\\Users\\saiki\\OneDrive - 東京理科大学\\This4年前期\\卒業研究1\\卒業研究使用データ\\販売価格・販売量\\ジュビロ磐田戦.csv', #予定販売数量：1340
    'C:\\Users\\saiki\\OneDrive - 東京理科大学\\This4年前期\\卒業研究1\\卒業研究使用データ\\販売価格・販売量\\浦和レッズ戦.csv', #予定販売数量：2350
    'C:\\Users\\saiki\\OneDrive - 東京理科大学\\This4年前期\\卒業研究1\\卒業研究使用データ\\販売価格・販売量\\名古屋グランパス戦.csv', #伸び #予定販売数量：1800
    'C:\\Users\\saiki\\OneDrive - 東京理科大学\\This4年前期\\卒業研究1\\卒業研究使用データ\\販売価格・販売量\\柏レイソル戦.csv', #伸び #予定販売数量：1400
    'C:\\Users\\saiki\\OneDrive - 東京理科大学\\This4年前期\\卒業研究1\\卒業研究使用データ\\販売価格・販売量\\町田ゼルビア戦.csv', #予定販売数量：1124
    'C:\\Users\\saiki\\OneDrive - 東京理科大学\\This4年前期\\卒業研究1\\卒業研究使用データ\\販売価格・販売量\\湘南ベルマーレ戦.csv', #予定販売数量：1200
    'C:\\Users\\saiki\\OneDrive - 東京理科大学\\This4年前期\\卒業研究1\\卒業研究使用データ\\販売価格・販売量\\横浜FM戦.csv', #伸び #予定販売数量：1849
    'C:\\Users\\saiki\\OneDrive - 東京理科大学\\This4年前期\\卒業研究1\\卒業研究使用データ\\販売価格・販売量\\サンフレッチェ広島戦.csv', #最後おかしすぎる⇒堺市民招待券 1080
    'C:\\Users\\saiki\\OneDrive - 東京理科大学\\This4年前期\\卒業研究1\\卒業研究使用データ\\販売価格・販売量\\サガン鳥栖戦.csv', #伸び #予定販売数量：1550
    'C:\\Users\\saiki\\OneDrive - 東京理科大学\\This4年前期\\卒業研究1\\卒業研究使用データ\\販売価格・販売量\\コンサドーレ札幌戦.csv', #予定販売数量：1250
    'C:\\Users\\saiki\\OneDrive - 東京理科大学\\This4年前期\\卒業研究1\\卒業研究使用データ\\販売価格・販売量\\ヴィッセル神戸戦.csv', #予定販売数量：2200
    'C:\\Users\\saiki\\OneDrive - 東京理科大学\\This4年前期\\卒業研究1\\卒業研究使用データ\\販売価格・販売量\\アルビレックス新潟戦.csv', #ほぼ伸び #予定販売数量：1750
    'C:\\Users\\saiki\\OneDrive - 東京理科大学\\This4年前期\\卒業研究1\\卒業研究使用データ\\販売価格・販売量\\東京ヴェルディ戦.csv', #予定販売数量：1000
    'C:\\Users\\saiki\\OneDrive - 東京理科大学\\This4年前期\\卒業研究1\\卒業研究使用データ\\販売価格・販売量\\アビスパ福岡戦.csv' #予定販売数量：1800
    #'C:\\Users\\saiki\\OneDrive - 東京理科大学\\This4年前期\\卒業研究1\\卒業研究使用データ\\販売価格・販売量\\FC東京戦.csv', #予定販売数量：2270
    #'C:\\Users\\saiki\\OneDrive - 東京理科大学\\This4年前期\\卒業研究1\\卒業研究使用データ\\販売価格・販売量\\鹿島アントラーズ戦.csv',#予定販売数量：2500
    #'C:\\Users\\saiki\\OneDrive - 東京理科大学\\This4年前期\\卒業研究1\\卒業研究使用データ\\販売価格・販売量\\ガンバ大阪戦.csv',#予定販売数量：2500
]
    #test_file = 'C:\\Users\\saiki\\OneDrive - 東京理科大学\\This4年前期\\卒業研究1\\卒業研究使用データ\\販売価格・販売量\\横浜FM戦.csv'

C0=10 # ゼロ除算を避けるための小さな値
C1=10

df_list = []
for file in learn_files:
    try:
        df_temp = pd.read_csv(file)
        df_temp['試合名'] = file.split(os.path.sep)[-1].replace('.csv', '') # os.path.sep を使用
        df_temp['席種'] = df_temp['席種'].str.replace(r'^\d+:\s*', '', regex=True)
        df_temp = df_temp[df_temp['席種'].str.contains("ホームバック自由席", na=False)].copy() #席種の限定
        time_last=df_temp["残り日数"].max() # 残り日数の最大値を取得
        df_temp = df_temp.iloc[6:].copy() 
        df_list.append(df_temp)
    except FileNotFoundError:
        print(f"エラー: ファイルが見つかりません - {file}")
        print("ファイルパスが正しいか、ファイルがその場所に存在するか確認してください。")
        continue

if not df_list:
    print("有効なデータが読み込まれませんでした。ファイルパスを確認してください。")
    exit()

df_raw = pd.concat(df_list, ignore_index=True)
df = df_raw.copy()

df_expected_sales = pd.Series(dtype=int)

# 各試合の予定販売数量を設定
for file in learn_files:
    if '川崎フロンターレ戦' in file:
        df_expected_sales['川崎フロンターレ戦'] = 1880
    elif '京都サンガ戦' in file:
        df_expected_sales['京都サンガ戦'] = 1554
    elif 'ジュビロ磐田戦' in file:
        df_expected_sales['ジュビロ磐田戦'] = 1340
    elif '浦和レッズ戦' in file:
        df_expected_sales['浦和レッズ戦'] = 2350
    elif '名古屋グランパス戦' in file:
        df_expected_sales['名古屋グランパス戦'] = 1800
    elif '柏レイソル戦' in file:
        df_expected_sales['柏レイソル戦'] = 1400
    elif '町田ゼルビア戦' in file:
        df_expected_sales['町田ゼルビア戦'] = 1124
    elif '湘南ベルマーレ戦' in file:
        df_expected_sales['湘南ベルマーレ戦'] = 1200
    elif '横浜FM戦' in file:
        df_expected_sales['横浜FM戦'] = 1849
    elif 'サンフレッチェ広島戦' in file:
        df_expected_sales['サンフレッチェ広島戦'] = 1080
    elif 'サガン鳥栖戦' in file:
        df_expected_sales['サガン鳥栖戦'] = 1550
    elif 'コンサドーレ札幌戦' in file:
        df_expected_sales['コンサドーレ札幌戦'] = 1250
    elif 'ヴィッセル神戸戦' in file:
        df_expected_sales['ヴィッセル神戸戦'] = 2200
    elif 'アルビレックス新潟戦' in file:
        df_expected_sales['アルビレックス新潟戦'] = 1750
    elif '東京ヴェルディ戦' in file:
        df_expected_sales['東京ヴェルディ戦'] = 1000
    elif 'アビスパ福岡戦' in file:
        df_expected_sales['アビスパ福岡戦'] = 1800
#print("販売予定数量:", df_expected_sales)

# --- 2. 目的変数と説明変数の準備 ------------------------------------------------------------------------
target_variable = '販売数量' # 目的変数：販売されたチケットの数量 (カウントデータ)
initial_variables = [
    '販売価格',           # 説明変数：チケットの販売価格
    '残り日数',           # 説明変数：販売期間の日数
    #'相手順位',           # 説明変数：対戦相手のリーグ順位など
    '相手ランク',
    #"順位"  ,             # 説明変数：自チームのリーグ順位
    #"価格変動"            # 説明変数：価格の前日との差
]
# df_model と df_test を必要なカラムで初期化
df_model = df[initial_variables + [target_variable, '試合名']].copy()

df_model['残り日数_squared'] = 1/(df_model['残り日数']+C0)  # ゼロ除算を避けるために小さな値を加える

explanatory_variables = [
    '販売価格',           # 説明変数：チケットの販売価格
    '残り日数',           # 説明変数：販売期間の日数
    '相手ランク',
    #"順位"  ,             # 説明変数：自チームのリーグ順位
    #"残り日数_squared",
#    "締め切り効果と価格変動"
]


#--- データの型変換と欠損値処理-------------------------
df_model[target_variable] = pd.to_numeric(df_model[target_variable], errors='coerce')
df_model[target_variable] = df_model[target_variable].fillna(0).astype(int) # NaNがあれば0として、整数型に変換
# 説明変数も数値型であることを確認し、NaNがあれば削除または補完
for col in explanatory_variables: # explanatory_variables に追加された新しい列も処理されます
    df_model[col] = pd.to_numeric(df_model[col], errors='coerce')
# 欠損値のある行を削除（簡単な方法ですが、データ損失に注意）
df_model.dropna(inplace=True)
# 念のため、販売数量が負の値でないことを確認
df_model = df_model[df_model[target_variable] >= 0].copy()

# 目的変数と説明変数の分離 (スケーリング前の生データ df_X_raw を保持)
df_y = df_model[target_variable]
#explanatory_variables.remove('残り日数') # 残り日数を削除
df_X_raw = df_model[explanatory_variables] 



# --- 説明変数の標準化（スケーリング）---
scaler = StandardScaler()

# 学習データの説明変数をスケーリング
# fit_transform で平均と標準偏差を計算し、データを変換
df_X_scaled = pd.DataFrame(scaler.fit_transform(df_X_raw), columns=df_X_raw.columns, index=df_X_raw.index)
# 定数項 (切片) はスケーリングしないので、別途追加
df_X_scaled = sm.add_constant(df_X_scaled) 


# モデル構築にはスケーリング済みのデータを使用
df_X_s = df_X_scaled


# --- 3. 負の二項回帰モデルの構築と学習 ---------------------------------------------------------------------
#print("--- 負の二項回帰モデルの学習開始 ---")
# statsmodels.discrete.discrete_model.NegativeBinomial を使用して alpha を推定
negbin_model = NegativeBinomial(df_y, df_X_s)
negbin_results = negbin_model.fit()
negbin_model2 = sm.GLM(df_y, df_X_s,family=sm.families.NegativeBinomial(alpha=negbin_results.params['alpha'])) # これにより疑似決定係数を求める
negbin_results2 = negbin_model2.fit() 
#print("--- 負の二項回帰モデルの学習完了 ---")
#print("\n")


# --- 4. モデル結果の表示 -------------------------------------------------------------------------------------
print("--- 負の二項回帰モデルのサマリー（sm.GLMメソッドを用いる） ---")
print(negbin_results.summary())
print(negbin_results2.summary())
print("\n")


    # --- 5. 予測値の計算とデータフレームへの追加 ----------------------------------------
for  i in learn_files:
    test_file=i
    try:
        df_test_raw = pd.read_csv(test_file) # df_test ではなく df_test_raw に変更
        df_test_raw['試合名'] = test_file.split(os.path.sep)[-1].replace('.csv', '') # os.path.sep を使用
        df_test_raw['席種'] = df_test_raw['席種'].str.replace(r'^\d+:\s*', '', regex=True)
        df_test_raw = df_test_raw[df_test_raw['席種'].str.contains("ホームバック自由席", na=False)].copy() #席種の限定
        time_last1=df_test_raw["残り日数"].max() # 残り日数の最大値を取得
        df_test_raw = df_test_raw.iloc[6:].copy() # 一般販売開始以降
    except FileNotFoundError:
        print(f"エラー: ファイルが見つかりません - {test_file}")
        print("ファイルパスが正しいか、ファイルがその場所に存在するか確認してください。")
        exit() # エラーが発生した場合は処理を中断
    # テストデータも同様の処理
    df_test = df_test_raw[initial_variables + [target_variable, '試合名']].copy() # df_test_rawから抽出
    #df_test['残り日数_squared'] = 1/(df_test['残り日数']+C1)  # ゼロ除算を避けるために小さな値を加える0
    df_test[target_variable] = pd.to_numeric(df_test[target_variable], errors='coerce')
    df_test[target_variable] = df_test[target_variable].fillna(0).astype(int)
    for col in explanatory_variables:
        df_test[col] = pd.to_numeric(df_test[col], errors='coerce')
    df_test.dropna(inplace=True)
    df_test = df_test[df_test[target_variable] >= 0].copy()

    df_y_test = df_test[target_variable]
    #explanatory_variables.remove(['残り日数']) # 交互作用項を追加
    df_X_test_raw = df_test[explanatory_variables] # df_X_test も生データを保持
    # テストデータの説明変数をスケーリング (学習データでフィットしたscalerを使用)
    df_X_test_scaled = pd.DataFrame(scaler.transform(df_X_test_raw), columns=df_X_test_raw.columns, index=df_X_test_raw.index)
    df_X_test_scaled = sm.add_constant(df_X_test_scaled, has_constant='add')
    # カラム順序を学習データと一致させる
    df_X_test_scaled = df_X_test_scaled[df_X_scaled.columns] 
    df_X_test_s = df_X_test_scaled
    df_test['predicted_sales'] = negbin_results.predict(df_X_test_s)


    # --- 6. 実際の販売数量と予測値を比較 --------------------------------------------------------------
    testdata_df = df_test.copy() 

    print("試合名：", df_test_raw['試合名'].values[0])
    print("販売予定数量:", df_expected_sales[df_test_raw['試合名'].values[0]])
    print("実際の販売数量合計:", testdata_df['販売数量'].sum())
    print("予定と実際の差:", df_expected_sales[df_test_raw['試合名'].values[0]] - testdata_df['販売数量'].sum())


    if not testdata_df.empty:
        #print(f"--- {test_file.split(os.path.sep)[-1].replace('.csv', '')} の実際の販売数量と予測値 (一部) ---")
        #print(testdata_df[['残り日数', '販売数量', 'predicted_sales']].head())

        plt.rcParams['text.usetex'] = True
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.size'] = 24
        
        plt.figure(figsize=(12, 7))
        #sns.lineplot(x='残り日数', y='predicted_sales', data=testdata_df, color='blue', label='Expected Sales')
        sns.scatterplot(x='残り日数', y='販売数量', data=testdata_df, color='red', s=100, label='Actual Sales', alpha=0.6)

        plt.xlabel('Remaining Days Until Match')
        plt.ylabel('Ticket Sales Quantity')
        #plt.title(f'Actual vs. Predicted Daily Sales Volume ') #{test_file.split(os.path.sep)[-1].replace(".csv", "")}
        #plt.grid(True)
        plt.legend()
        plt.gca().invert_xaxis() # 残り日数を降順に表示するため、X軸を反転
        plt.axhline(y=0, color='black', linestyle='-')
        plt.show()
    else:
        print(f"テストファイル {test_file.split(os.path.sep)[-1].replace('.csv', '')} のデータが見つかりませんでした。")

    # alphaとmuの取得方法をstatsmodels.discrete.discrete_model.NegativeBinomialに合わせて修正
    alpha = negbin_results.params['alpha']
    #print(f"推定された分散パラメータ alpha: {alpha}")

    # predicted_sales は既にmuの予測値なので、それを利用
    mu = df_test['predicted_sales'].mean()
    #print(f"テストデータ予測の平均値 mu: {mu}")
    # mu + mu^2 * alpha の計算
    #print(f"μ + αμ^2 = {mu + mu**2 * alpha}")

    # 以前の bellman_dynamic_pricing_stochastic 関数はユーザーが定義しているので、
    # ここではその関数呼び出し部分は含まれていません。
    # もし、このコードブロックが実行されるたびにベルマン関数も呼び出される場合は、
    # ベルマン関数内部の mu, alpha, scaler の渡し方と利用方法が正しく連動しているか確認してください。


    raw_predict_df_for_period = pd.DataFrame({
                        '販売価格': [3000],
                        '残り日数': [25],
                        '相手ランク':[3],
                        #'順位': [7],
                        #"残り日数_squared": [1/(25+C1)], # 2次項を追加
                        #"締め切り効果と価格変動":[0]
                    })
                    
                    # scalerでスケーリング
    scaled_predict_values = scaler.transform(raw_predict_df_for_period)
    predict_df_for_period = pd.DataFrame(scaled_predict_values, 
                                                        columns=raw_predict_df_for_period.columns, 
                                                        index=raw_predict_df_for_period.index)
                    
                    # 定数項を追加し、カラム順序を学習データと一致させる
    predict_df_for_period = sm.add_constant(predict_df_for_period, has_constant='add')
    predict_df_for_period = predict_df_for_period[df_X_s.columns] # カラム順序を合わせる
                    
                    # ここを修正！ statsmodelsのresultsオブジェクトから直接分布を取得する
    neg_binom_dist = negbin_results.get_distribution(exog=predict_df_for_period)
    # 予測値の販売数量に対する負の二項分布の確率質量関数を計算
    # 0から300までの販売数量に対する確率を計算
    expected_sales = 0  # 期待販売数量の初期化
    for k in range(1001):  # 0から1000までの販売数量に対して確率を計算
        prob_k = neg_binom_dist.pmf(k)  # k枚売れる確率を計算
        #print(f"販売価格: {raw_predict_df_for_period['販売価格'].values[0]}, 残り日数: {raw_predict_df_for_period['残り日数'].values[0]} のとき、{k}枚売れる確率: {prob_k[0]:.4f}")
        # 期待販売数量を計算
        expected_sales = expected_sales+prob_k[0]*k  # 期待値を計算



