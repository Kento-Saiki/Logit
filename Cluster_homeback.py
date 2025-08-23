import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tslearn.metrics import cdist_dtw # DTW距離行列を計算するためにインポート
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster # デンドログラム、クラスタ分割のためにインポート
import matplotlib.font_manager as fm # フォントマネージャーをインポート
import os # osモジュールをインポート

# 日本語フォントの設定例 (Windowsの場合)
try:
    plt.rcParams['font.family'] = "Yu Gothic" # 游ゴシック (Windowsの場合)
    plt.rcParams['axes.unicode_minus'] = False # 負の符号を正しく表示するために設定
except Exception:
    # フォントが見つからない場合や他のOSの場合のフォールバック
    print("指定された日本語フォントが見つからないか、他のOS環境です。デフォルトフォントで表示を試みます。")

# --- データ読み込みと前処理 ---
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
        df_temp_last5 = df_temp.iloc[-5:].copy()  # データの最後から5番目まで
        df_temp = df_temp_last5
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

df['時系列ID'] = df['試合名'] + '_' + df['席種']
# 各時系列IDごとに最初の販売価格で正規化
df['販売価格'] = df.groupby('時系列ID')['販売価格'].transform(lambda x: x / x.iloc[0])
# 各時系列IDごとに前日の販売数量で正規化
#df['販売数量'] = df.groupby('時系列ID')['販売数量'].transform(lambda x: x / x.iloc[0])

# `残り日数` は試合までの日数で、数字が大きいほど試合から遠い
df_sorted = df.sort_values(by=['時系列ID', '残り日数'], ascending=[True, False])

# ピボットテーブルを作成し、各時系列IDごとの販売数量の推移を取得
pivot_df = df_sorted.pivot_table(index='時系列ID', columns='残り日数', values='販売数量')
pivot_df = pivot_df.fillna(0)
pivot_df = pivot_df.reindex(columns=sorted(pivot_df.columns, reverse=True)) # 試合日に近い方が右になるようにソート

data_for_clustering = pivot_df.values

print("クラスタリング用のNumPy配列の形状:")
print(data_for_clustering.shape)
print(data_for_clustering)

# --- クラスタリング処理 ---

# 1. 時系列データを (n_ts, sz, d) の形状に変換 (d=1, 単一の時系列特徴量)
X_reshaped = data_for_clustering[:, :, np.newaxis]
print("時系列データの形状 (n_ts, sz, d):")
print(X_reshaped.shape)
print(X_reshaped)

# 2. 全ての時系列ペア間のDTW距離行列を計算
# cdist_dtwはNxNの正方行列を返す
distance_matrix = cdist_dtw(X_reshaped)

# 3. linkage関数に渡すために、距離行列を圧縮形式 (condensed form) に変換
# scipy.spatial.distance.squareform は、正方距離行列の対角要素を除く上三角部分を1D配列に変換します。
from scipy.spatial.distance import squareform
condensed_distance_matrix = squareform(distance_matrix) # この形式がlinkage関数の入力に必要

# 4. Ward法による連結 (Linkage) を実行
# 'ward' はWard法を指定します
Z = linkage(condensed_distance_matrix, method='ward')

# 5. デンドログラムを描画
plt.figure(figsize=(20, 10)) # グラフのサイズを調整
plt.title('時系列クラスタリング デンドログラム (Ward法 - DTW距離)', fontsize=20)
plt.xlabel('時系列ID (試合名_席種)', fontsize=16)
plt.ylabel('距離', fontsize=16)

dendrogram(
    Z,
    leaf_rotation=90.,  # x軸のラベルを90度回転
    leaf_font_size=8.,  # x軸のラベルのフォントサイズ
    labels=pivot_df.index # 各リーフのラベル
)
plt.tight_layout() # レイアウトの調整
plt.show()

# --- クラスタへの分割と結果の可視化 ---

# デンドログラムからクラスターを抽出
# 例えば、n_clusters=3で分割する場合
n_clusters = 4 # クラスタ数を指定
labels = fcluster(Z, n_clusters, criterion='maxclust') # 'maxclust' は指定したクラスター数で分割

print(f"\n--- 各クラスターに属する時系列データ (クラスタ数: {n_clusters}) ---")
unique_labels = np.unique(labels)
for cluster_label in unique_labels:
    print(f"\nCluster {cluster_label}:")
    indices_in_cluster = np.where(labels == cluster_label)[0]
    time_series_ids_in_cluster = pivot_df.index[indices_in_cluster]
    for ts_id in time_series_ids_in_cluster:
        print(f"   - {ts_id}")

# --- クラスタリング結果の可視化 ---
plt.rcParams["font.size"] = 18
for i, label in enumerate(unique_labels):
    plt.figure(figsize=(15, 10))
    cluster_series_indices = np.where(labels == label)[0]
    for idx in cluster_series_indices:
        plt.plot(pivot_df.columns, data_for_clustering[idx], color=plt.cm.jet(float(label) / len(unique_labels)), alpha=0.7)
    
    plt.plot([], [], color=plt.cm.jet(float(label) / len(unique_labels)), label=f'Cluster {label} (Count: {len(cluster_series_indices)})')

    plt.title(f'試合と席種ごとのチケット販売数量の時系列クラスタリング結果 (Ward法, {n_clusters}クラスタ)')
    plt.xlabel('試合までの残り日数')
    plt.ylabel('チケット販売数量')
    plt.gca().invert_xaxis() # 残り日数を降順に表示するため、X軸を反転
    plt.legend()
    plt.grid(True)
    plt.show()

plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 25
# --- 各クラスタのセントロイド（代表的な時系列パターン）を可視化 ---
plt.figure(figsize=(12, 7))
for i, label in enumerate(unique_labels):
    cluster_series_indices = np.where(labels == label)[0]
    series_in_cluster = data_for_clustering[cluster_series_indices]
    centroid = np.mean(series_in_cluster, axis=0)
    plt.plot(pivot_df.columns, centroid, label=f'Cluster {label}')
plt.xlabel('Remaining Days Until Match')
plt.ylabel('Ticket Sales Quantity')
plt.gca().invert_xaxis() # 残り日数を降順に表示するため、X軸を反転
plt.legend()
plt.show()
