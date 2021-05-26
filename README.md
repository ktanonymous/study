# study
**モデル(`.model`)ファイルのみを更新していく**

## ver1.(2021/5/25)
一先ず一通りの実装を終了  
<strong>ver1. では、`Movie_Market.model`には不備あり</strong>  
<strong>`Movie_Market_original.model`の一部をコメントアウトして再現モデルとして利用</strong>  
+ `Movie_Market_test.model`は実装テスト
+ `Movie_Market.model`は論文を再現したモデル
+ `Movie_Market_original.model`は自分自身の実装

## ver2.(2021/5/25)
再現したモデルの一部を改変  
<strong>`Movie_Market.model`には不備あり</strong>  
<strong>`Movie_Market_original.model`の一部をコメントアウトして再現モデルとして利用</strong>  
<strong>`myfunc.inc`がインクルードできない原因は不明</strong>  
### 変更点
+ ジャンル選好度 `W_g` について、「好き」、「嫌い」に加えて「普通 (0.1以上0.9以下)」を導入
+ 情報を受け取るたびに、満足度及び批評値の受信重みを2割ずつ減衰するようにした。

## ver2.1(2021/5/26)
ソースコードを整理
<strong>`Movie_Market.model`には不備あり</strong>  
<strong>`Movie_Market_original.model`の一部をコメントアウトして再現モデルとして利用</strong>  
<strong>`myfunc.inc`がインクルードできない原因は不明</strong>  
### 変更点
+ 各種操作を関数化（インクルードファイル使いてぇ...）

## ネットワークモデルの適用を構想中
