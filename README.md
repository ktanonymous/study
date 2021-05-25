# study
**モデル(`.model`)ファイルのみを更新していく**

- ver1.(2021/5/25)
    一先ず一通りの実装を終了  
    <strong>ver1. では、`Movie_Market.model`には不備あり</strong>  
    <strong>`Movie_Market_original.model`の一部をコメントアウトして再現モデルとして利用</strong>  
    + `Movie_Market_test.model`は実装テスト
    + `Movie_Market.model`は論文を再現したモデル
    + `Movie_Market_original.model`は自分自身の実装

- ver2.(2021/5/25)
    再現したモデルの一部を改変  
    <strong>`Movie_Market.model`には不備あり</strong>  
    <strong>`Movie_Market_original.model`の一部をコメントアウトして再現モデルとして利用</strong>  
    <strong>`myfunc.inc`がインクルードできない原因は不明</strong>  
    <strong>変更点は以下の通り</strong>  
    + ジャンル選好度 `W_g` について、「好き」、「嫌い」に加えて「普通 $\in (0.1, \, 0.9)$」を導入
    + 情報を受け取るたびに、満足度及び批評値の受信重みを2割ずつ減衰するようにした。

- ネットワークモデルの適用を構想中
