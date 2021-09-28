# study

**使用ツールは `artisoc`**  
**基本的にモデルの更新の記録用とする**

## ver1.(2021/5/25)

一先ず一通りの実装を終了  
<strong>ver1. では、`Movie_Market.model`には不備あり</strong>  
<strong>`Movie_Market_original.model`の一部をコメントアウトして再現モデルとして利用</strong>

- `Movie_Market_test.model`は実装テスト
- `Movie_Market.model`は論文を再現したモデル
- `Movie_Market_original.model`は自分自身の実装

## ver2.(2021/5/25)

再現したモデルの一部を改変  
<strong>`Movie_Market.model`には不備あり</strong>  
<strong>`Movie_Market_original.model`の一部をコメントアウトして再現モデルとして利用</strong>  
<strong>`myfunc.inc`がインクルードできない原因は不明</strong>

### 変更点

- ジャンル選好度 `W_g` について、「好き」、「嫌い」に加えて「普通 (0.1 以上 0.9 以下)」を導入
- 情報を受け取るたびに、満足度及び批評値の受信重みを 2 割ずつ減衰するようにした。

## ver2.1(2021/5/26)

ソースコードを整理  
<strong>`Movie_Market.model`には不備あり</strong>  
<strong>`Movie_Market_original.model`の一部をコメントアウトして再現モデルとして利用</strong>  
<strong>`myfunc.inc`がインクルードできない原因は不明</strong>

### 変更点

- 各種操作を関数化（インクルードファイル使いたい...）

## ver2.2(2021/6/19)

<strong>各種関数を外部ファイル(`.inc`)に記述</strong>  
(遂にインクルードファイルが使える！！！)

## ネットワークモデルの適用を構想中

## 参考文献

[1]. [山影進．"人工社会構築指南 artisoc によるマルチエージェント・シミュレーション入門"，書籍工房早山，2007.](https://www.kinokuniya.co.jp/f/dsg-01-9784886115034)  
[2]. [上村亮介，増田浩通，新井健．"消費者購買行動のマルチエージェントモデル 映画市場を事例として"，日本経営工学会論文誌，vol.57，No.5，pp.451-469，2006.](https://www.jstage.jst.go.jp/article/jima/57/5/57_KJ00005984238/_article/-char/ja/)  
