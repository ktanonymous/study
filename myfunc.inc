// シミュレーションの環境情報を記録する
Sub writeEnvInfo(){
    OpenFileCSV(Universe.dir & "/Environment.csv", 1, 2)
    WriteFileCSV(1, "series", False)
    WriteFileCSV(1, Universe.series, True)
    WriteFileCSV(1, "period", False)
    WriteFileCSV(1, Universe.period, True)
    WriteFileCSV(1, "Net-User", False)
    WriteFileCSV(1, CountAgt(Universe.Internet.user), True)
    WriteFileCSV(1, "Net-Follwer", False)
    WriteFileCSV(1, CountAgt(Universe.Internet.follower), True)
    WriteFileCSV(1, "Consumer", False)
    WriteFileCSV(1, CountAgt(Universe.Social.consumer), True)
    WriteFileCSV(1, "Movie", False)
    WriteFileCSV(1, CountAgt(Universe.movie), True)
    CloseFileCSV(1)
}

// 消費者カテゴリー取得する
Function getCategoryThreshold(rand As Double) As Double {
    If rand < 0.16 Then  // 16% が初期採用消費者
        return Rnd() * 50 + 940
    Elseif rand < 0.50 Then  // 34% が前期採用消費者
        return Rnd() * 50 + 990
    Elseif rand < 0.84 Then  // 34% が後期採用消費者
        return Rnd() * 100 + 1500
    Else  // 16% が採用遅滞消費者
        return Rnd() * 100 + 2100
    End if
}

// 消費者の作品ごとのパラメータを取得する
Sub getParams(self As Agt, movies As Agtset, num_genre As Integer) {
    Dim movie As Agt
    Dim norm As Double, mu As Double, sigma As Double

    For Each movie in movies
        // 鑑賞効用値の決定(r_U = 0.66, n_U = 2)
        self.Utility(movie.IDNo) = 0.66 * (movie.review^2) * self.W_g(movie.genre-1)
        // 初期化
        self.Info(movie.IDNo) = 0  // 作品情報量の初期化
        self.wGreed(movie.IDNo) = 1  // 満足度受信重みの初期化
        self.wCriticism(movie.IDNo) = 1  // 批評値受信重みの初期化
        self.viewStep(0, movie.IDNo) = 0  // 鑑賞回数の初期化
        self.viewStep(1, movie.IDNo) = 0  // 観賞後の経過時間の初期化
    Next movie
    // 消費者(follower, consumer)の情報電探つ範囲は周囲1近傍(3*3 セルの正方形内)
    self.transmission_range = 1
    // ジャンルごとの選好度を決定する
    getGenreWeight(self, num_genre)
    // 商品の採用時期による消費者集団の分類に応じて閾値を決定(最終的な鑑賞閾値決定に使用)
    self.categoryThreshold = getCategoryThreshold(Rnd())
    // 消費者の被影響性を決定
    self.effectiveness = Rnd() + 0.7
    // 映画鑑賞回数の初期化
    // self.viewNum = 0
    // 映画鑑賞回数の初期化、対数正規分布(https://sci-fx.net/math-log-norm-dist/)を利用するパターン
    // 対数正規分布のピーク(最頻値)は exp(mu - sigma ^ 2) となるらしい
    // 対数正規分布の累積分布関数の値が 1/2 となるときが中央値になる定義で、その値は exp(mu) になるらしい
    mu = Log(Universe.medianDist)
    sigma = Sqr(mu - Log(Universe.modeDist))
    norm = Exp(NormInv(Rnd(), mu, sigma))  // NormInv は「逆関数法」というものを利用しているらしい
    norm = Int(norm)
    self.viewNum = norm
    self.initialViewNum = norm
    If Universe.maxInitialViewNum < norm Then
        Universe.maxInitialViewNum = norm
    End if
    // PrintLn(self.viewNum)
    // 未鑑賞かつ情報量最大の作品IDを初期化
    self.max_info_movie = 0
}

// ジャンルごとの選好度を取得する
Sub getGenreWeight(self As Agt, num_genre As Integer){
    Dim i As Integer
    Dim rand As Double

    For i=0 To num_genre-1
        rand = Rnd()
        If rand < Universe.genreRatio(i) Then  // ジャンル選好割合に応じて重み付けを行う
            self.W_g(i) = 1
        Elseif rand > 0.9 Then  // 10% がジャンルを嫌う
            self.W_g(i) = 0
        Else  // ジャンルの選好度「普通」の場合は 区間(0.1, 0.9) から選ばれる
            self.W_g(i) = Rnd() * 0.8 + 0.1
        End if
    Next i
}

// ネットユーザーから受け取る作品の情報量を取得する
Function getInformationFromVirtual(self As Agt, movie As Agt, info As Double, rand As Double, users As Agtset) As Double {
    Dim one As Agt

    For Each one in users
        // 満足度を受け取るかどうかの判定
        If rand < (0.013 * movie.review^2) ^ one.viewStep(1, movie.IDNo) Then
            info = info + movie.greed * self.wGreed(movie.IDNo)
            // 次に受け取る情報の重みは減衰する
            self.wGreed(movie.IDNo) = self.wGreed(movie.IDNo) * 0.5
        End if
        // 批評値を受け取るかどうかの判定
        If rand < 0.374 Then
            info = info + movie.criticism * self.wCriticism(movie.IDNo)
            // 次に受け取る情報の重みは減衰する
            self.wCriticism(movie.IDNo) = self.wGreed(movie.IDNo) * 0.5
        End if
    Next one

    return info
}

// ネットフォロワー及び一般消費者から受け取る作品の情報量を取得する
Function getInformationFromReal(self As Agt, movie As Agt, info As Double, rand As Double, neighbors As Agtset) As Double {
    Dim one As Agt

    // 周囲の各エージェントから受信する情報量を蓄積させる
    For Each one in neighbors
        If one.viewStep(0, movie.IDNo) == 0 Then
            // 周囲の対象エージェントが作品を未鑑賞の場合、そのエージェントの保有最大情報量を受信する
            If movie.IDNo == one.max_info_movie Then
                // 鑑賞済みのエージェントよりも影響が小さい(らしい)ので実際の被影響性以上に減衰させる
                info = info + one.Info(movie.IDNo) * 0.5
            End if
        Else
            // 周囲の対象エージェントが作品を鑑賞済みの場合、満足度と批評値を受信する
            // 満足度を受け取るかどうかの判定
            If rand < (0.013 * movie.review^2) ^ one.viewStep(1, movie.IDNo) Then
                info = info + movie.greed * self.wGreed(movie.IDNo)
                // 次に受け取る情報の重みは減衰する
                self.wGreed(movie.IDNo) = self.wGreed(movie.IDNo) * 0.5
            End if
            // 批評値を受け取るかどうかの判定
            If rand < 0.374 Then
                info = info + movie.criticism * self.wCriticism(movie.IDNo)
                // 次に受け取る情報の重みは減衰する
                self.wCriticism(movie.IDNo) = self.wCriticism(movie.IDNo) * 0.5
            End if
        End if
    Next one

    return info
}

// agt_step を関数化したもの
Sub functionAgtStep(self As Agt, movies As Agtset, neighbors As Agtset, followers As Agtset, date As Integer, is_follower As Boolean){
    Dim movie As Agt, one As Agt
    Dim info As Double, rand As Double

    // 鑑賞閾値は映画の鑑賞回数に応じて減衰する(sigma = 0.99)
    self.infoThreshold = self.categoryThreshold  / (0.99 ^ self.viewNum)

    // 鑑賞意思決定
    For Each movie in movies

        If (date < movie.startDate - 30) Or (self.viewStep(0, movie.IDNo) >= 1) Then
            // 公開の30日以上前は何も行わない
            // 鑑賞済みの作品についても情報受信は行わない
        Else
            info = 0  // 情報受信量の初期化
            rand = Rnd()  // その日の乱数を確定する
            // 宣伝に接触するかどうかを判定
            If rand < movie.promProb Then
                info = info + movie.promotion  // 宣伝情報量を追加
            End if
            // 周囲の各エージェントから、知覚している作品の情報を受信する
            If date > movie.startDate Then  // 実評価値は公開以降のみ受け取り可能？
                If self.Info(movie.IDNo) != 0 Then  // 知覚していない作品の情報は受信できない
                    If is_follower Then
                        // ネットフォロワーの場合、ネットユーザーから情報を受信するかどうかの判定
                        If CountAgtSet(neighbors) > 0 Then
                            info = getInformationFromVirtual(self, movie, info, rand, neighbors)
                        End if
                    Else
                        // フォロワーでない、一般消費者の場合、一般消費者から情報を受信するかどうかの判定
                        If CountAgtSet(neighbors) > 0 Then
                            // consumers から満足度と批評値を受信する
                            info = getInformationFromReal(self, movie, info, rand, neighbors)
                        End if
                    End if
                    // ネットフォロワーから情報を受信するかどうかの判定
                    If CountAgtSet(followers) > 0 Then
                        // followers から満足度と批評値を受信する
                        info = getInformationFromReal(self, movie, info, rand, followers)
                    End if
                End if
            End if

            // 情報量を蓄積
            self.Info(movie.IDNo) = self.Info(movie.IDNo) + info * self.effectiveness

            // 閾値比較による鑑賞意思決定
            If (date > movie.startDate) And  // 作品が公開済み
               (
                   self.Info(movie.IDNo) > self.infoThreshold Or  // 情報量が閾値を超える
                   self.Utility(movie.IDNo) > rand * 35 + 30  // 作品効用値が閾値を超える(h = 35, l = 30)
               ) Then
                // 映画を鑑賞する場合の操作
                self.viewStep(0, movie.IDNo) = self.viewStep(0, movie.IDNo) + 1  // 作品(movie)の鑑賞回数を増やす(フラグを立てる)
                self.viewNum = self.viewNum + 1  // 映画の鑑賞回数を増やす
                movie.numAudience = movie.numAudience + 1  // 作品の動員数を増やす
            End if

        End if
    Next movie

    //// ランダムに移動
    //ForwardDirectionCell(Int(Rnd() * 8), 1)
}