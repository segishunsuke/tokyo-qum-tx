# -*- coding: utf-8 -*-
import numpy as np

# 下三角行列の逆行列を計算する関数
# Cholesky分解後の行列Lから、L^{-1}を求めるために使用
def low_tri_inverse(L):
    LI = np.identity(L.shape[0])
    for i in range(L.shape[0]):
        for j in range(i):
            LI[i,j] = -np.dot(L[i,j:i], LI[j:i,j])/L[i,i]
        LI[i,i] = 1.0/L[i,i]
    return LI

class Errorterms(object):
    def __init__(self, T, N, M):
        self.T = T  # 時点数
        self.N = N  # 地域数
        self.M = M  # 誤差項の変量数（次元）
        self.ehat = None
        self.kappa = 0.0      # 通時的な不変部分の割合パラメータ
        self.Sigma = None     # 誤差項の共分散行列
        self.Sigmainv = None  # Sigmaの逆行列
        self.log_likelihood = 0.0
        
        # 行列Bの作成： (N-1)個の差分データから、N個の誤差項（合計0制約）へ変換する行列
        self.B = np.zeros((self.N, self.N - 1))
        self.B[:self.N - 1, :] = np.identity(self.N - 1)
        self.B -= 1.0 / self.N * np.ones((self.N, self.N - 1))
    
    def compute_log_likelihood(self, epstilde): # epstildeはT×(N-1)×Mの復元誤差項テンソル
        # 1. 誤差項の復元
        # 観測された(N-1)個の差分データ(epstilde)に行列Bをかけ、N個の元の誤差項(ehat)を復元
        self.ehat = np.zeros((self.T, self.N, self.M))
        for t in range(self.T):
            self.ehat[t,:,:] = self.B @ epstilde[t,:,:]
        
        # 2. 共分散行列の構成要素の計算
        # Sigma_diag: 通常の標本共分散に近い成分（同一時点・同一地域での積和）
        Sigma_diag = np.zeros((self.M, self.M))
        for t in range(self.T):
            for i in range(self.N):
                Sigma_diag += self.ehat[t,i,:].reshape(-1, 1) @ self.ehat[t,i,:].reshape(1, -1)
        Sigma_diag /= self.T * (self.N - 1)
        
        # Sigma_all: 通時的な構造をとらえるための成分（異なる時点間も含めた総当たり積和）
        Sigma_all = np.zeros((self.M, self.M))
        for t in range(self.T):
            for t2 in range(self.T):
                for i in range(self.N):
                    Sigma_all += self.ehat[t,i,:].reshape(-1, 1) @ self.ehat[t2,i,:].reshape(1, -1)
        Sigma_all /= self.T * self.T * (self.N - 1)
        
        kappa_prev = self.kappa
        error_prev = 1.0e100
        
        # 3. kappa（通時相関パラメータ）の最尤推定：ニュートン法による反復計算
        # 目的：尤度関数を最大化する kappa を見つける
        while True:
            # 現在のkappaに基づいて分散共分散行列 Sigma を構築
            # 定義： Sigma(kappa) = Sigma_diag - (1 - kappa) * Sigma_all
            self.Sigma = Sigma_diag - (1.0 - self.kappa) * Sigma_all
            
            # 【数値安定化】計算誤差による非対称性を補正（対称行列化）
            self.Sigma = (self.Sigma + self.Sigma.T) / 2.0
            
            # 逆行列の計算（Cholesky分解を利用して高速・安定に計算）
            SigmaL = np.linalg.cholesky(self.Sigma)
            SigmaLinv = low_tri_inverse(SigmaL)
            self.Sigmainv = np.dot(SigmaLinv.T, SigmaLinv)
            
            # 尤度方程式の1階微分(f)と2階微分(fdash)に必要なスカラー量 s, s2 の計算
            
            # s: データと現在の逆行列との二次形式の総和
            s = 0.0
            for t in range(self.T):
                for t2 in range(self.T):
                    for i in range(self.N):
                        s += self.ehat[t,i,:] @ self.Sigmainv @ self.ehat[t2,i,:]
            
            # s2: 2階微分に必要な項（逆行列の微分の寄与）
            s2 = 0.0
            for t in range(self.T):
                for t2 in range(self.T):
                    for i in range(self.N):
                        tmp = self.Sigmainv @ self.ehat[t2,i,:]
                        s2 += tmp @ Sigma_all @ tmp
            
            # f: log尤度のkappaに関する1階微分
            # f = 0 となる点が解
            f = self.kappa - self.T * (self.N - 1) * self.M / s
            
            # fdash: log尤度のkappaに関する2階微分
            fdash = 1.0  - self.T * (self.N - 1) * self.M / (s * s) * s2
            
            # ニュートン法による更新ステップ： x_new = x_old - f(x)/f'(x)
            new_kappa = kappa_prev - f / fdash
            
            # 【数値安定化】kappaは0より大きく1未満
            self.kappa = min(max(new_kappa, 1.0e-14), 1.0 - 1.0e-12)
            
            # 収束判定（変化率が十分小さくなれば終了）
            error = np.fabs( (self.kappa - kappa_prev) / self.kappa )

            if error <= 1.0e-7 and error >= error_prev * (1.0 - 1.0e-7):
                break
            else:
                kappa_prev = self.kappa
                error_prev = error
        
        # 【修正】ニュートン法の結果を検証し，必要なら κ の1次元最大化で求め直す．
        #   ニュートン法が負の値に飛ぶと下限 1e-8 に張り付いたまま終了することがあり，
        #   その場合の対数尤度は最大値より大幅に小さく計算されていた（η = 0 付近で LL が −3500 程度になる症状）．
        #   κ を固定したときの Σ の最尤推定量は Σ(κ) = D + κP（D = Sigma_diag − Sigma_all，P = Sigma_all）で，
        #   対数尤度は κ の1変数関数 ll(κ) になる（二次形式の項は定数 T(N−1)M に等しいので省略）．
        D = Sigma_diag - Sigma_all
        P = Sigma_all
        D = (D + D.T) / 2.0
        P = (P + P.T) / 2.0

        def ll_of(log_k):
            k = np.exp(log_k)
            sign, ld = np.linalg.slogdet(D + k * P)
            if sign <= 0:
                return -np.inf
            return -0.5 * (self.T * (self.N - 1.0) * ld - (self.N - 1.0) * self.M * log_k
                           + self.T * self.M * np.log(self.N))

        lo, hi = np.log(1.0e-14), np.log(1.0 - 1.0e-12)
        ll_newton = ll_of(np.log(self.kappa))
        grid = np.linspace(lo, hi, 80)
        vals = np.array([ll_of(g) for g in grid])
        k = int(np.argmax(vals))
        a, b = grid[max(k - 1, 0)], grid[min(k + 1, len(grid) - 1)]
        gr = (np.sqrt(5.0) - 1.0) / 2.0                  # 黄金分割法で精密化
        c, d = b - gr * (b - a), a + gr * (b - a)
        fc, fd = ll_of(c), ll_of(d)
        for _ in range(200):
            if fc > fd:
                b, d, fd = d, c, fc
                c = b - gr * (b - a)
                fc = ll_of(c)
            else:
                a, c, fc = c, d, fd
                d = a + gr * (b - a)
                fd = ll_of(d)
            if b - a < 1.0e-12:
                break
        log_k_best = (a + b) / 2.0
        if ll_of(log_k_best) > ll_newton + 1.0e-6 * max(1.0, abs(ll_newton)):
            print(f"    [誤差項の尤度] ニュートン法の κ = {self.kappa:.3e}（LL {ll_newton:.2f}）を "
                  f"κ = {np.exp(log_k_best):.3e}（LL {ll_of(log_k_best):.2f}）に修正", flush=True)
            self.kappa = float(np.exp(log_k_best))
        self.Sigma = D + self.kappa * P
        SigmaL = np.linalg.cholesky(self.Sigma)
        SigmaLinv = low_tri_inverse(SigmaL)
        self.Sigmainv = np.dot(SigmaLinv.T, SigmaLinv)

        # 4. 最終的な対数尤度の計算
        # logdetSigma: log(|Sigma|) の計算
        logdetSigma = 2.0 * np.sum(np.log(SigmaL.diagonal()))
        
        # 対数尤度関数の本体
        # 第1項: 分散行列の行列式由来
        # 第2項: kappa由来の行列式部分 (- (N-1)M * log(kappa))
        # 第3項: 空間差分の行列式部分
        self.log_likelihood = - 0.5 * ( self.T * (self.N - 1.0) * logdetSigma - (self.N - 1.0) * self.M * np.log(self.kappa) + self.T * self.M * np.log(self.N) )
        
        return self.log_likelihood
        
    # 対数尤度のパラメータに関する微分の計算
    def compute_dlog_likelihood_dtheta(self, detildedtheta): # detildedthetaは復元誤差項テンソルのパラメータに関する微分結果
        # 1. 微分の復元: N-1次元の微分を行列BでN次元へ変換
        dehatdtheta = np.zeros((self.T, self.N, self.M))

        for t in range(self.T):
            dehatdtheta[t,:,:] = self.B @ detildedtheta[t,:,:]
        
        # 2. クロス項の計算
        dehatdthetaTehat = dehatdtheta.ravel().reshape(-1, 1) @ self.ehat.ravel().reshape(1, -1)
        
        # 3. トレース計算による微分の導出
        # np.kron(...) 部分は、時空間全体の巨大な分散共分散行列の逆行列に相当
        dlog_likelihood_dtheta = - np.trace( np.kron( np.kron(np.identity(self.T) - 1 / self.T * (1 - self.kappa) * np.ones((self.T,self.T)), np.identity(self.N)), self.Sigmainv ) @ dehatdthetaTehat )
        return dlog_likelihood_dtheta