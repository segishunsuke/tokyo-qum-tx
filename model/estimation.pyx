# cython: boundscheck=False
# cython: wraparound=False

from libc.math cimport fmax, fabs, sqrt, log, pow, exp
from cython.parallel cimport prange
import numpy as np
import pandas as pd
import csv
import os
import time
import myoptimize
import liberrortermsre
import sys

# ディレクトリの設定
# 入出力フォルダ（run_cases.py から環境変数で指定される．単独で実行する場合は input/ と output/）
INPUT_DIR = os.environ.get('QUM_INPUT_DIR', 'input')
OUTPUT_DIR = os.environ.get('QUM_OUTPUT_DIR', 'output')

# 集積の外部性の定式化（環境変数 QUM_ACCESSIBILITY で切り替え）
#   mass（既定）: Ω_i = Σ_j L_j exp(-δ τ_ij)   （本文の基準ケース）
#   own_density : Ω_i = L_i / G_i              （4. (10) d) の感度分析）
ACCESSIBILITY = os.environ.get('QUM_ACCESSIBILITY', 'mass')
if ACCESSIBILITY not in ('mass', 'own_density'):
    raise ValueError("QUM_ACCESSIBILITY は mass または own_density を指定してください。")
print('集積の外部性の定式化:', ACCESSIBILITY)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 交通費用の関数形（環境変数 QUM_TAU_TRANSFORM で切り替え．論文 4. (10) d) の感度分析）
#   ''（既定）: 通勤不効用 exp(−κτ)，アクセシビリティ exp(−δτ)
#   'log:<τ0>' : τ を max(0, ln(τ/τ0)) に置き換える．これにより
#                通勤不効用は min(1, (τ/τ0)^(−κ))，アクセシビリティの減衰は min(1, (τ/τ0)^(−δ)) となる（べき乗型）
TAU_TRANSFORM = os.environ.get('QUM_TAU_TRANSFORM', '').strip()
if TAU_TRANSFORM:
    if not TAU_TRANSFORM.startswith('log:'):
        raise ValueError("QUM_TAU_TRANSFORM は 'log:<τ0>' の形式で指定してください。")
    TAU0 = float(TAU_TRANSFORM.split(':', 1)[1])
    print(f'交通費用の関数形: べき乗型（τ0 = {TAU0}）')

# ゾーン内の費用（環境変数 QUM_TAU_II_FILE．論文 4. (8)）
#   指定した場合：CSV（列 id, tau_ii）の値を τ_ii とする（全時点・全シナリオ共通）
#   指定しない場合：τ_ii = cost_per_root_area × sqrt(G_i)
TAU_II_FILE = os.environ.get('QUM_TAU_II_FILE', '').strip()
TAU_II_VALUES = None
if TAU_II_FILE:
    _d = pd.read_csv(TAU_II_FILE, encoding='utf-8-sig').sort_values('id')
    TAU_II_VALUES = _d['tau_ii'].to_numpy(dtype=float)
    print(f'ゾーン内の費用: {TAU_II_FILE}（中央値 {np.median(TAU_II_VALUES):.1f}）')

def apply_tau_ii_file(M):
    if TAU_II_VALUES is not None:
        A = np.asarray(M)
        if len(TAU_II_VALUES) != A.shape[0]:
            raise ValueError('QUM_TAU_II_FILE のゾーン数が一致しません。')
        A[np.arange(A.shape[0]), np.arange(A.shape[0])] = TAU_II_VALUES

def transform_tau_matrix(M):
    if TAU_TRANSFORM:
        A = np.asarray(M)
        A[:, :] = np.maximum(0.0, np.log(A / TAU0))

# 集積の外部性の減衰の関数形（環境変数 QUM_DECAY_TRANSFORM．通勤不効用とは別に指定できる）
#   未指定：QUM_TAU_TRANSFORM と同じ（通勤不効用と同じ関数形）
#   ''（空文字列を明示するときは 'exp'）：exp(−δτ)
#   'log:<τ0>'：τ を max(0, ln(τ/τ0)) に置き換える → min(1, (τ/τ0)^(−δ))（τ0 = 1 なら切り捨てのない τ^(−δ)）
DECAY_TRANSFORM = os.environ.get('QUM_DECAY_TRANSFORM', None)
DECAY_TRANSFORM = TAU_TRANSFORM if DECAY_TRANSFORM is None else DECAY_TRANSFORM.strip()
if DECAY_TRANSFORM in ('exp', 'none'):
    DECAY_TRANSFORM = ''
if DECAY_TRANSFORM:
    if not DECAY_TRANSFORM.startswith('log:'):
        raise ValueError("QUM_DECAY_TRANSFORM は 'log:<τ0>' または 'exp' の形式で指定してください。")
    DECAY_TAU0 = float(DECAY_TRANSFORM.split(':', 1)[1])
print('通勤不効用の関数形:', f'べき乗型（τ0 = {TAU0}）' if TAU_TRANSFORM else 'exp 型',
      '／集積の減衰の関数形:', f'べき乗型（τ0 = {DECAY_TAU0}）' if DECAY_TRANSFORM else 'exp 型')

def set_decay_tau(D, M):
    """集積の減衰に使う費用の行列 D を，ゾーン内の費用を設定済み・変換前の τ の行列 M から作る"""
    A = np.asarray(D)
    A[:, :] = np.asarray(M)
    if DECAY_TRANSFORM:
        A[:, :] = np.maximum(0.0, np.log(A / DECAY_TAU0))

cdef double tol_model = 1.0e-7
cdef double tol_inversion = 1.0e-14

# Excelファイルから各シートのデータを読み込む
excel_path = os.path.join(INPUT_DIR, 'input_data.xlsx')
df_parameters = pd.read_excel(excel_path, sheet_name='parameters', header=None).fillna('')
data_parameters = df_parameters.values.tolist()

df_area = pd.read_excel(excel_path, sheet_name='area', header=None).fillna('')
data_area = df_area.values.tolist()

df_lam_R = pd.read_excel(excel_path, sheet_name='lam_R', header=None).fillna('')
data_lam_R = df_lam_R.values.tolist()

df_lam_M = pd.read_excel(excel_path, sheet_name='lam_M', header=None).fillna('')
data_lam_M = df_lam_M.values.tolist()

df_p_G = pd.read_excel(excel_path, sheet_name='p_G', header=None).fillna('')
data_p_G = df_p_G.values.tolist()

# parameterの読み込み
cdef int it
cdef int t, i, j, r, s
cdef double summ

it = 1

cdef int S
S = int(data_parameters[it][1])
it += 1

cdef int T
T = int(data_parameters[it][1])
it += 1

cdef double cost_per_root_area
cost_per_root_area = float(data_parameters[it][1])
it += 1

cdef double alpha
alpha = float(data_parameters[it][1])
it += 1

cdef double beta
beta = float(data_parameters[it][1])
it += 1

cdef double mu
mu = float(data_parameters[it][1])
it += 1

cdef double kappa_theta
kappa_theta = float(data_parameters[it][1])
it += 1

def _fixed_flag(row):
    # C列の固定フラグ（1 = 固定，0 = 推定）．空欄は固定とみなす
    try:
        return int(float(row[2])) != 0
    except (TypeError, ValueError, IndexError):
        return True

cdef double delta_A
delta_A = float(data_parameters[it][1])
delta_A_fixed = _fixed_flag(data_parameters[it])
it += 1

cdef double delta_B
delta_B = float(data_parameters[it][1])
delta_B_fixed = _fixed_flag(data_parameters[it])
it += 1

# 推定対象のパラメータ：theta, eta_A, eta_B, delta_A, delta_B（この順）
PARAM_NAMES = ['theta', 'eta_A', 'eta_B', 'delta_A', 'delta_B']
parameters_fixed = np.zeros(5, dtype=bool) 
dim_parameters = 0

parameters = np.zeros(5) 
it2 = 0
it3 = 0

cdef double theta
theta = float(data_parameters[it][1])
if int(data_parameters[it][2]) != 0:
    parameters_fixed[it3] = True
else:
    parameters_fixed[it3] = False
    parameters[it2] = theta
    it2 += 1
it3 += 1
it += 1

# kappa変数は計算用の変数として定義
cdef double kappa
kappa = kappa_theta / theta

cdef double eta_A
eta_A = float(data_parameters[it][1])
if int(data_parameters[it][2]) != 0:
    parameters_fixed[it3] = True
else:
    parameters_fixed[it3] = False
    parameters[it2] = eta_A
    it2 += 1
it3 += 1
it += 1

cdef double eta_B
eta_B = float(data_parameters[it][1])
if int(data_parameters[it][2]) != 0:
    parameters_fixed[it3] = True
else:
    parameters_fixed[it3] = False
    parameters[it2] = eta_B
    it2 += 1
it3 += 1
it += 1

# delta_A, delta_B（減衰パラメータ．【追加】最尤推定の対象にできる）
parameters_fixed[3] = delta_A_fixed
if not delta_A_fixed:
    parameters[it2] = delta_A
    it2 += 1
parameters_fixed[4] = delta_B_fixed
if not delta_B_fixed:
    parameters[it2] = delta_B
    it2 += 1
FREE_NAMES = [PARAM_NAMES[k] for k in range(5) if not parameters_fixed[k]]
print('推定するパラメータ:', FREE_NAMES)

# 実際のパラメータ数に合わせて配列をリサイズ
dim_parameters = it2
parameters = parameters[:dim_parameters]

cdef double Abar
Abar = float(data_parameters[it][1])
it += 1

cdef double Hbar
Hbar = float(data_parameters[it][1])
it += 1

cdef double Rbar
Rbar = float(data_parameters[it][1])
it += 1

cdef double[::1] Lbar
Lbar = np.zeros(T)
for t in range(T):
    Lbar[t] = float(data_parameters[it][1])
    it += 1

cdef double[::1] G
G = np.zeros(S)
for i in range(S):
    G[i] = float(data_area[1+i][1])

cdef double[:,::1] exp_epsilon_A
exp_epsilon_A = np.ones((T,S))

cdef double[:,::1] exp_epsilon_B
exp_epsilon_B = np.ones((T,S))

cdef double[:,::1] exp_epsilon_G
exp_epsilon_G = np.ones((T,S))

cdef double[:,::1] epsilon_A
epsilon_A = np.zeros((T,S))

try:
    with open (os.path.join(INPUT_DIR, 'epsilon_A.csv'), 'r', encoding='utf-8-sig') as fin:
        cin = csv.reader(fin)
        data_eps_A = [row for row in cin]
    for i in range(S):
        for t in range(T):
            epsilon_A[t,i] = float(data_eps_A[1+i][1+t])
    for t in range(T):
        for i in prange(S, nogil=True):
            exp_epsilon_A[t,i] = exp(epsilon_A[t,i])
except:
    for i in range(S):
        for t in range(T):
            epsilon_A[t,i] = 0.0
    for t in range(T):
        for i in prange(S, nogil=True):
            exp_epsilon_A[t,i] = 1.0

cdef double[:,::1] epsilon_B
epsilon_B = np.zeros((T,S))

try:
    with open (os.path.join(INPUT_DIR, 'epsilon_B.csv'), 'r', encoding='utf-8-sig') as fin:
        cin = csv.reader(fin)
        data_eps_B = [row for row in cin]
    for i in range(S):
        for t in range(T):
            epsilon_B[t,i] = float(data_eps_B[1+i][1+t])
    for t in range(T):
        for i in prange(S, nogil=True):
            exp_epsilon_B[t,i] = exp(epsilon_B[t,i])
except:
    for i in range(S):
        for t in range(T):
            epsilon_B[t,i] = 0.0
    for t in range(T):
        for i in prange(S, nogil=True):
            exp_epsilon_B[t,i] = 1.0

cdef double[:,::1] epsilon_G
epsilon_G = np.zeros((T,S))

try:
    with open (os.path.join(INPUT_DIR, 'epsilon_G.csv'), 'r', encoding='utf-8-sig') as fin:
        cin = csv.reader(fin)
        data_eps_G = [row for row in cin]
    for i in range(S):
        for t in range(T):
            epsilon_G[t,i] = float(data_eps_G[1+i][1+t])
    for t in range(T):
        for i in prange(S, nogil=True):
            exp_epsilon_G[t,i] = exp(epsilon_G[t,i])
except:
    for i in range(S):
        for t in range(T):
            epsilon_G[t,i] = 0.0
    for t in range(T):
        for i in prange(S, nogil=True):
            exp_epsilon_G[t,i] = 1.0

cdef double[:,:,::1] tau
tau = np.zeros((T,S,S))
cdef double[:,:,::1] tau_decay
tau_decay = np.zeros((T,S,S))
for t in range(T):
    with open (os.path.join(INPUT_DIR, 'tau['+str(t)+'].csv'), 'r', encoding='utf-8-sig') as fin:
        cin = csv.reader(fin)
        data_tau = [row for row in cin]
    for it in range(1, len(data_tau)):
        i = int(data_tau[it][0])
        j = int(data_tau[it][1])
        tau[t,i,j] = float(data_tau[it][2])
    for i in range(S):
        tau[t,i,i] = cost_per_root_area * sqrt(G[i])
    apply_tau_ii_file(tau[t])
    set_decay_tau(tau_decay[t], tau[t])
    transform_tau_matrix(tau[t])
    
cdef double[:,::1] v
v = np.zeros((S,S))

cdef double[:,::1] lam
lam = np.ones((S,S))

cdef double[:,::1] lam_R
lam_R = np.zeros((T,S))

cdef double[::1] dlam_R
dlam_R = np.zeros(S)

for i in range(S):
    for t in range(T):
        lam_R[t,i] = float(data_lam_R[1+i][1+t])

for t in range(T):
    summ = 0.0
    for i in range(S):
        summ += lam_R[t,i]
    for i in range(S):
        lam_R[t,i] /= summ

cdef double[:,::1] lam_M
lam_M = np.zeros((T,S))

cdef double[::1] dlam_M
dlam_M = np.zeros(S)

for i in range(S):
    for t in range(T):
        lam_M[t,i] = float(data_lam_M[1+i][1+t])

for t in range(T):
    summ = 0.0
    for i in range(S):
        summ += lam_M[t,i]
    for i in range(S):
        lam_M[t,i] /= summ

cdef double[:,::1] exp_theta_v
exp_theta_v = np.zeros((S,S))

cdef double[::1] sum_exp_theta_v_M
sum_exp_theta_v_M = np.zeros(S)

cdef double[::1] sum_exp_theta_v_R
sum_exp_theta_v_R = np.zeros(S)

cdef double max_v[1]

cdef double[:,:,::1] exp_delta_A_tau
exp_delta_A_tau = np.zeros((T,S,S))

cdef double[:,:,::1] exp_delta_B_tau
exp_delta_B_tau = np.zeros((T,S,S))

cdef double[::1] Omega_A
Omega_A = np.zeros(S)

cdef double[::1] Omega_B
Omega_B = np.zeros(S)

cdef double[::1] A
A = np.zeros(S)

cdef double[::1] ln_B
ln_B = np.zeros(S)

cdef double[::1] G_effective
G_effective = np.zeros(S)

cdef double[:,::1] p_G
p_G = np.ones((T,S))

cdef double[:,::1] p_G_observed
p_G_observed = np.zeros((T,S))

for i in range(S):
    for t in range(T):
        p_G_observed[t,i] = float(data_p_G[1+i][1+t])

cdef double[::1] p_H
p_H = np.zeros(S)

cdef double[::1] w
w = np.zeros(S)

cdef double[::1] H
H = np.zeros(S)

cdef double[::1] H_M
H_M = np.zeros(S)

cdef double[:,::1] h
h = np.zeros((S,S))

cdef double[::1] G_demand
G_demand = np.zeros(S)

cdef double[::1] dp_G
dp_G = np.zeros(S)

cdef double[::1] dp_H
dp_H = np.zeros(S)

cdef double[::1] dw
dw = np.zeros(S)

cdef double[::1] dH
dH = np.zeros(S)

cdef double[::1] dH_M
dH_M = np.zeros(S)

cdef double[:,::1] dh
dh = np.zeros((S,S))

cdef double[::1] dG_demand
dG_demand = np.zeros(S)

cdef double[::1] dOmega_A
dOmega_A = np.zeros(S)

cdef double[::1] dOmega_B
dOmega_B = np.zeros(S)

cdef double[::1] dA
dA = np.zeros(S)

cdef double[::1] dexp_epsilon_A
dexp_epsilon_A = np.zeros(S)

cdef double[::1] dln_B
dln_B = np.zeros(S)

cdef double[::1] dexp_epsilon_B
dexp_epsilon_B = np.zeros(S)

cdef double[::1] dG_effective
dG_effective = np.zeros(S)

cdef double[::1] dexp_epsilon_G
dexp_epsilon_G = np.zeros(S)

cdef double[:,::1] dlam
dlam = np.zeros((S,S))

cdef double[:,::1] dv
dv = np.zeros((S,S))

cdef double[:,::1] dexp_theta_v
dexp_theta_v = np.zeros((S,S))

cdef double[::1] dsum_exp_theta_v_M
dsum_exp_theta_v_M = np.zeros(S)

cdef double[::1] dsum_exp_theta_v_R
dsum_exp_theta_v_R = np.zeros(S)

cdef double[::1] v_M
v_M = np.zeros(S)

cdef double[::1] dv_M
dv_M = np.zeros(S)

cdef double[::1] v_R
v_R = np.zeros(S)

cdef double[::1] dv_R
dv_R = np.zeros(S)

def get_param(name):
    return {'theta': theta, 'eta_A': eta_A, 'eta_B': eta_B, 'delta_A': delta_A, 'delta_B': delta_B}[name]

def set_param(name, value):
    """パラメータを設定する．theta なら kappa を，delta なら減衰の行列を計算し直す"""
    global theta, kappa, eta_A, eta_B, delta_A, delta_B
    if value <= 0.0:
        raise ValueError
    if name == 'theta':
        theta = value
        kappa = kappa_theta / theta
    elif name == 'eta_A':
        eta_A = value
    elif name == 'eta_B':
        eta_B = value
    elif name == 'delta_A':
        delta_A = value
        update_decay()
    elif name == 'delta_B':
        delta_B = value
        update_decay()

def set_free_parameters(params):
    for k, name in enumerate(FREE_NAMES):
        if params[k] <= 0.0:
            raise ValueError
    for k, name in enumerate(FREE_NAMES):
        set_param(name, params[k])

def update_decay():
    """exp(−δτ) の行列を計算し直す（自ゾーン密度型では δ は使わない）"""
    cdef int t, i, j
    if ACCESSIBILITY == 'own_density':
        return
    for t in range(T):
        for i in prange(S, nogil=True):
            for j in range(S):
                exp_delta_A_tau[t,i,j] = exp(- delta_A * tau_decay[t,i,j])
                exp_delta_B_tau[t,i,j] = exp(- delta_B * tau_decay[t,i,j])

def calculate_basic_variables():
    cdef int t, i, j
    
    for t in range(T):
        if ACCESSIBILITY == 'own_density':
            # 感度分析：自ゾーン密度型 Ω_i = L_i / G_i（集積の外部性が自ゾーン内でのみ働く）
            # Ω_i = Σ_j L_j K_ij の K_ij を対角要素 1/G_i の対角行列とすることで実装する
            for i in range(S):
                for j in range(S):
                    exp_delta_A_tau[t,i,j] = 0.0
                    exp_delta_B_tau[t,i,j] = 0.0
                exp_delta_A_tau[t,i,i] = 1.0 / G[i]
                exp_delta_B_tau[t,i,i] = 1.0 / G[i]
        else:
            for i in prange(S, nogil=True):
                for j in range(S):
                    exp_delta_A_tau[t,i,j] = exp(- delta_A * tau_decay[t,i,j])
            
            for i in prange(S, nogil=True):
                for j in range(S):
                    exp_delta_B_tau[t,i,j] = exp(- delta_B * tau_decay[t,i,j])
        
        for i in prange(S, nogil=True):
            exp_epsilon_A[t,i] = exp(epsilon_A[t,i])
        
        for i in prange(S, nogil=True):
            exp_epsilon_B[t,i] = exp(epsilon_B[t,i])
        
        for i in prange(S, nogil=True):
            exp_epsilon_G[t,i] = exp(epsilon_G[t,i])

def calculate_auxiliary_variables():
    cdef int i, j
    
    for i in prange(S, nogil=True):
        Omega_A[i] = 0.0
        for j in range(S):
            Omega_A[i] = Omega_A[i] + lam_M[t,j] * Lbar[t] * exp_delta_A_tau[t,i,j]
    
    for i in prange(S, nogil=True):
        Omega_B[i] = 0.0
        for j in range(S):
            Omega_B[i] = Omega_B[i] + lam_R[t,j] * Lbar[t] * exp_delta_B_tau[t,i,j]
    
    for i in prange(S, nogil=True):
        A[i] = Abar * exp_epsilon_A[t,i] * pow(Omega_A[i], eta_A)
    
    for i in prange(S, nogil=True):
        ln_B[i] = log(exp_epsilon_B[t,i]) + eta_B * log(Omega_B[i])
    
    for i in prange(S, nogil=True):
        G_effective[i] = G[i] / exp_epsilon_G[t,i]
    
    for i in prange(S, nogil=True):
        p_H[i] = 1.0/Hbar * pow(Rbar/mu, mu) * pow(p_G[t,i]/(1.0 - mu), 1.0 - mu)
    
    for i in prange(S, nogil=True):
        w[i] = alpha * pow(A[i], 1.0 / alpha) * pow((1.0 - alpha) / p_H[i], (1.0 - alpha) / alpha)
    
    for i in prange(S, nogil=True):
        for j in range(S):
            v[i,j] = ln_B[i] - kappa * tau[t,i,j] + log(w[j]) - (1.0 - beta) * log(p_H[i])
    
    max_v[0] = -1.0e100
    for i in range(S):
        for j in range(S):
            if max_v[0] < v[i,j]:
                max_v[0] = v[i,j]
    
    for i in prange(S, nogil=True):
        for j in range(S):
            exp_theta_v[i,j] = exp( theta * (v[i,j] - max_v[0]) )
    
    for j in prange(S, nogil=True):
        sum_exp_theta_v_M[j] = 0.0
        for i in range(S):
            sum_exp_theta_v_M[j] = sum_exp_theta_v_M[j] + exp_theta_v[i,j]
    
    for i in prange(S, nogil=True):
        sum_exp_theta_v_R[i] = 0.0
        for j in range(S):
            sum_exp_theta_v_R[i] = sum_exp_theta_v_R[i] + exp_theta_v[i,j]
    
    for i in prange(S, nogil=True):
        for j in range(S):
            lam[i,j] = lam_M[t,j] * exp_theta_v[i,j] / sum_exp_theta_v_M[j]
    
    for i in prange(S, nogil=True):
        H_M[i] = pow(A[i] * (1.0 - alpha) / p_H[i], 1.0 / alpha) * lam_M[t,i] * Lbar[t]
    
    for i in prange(S, nogil=True):
        for j in range(S):
            h[i,j] = (1.0 - beta) * w[j] / p_H[i]
    
    for i in prange(S, nogil=True):
        H[i] = H_M[i]
        for j in range(S):
            H[i] = H[i] + lam[i,j] * Lbar[t] * h[i,j]
    
    for i in prange(S, nogil=True):
        G_demand[i] = (1.0 - mu) * p_H[i] * H[i] / p_G[t,i]

def calculate_auxiliary_variables2():
    cdef int i
    
    for i in range(S):
        v_R[i] = max_v[0] + 1.0/theta * log(sum_exp_theta_v_R[i] / lam_R[t,i])
        v_M[i] = max_v[0] + 1.0/theta * log(sum_exp_theta_v_M[i] / lam_M[t,i])

def func(y):
    cdef int i
    cdef double[::1] value
    
    for i in range(S):
        p_G[t,i] = y[i]
        if p_G[t,i] <= 0.0:
            raise ValueError
    
    calculate_auxiliary_variables()
    
    Value = np.zeros(S)
    value = Value
    
    for i in prange(S, nogil=True):
        value[i] = log( G_demand[i] / G_effective[i] )
    
    return Value

def func2(y):
    cdef int i, j
    cdef double[::1] value
    cdef int it
    
    it = 0
    
    exp_epsilon_A[t,S-1] = 1.0
    for i in range(S-1):
        exp_epsilon_A[t,i] = y[it]
        it += 1
        if exp_epsilon_A[t,i] <= 0.0:
            raise ValueError
    
    exp_epsilon_B[t,S-1] = 1.0
    for i in range(S-1):
        exp_epsilon_B[t,i] = y[it]
        it += 1
        if exp_epsilon_B[t,i] <= 0.0:
            raise ValueError
    
    for i in prange(S, nogil=True):
        p_G[t,i] = p_G_observed[t,i]
    
    for i in range(S):
        exp_epsilon_G[t,i] = y[it]
        it += 1
        if exp_epsilon_G[t,i] <= 0.0:
            raise ValueError
    
    calculate_auxiliary_variables()
    calculate_auxiliary_variables2()
    
    Value = np.zeros(3*S-2)
    value = Value
    
    it = 0
    for i in range(S-1):
        value[it] = v_M[i] - v_M[S-1] 
        it += 1
    for i in range(S-1):
        value[it] = v_R[i] - v_R[S-1]
        it += 1
    for i in range(S):
        value[it] = log( G_demand[i] / G_effective[i] )
        it += 1
    
    return Value

def calculate_auxiliary_dvariables():
    cdef int i, j
    
    for i in prange(S, nogil=True):
        dOmega_A[i] = 0.0
        for j in range(S):
            dOmega_A[i] = dOmega_A[i] + dlam_M[j] * Lbar[t] * exp_delta_A_tau[t,i,j]
    
    for i in prange(S, nogil=True):
        dOmega_B[i] = 0.0
        for j in range(S):
            dOmega_B[i] = dOmega_B[i] + dlam_R[j] * Lbar[t] * exp_delta_B_tau[t,i,j]
    
    for i in prange(S, nogil=True):
        dA[i] = A[i] * ( dexp_epsilon_A[i]/exp_epsilon_A[t,i] + eta_A * dOmega_A[i]/Omega_A[i] )
    
    for i in prange(S, nogil=True):
        dln_B[i] = dexp_epsilon_B[i]/exp_epsilon_B[t,i] + eta_B * dOmega_B[i]/Omega_B[i]
    
    for i in prange(S, nogil=True):
        dG_effective[i] = G_effective[i] * (- dexp_epsilon_G[i]/exp_epsilon_G[t,i])
    
    for i in prange(S, nogil=True):
        dp_H[i] = p_H[i] * (1.0 - mu) * dp_G[i]/p_G[t,i]
    
    for i in prange(S, nogil=True):
        dw[i] = w[i] * (1.0 / alpha * dA[i]/A[i] - (1.0 - alpha) / alpha * dp_H[i]/p_H[i])
    
    for i in prange(S, nogil=True):
        for j in range(S):
            dv[i,j] = dln_B[i] + dw[j]/w[j] - (1.0 - beta) * dp_H[i]/p_H[i]
    
    for i in prange(S, nogil=True):
        for j in range(S):
            dexp_theta_v[i,j] = exp_theta_v[i,j] * theta * dv[i,j]
    
    for j in prange(S, nogil=True):
        dsum_exp_theta_v_M[j] = 0.0
        for i in range(S):
            dsum_exp_theta_v_M[j] = dsum_exp_theta_v_M[j] + dexp_theta_v[i,j]
    
    for i in prange(S, nogil=True):
        dsum_exp_theta_v_R[i] = 0.0
        for j in range(S):
            dsum_exp_theta_v_R[i] = dsum_exp_theta_v_R[i] + dexp_theta_v[i,j]
    
    for i in prange(S, nogil=True):
        for j in range(S):
            dlam[i,j] = lam[i,j] * ( dlam_M[j]/lam_M[t,j] + dexp_theta_v[i,j]/exp_theta_v[i,j] - dsum_exp_theta_v_M[j]/sum_exp_theta_v_M[j] )
    
    for i in prange(S, nogil=True):
        dH_M[i] = H_M[i] * ( 1.0 / alpha * dA[i]/A[i] - 1.0 / alpha * dp_H[i]/p_H[i] + dlam_M[i]/lam_M[t,i] )
    
    for i in prange(S, nogil=True):
        for j in range(S):
            dh[i,j] = h[i,j] * ( dw[j]/w[j] - dp_H[i]/p_H[i] )
    
    for i in prange(S, nogil=True):
        dH[i] = dH_M[i]
        for j in range(S):
            dH[i] = dH[i] + lam[i,j] * Lbar[t] * h[i,j] * (dh[i,j]/h[i,j] + dlam[i,j]/lam[i,j])
    
    for i in prange(S, nogil=True):
        dG_demand[i] = G_demand[i] * ( dp_H[i]/p_H[i] + dH[i]/H[i] - dp_G[i]/p_G[t,i] )

def calculate_auxiliary_dvariables2():
    cdef int i
    
    for i in range(S):
        dv_M[i] = 1.0/theta * dsum_exp_theta_v_M[i]/sum_exp_theta_v_M[i] - 1.0/theta * dlam_M[i]/lam_M[t,i]
        dv_R[i] = 1.0/theta * dsum_exp_theta_v_R[i]/sum_exp_theta_v_R[i] - 1.0/theta * dlam_R[i]/lam_R[t,i]

def zeroclear_main_dvariables():
    cdef int i
    
    for i in prange(S, nogil=True):
        dlam_M[i] = 0.0
    
    for i in prange(S, nogil=True):
        dlam_R[i] = 0.0
    
    for i in prange(S, nogil=True):
        dp_G[i] = 0.0
    
    for i in prange(S, nogil=True):
        dexp_epsilon_A[i] = 0.0
    
    for i in prange(S, nogil=True):
        dexp_epsilon_B[i] = 0.0
    
    for i in prange(S, nogil=True):
        dexp_epsilon_G[i] = 0.0

def jac_func(y):
    cdef int i, j
    cdef double[:,::1] J
    JJ = np.zeros((S,S))
    J = JJ
    for j in range(S):
        zeroclear_main_dvariables()
        dp_G[j] = 1.0
        calculate_auxiliary_dvariables()
        for i in prange(S, nogil=True):
            J[i,j] = dG_demand[i]/G_demand[i]
    return JJ

def jac_func2(y):
    cdef int i1, i2
    cdef int it1, it2
    cdef double[:,::1] J
    
    JJ = np.zeros((3*S-2,3*S-2))
    J = JJ
    
    it2 = 0
    
    for i2 in range(S-1):
        zeroclear_main_dvariables()
        dexp_epsilon_A[i2] = 1.0
        calculate_auxiliary_dvariables()
        calculate_auxiliary_dvariables2()
        
        it1 = 0
        for i1 in range(S-1):
            J[it1,it2] = dv_M[i1] - dv_M[S-1]
            it1 += 1
        for i1 in range(S-1):
            J[it1,it2] = dv_R[i1] - dv_R[S-1] 
            it1 += 1
        for i1 in range(S):
            J[it1,it2] = dG_demand[i1]/G_demand[i1] - dG_effective[i1]/G_effective[i1]
            it1 += 1
        
        it2 += 1
    
    for i2 in range(S-1):
        zeroclear_main_dvariables()
        dexp_epsilon_B[i2] = 1.0
        calculate_auxiliary_dvariables()
        calculate_auxiliary_dvariables2()
        
        it1 = 0
        for i1 in range(S-1):
            J[it1,it2] = dv_M[i1] - dv_M[S-1]
            it1 += 1
        for i1 in range(S-1):
            J[it1,it2] = dv_R[i1] - dv_R[S-1]
            it1 += 1
        for i1 in range(S):
            J[it1,it2] = dG_demand[i1]/G_demand[i1] - dG_effective[i1]/G_effective[i1]
            it1 += 1
        
        it2 += 1
    
    for i2 in range(S):
        zeroclear_main_dvariables()
        dexp_epsilon_G[i2] = 1.0
        calculate_auxiliary_dvariables()
        calculate_auxiliary_dvariables2()
        
        it1 = 0
        for i1 in range(S-1):
            J[it1,it2] = dv_M[i1] - dv_M[S-1]
            it1 += 1
        for i1 in range(S-1):
            J[it1,it2] = dv_R[i1] - dv_R[S-1]
            it1 += 1
        for i1 in range(S):
            J[it1,it2] = dG_demand[i1]/G_demand[i1] - dG_effective[i1]/G_effective[i1]
            it1 += 1
        
        it2 += 1
    
    return JJ

cdef double[:,::1] J1man, J1eps

JJ1man = np.zeros(( 3*S-2, 3*S-2 ))
JJ1eps = np.zeros(( 3*S-2, 3*S-2 ))

J1man = JJ1man
J1eps = JJ1eps

"""
メイン変数と誤差項の数は(lam_M, lam_R, p_L)と(eps_A, eps_B, eps_G + base_eps_G)が3 * S - 2
補助変数の数は0
"""

def determinant():
    cdef int i1, i2
    cdef int it1, it2
    
    it2 = 0
    
    for i2 in range(S-1):
        zeroclear_main_dvariables()
        dlam_M[i2] = 1.0
        dlam_M[S-1] = -1.0
        calculate_auxiliary_dvariables()
        calculate_auxiliary_dvariables2()
        
        it1 = 0
        for i1 in range(S-1):
            J1man[it1,it2] = dv_M[i1] - dv_M[S-1]
            it1 += 1
        for i1 in range(S-1):
            J1man[it1,it2] = dv_R[i1] - dv_R[S-1] 
            it1 += 1
        for i1 in range(S):
            J1man[it1,it2] = dG_demand[i1]/G_demand[i1] - dG_effective[i1]/G_effective[i1]
            it1 += 1
        
        it2 += 1
    
    for i2 in range(S-1):
        zeroclear_main_dvariables()
        dlam_R[i2] = 1.0
        dlam_R[S-1] = -1.0
        calculate_auxiliary_dvariables()
        calculate_auxiliary_dvariables2()
        
        it1 = 0
        for i1 in range(S-1):
            J1man[it1,it2] = dv_M[i1] - dv_M[S-1]
            it1 += 1
        for i1 in range(S-1):
            J1man[it1,it2] = dv_R[i1] - dv_R[S-1] 
            it1 += 1
        for i1 in range(S):
            J1man[it1,it2] = dG_demand[i1]/G_demand[i1] - dG_effective[i1]/G_effective[i1]
            it1 += 1
        
        it2 += 1
    
    for i2 in range(S):
        zeroclear_main_dvariables()
        dp_G[i2] = 1.0
        calculate_auxiliary_dvariables()
        calculate_auxiliary_dvariables2()
        
        it1 = 0
        for i1 in range(S-1):
            J1man[it1,it2] = dv_M[i1] - dv_M[S-1]
            it1 += 1
        for i1 in range(S-1):
            J1man[it1,it2] = dv_R[i1] - dv_R[S-1] 
            it1 += 1
        for i1 in range(S):
            J1man[it1,it2] = dG_demand[i1]/G_demand[i1] - dG_effective[i1]/G_effective[i1]
            it1 += 1
        
        it2 += 1
    
    
    it2 = 0
    
    for i2 in range(S-1):
        zeroclear_main_dvariables()
        dexp_epsilon_A[i2] = exp_epsilon_A[t,i2]
        calculate_auxiliary_dvariables()
        calculate_auxiliary_dvariables2()
        
        it1 = 0
        for i1 in range(S-1):
            J1eps[it1,it2] = dv_M[i1] - dv_M[S-1]
            it1 += 1
        for i1 in range(S-1):
            J1eps[it1,it2] = dv_R[i1] - dv_R[S-1] 
            it1 += 1
        for i1 in range(S):
            J1eps[it1,it2] = dG_demand[i1]/G_demand[i1] - dG_effective[i1]/G_effective[i1]
            it1 += 1
        
        it2 += 1
    
    for i2 in range(S-1):
        zeroclear_main_dvariables()
        dexp_epsilon_B[i2] = exp_epsilon_B[t,i2]
        calculate_auxiliary_dvariables()
        calculate_auxiliary_dvariables2()
        
        it1 = 0
        for i1 in range(S-1):
            J1eps[it1,it2] = dv_M[i1] - dv_M[S-1]
            it1 += 1
        for i1 in range(S-1):
            J1eps[it1,it2] = dv_R[i1] - dv_R[S-1]
            it1 += 1
        for i1 in range(S):
            J1eps[it1,it2] = dG_demand[i1]/G_demand[i1] - dG_effective[i1]/G_effective[i1]
            it1 += 1
        
        it2 += 1
    
    for i2 in range(S):
        zeroclear_main_dvariables()
        dexp_epsilon_G[i2] = exp_epsilon_G[t,i2]
        calculate_auxiliary_dvariables()
        calculate_auxiliary_dvariables2()
        
        it1 = 0
        for i1 in range(S-1):
            J1eps[it1,it2] = dv_M[i1] - dv_M[S-1]
            it1 += 1
        for i1 in range(S-1):
            J1eps[it1,it2] = dv_R[i1] - dv_R[S-1]
            it1 += 1
        for i1 in range(S):
            J1eps[it1,it2] = dG_demand[i1]/G_demand[i1] - dG_effective[i1]/G_effective[i1]
            it1 += 1
        
        it2 += 1
    
    
    (signeps, logdeteps) = np.linalg.slogdet(- JJ1eps)
    (signman, logdetman) = np.linalg.slogdet(  JJ1man)
    
    return (signeps * signman, logdetman - logdeteps)

JI2 = []
for t in range(T):
    JI2.append(None)

def calculate_error_terms():
    global t
    cdef int it
    cdef int i, j
    cdef double err, maxerr
    
    y1 = np.zeros(S)
    y2 = np.zeros(3*S-2)
    
    for t in range(T):
        print(f"  構造残差の初期値の計算：時点 {t}（土地賃料を観測値に合わせる反復）", flush=True)
        t_start = time.time()
        for i in range(S):
            y1[i] = p_G[t,i]
        
        res = myoptimize.solve1(func, y1, jac = jac_func, tol = tol_model, label = f"t={t} 均衡")
        JI = res.jac_inv
        
        maxerr = 0.0
        for i in range(S):
            exp_epsilon_G[t,i] *= p_G_observed[t,i] / p_G[t,i]
            err = fabs(p_G_observed[t,i] / p_G[t,i] - 1.0)
            if maxerr < err:
                maxerr = err
        
        n_outer = 1
        t_last = time.time()
        while maxerr > 1.0e-2:
            if time.time() - t_last >= 10.0:
                t_last = time.time()
                print(f"    [t={t} 土地賃料の調整] 反復 {n_outer:,}，経過 {t_last - t_start:,.0f} 秒，"
                      f"観測値との最大相対誤差 {maxerr:.3e}（目標 1e-2）", flush=True)
            n_outer += 1
            res = myoptimize.solve1(func, y1, jac = jac_func, tol = tol_model, label = f"t={t} 均衡")
            JI = res.jac_inv
            
            maxerr = 0.0
            for i in range(S):
                exp_epsilon_G[t,i] *= p_G_observed[t,i] / p_G[t,i]
                err = fabs(p_G_observed[t,i] / p_G[t,i] - 1.0)
                if maxerr < err:
                    maxerr = err
        
        calculate_auxiliary_variables2()
        exp_epsilon_A[t,i] *= exp(v_M[S-1] - v_M[i])
        exp_epsilon_B[t,i] *= exp(v_R[S-1] - v_R[i])
        
        it = 0
        for i in range(S-1):
            y2[it] = exp_epsilon_A[t,i]
            it += 1
        for i in range(S-1):
            y2[it] = exp_epsilon_B[t,i] 
            it += 1
        for i in range(S):
            y2[it] = exp_epsilon_G[t,i]
            it += 1
        
        print(f"  時点 {t}：土地賃料の調整 {n_outer:,} 回（{time.time() - t_start:,.0f} 秒）．構造残差の逆算（ニュートン法）を開始", flush=True)
        res = myoptimize.solve2(func2, y2, jac = jac_func2, tol = tol_inversion, label = f"t={t} 構造残差の逆算")
        JI2[t] = res.jac_inv
        print(f"  時点 {t}：構造残差の逆算が完了（計 {time.time() - t_start:,.0f} 秒）", flush=True)
        
        for i in prange(S, nogil=True):
            epsilon_A[t,i] = log(exp_epsilon_A[t,i])
        
        for i in prange(S, nogil=True):
            epsilon_B[t,i] = log(exp_epsilon_B[t,i])
        
        for i in prange(S, nogil=True):
            epsilon_G[t,i] = log(exp_epsilon_G[t,i])

logdet = np.zeros(T)
dlogdet = np.zeros(T)

def calculate_error_terms_and_determinant(logdet):
    global t
    cdef int it
    cdef int i, j
    cdef double err, maxerr, summ, summ2
    
    y = np.zeros(3*S-2)
    
    for t in range(T):
        it = 0
        for i in range(S-1):
            y[it] = exp_epsilon_A[t,i]
            it += 1
        for i in range(S-1):
            y[it] = exp_epsilon_B[t,i] 
            it += 1
        for i in range(S):
            y[it] = exp_epsilon_G[t,i]
            it += 1
        
        res = myoptimize.solve2(func2, y, jac = jac_func2, tol = tol_inversion, label = f"t={t} 構造残差の逆算（尤度の計算）")
        JI2[t] = res.jac_inv
        
        for i in prange(S, nogil=True):
            epsilon_A[t,i] = log(exp_epsilon_A[t,i])
        
        for i in prange(S, nogil=True):
            epsilon_B[t,i] = log(exp_epsilon_B[t,i])
        
        for i in prange(S, nogil=True):
            epsilon_G[t,i] = log(exp_epsilon_G[t,i])
        
        (sign, logdet[t]) = determinant()
        if sign == -1:
            raise ValueError

errorterms = liberrortermsre.Errorterms(T, S, 3)
epstilde = np.zeros((T, S - 1, 3))
depstilde = np.zeros((T, S - 1, 3))
ll = np.zeros(1)

def log_likelihood(parameters):
    global t, kappa, theta, eta_A, eta_B, delta_A, delta_B
    cdef int i, j
    
    set_free_parameters(parameters)
    
    calculate_error_terms_and_determinant(logdet)
    
    for t in range(T):
        for i in range(S-1):
            epstilde[t,i,0] = epsilon_A[t,i]
            epstilde[t,i,1] = epsilon_B[t,i]
            epstilde[t,i,2] = epsilon_G[t,i] - epsilon_G[t,S-1]
    
    llfeps = errorterms.compute_log_likelihood(epstilde)
    
    ll[0] = llfeps + np.sum(logdet)
    print('parameters:', parameters)
    print('LL:', ll[0])
    
    with open (os.path.join(OUTPUT_DIR, 'epsilon_A_calibrated.csv'), 'w') as fout:
        fout.write('Location')
        fout.write(',')
        for t in range(T):
            fout.write('Period '+str(t))
            fout.write(',')
        fout.write('\n')
        for i in range(S):
            fout.write('Location '+str(i))
            fout.write(',')
            for t in range(T):
                fout.write(str(epsilon_A[t,i]))
                fout.write(',')
            fout.write('\n')

    with open (os.path.join(OUTPUT_DIR, 'epsilon_B_calibrated.csv'), 'w') as fout:
        fout.write('Location')
        fout.write(',')
        for t in range(T):
            fout.write('Period '+str(t))
            fout.write(',')
        fout.write('\n')
        for i in range(S):
            fout.write('Location '+str(i))
            fout.write(',')
            for t in range(T):
                fout.write(str(epsilon_B[t,i]))
                fout.write(',')
            fout.write('\n')

    with open (os.path.join(OUTPUT_DIR, 'epsilon_G_calibrated.csv'), 'w') as fout:
        fout.write('Location')
        fout.write(',')
        for t in range(T):
            fout.write('Period '+str(t))
            fout.write(',')
        fout.write('\n')
        for i in range(S):
            fout.write('Location '+str(i))
            fout.write(',')
            for t in range(T):
                fout.write(str(epsilon_G[t,i]))
                fout.write(',')
            fout.write('\n')
    
    with open (os.path.join(OUTPUT_DIR, 'parameters_estimated.csv'), 'w') as fout:
        fout.write('kappa,'+str(kappa)+'\n')
        fout.write('theta,'+str(theta)+'\n')
        fout.write('eta_A,'+str(eta_A)+'\n')
        fout.write('eta_B,'+str(eta_B)+'\n')
        fout.write('delta_A,'+str(delta_A)+'\n')
        fout.write('delta_B,'+str(delta_B)+'\n')
        fout.write('LL,'+str(ll[0])+'\n')
    
    return - ll[0]

def grad_log_likelihood(parameters):
    global t, kappa, theta, eta_A, eta_B, delta_A, delta_B
    cdef int i, it3
    
    grad = np.zeros(dim_parameters)
    
    for it3 in range(dim_parameters):
        name = FREE_NAMES[it3]
        original = get_param(name)
        diff = original * 1.0e-7
        set_param(name, original + diff)
        
        calculate_error_terms_and_determinant(dlogdet)
        dlogdet[:] -= logdet[:]
        
        for t in range(T):
            for i in range(S-1):
                depstilde[t,i,0] = epsilon_A[t,i] - epstilde[t,i,0]
                depstilde[t,i,1] = epsilon_B[t,i] - epstilde[t,i,1]
                depstilde[t,i,2] = epsilon_G[t,i] - epsilon_G[t,S-1] - epstilde[t,i,2]
        
        dlleps = errorterms.compute_dlog_likelihood_dtheta(depstilde)
        
        grad[it3] = (dlleps + np.sum(dlogdet)) / diff
        
        set_param(name, original)
    
    print('grad:', grad)
    
    return - grad

calculate_basic_variables()

t = 0

for t in range(T):
    if t > 0:
        for i in prange(S, nogil=True):
            p_G[t,i] = p_G[t-1,i]
        for i in prange(S, nogil=True):
            exp_epsilon_G[t,i] = exp_epsilon_G[t-1,i]
        for i in prange(S, nogil=True):
            exp_epsilon_A[t,i] = exp_epsilon_A[t-1,i]
        for i in prange(S, nogil=True):
            exp_epsilon_B[t,i] = exp_epsilon_B[t-1,i]
    
    calculate_error_terms()

with open (os.path.join(OUTPUT_DIR, 'epsilon_A_calibrated.csv'), 'w') as fout:
    fout.write('Location')
    fout.write(',')
    for t in range(T):
        fout.write('Period '+str(t))
        fout.write(',')
    fout.write('\n')
    for i in range(S):
        fout.write('Location '+str(i))
        fout.write(',')
        for t in range(T):
            fout.write(str(epsilon_A[t,i]))
            fout.write(',')
        fout.write('\n')

with open (os.path.join(OUTPUT_DIR, 'epsilon_B_calibrated.csv'), 'w') as fout:
    fout.write('Location')
    fout.write(',')
    for t in range(T):
        fout.write('Period '+str(t))
        fout.write(',')
    fout.write('\n')
    for i in range(S):
        fout.write('Location '+str(i))
        fout.write(',')
        for t in range(T):
            fout.write(str(epsilon_B[t,i]))
            fout.write(',')
        fout.write('\n')

with open (os.path.join(OUTPUT_DIR, 'epsilon_G_calibrated.csv'), 'w') as fout:
    fout.write('Location')
    fout.write(',')
    for t in range(T):
        fout.write('Period '+str(t))
        fout.write(',')
    fout.write('\n')
    for i in range(S):
        fout.write('Location '+str(i))
        fout.write(',')
        for t in range(T):
            fout.write(str(epsilon_G[t,i]))
            fout.write(',')
        fout.write('\n')

myoptimize.optimize(log_likelihood, parameters, grad = grad_log_likelihood, tol1 = 1.0e-5)

def L(parameters):
    global t, kappa, theta, eta_A, eta_B, delta_A, delta_B
    cdef int i, j
    
    set_free_parameters(parameters)
    
    calculate_error_terms_and_determinant(logdet)
    
    for t in range(T):
        for i in range(S-1):
            epstilde[t,i,0] = epsilon_A[t,i]
            epstilde[t,i,1] = epsilon_B[t,i]
            epstilde[t,i,2] = epsilon_G[t,i] - epsilon_G[t,S-1]
    
    llfeps = errorterms.compute_log_likelihood(epstilde)
    ll[0] = llfeps + np.sum(logdet)
    return - ll[0]

def hesse_L(theta):
    print("Estimation completed. Begins the computation of Hessian matrix. Please wait for a while.")
    n = theta.size
    hbase = sys.float_info.epsilon ** (1.0/4.0)
    # 刻み幅：theta, eta は従来通り絶対値 hbase*10，delta は相対値（値に比例）
    h = np.array([hbase * 10.0 * (abs(theta[k]) if FREE_NAMES[k].startswith('delta') else 1.0) for k in range(n)])
    H = np.zeros((n,n))
    thetas = theta.copy()
    fs = L(thetas)
    for i in range(n):
        print("(i, j) = ("+str(i)+", "+str(i)+")")
        thetas[i] = theta[i] + h[i]
        fp = L(thetas)
        thetas[i] = theta[i] - h[i]
        fm = L(thetas)
        H[i,i] = (fp + fm - 2.0 * fs) / (h[i] * h[i])
        thetas[i] = theta[i]
    for i in range(n):
        for j in range(i):
            print("(i, j) = ("+str(i)+", "+str(j)+")")
            thetas[i] = theta[i] + h[i]; thetas[j] = theta[j] + h[j]
            fpp = L(thetas)
            thetas[i] = theta[i] + h[i]; thetas[j] = theta[j] - h[j]
            fpm = L(thetas)
            thetas[i] = theta[i] - h[i]; thetas[j] = theta[j] + h[j]
            fmp = L(thetas)
            thetas[i] = theta[i] - h[i]; thetas[j] = theta[j] - h[j]
            fmm = L(thetas)
            H[i,j] = (fpp + fmm - fpm - fmp) / (4.0 * h[i] * h[j])
            H[j,i] = H[i,j]
            thetas[i] = theta[i]; thetas[j] = theta[j]
    return H

def low_tri_inverse(L):
    LI = np.identity(L.shape[0])
    for i in range(L.shape[0]):
        for j in range(i):
            LI[i,j] = -np.dot(L[i,j:i], LI[j:i,j])/L[i,i]
        LI[i,i] = 1.0/L[i,i]
    return LI

Hessian = hesse_L(parameters)
HL = np.linalg.cholesky(Hessian)
HLinv = low_tri_inverse(HL)
Hinv = HLinv.T @ HLinv

with open (os.path.join(OUTPUT_DIR, 'standard error.csv'), 'w') as fout:
    for i in range(parameters.size):
        fout.write(str(np.sqrt(Hinv[i,i]))+"\n")

print("=== 誤差項の共分散に関する出力 ===")
print(f"個体内相関係数: {(1 - errorterms.kappa ** 2) / (1 + (T - 1) * errorterms.kappa ** 2)}")

D_inv = np.diag(1.0 / np.sqrt(np.diag(errorterms.Sigma)))
corr_matrix = D_inv @ errorterms.Sigma @ D_inv
for i in range(3):
    print(f"sigma[{i},{i}]: {sqrt(errorterms.Sigma[i,i])}")
for i in range(3):
    for j in range(i):
        print(f"rho[{i},{j}]: {corr_matrix[i,j]}")