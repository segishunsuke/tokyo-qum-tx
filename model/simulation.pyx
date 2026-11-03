# cython: boundscheck=False
# cython: wraparound=False

from libc.math cimport fmax, fabs, sqrt, log, pow, exp
from cython.parallel cimport prange
import numpy as np
import pandas as pd
import csv
import os
import myoptimize

cdef double tol_model = 1.0e-10

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

cdef int it
cdef int t, i, j, r, s

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

cdef double delta_A
delta_A = float(data_parameters[it][1])
it += 1

cdef double delta_B
delta_B = float(data_parameters[it][1])
it += 1

cdef double theta
theta = float(data_parameters[it][1])
it += 1

# thetaが決まったので kappa を計算
cdef double kappa
kappa = kappa_theta / theta

cdef double eta_A
eta_A = float(data_parameters[it][1])
it += 1

cdef double eta_B
eta_B = float(data_parameters[it][1])
it += 1

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

cdef double[:,::1] epsilon_A
epsilon_A = np.zeros((T,S))

try:
    with open (os.path.join(INPUT_DIR, 'epsilon_A.csv'), 'r', encoding='utf-8-sig') as fin:
        cin = csv.reader(fin)
        data_eps_A = [row for row in cin]
    for i in range(S):
        for t in range(T):
            epsilon_A[t,i] = float(data_eps_A[1+i][1+t])
except:
    for i in range(S):
        for t in range(T):
            epsilon_A[t,i] = 0.0

cdef double[:,::1] epsilon_B
epsilon_B = np.zeros((T,S))

try:
    with open (os.path.join(INPUT_DIR, 'epsilon_B.csv'), 'r', encoding='utf-8-sig') as fin:
        cin = csv.reader(fin)
        data_eps_B = [row for row in cin]
    for i in range(S):
        for t in range(T):
            epsilon_B[t,i] = float(data_eps_B[1+i][1+t])
except:
    for i in range(S):
        for t in range(T):
            epsilon_B[t,i] = 0.0

cdef double[:,::1] epsilon_G
epsilon_G = np.zeros((T,S))
try:
    with open (os.path.join(INPUT_DIR, 'epsilon_G.csv'), 'r', encoding='utf-8-sig') as fin:
        cin = csv.reader(fin)
        data_eps_G = [row for row in cin]
    for i in range(S):
        for t in range(T):
            epsilon_G[t,i] = float(data_eps_G[1+i][1+t])
except:
    for i in range(S):
        for t in range(T):
            epsilon_G[t,i] = 0.0
    
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

cdef double[:,::1] lam_M
lam_M = np.zeros((T,S))

for i in range(S):
    for t in range(T):
        lam_R[t,i] = float(data_lam_R[1+i][1+t])

for t in range(T):
    summ = 0.0
    for i in range(S):
        summ += lam_R[t,i]
    for i in range(S):
        lam_R[t,i] /= summ

for i in range(S):
    for t in range(T):
        lam_M[t,i] = float(data_lam_M[1+i][1+t])

for t in range(T):
    summ = 0.0
    for i in range(S):
        summ += lam_M[t,i]
    for i in range(S):
        lam_M[t,i] /= summ

for i in range(S):
    for j in range(S):
        lam[i,j] = lam_R[0,i] * lam_M[0,j]

cdef double[:,::1] dlam
dlam = np.ones((S,S))

cdef double[::1] dlam_R
dlam_R = np.zeros(S)

cdef double[::1] dlam_M
dlam_M = np.zeros(S)

cdef double[:,::1] exp_theta_v
exp_theta_v = np.zeros((S,S))

cdef double[::1] sum_exp_theta_v
sum_exp_theta_v = np.zeros(S)

cdef double max_v[1]

cdef double sum_all_exp_theta_v[1]

cdef double[:,:,::1] exp_delta_A_tau
exp_delta_A_tau = np.zeros((T,S,S))

cdef double[:,:,::1] exp_delta_B_tau
exp_delta_B_tau = np.zeros((T,S,S))

cdef double[::1] Omega_A
Omega_A = np.zeros(S)

cdef double[::1] Omega_B
Omega_B = np.zeros(S)

cdef double[:,::1] exp_epsilon_A
exp_epsilon_A = np.ones((T,S))

cdef double[:,::1] exp_epsilon_G
exp_epsilon_G = np.ones((T,S))

cdef double[::1] A
A = np.zeros(S)

cdef double[::1] ln_B
ln_B = np.zeros(S)

cdef double[::1] G_effective
G_effective = np.zeros(S)

cdef double[:,::1] p_G
p_G = np.ones((T,S))

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

# ========== 追加の出力用変数の宣言 ==========
cdef double[:,::1] w_predicted = np.zeros((T,S))
cdef double[:,::1] H_predicted = np.zeros((T,S))
cdef double[::1] landlord_income = np.zeros(T)
cdef double[::1] total_household_income = np.zeros(T)
cdef double[::1] logsum_V = np.zeros(T)
cdef double[::1] average_wage = np.zeros(T)
cdef double[::1] average_commuting_cost = np.zeros(T)
cdef double[::1] average_tau = np.zeros(T)
# ============================================

def calculate_basic_variables1():
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
            exp_epsilon_G[t,i] = exp(epsilon_G[t,i])

def calculate_basic_variables2():
    cdef int i, j
    
    for i in prange(S, nogil=True):
        lam_M[t,i] = 0.0
        for j in range(S):
            lam_M[t,i] += lam[j,i]
    
    for i in prange(S, nogil=True):
        lam_R[t,i] = 0.0
        for j in range(S):
            lam_R[t,i] += lam[i,j]
    
    for i in prange(S, nogil=True):
        Omega_A[i] = 0.0
        for j in range(S):
            Omega_A[i] = Omega_A[i] + lam_M[t,j] * Lbar[t] * exp_delta_A_tau[t,i,j]
    
    for i in prange(S, nogil=True):
        Omega_B[i] = 0.0
        for j in range(S):
            Omega_B[i] = Omega_B[i] + lam_R[t,j] * Lbar[t] * exp_delta_B_tau[t,i,j]
    
def calculate_basic_variables3():
    cdef int i, j
    
    for i in prange(S, nogil=True):
        A[i] = Abar * exp_epsilon_A[t,i] * pow(Omega_A[i], eta_A)
    
    for i in prange(S, nogil=True):
        ln_B[i] = epsilon_B[t,i] + eta_B * log(Omega_B[i])
    
    for i in prange(S, nogil=True):
        G_effective[i] = G[i] / exp_epsilon_G[t,i]

def calculate_auxiliary_variables():
    cdef int i, j
    
    for i in prange(S, nogil=True):
        p_H[i] = 1.0/Hbar * pow(Rbar/mu, mu) * pow(p_G[t,i]/(1.0 - mu), 1.0 - mu)
    
    for i in prange(S, nogil=True):
        H_M[i] = pow(A[i] * (1.0 - alpha) / p_H[i], 1.0 / alpha) * lam_M[t,i] * Lbar[t]
    
    for i in prange(S, nogil=True):
        w[i] = alpha * pow(A[i], 1.0 / alpha) * pow((1.0 - alpha) / p_H[i], (1.0 - alpha) / alpha)
    
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
    cdef int i, j
    
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
        sum_exp_theta_v[j] = 0.0
        for i in range(S):
            sum_exp_theta_v[j] = sum_exp_theta_v[j] + exp_theta_v[i,j]
    
    sum_all_exp_theta_v[0] = 0.0
    for j in range(S):
        sum_all_exp_theta_v[0] += sum_exp_theta_v[j]
        
    for j in prange(S, nogil=True):
        lam_M[t,j] = sum_exp_theta_v[j] / sum_all_exp_theta_v[0]

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

def calculate_auxiliary_dvariables():
    cdef int i, j
    
    for i in prange(S, nogil=True):
        dp_H[i] = p_H[i] * (1.0 - mu) * dp_G[i]/p_G[t,i]
    
    for i in prange(S, nogil=True):
        dH_M[i] = H_M[i] * ( - 1.0 / alpha * dp_H[i]/p_H[i] + dlam_M[i]/lam_M[t,i] )
    
    for i in prange(S, nogil=True):
        dw[i] = w[i] * ( - (1.0 - alpha) / alpha * dp_H[i]/p_H[i] )
    
    for i in prange(S, nogil=True):
        for j in range(S):
            dh[i,j] = h[i,j] * ( dw[j]/w[j] - dp_H[i]/p_H[i] )
    
    for i in prange(S, nogil=True):
        dH[i] = dH_M[i]
        for j in range(S):
            dH[i] = dH[i] + lam[i,j] * Lbar[t] * h[i,j] * ( dlam[i,j]/lam[i,j] + dh[i,j]/h[i,j] )
    
    for i in prange(S, nogil=True):
        dG_demand[i] = G_demand[i] * ( dp_H[i]/p_H[i] + dH[i]/H[i] - dp_G[i]/p_G[t,i] )

def zeroclear_main_dvariables():
    cdef int i, j
    
    for i in prange(S, nogil=True):
        dp_G[i] = 0.0
    for i in prange(S, nogil=True):
        for j in range(S):
            dlam[i,j] = 0.0
    for i in prange(S, nogil=True):
        dlam_R[i] = 0.0
    for i in prange(S, nogil=True):
        dlam_M[i] = 0.0

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

JI = None
y = np.zeros(S)

def replicator_dynamic():
    global JI
    cdef double gam = 1.0, Deltalam, err, summ
    cdef double[:,::1] lam2
    cdef int i, j
    
    lam2 = np.zeros((S,S))
    
    calculate_basic_variables1()
    
    while True:
        calculate_basic_variables2()
        calculate_basic_variables3()
        
        for i in range(S):
            y[i] = p_G[t,i]
        res = myoptimize.solve2(func, y, JI = JI, jac = jac_func, tol = tol_model)
        JI = res.jac_inv
        
        calculate_auxiliary_variables2()
        
        for i in prange(S, nogil=True):
            for j in range(S):
                lam2[i,j] = exp_theta_v[i,j] / sum_all_exp_theta_v[0]
        
        Deltalam = 0.0
        for i in prange(S, nogil=True):
            for j in range(S):
                Deltalam += gam * fabs(lam2[i,j] - lam[i,j])
        
        if Deltalam > 0.05:
            gam *= 0.05 / Deltalam
        
        err = 0.0
        for i in range(S):
            for j in range(S):
                if err < fabs( gam * (lam2[i,j] - lam[i,j]) / lam[i,j] ):
                    err = fabs( gam * (lam2[i,j] - lam[i,j]) / lam[i,j] )
        
        print('t:', t, ', error:', err)
        if err < 1.0e-8:
            break
        
        for i in prange(S, nogil=True):
            for j in range(S):
                lam[i,j] = lam[i,j] + gam * (lam2[i,j] - lam[i,j])

t = 0

calculate_basic_variables1()
calculate_basic_variables2()
calculate_basic_variables3()
for i in range(S):
    y[i] = p_G[t,i]
res = myoptimize.solve1(func, y, jac = jac_func, tol = tol_model)
JI = res.jac_inv

for t in range(T):
    if t > 0:
        for i in prange(S, nogil=True):
            p_G[t,i] = p_G[t-1,i]
    
    replicator_dynamic()

    # ========== 追加部分：シミュレーション結果（各期の均衡値）の保存 ==========
    for i in range(S):
        w_predicted[t,i] = w[i]
        H_predicted[t,i] = H[i]
        
    landlord_income[t] = 0.0
    for i in range(S):
        landlord_income[t] += p_G[t,i] * G_effective[i]
        
    total_household_income[t] = 0.0
    average_commuting_cost[t] = 0.0
    average_tau[t] = 0.0
    for i in range(S):
        for j in range(S):
            total_household_income[t] += Lbar[t] * lam[i,j] * w[j]
            average_commuting_cost[t] += lam[i,j] * w[j] * (1.0 - exp(- kappa * tau[t,i,j]))
            average_tau[t] += lam[i,j] * tau[t,i,j]
            
    logsum_V[t] = max_v[0] + (1.0 / theta) * log(sum_all_exp_theta_v[0])
    
    average_wage[t] = 0.0
    for j in range(S):
        average_wage[t] += lam_M[t,j] * w[j]
    # ==========================================================================
    
    # OD交通量出力用
    with open (os.path.join(OUTPUT_DIR, f'OD_demand[{t}]_predict.csv'), 'w') as fout:
        fout.write('i,j,demand\n')
        for i in range(S):
            for j in range(S):
                fout.write(f'{i},{j},{Lbar[t] * lam[i,j]}\n')

with open (os.path.join(OUTPUT_DIR, 'lam_M_predict.csv'), 'w') as fout:
    fout.write('Location')
    fout.write(',')
    for t in range(T):
        fout.write('Period '+str(t))
        fout.write(',')
    fout.write('\n')
    for i in range(S):
        fout.write(str(i))
        fout.write(',')
        for t in range(T):
            fout.write(str(lam_M[t,i]))
            fout.write(',')
        fout.write('\n')

with open (os.path.join(OUTPUT_DIR, 'lam_R_predict.csv'), 'w') as fout:
    fout.write('Location')
    fout.write(',')
    for t in range(T):
        fout.write('Period '+str(t))
        fout.write(',')
    fout.write('\n')
    for i in range(S):
        fout.write(str(i))
        fout.write(',')
        for t in range(T):
            fout.write(str(lam_R[t,i]))
            fout.write(',')
        fout.write('\n')

with open (os.path.join(OUTPUT_DIR, 'p_G_predict.csv'), 'w') as fout:
    fout.write('Location')
    fout.write(',')
    for t in range(T):
        fout.write('Period '+str(t))
        fout.write(',')
    fout.write('\n')
    for i in range(S):
        fout.write(str(i))
        fout.write(',')
        for t in range(T):
            fout.write(str(p_G[t,i]))
            fout.write(',')
        fout.write('\n')

# ========== 追加部分：新規指標のエクスポート ==========
with open (os.path.join(OUTPUT_DIR, 'w_predicted.csv'), 'w') as fout:
    fout.write('Location')
    fout.write(',')
    for t in range(T):
        fout.write('Period '+str(t))
        fout.write(',')
    fout.write('\n')
    for i in range(S):
        fout.write(str(i))
        fout.write(',')
        for t in range(T):
            fout.write(str(w_predicted[t,i]))
            fout.write(',')
        fout.write('\n')

with open (os.path.join(OUTPUT_DIR, 'H_predicted.csv'), 'w') as fout:
    fout.write('Location')
    fout.write(',')
    for t in range(T):
        fout.write('Period '+str(t))
        fout.write(',')
    fout.write('\n')
    for i in range(S):
        fout.write(str(i))
        fout.write(',')
        for t in range(T):
            fout.write(str(H_predicted[t,i]))
            fout.write(',')
        fout.write('\n')

with open (os.path.join(OUTPUT_DIR, 'welfare_indicators.csv'), 'w') as fout:
    fout.write('Period,Landlord_Income,Total_Household_Income,Logsum_V,Average_Wage,Average_Commuting_Cost,Average_tau\n')
    for t in range(T):
        fout.write(str(t) + ',' + 
                   str(landlord_income[t]) + ',' + 
                   str(total_household_income[t]) + ',' + 
                   str(logsum_V[t]) + ',' + 
                   str(average_wage[t]) + ',' + 
                   str(average_commuting_cost[t]) + ',' + 
                   str(average_tau[t]) + '\n')
# ======================================================