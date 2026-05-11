## ==============================================================================
## SECTION 1: LIBRARIES AND DATA LOADING
## ==============================================================================

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import skew, kurtosis, jarque_bera
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt

# Load database with specific formatting for missing values and dates
data = pd.read_excel('database.xlsx',
                    sheet_name='database',
                    na_values=['n.e.'],
                    index_col='Index',
                    header=0,
                    parse_dates=True)

## ==============================================================================
## SECTION 2: ASSET UNIVERSE DEFINITION
## ==============================================================================

# Assets available for non-informed clients (Basic Universe)
noninformed_assets = [
    "EURUSD Tasso di cambio spot - Prezzo : 1 EUR in USD",
    
    # --- BOND & MONEY MARKET ---
    "DB Euro Overnight Rate Index",                                     # Euro Liquidity
    "Bloomberg Global Aggregate 1-3 Year Total Return Index Unhedged EUR",    # Short Term Sovereign and Corporate
    "Bloomberg Global Agg Treasuries Total Return Index Value Unhedged USD",  # Sovereign Global all duration
    "Bloomberg Global Agg Corporate Total Return Index Value Unhedged USD", # Global Corporate Investment Grade 
    "Bloomberg Global High Yield Total Return Index Value Unhedge",                 # Global Corporate High Yield      
    "Bloomberg Emerging Markets Sovereign TR Index Value Unhedged USD",     # Emerging Market Bonds
    "Bloomberg EuroAgg Government Total Return Index Value Unhedged EUR", # Tilt on European Sovereign
    
    # --- GEOGRAPHIC EQUITIES (Local Indices for pure optimization) ---
    "MSCI Europe Gross Total Return Local Index",
    "MSCI North America Gross Total Return Local Index",
    "MSCI Pacific Gross Total Return Local Index",
    "MSCI EM Gross Total Return Local Index",
    
    # --- COMMODITIES ---
    "Bloomberg-Gold Sub Index TR - RETURN IND. (OFCL)"            # Gold
]

# Extended assets for informed clients (Sectors & Commodities)
informed_assets = [
    "EURUSD Tasso di cambio spot - Prezzo : 1 EUR in USD",
    
    # --- BOND & MONEY MARKET ---
    "DB Euro Overnight Rate Index",                                     # Euro Liquidity
    "Bloomberg Global Aggregate 1-3 Year Total Return Index Unhedged EUR",    # Short Term Sovereign and Corporate
    "Bloomberg Global Agg Treasuries Total Return Index Value Unhedged USD",  # Sovereign Global all duration
    "Bloomberg Global Agg Corporate Total Return Index Value Unhedged USD", # Global Corporate Investment Grade 
    "Bloomberg Global High Yield Total Return Index Value Unhedge",                 # Global Corporate High Yield      
    "Bloomberg Emerging Markets Sovereign TR Index Value Unhedged USD", # Emerging Bonds
    "Bloomberg EuroAgg Government Total Return Index Value Unhedged EUR", # Tilt on European Sovereign
    
    # --- GEOGRAPHIC EQUITIES ---
    "MSCI Europe Gross Total Return Local Index",
    "MSCI North America Gross Total Return Local Index",
    "MSCI Pacific Gross Total Return Local Index",
    "MSCI EM Gross Total Return Local Index",
    
    # --- INDUSTRY SECTORS - WORLD ---
    "MSCI WORLD ENERGY $ - TOT RETURN IND",
    "MSCI WORLD MATERIALS $ - TOT RETURN IND",
    "MSCI WORLD INDUSTRIALS $ - TOT RETURN IND",
    "MSCI WORLD CONS DISCR $ - TOT RETURN IND",
    "MSCI WORLD CONS STAPLES $ - TOT RETURN IND",
    "MSCI WORLD HEALTH CARE $ - TOT RETURN IND",
    "MSCI EUROPE FIn.e.NCIALS $ - TOT RETURN IND",
    "MSCI WORLD IT $ - TOT RETURN IND",
    "MSCI WORLD COMMUNICATION SERVICES $ - TOT RETURN IND",
    "MSCI WORLD UTILITIES $ - TOT RETURN IND",
    "MSCI WORLD MEDIA & ENTERTAINMENT $ - TOT RETURN IND",
    "MSCI WORLD PHARM $ - TOT RETURN IND",
    "MSCI WORLD PH/BIO L SCI $ - TOT RETURN IND",
    "MSCI WORLD BANKS $ - TOT RETURN IND",
    "MSCI WORLD BIOTEC $ - TOT RETURN IND",
    "MSCI WORLD CHEMICALS $ - TOT RETURN IND",
    "MSCI WORLD REAL ESTATE $ - TOT RETURN IND",
    "MSCI WORLD SOFTWARE $ - TOT RETURN IND",

    # --- COMMODITIES ---
    "Bloomberg-Gold Sub Index TR - RETURN IND. (OFCL)",                # Gold
    "Bloomberg- Commodity TR - RETURN IND. (OFCL)"
]

## ==============================================================================
## SECTION 3: DATA PREPARATION AND MAPPING
## ==============================================================================

# Mapping for core assets (Non-Informed)
map_not = {
    "EURUSD Tasso di cambio spot - Prezzo : 1 EUR in USD": "FX_EURUSD",
    "DB Euro Overnight Rate Index": "CASH_EUR",
    "Bloomberg Global Aggregate 1-3 Year Total Return Index Unhedged EUR": "BOND_ST_AGG",
    "Bloomberg Global Agg Treasuries Total Return Index Value Unhedged USD": "BOND_GOV_GLO",
    "Bloomberg Global Agg Corporate Total Return Index Value Unhedged USD": "BOND_CORP_IG",
    "Bloomberg Global High Yield Total Return Index Value Unhedge": "BOND_HY",
    "Bloomberg Emerging Markets Sovereign TR Index Value Unhedged USD": "BOND_EM",
    "Bloomberg EuroAgg Government Total Return Index Value Unhedged EUR": "BOND_GOV_EUR",
    "MSCI Europe Gross Total Return Local Index": "EQ_EUR",
    "MSCI North America Gross Total Return Local Index": "EQ_US",
    "MSCI Pacific Gross Total Return Local Index": "EQ_PAC",
    "MSCI EM Gross Total Return Local Index": "EQ_EM",
    "Bloomberg-Gold Sub Index TR - RETURN IND. (OFCL)": "GOLD"
}

# Mapping for sector assets (Informed)
map_sectors = {
    "MSCI WORLD ENERGY $ - TOT RETURN IND": "SEC_ENERGY",
    "MSCI WORLD MATERIALS $ - TOT RETURN IND": "SEC_MAT",
    "MSCI WORLD INDUSTRIALS $ - TOT RETURN IND": "SEC_IND",
    "MSCI WORLD CONS DISCR $ - TOT RETURN IND": "SEC_DISCR",
    "MSCI WORLD CONS STAPLES $ - TOT RETURN IND": "SEC_STAPLES",
    "MSCI WORLD HEALTH CARE $ - TOT RETURN IND": "SEC_HEALTH",
    "MSCI EUROPE FIn.e.NCIALS $ - TOT RETURN IND": "SEC_FIN",
    "MSCI WORLD IT $ - TOT RETURN IND": "SEC_IT",
    "MSCI WORLD COMMUNICATION SERVICES $ - TOT RETURN IND": "SEC_COMM",
    "MSCI WORLD UTILITIES $ - TOT RETURN IND": "SEC_UTIL",
    "MSCI WORLD MEDIA & ENTERTAINMENT $ - TOT RETURN IND": "SEC_MEDIA",
    "MSCI WORLD PHARM $ - TOT RETURN IND": "SEC_PHARMA",
    "MSCI WORLD PH/BIO L SCI $ - TOT RETURN IND": "SEC_BIO",
    "MSCI WORLD BANKS $ - TOT RETURN IND": "SEC_BANKS",
    "MSCI WORLD BIOTEC $ - TOT RETURN IND": "SEC_BIOTECH",
    "MSCI WORLD CHEMICALS $ - TOT RETURN IND": "SEC_CHEM",
    "MSCI WORLD REAL ESTATE $ - TOT RETURN IND": "SEC_RE",
    "MSCI WORLD SOFTWARE $ - TOT RETURN IND": "SEC_SOFT",
    "Bloomberg- Commodity TR - RETURN IND. (OFCL)": "COMMOD_AGG"
}

# Merge maps for the 'informed' dataset
map_inf = {**map_not, **map_sectors}

# Process data and calculate log-returns
data_not = data[noninformed_assets].apply(pd.to_numeric, errors='coerce')
data_inf = data[informed_assets].apply(pd.to_numeric, errors='coerce')

logrets_not = np.log(data_not / data_not.shift(1)).dropna()
logrets_inf = np.log(data_inf / data_inf.shift(1)).dropna()

# Rename columns according to maps
logrets_not.rename(columns=map_not, inplace=True)
logrets_inf.rename(columns=map_inf, inplace=True)

# Store Cash/FX indices separately
CASH_not = logrets_inf["CASH_EUR"]
CASH_inf = logrets_inf["CASH_EUR"]

# Remove FX and Cash from optimization sets
logrets_not = logrets_not.iloc[:, 2:] 
logrets_inf = logrets_inf.iloc[:, 2:] 

logrets_inf.index = pd.to_datetime(logrets_inf.index)
logrets_not.index = pd.to_datetime(logrets_not.index)

# Train/Test Split
TRAIN_logrets_not = logrets_not.iloc[:200,:]
TRAIN_logrets_inf = logrets_inf.iloc[:200,:] 

TEST_logrets_not = logrets_not.iloc[201:,:]
TEST_logrets_inf = logrets_inf.iloc[201:,:] 

## ==============================================================================
## SECTION 4: DESCRIPTIVE STATISTICS FUNCTION
## ==============================================================================

def analyze_log_returns(df, annualization_factor=12):
    """
    Analyzes log-returns providing descriptive statistics, normality, and stationarity tests.
    """
    stats_list = []
    
    for col in df.columns:
        data = df[col].dropna()
        
        # Core statistics
        mean_ret = data.mean()
        vol = data.std()
        
        # Annualization
        ann_mean = mean_ret * annualization_factor
        ann_vol = vol * np.sqrt(annualization_factor)
        
        # Distribution moments
        sk = skew(data)
        kt = kurtosis(data) # Excess Kurtosis (Fisher)
        
        # Jarque-Bera test (Normality)
        jb_stat, jb_pvalue = jarque_bera(data)
        
        # Augmented Dickey-Fuller (Stationarity)
        adf_result = adfuller(data)
        adf_pvalue = adf_result[1]
        
        # Max Drawdown calculation (Approximated on log-returns)
        cum_rets = data.cumsum()
        running_max = cum_rets.cummax()
        drawdown = cum_rets - running_max
        max_dd = drawdown.min()

        stats_list.append({
            'Asset': col,
            'Mean (Ann.)': ann_mean,
            'Volatility (Ann.)': ann_vol,
            'Skewness': sk,
            'Kurtosis': kt,
            'JB p-value': jb_pvalue,
            'ADF p-value': adf_pvalue,
            'Max Drawdown (Log)': max_dd,
            'Sharpe Ratio (Ann.)': ann_mean / ann_vol if ann_vol != 0 else np.nan
        })
    
    return pd.DataFrame(stats_list).set_index('Asset')

# Run analysis
stats_not = analyze_log_returns(TRAIN_logrets_not)
stats_inf = analyze_log_returns(TRAIN_logrets_inf)

## ==============================================================================
## SECTION 5: STRATEGIC ASSET ALLOCATION (SAA) CONFIGURATION
## ==============================================================================

# UNIVERSAL ASSET CONFIGURATION
ASSET_COLS = [
    'BOND_ST_AGG', 'BOND_GOV_GLO', 'BOND_CORP_IG', 'BOND_HY', 'BOND_EM',
    'BOND_GOV_EUR', 'EQ_EUR', 'EQ_US', 'EQ_PAC', 'EQ_EM', 'SEC_ENERGY',
    'SEC_MAT', 'SEC_IND', 'SEC_DISCR', 'SEC_STAPLES', 'SEC_HEALTH',
    'SEC_FIN', 'SEC_IT', 'SEC_COMM', 'SEC_UTIL', 'SEC_MEDIA',
    'SEC_PHARMA', 'SEC_BIO', 'SEC_BANKS', 'SEC_BIOTECH', 'SEC_CHEM',
    'SEC_RE', 'SEC_SOFT', 'GOLD', 'COMMOD_AGG'
]

IDX = {col: i for i, col in enumerate(ASSET_COLS)}

# Dynamic identification of buckets for constraints
def get_idx_by_prefix(prefixes):
    return [i for i, name in enumerate(ASSET_COLS) if name.startswith(prefixes)]

EQUITY_MACRO = get_idx_by_prefix(('EQ_', 'SEC_'))
BOND_MACRO   = get_idx_by_prefix('BOND_')
COMMOD_MACRO = get_idx_by_prefix(('GOLD', 'COMMOD_'))

# RELATIVE BOUNDS (Strategic Base)
RELATIVE_BOUNDS = {
    "Ultra-Conservative": {
        IDX["BOND_ST_AGG"]:  [0, 0.60], IDX["BOND_GOV_GLO"]: [0, 0.5],
        IDX["BOND_CORP_IG"]: [0, 0.25], IDX["BOND_HY"]: [0.00, 0.05],
        IDX["BOND_EM"]: [0.00, 0.1], IDX["BOND_GOV_EUR"]: [0.0, 0.0],
        IDX["EQ_EUR"]: [0.10, 0.50], IDX["EQ_US"]: [0.40, 0.60],
        IDX["EQ_PAC"]: [0.00, 0.15], IDX["EQ_EM"]: [0.00, 0.15],
    },
    "Conservative": {
        IDX["BOND_ST_AGG"]: [0, 0.60], IDX["BOND_GOV_GLO"]: [0, 0.5],
        IDX["BOND_CORP_IG"]: [0, 0.25], IDX["BOND_HY"]: [0.00, 0.05],
        IDX["BOND_EM"]: [0.00, 0.1], IDX["BOND_GOV_EUR"]: [0.0, 0.0],
        IDX["EQ_EUR"]: [0.10, 0.50], IDX["EQ_US"]: [0.40, 0.60],
        IDX["EQ_PAC"]: [0.00, 0.15], IDX["EQ_EM"]: [0.00, 0.15],
    },
    "Balanced": {
        IDX["BOND_ST_AGG"]: [0, 0.60], IDX["BOND_GOV_GLO"]: [0, 0.5],
        IDX["BOND_CORP_IG"]: [0.05, 0.40], IDX["BOND_HY"]: [0.00, 0.10],
        IDX["BOND_EM"]: [0.00, 0.20], IDX["BOND_GOV_EUR"]: [0.0, 0.0],
        IDX["EQ_EUR"]: [0.10, 0.50], IDX["EQ_US"]: [0.40, 0.60],
        IDX["EQ_PAC"]: [0.00, 0.15], IDX["EQ_EM"]: [0.00, 0.15],
    },
    "Aggressive": {
        IDX["BOND_ST_AGG"]: [0, 0.60], IDX["BOND_GOV_GLO"]: [0, 0.50],
        IDX["BOND_CORP_IG"]: [0.05, 0.50], IDX["BOND_HY"]: [0.00, 0.30],
        IDX["BOND_EM"]: [0.00, 0.30], IDX["BOND_GOV_EUR"]: [0.0, 0.0],
        IDX["EQ_EUR"]: [0.10, 0.50], IDX["EQ_US"]: [0.40, 0.60],
        IDX["EQ_PAC"]: [0.00, 0.20], IDX["EQ_EM"]: [0.00, 0.20],
    },
    "Ultra-Aggressive": {
        IDX["BOND_ST_AGG"]: [0, 0.60], IDX["BOND_GOV_GLO"]: [0.05, 0.50],
        IDX["BOND_CORP_IG"]: [0.10, 0.40], IDX["BOND_HY"]: [0.00, 0.30],
        IDX["BOND_EM"]: [0.00, 0.30], IDX["BOND_GOV_EUR"]: [0.0, 0.0],
        IDX["EQ_EUR"]: [0.10, 0.50], IDX["EQ_US"]: [0.30, 0.60],
        IDX["EQ_PAC"]: [0, 0.20], IDX["EQ_EM"]: [0, 0.20],
    }
}

# CLIENT OVERLAY (Tilts & Activation)
CLIENT_MODS = {}

def add_client_constraint(profile_name, asset_name, min_delta=0.0, max_delta=0.0, active=True):
    if profile_name not in CLIENT_MODS: CLIENT_MODS[profile_name] = {}
    CLIENT_MODS[profile_name][asset_name] = {"min_delta": min_delta, "max_delta": max_delta, "active": active}

def get_universal_rel_bounds(p_name, asset_idx):
    profile_base = RELATIVE_BOUNDS.get(p_name, RELATIVE_BOUNDS.get("Balanced", {}))
    low, high = profile_base.get(asset_idx, [0.0, 1.0])
    
    asset_name = ASSET_COLS[asset_idx]
    if p_name in CLIENT_MODS and asset_name in CLIENT_MODS[p_name]:
        mod = CLIENT_MODS[p_name][asset_name]
        low = np.clip(low + mod["min_delta"], 0.0, 1.0)
        high = np.clip(high + mod["max_delta"], 0.0, 1.0)
        if low > high: high = low
    return [low, high]

def get_universal_config(profile_name, current_assets):
    target_ret_map = {"Ultra-Conservative": 0.033, "Conservative": 0.048, "Balanced": 0.063, "Aggressive": 0.083, "Ultra-Aggressive": 0.103}
    saa_map = {
        "Ultra-Conservative": [0.05, 0.15, 0.00, 0.10],
        "Conservative": [0.15, 0.35, 0.00, 0.10],
        "Balanced": [0.35, 0.50, 0.00, 0.10],
        "Aggressive": [0.50, 0.75, 0.00, 0.12],
        "Ultra-Aggressive": [0.75, 1.00, 0.00, 0.12]
    }
    
    n = len(ASSET_COLS)
    lb, ub = np.zeros(n), np.ones(n)
    
    # 1. Default Deactivation (Sectors and Aggregated Commodities)
    core_pref = ('EQ_EUR', 'EQ_US', 'EQ_PAC', 'EQ_EM', 'BOND_ST', 'BOND_GOV', 'BOND_CORP', 'BOND_HY', 'BOND_EM', 'GOLD')
    for i, name in enumerate(ASSET_COLS):
        if not name.startswith(core_pref) or name == 'BOND_GOV_EUR':
            ub[i] = 0.0
            
    # 2. Presence Check in provided DataFrame
    for i, name in enumerate(ASSET_COLS):
        if name not in current_assets: ub[i] = 0.0

    # 3. Client Overlay
    if profile_name in CLIENT_MODS:
        for a_name, mod in CLIENT_MODS[profile_name].items():
            if mod["active"] and a_name in IDX: ub[IDX[a_name]] = 1.0

    res = saa_map[profile_name]
    return {
        "profile_name": profile_name, "target_return": target_ret_map[profile_name],
        "min_equity": res[0], "max_equity": res[1], "min_commod": res[2], "max_commod": res[3],
        "lb": lb, "ub": ub
    }

## ==============================================================================
## SECTION 6: OPTIMIZATION ENGINE
## ==============================================================================

def optimize_universal(mu_s, sigma_s, config, n_restarts=7):
    lb, ub = config["lb"], config["ub"]
    p_name = config["profile_name"]
    
    active_eq = [i for i in EQUITY_MACRO if ub[i] > 0]
    active_bn = [i for i in BOND_MACRO if ub[i] > 0]
    active_cm = [i for i in COMMOD_MACRO if ub[i] > 0]
    
    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1},
        {"type": "ineq", "fun": lambda w: (w @ mu_s) - config["target_return"]},
        {"type": "ineq", "fun": lambda w: np.sum(w[active_eq]) - config["min_equity"]},
        {"type": "ineq", "fun": lambda w: config["max_equity"] - np.sum(w[active_eq])},
        {"type": "ineq", "fun": lambda w: np.sum(w[active_cm]) - config["min_commod"]},
        {"type": "ineq", "fun": lambda w: config["max_commod"] - np.sum(w[active_cm])},
    ]

    for bucket in [active_eq, active_bn]:
        for i in bucket:
            rmin, rmax = get_universal_rel_bounds(p_name, i)
            constraints.append({"type": "ineq", "fun": lambda w, idx=i, b=bucket, r=rmin: w[idx] - r * np.sum(w[b])})
            constraints.append({"type": "ineq", "fun": lambda w, idx=i, b=bucket, r=rmax: r * np.sum(w[b]) - w[idx]})

    best_w, best_vol = None, np.inf
    for _ in range(n_restarts):
        w0 = np.random.uniform(lb + 1e-6, np.minimum(ub, 1))
        w0 /= w0.sum()
        res = minimize(lambda w: np.sqrt(w @ sigma_s @ w), w0, method="SLSQP", 
                       bounds=list(zip(lb, ub)), constraints=constraints, 
                       options={"maxiter": 1500, "ftol": 1e-7})
        if res.success and res.fun < best_vol:
            best_vol, best_w = res.fun, res.x
    return best_w

# RESAMPLING ENGINE
def resample_universal_profile(mu, sigma, p_name, T, current_assets):
    config = get_universal_config(p_name, current_assets)
    weights = []
    for _ in range(100):
        mu_s = np.random.multivariate_normal(mu, sigma / T)
        w = optimize_universal(mu_s, sigma, config)
        if w is not None: weights.append(w)
    return np.mean(weights, axis=0) if weights else None

def run_universal_optimization(log_returns, profiles=None):
    if profiles is None: 
        profiles = ["Ultra-Conservative", "Conservative", "Balanced", "Aggressive", "Ultra-Aggressive"]
    if isinstance(profiles, str): 
        profiles = [profiles]
    
    # 1. Calculate annualized parameters for ACTIVE assets
    simple = np.exp(log_returns) - 1
    mu_active = simple.mean().values * 12
    sigma_active = simple.cov().values * 12
    T = len(simple)
    
    # 2. EXPANSION: Create 30-dimension mu and sigma (full universe)
    n_total = len(ASSET_COLS)
    mu_full = np.zeros(n_total)
    sigma_full = np.zeros((n_total, n_total))
    
    current_asset_names = log_returns.columns.tolist()
    indices_in_full = [IDX[name] for name in current_asset_names]
    
    mu_full[indices_in_full] = mu_active
    sigma_full[np.ix_(indices_in_full, indices_in_full)] = sigma_active
    
    results = {}

    # Table Header
    print(f"{'Profile':20} | {'Vol (Ann)':>10} | {'Ret (Ann)':>10} | {'Equity %':>10}")
    print("-" * 60)

    for p in profiles:
        w = resample_universal_profile(mu_full, sigma_full, p, T, current_asset_names)
        
        if w is not None:
            vol = np.sqrt(w @ sigma_full @ w)
            ret = w @ mu_full
            equity_comp = np.sum(w[EQUITY_MACRO])
            
            print(f"{p:20} | {vol:10.2%} | {ret:10.2%} | {equity_comp:10.2%}")
            results[p] = w
        else:
            print(f"{p:20} | Error: Optimization failed.")
            
    return pd.DataFrame(results, index=ASSET_COLS).T

## ==============================================================================
## SECTION 7: BACKTESTING ENGINE
## ==============================================================================

def backtest_portfolio(returns_df, target_weights, rebalance_period=6,
                       v_cost=0.0010, f_cost=10, initial_aum=1_000_000,
                       tolerance_band=0.01):
    """
    Backtest a static SAA portfolio (non-informed clients).
    """

    # 0. Normalise inputs
    if hasattr(target_weights, "values"):
        target_weights = target_weights.values.flatten()
    target_weights = target_weights / target_weights.sum()

    common_cols = [c for c in returns_df.columns if c in
                       {col for col, w in zip(returns_df.columns, target_weights) if w > 0}]
    active_mask = np.array([c in common_cols for c in returns_df.columns])
    tw_active = target_weights[active_mask]
    tw_active = tw_active / tw_active.sum()
    ret_active = returns_df.loc[:, active_mask]

    n_periods, n_assets = ret_active.shape

    # 1. Pre-allocate
    aum_over_time = np.zeros(n_periods)
    weights_over_time = np.zeros((n_periods, n_assets))

    current_weights = tw_active.copy()
    current_aum = float(initial_aum)
    rebal_log = []

    # 2. Main loop
    for t in range(n_periods):
        period_ret = ret_active.iloc[t].values

        # a) Update AUM and drift weights
        port_ret = np.dot(current_weights, period_ret)
        current_aum *= (1.0 + port_ret)

        # b) Drift weights calculation
        denom = 1.0 + port_ret
        if abs(denom) > 1e-10:
            current_weights = (current_weights * (1.0 + period_ret)) / denom
        
        # c) Save state
        aum_over_time[t] = current_aum
        weights_over_time[t, :] = current_weights

        # d) Rebalancing logic
        if t > 0 and t % rebalance_period == 0:
            drift = np.sum(np.abs(current_weights - tw_active))
            if drift > tolerance_band:
                trades = np.abs(tw_active - current_weights)
                var_cost = np.sum(trades) * v_cost * current_aum
                fix_cost = np.sum(trades > 1e-4) * f_cost
                total_cost = var_cost + fix_cost
                current_aum -= total_cost
                current_weights = tw_active.copy()
                rebal_log.append({
                    "date": ret_active.index[t],
                    "drift": drift,
                    "total_cost": total_cost,
                    "aum_post": current_aum
                })

    # 3. Output construction
    aum_series = pd.Series(aum_over_time, index=ret_active.index, name="AUM")
    weights_df = pd.DataFrame(weights_over_time, index=ret_active.index, columns=ret_active.columns)
    rebal_df = pd.DataFrame(rebal_log) if rebal_log else pd.DataFrame(columns=["date", "drift", "total_cost", "aum_post"])

    # 4. Metrics computation
    metrics = _compute_metrics(aum_series, annualization=12)
    return aum_series, weights_df, rebal_df, metrics

def _compute_metrics(aum_series, annualization=12):
    """Calculates annualized performance metrics."""
    rets = aum_series.pct_change().dropna()
    ann_ret = rets.mean() * annualization
    ann_vol = rets.std() * np.sqrt(annualization)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan

    rolling_max = aum_series.cummax()
    drawdown = (aum_series - rolling_max) / rolling_max
    max_dd = drawdown.min()
    
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else np.nan
    hit_rate = (rets > 0).mean()
    worst_month = rets.min()
    best_month = rets.max()

    return {
        "Ann. Return": ann_ret,
        "Ann. Volatility": ann_vol,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_dd,
        "Calmar Ratio": calmar,
        "Hit Rate": hit_rate,
        "Worst Month": worst_month,
        "Best Month": best_month,
        "N Periods": len(rets)
    }

def run_backtest_all_profiles(portfolio_weights, simple_returns, rebalance_period=6):
    """Backtest wrapper for all risk profiles."""
    results = {}
    header = f"{'Profile':20} | {'Ann.Ret':>8} | {'Ann.Vol':>8} | {'Sharpe':>7} | {'MaxDD':>8} | {'Calmar':>7} | {'Rebals':>6}"
    print(header)
    print("─" * len(header))

    for profile in portfolio_weights.index:
        w = portfolio_weights.loc[profile]
        common = [c for c in w.index if c in simple_returns.columns and w[c] > 1e-6]
        w_active = w[common].values
        w_active = w_active / w_active.sum()

        aum, weights, rebal_log, metrics = backtest_portfolio(
            returns_df = simple_returns[common],
            target_weights = pd.Series(w_active, index=common),
            rebalance_period = rebalance_period
            )
        
        results[profile] = {"aum": aum, "weights": weights, "rebal": rebal_log, "metrics": metrics}
        m = metrics
        print(f"{profile:20} | {m['Ann. Return']:>8.2%} | {m['Ann. Volatility']:>8.2%} | {m['Sharpe Ratio']:>7.2f} | {m['Max Drawdown']:>8.2%} | {m['Calmar Ratio']:>7.2f} | {len(rebal_log):>6}")
    return results

## ==============================================================================
## SECTION 8: EXECUTION AND VISUALIZATION (SAA)
## ==============================================================================

# Run SAA Optimization on Train set
portfolio_11 = run_universal_optimization(TRAIN_logrets_not)

# Backtesting (Non-Tilted Portfolios)
results_not_IS = run_backtest_all_profiles(portfolio_11, np.exp(TRAIN_logrets_not) - 1)
results_not_OOS = run_backtest_all_profiles(portfolio_11, np.exp(TEST_logrets_not) - 1)

# Equity Curve Visualization
plt.figure(figsize=(12, 6))
results_not_OOS["Ultra-Conservative"]["aum"].plot(color='royalblue', linewidth=2, label='Ultra-Conservative')
results_not_OOS["Conservative"]["aum"].plot(color='lightblue', linewidth=2, label='Conservative')
results_not_OOS["Balanced"]["aum"].plot(color='green', linewidth=2, label='Balanced')
results_not_OOS["Aggressive"]["aum"].plot(color='orange', linewidth=2, label='Aggressive')
results_not_OOS["Ultra-Aggressive"]["aum"].plot(color='red', linewidth=2, label='Ultra-Aggressive')

plt.title('Backtesting OOS: Equity Curve (All SAA Base Profiles)', fontsize=14, fontweight='bold')
plt.xlabel('Date')
plt.ylabel('Portfolio Value (AUM)')
plt.grid(True, linestyle='--', alpha=0.7)

aum_iniziale = results_not_OOS["Balanced"]["aum"].iloc[0] 
plt.axhline(y=aum_iniziale, color='black', linestyle='-', alpha=0.5, label='Initial AUM') 
plt.legend(loc='best', fontsize=10)
plt.tight_layout()
plt.show()

# Performance Table Construction
data_to_table = {profile: results_not_OOS[profile]['metrics'] for profile in results_not_OOS}
df_performance = pd.DataFrame(data_to_table).T
if 'N Periods' in df_performance.columns:
    df_performance = df_performance.drop(columns=['N Periods'])
table_performance_not = df_performance.copy()
cols_pct = ['Ann. Return', 'Ann. Volatility', 'Max Drawdown', 'Hit Rate', 'Worst Month', 'Best Month']
cols_float = ['Sharpe Ratio', 'Calmar Ratio']
for col in cols_pct:
    table_performance_not[col] = table_performance_not[col].apply(lambda x: f"{x:.2%}")
for col in cols_float:
    table_performance_not[col] = table_performance_not[col].apply(lambda x: f"{x:.2f}")

print(table_performance_not)

## ==============================================================================
## SECTION 9: BLACK-LITTERMAN (INFORMED CLIENT FRAMEWORK)
## ==============================================================================

# CONVERT VIEWS TO P AND Q MATRICES
def views_to_PQ(views, asset_cols):
    """Converts structured views into P and Q matrices."""
    n = len(asset_cols)
    idx = {name: i for i, name in enumerate(asset_cols)}
    P_rows, Q_vals = [], []
    for v in views:
        row = np.zeros(n)
        if v["type"] == "absolute":
            if v["asset"] not in idx: raise ValueError(f"Asset '{v['asset']}' not found")
            row[idx[v["asset"]]] = 1.0
        elif v["type"] == "relative":
            if v["long"] not in idx or v["short"] not in idx: raise ValueError("Asset not found")
            row[idx[v["long"]]] = +1.0
            row[idx[v["short"]]] = -1.0
        else: raise ValueError(f"Unknown view type: {v['type']}")
        P_rows.append(row)
        Q_vals.append(v["view_ret"])
    return np.array(P_rows), np.array(Q_vals)

# OMEGA CALCULATION (Confidence level)
def compute_omega(views, P, sigma, tau):
    """Builds the diagonal Omega matrix based on He & Litterman scaled by confidence."""
    scale_map = {"high": 0.5, "medium": 1.0, "low": 2.0}
    omega_diag = []
    for k, v in enumerate(views):
        scale = scale_map.get(v["confidence"], 1.0)
        base_var = tau * float(P[k] @ sigma @ P[k])
        omega_diag.append(scale * base_var)
    return np.diag(omega_diag)

# BLACK-LITTERMAN CORE ALGORITHM
def black_litterman(w_prior, sigma, views, profile_name,
                    asset_cols=ASSET_COLS, tau=None, n_obs=None,
                    lambda_override=None, n_restarts=7):
    """Complete Black-Litterman implementation with framework constraints."""

    # 1. PARAMETERS
    target_ret_map = {"Ultra-Conservative": 0.03, "Conservative": 0.045, "Balanced": 0.06, "Aggressive": 0.08, "Ultra-Aggressive": 0.10}
    saa_map = {"Ultra-Conservative": [0.05, 0.15, 0.00, 0.10], "Conservative": [0.15, 0.35, 0.00, 0.10], "Balanced": [0.35, 0.50, 0.00, 0.10], "Aggressive": [0.50, 0.75, 0.00, 0.12], "Ultra-Aggressive": [0.75, 1.00, 0.00, 0.12]}
    tau = tau if tau is not None else (1.0 / n_obs if n_obs else 0.025)
    n = len(asset_cols)
    port_var = float(w_prior @ sigma @ w_prior)
    lam = lambda_override if lambda_override is not None else (target_ret_map[profile_name] / port_var)

    # 2. IMPLICIT EQUILIBRIUM RETURNS (PRIOR)
    pi = lam * sigma @ w_prior

    # 3. P, Q, OMEGA
    P, Q = views_to_PQ(views, asset_cols)
    omega = compute_omega(views, P, sigma, tau)

    # 4. BL UPDATE
    tau_sigma_inv = np.linalg.inv(tau * sigma + np.eye(n) * 1e-8)
    omega_inv = np.linalg.inv(omega + np.eye(len(views)) * 1e-10)
    A = tau_sigma_inv + P.T @ omega_inv @ P
    b = tau_sigma_inv @ pi + P.T @ omega_inv @ Q
    mu_bl = np.linalg.solve(A, b)
    sigma_bl = sigma + np.linalg.inv(A)

    # 5. BOUNDS CONSTRUCTION (lb, ub)
    DEFAULT_CAP_NEW = 0.15
    DEFAULT_EXPANSION = 0.10
    lb, ub = np.zeros(n), np.zeros(n)
    core_pref = ('EQ_EUR', 'EQ_US', 'EQ_PAC', 'EQ_EM', 'BOND_ST', 'BOND_GOV_GLO', 'BOND_CORP', 'BOND_HY', 'BOND_EM', 'GOLD')

    for i, name in enumerate(asset_cols):
        if name.startswith(core_pref) and name != 'BOND_GOV_EUR': ub[i] = 1.0
        else: ub[i] = 0.0

    if profile_name in CLIENT_MODS:
        for asset_name, mod in CLIENT_MODS[profile_name].items():
            if mod["active"] and asset_name in IDX: ub[IDX[asset_name]] = 1.0

    for v in views:
        if v["type"] == "absolute": candidates = [(v["asset"], v.get("max_weight"), True)]
        elif v["type"] == "relative": candidates = [(v["long"], v.get("max_weight"), True), (v["short"], None, False)]
        else: continue
        for asset_name, max_w, is_bullish in candidates:
            if asset_name not in IDX: continue
            i = IDX[asset_name]
            already_in_prior = w_prior[i] > 1e-6
            if not already_in_prior:
                cap = max_w if max_w is not None else DEFAULT_CAP_NEW
                ub[i] = float(np.clip(cap, 0.0, 1.0))
            else:
                if is_bullish and v["view_ret"] > 0:
                    expansion = max_w if max_w is not None else DEFAULT_EXPANSION
                    ub[i] = float(np.clip(ub[i] + expansion, 0.0, 1.0))

    # 6. OPTIMIZATION CONSTRAINTS
    saa = saa_map[profile_name]
    active_eq = [i for i in EQUITY_MACRO if ub[i] > 0]
    active_bn = [i for i in BOND_MACRO if ub[i] > 0]
    active_cm = [i for i in COMMOD_MACRO if ub[i] > 0]

    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1},
        {"type": "ineq", "fun": lambda w: (w @ mu_bl) - target_ret_map[profile_name]},
        {"type": "ineq", "fun": lambda w: np.sum(w[active_eq]) - saa[0]},
        {"type": "ineq", "fun": lambda w: saa[1] - np.sum(w[active_eq])},
        {"type": "ineq", "fun": lambda w: np.sum(w[active_cm]) - saa[2]},
        {"type": "ineq", "fun": lambda w: saa[3] - np.sum(w[active_cm])},
    ]

    for bucket in [active_eq, active_bn]:
        for i in bucket:
            rmin, rmax = get_universal_rel_bounds(profile_name, i)
            if rmax > 0:
                constraints.append({"type": "ineq", "fun": lambda w, idx=i, b=bucket, r=rmin: w[idx] - r * np.sum(w[b])})
                constraints.append({"type": "ineq", "fun": lambda w, idx=i, b=bucket, r=rmax: r * np.sum(w[b]) - w[idx]})

    # 7. OPTIMIZATION WITH MULTI-RESTART
    best_w, best_vol = None, np.inf
    for _ in range(n_restarts):
        w0 = np.random.uniform(lb + 1e-6, np.maximum(ub, 1e-6))
        w0 /= w0.sum()
        res = minimize(fun = lambda w: np.sqrt(w @ sigma_bl @ w), x0 = w0, method = "SLSQP", 
                       bounds = list(zip(lb, ub)), constraints = constraints, 
                       options = {"maxiter": 2000, "ftol": 1e-8})
        if res.success and res.fun < best_vol:
            best_vol, best_w = res.fun, res.x

    # 8. DIAGNOSTICS
    if best_w is not None:
        ret_bl = float(best_w @ mu_bl)
        vol_bl = float(np.sqrt(best_w @ sigma_bl @ best_w))
        eq_pct = float(np.sum(best_w[EQUITY_MACRO]))
    else:
        ret_bl = vol_bl = eq_pct = np.nan

    diagnostics = {"lambda": lam, "tau": tau, "ret_bl": ret_bl, "vol_bl": vol_bl, "equity_pct": eq_pct, "optimization_ok": best_w is not None}
    return best_w, mu_bl, pi, diagnostics

## ==============================================================================
## SECTION 10: BL BACKTEST AND SCHEDULING
## ==============================================================================

# Views schedule based on market events 2023-2026
views_schedule = {
    0: [{"type": "absolute", "asset": "SEC_IT", "view_ret": 0.30, "confidence": "high"}, {"type": "relative", "long": "EQ_US", "short": "EQ_EUR", "view_ret": 0.06, "confidence": "medium"}],
    6: [{"type": "relative", "long": "SEC_IT", "short": "SEC_STAPLES", "view_ret": 0.14, "confidence": "high"}, {"type": "absolute", "asset": "GOLD", "view_ret": 0.10, "confidence": "medium"}],
    12: [{"type": "absolute", "asset": "EQ_US", "view_ret": 0.20, "confidence": "medium"}, {"type": "absolute", "asset": "BOND_ST_AGG", "view_ret": 0.05, "confidence": "high"}],
    18: [{"type": "relative", "long": "SEC_FIN", "short": "SEC_IT", "view_ret": 0.08, "confidence": "medium"}, {"type": "absolute", "asset": "GOLD", "view_ret": 0.16, "confidence": "high"}],
    24: [{"type": "absolute", "asset": "SEC_ENERGY", "view_ret": 0.24, "confidence": "high"}, {"type": "relative", "long": "EQ_US", "short": "EQ_EM", "view_ret": 0.12, "confidence": "medium"}],
    30: [{"type": "absolute", "asset": "SEC_HEALTH", "view_ret": 0.14, "confidence": "medium"}, {"type": "relative", "long": "BOND_CORP_IG", "short": "BOND_HY", "view_ret": 0.04, "confidence": "high"}],
    36: [{"type": "absolute", "asset": "SEC_SOFT", "view_ret": 0.20, "confidence": "high"}, {"type": "absolute", "asset": "SEC_RE", "view_ret": 0.10, "confidence": "medium"}]
}

# Functions for BL Execution
def _build_sigma_full(simple_returns):
    sigma_active = simple_returns.cov().values * 12
    n_total = len(ASSET_COLS)
    sigma_full = np.zeros((n_total, n_total))
    current_names = simple_returns.columns.tolist()
    indices_in_full = [IDX[name] for name in current_names]
    sigma_full[np.ix_(indices_in_full, indices_in_full)] = sigma_active
    return sigma_full, indices_in_full

# BL Optimization Wrapper
def run_bl_optimization(portfolio_30, sigma_full, views, asset_cols=ASSET_COLS, n_obs=None):
    tau = 1.0 / n_obs if n_obs else 0.025
    bl_weights, bl_mu, bl_diag = {}, {}, {}
    for profile in portfolio_30.index:
        w_prior = portfolio_30.loc[profile].values
        w_bl, mu_bl, pi, diag = black_litterman(w_prior=w_prior, sigma=sigma_full, views=views, profile_name=profile, asset_cols=asset_cols, tau=tau, n_obs=n_obs)
        bl_weights[profile] = w_bl if w_bl is not None else np.zeros(len(asset_cols))
        bl_mu[profile] = mu_bl
        bl_diag[profile] = diag
    return pd.DataFrame(bl_weights, index=asset_cols).T, pd.DataFrame(bl_mu, index=asset_cols).T, pd.DataFrame(bl_diag).T

# Backtest Function for BL
def backtest_bl_6m(logrets_inf, views_input, portfolio_30_prior, sigma_full_prior, train_size=200, rebalance_period=6, profiles=None, v_cost=0.0010, f_cost=10, initial_aum=1_000_000, tolerance_band=0.01):
    if profiles is None: profiles = ["Ultra-Conservative", "Conservative", "Balanced", "Aggressive", "Ultra-Aggressive"]
    test_data = logrets_inf.iloc[train_size:]
    n_test = len(test_data)
    test_index = test_data.index
    names = logrets_inf.columns.tolist()

    bl_weights_cache = {}
    for t_rebal in [t for t in range(n_test) if t % rebalance_period == 0]:
        views_t = views_input.get(t_rebal, [])
        if len(views_t) > 0:
            bl_w, _, _ = run_bl_optimization(portfolio_30=portfolio_30_prior, sigma_full=sigma_full_prior, views=views_t, n_obs=train_size)
        else: bl_w = portfolio_30_prior.copy()
        bl_weights_cache[t_rebal] = bl_w

    all_results, rebal_log = {}, []
    for p in profiles:
        aum_over_time = np.zeros(n_test)
        weights_over_time = np.zeros((n_test, len(names)))
        current_aum = float(initial_aum)
        current_weights = None
        rebal_log_p = []

        for t in range(n_test):
            if t % rebalance_period == 0:
                bl_w = bl_weights_cache[t]
                w_full = bl_w.loc[p].values
                active_cols = [n for n in names if w_full[IDX[n]] > 1e-6]
                w_active = np.array([w_full[IDX[n]] for n in active_cols])
                w_active /= w_active.sum()
                new_target = pd.Series(w_active, index=active_cols)

                if current_weights is None: current_weights = new_target.copy()
                else:
                    all_assets = list(set(current_weights.index) | set(new_target.index))
                    drift = np.sum(np.abs(current_weights.reindex(all_assets, fill_value=0.0) - new_target.reindex(all_assets, fill_value=0.0)))
                    if drift > tolerance_band:
                        total_cost = np.sum(np.abs(new_target.reindex(all_assets, fill_value=0.0) - current_weights.reindex(all_assets, fill_value=0.0))) * v_cost * current_aum + np.sum(np.abs(new_target.reindex(all_assets, fill_value=0.0) - current_weights.reindex(all_assets, fill_value=0.0)) > 1e-4) * f_cost
                        current_aum -= total_cost
                        current_weights = new_target.copy()
                        rebal_log_p.append({"t": t, "date": test_index[t], "profile": p, "drift": drift, "total_cost": total_cost, "aum_post": current_aum})

            period_ret = np.exp(test_data.iloc[t]) - 1
            common = [c for c in current_weights.index if c in period_ret.index]
            cw, r_t = current_weights[common].values / current_weights[common].sum(), period_ret[common].values
            port_ret = np.dot(cw, r_t)
            current_aum *= (1.0 + port_ret)
            if abs(1.0 + port_ret) > 1e-10: current_weights[common] = cw * (1.0 + r_t) / (1.0 + port_ret)
            aum_over_time[t] = current_aum
            for col in common: weights_over_time[t, names.index(col)] = current_weights[col]

        all_results[p] = {"aum": pd.Series(aum_over_time, index=test_index), "metrics": _compute_metrics(pd.Series(aum_over_time, index=test_index)), "weights": pd.DataFrame(weights_over_time, index=test_index, columns=names), "rebal": pd.DataFrame(rebal_log_p)}
    return all_results, pd.DataFrame(rebal_log)

# Final Execution
simple_train = np.exp(TRAIN_logrets_inf) - 1
sigma_full_prior, _ = _build_sigma_full(simple_train)
portfolio_30 = run_universal_optimization(TRAIN_logrets_inf)
results_bl, rebal_log = backtest_bl_6m(logrets_inf=logrets_inf, views_input=views_schedule, portfolio_30_prior=portfolio_30, sigma_full_prior=sigma_full_prior, train_size=200)

## ==============================================================================
## SECTION 11: FINAL RESULTS ANALYSIS AND PLOTTING
## ==============================================================================

# EQUITY CURVE - TAA (Black-Litterman)
plt.figure(figsize=(12, 6))
aum_iniziale_BL = 1000000
profili_colori_BL = {"Ultra-Conservative": "royalblue", "Conservative": "lightblue", "Balanced": "green", "Aggressive": "orange", "Ultra-Aggressive": "red"}

for nome, colore in profili_colori_BL.items():
    if nome in results_bl:
        series_BL = results_bl[nome]["aum"]
        display_series_BL = (series_BL / series_BL.iloc[0]) * aum_iniziale_BL
        display_series_BL.plot(color=colore, linewidth=2, label=nome)

plt.title('Backtesting OOS: Equity Curve - TAA (Black-Litterman)', fontsize=14, fontweight='bold')
plt.xlabel('Date')
plt.ylabel('AUM')
plt.grid(True, linestyle='--', alpha=0.6)
plt.axhline(y=aum_iniziale_BL, color='black', linestyle='-', alpha=0.4, label='Initial Capital') 
plt.legend(loc='upper left', fontsize=10)
plt.tight_layout()
plt.show()

# SUMMARY PERFORMANCE TABLE - BL
data_to_table_BL = {profile: results_bl[profile]['metrics'] for profile in results_bl}
df_performance_BL = pd.DataFrame(data_to_table_BL).T
df_performance_BL = df_performance_BL.reindex(["Ultra-Conservative", "Conservative", "Balanced", "Aggressive", "Ultra-Aggressive"])
if 'N Periods' in df_performance_BL.columns: df_performance_BL = df_performance_BL.drop(columns=['N Periods'])

table_performance_BL_not = df_performance_BL.copy()
for col in ['Ann. Return', 'Ann. Volatility', 'Max Drawdown', 'Hit Rate', 'Worst Month', 'Best Month']:
    table_performance_BL_not[col] = pd.to_numeric(table_performance_BL_not[col]).map("{:.2%}".format)
for col in ['Sharpe Ratio', 'Calmar Ratio']:
    table_performance_BL_not[col] = pd.to_numeric(table_performance_BL_not[col]).map("{:.2f}".format)

print(table_performance_BL_not)
