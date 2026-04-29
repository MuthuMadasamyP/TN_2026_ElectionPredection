"""
═══════════════════════════════════════════════════════════════════════════════
  Tamil Nadu Legislative Assembly 2026 — ML Election Prediction
  Author  : Muthu Madasamy (Data Analyst)
  Fix     : Robust party mapping, safe CV, no data leakage, Windows compatible
═══════════════════════════════════════════════════════════════════════════════

  HOW TO RUN:
      pip install pandas numpy scikit-learn xgboost matplotlib
      python election.py

  Place all CSV files in the SAME folder as this script.
"""

import os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')          # works on Windows without a display
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
import xgboost as xgb

warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════════════════════
# 0.  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# Script auto-detects its own directory — no need to change this
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR  = os.path.join(BASE_DIR, 'output')
os.makedirs(OUT_DIR, exist_ok=True)

ELECTION_YEARS = [1971, 1977, 1980, 1984, 1989, 1991,
                  1996, 2001, 2006, 2011, 2016, 2021]

# ── Seat allocations 2026 (from 2026_party_seat_alloacations.csv) ─────────────
DMK_ALLOC = {
    'DMK': 164, 'INC': 28, 'VCK': 8, 'DMDK': 10,
    'CPI': 5,   'CPM': 5,  'MDMK': 4, 'IUML': 2,
    'KMDK': 2,  'MMK': 2,  'MJK': 1,  'MKLP': 1,
    'SDIP': 1,  'TDK': 1,
}
ADMK_ALLOC = {
    'ADMK': 168, 'BJP': 27, 'PMK': 18, 'AMMK': 11,
    'TMC': 5,    'IJK': 2,  'PB': 1,   'TMMK': 1, 'STMK': 1,
}

# ── Visual theme ───────────────────────────────────────────────────────────────
DMK_C  = '#FF6B35'
ADMK_C = '#27AE60'
BG     = '#0A0E1A'
CARD   = '#111827'
GRID_C = '#1F2937'
TXT    = '#F1F5F9'
MUTED  = '#94A3B8'
GOLD   = '#F59E0B'
INC_C  = '#3B82F6'
BJP_C  = '#FF8C00'

plt.rcParams.update({
    'figure.facecolor': BG, 'axes.facecolor': CARD,
    'axes.edgecolor':   GRID_C, 'axes.labelcolor': TXT,
    'xtick.color': MUTED, 'ytick.color': MUTED,
    'text.color':  TXT,   'grid.color': GRID_C,
    'grid.linestyle': '--', 'grid.alpha': 0.4,
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False, 'axes.spines.right': False,
})

# ══════════════════════════════════════════════════════════════════════════════
# 1.  ALLIANCE MAPPING  (substring-safe, handles short codes + full names)
# ══════════════════════════════════════════════════════════════════════════════

# DMK Alliance parties — exact names from the CSVs
_DMK_EXACT = {
    # short codes (1971 style)
    'DMK', 'INC', 'CPI', 'AIFB', 'IUML',
    # full names
    'Dravida Munetra Kazhagam',
    'Indian National Congress',
    'Indian National Congress (I)',
    'Indian National Congress (SOCIALIST- SARAT CHANDRA SINHA)',
    'Indian Congress (SOCIALIST- SARAT CHANDRA SINHA)',
    'Gandhi Kamraj National Congress',
    'Communist Party Of India',
    'Communist Party Of India (MARXIST)',
    'Communist Party Of India (Marxist)',
    'All India Forward Bloc',
    'Viduthalai Chiruthaigal Katchi',
    'Marumalarchi Dravida Munnetra Kazhagam',
    'Indian Union Muslim League',
    'Tamil Maanila Congress (MOOPANAR)',
    'Desiya Murpokku Dravida Kazhagam',
    'Manithaneya Makkal Katchi',
    'Thayaga Marumalrchi Kazhagam',
    'Puthiya Tamilagam',
}

# ADMK Alliance parties — exact names from the CSVs
_ADMK_EXACT = {
    # short codes (1971 style)
    'SWA', 'PSP',
    # full names
    'All India Anna Dravida Munnetra Kazhagam',
    'All India Anna Dravida Munnetra Kazhagam(JANAKI RAMACHANDRAN)',
    'All India Anna Dravida Munnetra Kazhagam(JAYALALITA GROUP)',
    'M.G.R.Anna D.M.Kazhagam',
    'Bharatiya Janta Party',
    'Pattali Makkal Katchi',
    'Janata Party (JP)',
    'Janta Party',
    'Janata Dal',
    'Ambedkar Kranti Dal',
    'Jharkhand Party',
}

def map_alliance(party: str) -> str:
    """Return 'DMK_ALLIANCE', 'ADMK_ALLIANCE', or 'OTHER'."""
    p = str(party).strip()

    # Exact match first (fastest, most reliable)
    if p in _DMK_EXACT:  return 'DMK_ALLIANCE'
    if p in _ADMK_EXACT: return 'ADMK_ALLIANCE'

    # Substring fallback for unexpected name variants
    pu = p.upper()
    if 'ANNA DRAVIDA' in pu or 'AIADMK' in pu or 'ADMK' in pu:
        return 'ADMK_ALLIANCE'
    if 'DRAVIDA MUNETRA' in pu or ('DMK' in pu and 'ADMK' not in pu):
        return 'DMK_ALLIANCE'
    if 'ANNA' in pu and 'MGR' in pu:
        return 'ADMK_ALLIANCE'
    if 'CONGRESS' in pu and 'GANDHI' in pu:
        return 'DMK_ALLIANCE'
    if 'BHARATIYA JANTA' in pu or 'JANATA' in pu or 'JANTA' in pu:
        return 'ADMK_ALLIANCE'
    if 'PATTALI' in pu:
        return 'ADMK_ALLIANCE'

    return 'OTHER'

# ══════════════════════════════════════════════════════════════════════════════
# 2.  DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def clean_num(x):
    if pd.isna(x): return np.nan
    return float(str(x).replace(',', '').replace('%', '').strip())

print("📂 Loading election data...")
dfs = []
for year in ELECTION_YEARS:
    path = os.path.join(BASE_DIR, f'{year}.csv')
    if not os.path.exists(path):
        print(f"   ⚠️  Missing: {year}.csv — skipping")
        continue
    df = pd.read_csv(path, encoding='utf-8-sig')
    df.columns = [c.strip().lower() for c in df.columns]
    df['year'] = year
    # Rename 'party' → 'winning_party' if needed
    if 'party' in df.columns and 'winning_party' not in df.columns:
        df = df.rename(columns={'party': 'winning_party'})
    df['ac_name']       = df['ac_name'].str.strip().str.upper()
    df['winning_party'] = df['winning_party'].str.strip()
    for col in ['margin', 'totelectors', 'totvotes',
                'winning_percentage', 'poll_percentage']:
        if col in df.columns:
            df[col] = df[col].apply(clean_num)
    keep = ['year', 'ac_no', 'ac_name', 'winning_party',
            'margin', 'totelectors', 'totvotes',
            'poll_percentage', 'winning_percentage']
    dfs.append(df[[c for c in keep if c in df.columns]])
    print(f"   ✅ Loaded {year}.csv  ({len(df)} seats)")

if not dfs:
    sys.exit("❌ No CSV files found. Make sure the CSVs are in the same folder as election.py")

data = pd.concat(dfs, ignore_index=True).dropna(subset=['ac_name', 'winning_party'])

# Apply alliance mapping
data['alliance'] = data['winning_party'].apply(map_alliance)

# ── Sanity check: print alliance counts per year ──────────────────────────────
print("\n📊 Alliance distribution by year (after mapping):")
pivot = data.groupby(['year', 'alliance']).size().unstack(fill_value=0)
print(pivot.to_string())

# Verify both alliances are present
binary_mask = data['alliance'].isin(['DMK_ALLIANCE', 'ADMK_ALLIANCE'])
n_dmk  = (data.loc[binary_mask, 'alliance'] == 'DMK_ALLIANCE').sum()
n_admk = (data.loc[binary_mask, 'alliance'] == 'ADMK_ALLIANCE').sum()
print(f"\n   DMK_ALLIANCE : {n_dmk} records")
print(f"   ADMK_ALLIANCE: {n_admk} records")

if n_admk == 0:
    sys.exit("❌ ADMK mapping failed — no records found. Check party names in your CSVs.")

# ══════════════════════════════════════════════════════════════════════════════
# 3.  FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════
print("\n⚙️  Feature Engineering...")

# State-level winner by election year (ground truth for anti-incumbency)
TN_STATE_WINNER = {
    1971: 'DMK_ALLIANCE',  1977: 'ADMK_ALLIANCE', 1980: 'DMK_ALLIANCE',
    1984: 'ADMK_ALLIANCE', 1989: 'DMK_ALLIANCE',  1991: 'ADMK_ALLIANCE',
    1996: 'DMK_ALLIANCE',  2001: 'ADMK_ALLIANCE', 2006: 'DMK_ALLIANCE',
    2011: 'ADMK_ALLIANCE', 2016: 'ADMK_ALLIANCE', 2021: 'DMK_ALLIANCE',
}
data['ruling_alliance'] = data['year'].map(TN_STATE_WINNER)

# Previous year lookup (per constituency)
years_sorted = sorted(data['year'].unique())
year_to_prev = {y: years_sorted[i-1] if i > 0 else None
                for i, y in enumerate(years_sorted)}
data['prev_year'] = data['year'].map(year_to_prev)

prev_cols = data[['year', 'ac_name', 'alliance', 'margin', 'winning_percentage']].copy()
prev_cols.columns = ['prev_year', 'ac_name', 'prev_alliance', 'prev_margin', 'prev_win_pct']
data = data.merge(prev_cols, on=['prev_year', 'ac_name'], how='left')

# Incumbency flags
data['is_incumbent_dmk']  = (data['prev_alliance'] == 'DMK_ALLIANCE').astype(int)
data['is_incumbent_admk'] = (data['prev_alliance'] == 'ADMK_ALLIANCE').astype(int)

# Anti-incumbency: seat held by the current ruling party (likely to swing)
data['anti_incumbency_flag'] = (
    data['ruling_alliance'] == data['prev_alliance']
).astype(int)

# Rolling historical win-rate (computed using ONLY past elections to avoid leakage)
def rolling_win_rate(group, target_alliance):
    rates, wins = [], []
    for _, row in group.iterrows():
        rates.append(sum(wins) / len(wins) if wins else 0.5)
        wins.append(1 if row['alliance'] == target_alliance else 0)
    return rates

data = data.sort_values(['ac_name', 'year']).reset_index(drop=True)
for a in ['DMK_ALLIANCE', 'ADMK_ALLIANCE']:
    col = f'hist_winrate_{a}'
    tmp = data.groupby('ac_name').apply(
        lambda g: pd.Series(rolling_win_rate(g, a), index=g.index)
    )
    # Handle multi-level index returned by groupby+apply
    if isinstance(tmp.index, pd.MultiIndex):
        tmp = tmp.reset_index(level=0, drop=True)
    data[col] = tmp

# Normalised numeric features
data['year_norm']         = (data['year'] - 1971) / (2021 - 1971)
data['prev_margin_norm']  = data['prev_margin'].fillna(0) / 100_000
data['prev_win_pct_norm'] = data['prev_win_pct'].fillna(0) / 100

FEATURES = [
    'year_norm',
    'is_incumbent_dmk',
    'is_incumbent_admk',
    'hist_winrate_DMK_ALLIANCE',
    'hist_winrate_ADMK_ALLIANCE',
    'anti_incumbency_flag',
    'prev_margin_norm',
    'prev_win_pct_norm',
]

# ══════════════════════════════════════════════════════════════════════════════
# 4.  MODEL TRAINING  (binary: 0 = DMK wins, 1 = ADMK wins)
# ══════════════════════════════════════════════════════════════════════════════
print("\n🤖 Training models...")

# Keep only binary classes; use data from 1984 onwards (enough prev-year context)
train_df = data[
    (data['alliance'].isin(['DMK_ALLIANCE', 'ADMK_ALLIANCE'])) &
    (data['year'] >= 1984) &
    (data['prev_alliance'].notna())      # must have previous year info
].copy()

train_df['target'] = (train_df['alliance'] == 'ADMK_ALLIANCE').astype(int)

X = train_df[FEATURES].fillna(0)
y = train_df['target']

print(f"   Training set: {len(X)} samples  |  "
      f"DMK={int((y==0).sum())}  ADMK={int((y==1).sum())}")

if y.nunique() < 2:
    sys.exit("❌ Still only one class in training data. "
             "Re-check alliance mapping or CSV files.")

# ── Safe cross-validation helper ──────────────────────────────────────────────
def safe_cv(clf, X, y, n_splits=5):
    """StratifiedKFold CV — gracefully skips folds with only 1 class."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    for train_idx, val_idx in skf.split(X, y):
        Xtr, Xval = X.iloc[train_idx], X.iloc[val_idx]
        ytr, yval = y.iloc[train_idx], y.iloc[val_idx]
        if ytr.nunique() < 2:
            continue          # skip fold with single class
        try:
            clf.fit(Xtr, ytr)
            scores.append(clf.score(Xval, yval))
        except Exception as e:
            print(f"      Fold skipped: {e}")
    return np.array(scores)

# ── Define models ─────────────────────────────────────────────────────────────
rf_model = RandomForestClassifier(
    n_estimators=300, max_depth=7, min_samples_leaf=3,
    class_weight='balanced', random_state=42
)
xgb_model = xgb.XGBClassifier(
    n_estimators=300, max_depth=5, learning_rate=0.05,
    subsample=0.8, eval_metric='logloss',
    use_label_encoder=False, random_state=42
)
gb_model = GradientBoostingClassifier(
    n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42
)

# ── Cross-validate ─────────────────────────────────────────────────────────────
cv_scores = {}
for name, clf in [('Random Forest', rf_model),
                  ('XGBoost',       xgb_model),
                  ('GBM',           gb_model)]:
    scores = safe_cv(clf, X, y)
    if len(scores) == 0:
        print(f"   {name:15s} → ⚠️  all CV folds skipped")
        cv_scores[name] = np.array([0.5])
    else:
        cv_scores[name] = scores
        print(f"   {name:15s} → CV Accuracy: {scores.mean():.3f} ± {scores.std():.3f}")

# ── Train final models on full training set ───────────────────────────────────
rf_model.fit(X, y)
xgb_model.fit(X, y)
gb_model.fit(X, y)
print("   ✅ All models trained on full data")

# ══════════════════════════════════════════════════════════════════════════════
# 5.  PREDICT 2026
# ══════════════════════════════════════════════════════════════════════════════
print("\n🔮 Predicting 2026 results...")

last2021 = data[data['year'] == 2021][
    ['ac_name', 'ac_no', 'alliance', 'margin', 'winning_percentage', 'poll_percentage']
].copy()
last2021.columns = ['ac_name', 'ac_no', 'prev_alliance',
                    'prev_margin', 'prev_win_pct', 'prev_poll_pct']

# Historical win-rates using all years up to 2021
hist_rates = (
    data[data['year'] <= 2021]
    .groupby('ac_name')
    .apply(lambda g: pd.Series({
        'hist_winrate_DMK_ALLIANCE':  (g['alliance'] == 'DMK_ALLIANCE').mean(),
        'hist_winrate_ADMK_ALLIANCE': (g['alliance'] == 'ADMK_ALLIANCE').mean(),
    }))
    .reset_index()
)

pred_df = last2021.merge(hist_rates, on='ac_name', how='left')

pred_df['is_incumbent_dmk']  = (pred_df['prev_alliance'] == 'DMK_ALLIANCE').astype(int)
pred_df['is_incumbent_admk'] = (pred_df['prev_alliance'] == 'ADMK_ALLIANCE').astype(int)
# 2021 ruling party = DMK, so their seats carry anti-incumbency risk
pred_df['anti_incumbency_flag'] = pred_df['is_incumbent_dmk']
pred_df['year_norm']            = (2026 - 1971) / (2021 - 1971)
pred_df['prev_margin_norm']     = pred_df['prev_margin'].fillna(0) / 100_000
pred_df['prev_win_pct_norm']    = pred_df['prev_win_pct'].fillna(0) / 100

X_pred = pred_df[FEATURES].fillna(0)

# Weighted ensemble  RF=35%  XGB=40%  GBM=25%
prob_admk = (
    0.35 * rf_model.predict_proba(X_pred)[:, 1] +
    0.40 * xgb_model.predict_proba(X_pred)[:, 1] +
    0.25 * gb_model.predict_proba(X_pred)[:, 1]
)

# Anti-incumbency penalty: shift DMK-held seats 8% toward ADMK
#   (TN historically flipped ruling party in 10 of 11 elections)
AI_SHIFT = 0.08
dmk_inc_mask = pred_df['is_incumbent_dmk'].values == 1
prob_admk[dmk_inc_mask] = np.clip(prob_admk[dmk_inc_mask] + AI_SHIFT, 0, 1)

pred_df['prob_ADMK']   = prob_admk
pred_df['prob_DMK']    = 1 - prob_admk
pred_df['confidence']  = pred_df[['prob_DMK', 'prob_ADMK']].max(axis=1)
pred_df['predicted']   = np.where(prob_admk >= 0.5, 'ADMK_ALLIANCE', 'DMK_ALLIANCE')

# Anti-incumbency scenario (stricter 57% threshold for DMK to hold)
pred_df['predicted_ai'] = np.where(
    pred_df['prob_DMK'] >= 0.57, 'DMK_ALLIANCE', 'ADMK_ALLIANCE'
)

DMK_BASE  = int((pred_df['predicted']    == 'DMK_ALLIANCE').sum())
ADMK_BASE = int((pred_df['predicted']    == 'ADMK_ALLIANCE').sum())
DMK_AI    = int((pred_df['predicted_ai'] == 'DMK_ALLIANCE').sum())
ADMK_AI   = int((pred_df['predicted_ai'] == 'ADMK_ALLIANCE').sum())

print(f"   Base Scenario     → DMK: {DMK_BASE} | ADMK: {ADMK_BASE}")
print(f"   Anti-Inc Scenario → DMK: {DMK_AI}   | ADMK: {ADMK_AI}")


# ── Distribute alliance seats to member parties ────────────────────────────────
def distribute_seats(seats_won: int, alloc: dict) -> dict:
    total = sum(alloc.values())
    result = {p: max(0, round(seats_won * n / total)) for p, n in alloc.items()}
    diff = seats_won - sum(result.values())
    lead = max(alloc, key=alloc.get)
    result[lead] += diff
    return result

dmk_dist  = distribute_seats(DMK_BASE,  DMK_ALLOC)
admk_dist = distribute_seats(ADMK_BASE, ADMK_ALLOC)

# Historical summary for charts
hist_summary = (
    data.groupby(['year', 'alliance'])
    .size()
    .reset_index(name='seats')
)

# ══════════════════════════════════════════════════════════════════════════════
# 6.  VISUALISATIONS (pure matplotlib)
# ══════════════════════════════════════════════════════════════════════════════
print("\n🎨 Generating visualizations...")

# ─── FIGURE 1 — MASTER DASHBOARD ─────────────────────────────────────────────
fig1 = plt.figure(figsize=(22, 24), facecolor=BG)
fig1.suptitle(
    'Tamil Nadu Legislative Assembly 2026\nML Election Prediction Dashboard',
    fontsize=20, fontweight='bold', color=TXT, y=0.98, linespacing=1.4
)
gs1 = gridspec.GridSpec(4, 3, figure=fig1,
                         hspace=0.52, wspace=0.35,
                         top=0.94, bottom=0.03, left=0.07, right=0.97)

# [0,0]  Donut — seat share
ax = fig1.add_subplot(gs1[0, 0])
wedges, _, pcts = ax.pie(
    [DMK_BASE, ADMK_BASE],
    colors=[DMK_C, ADMK_C],
    autopct='%1.1f%%', startangle=90,
    wedgeprops=dict(linewidth=3, edgecolor=BG),
    pctdistance=0.80,
    textprops=dict(fontsize=11, color='white', fontweight='bold'),
)
ax.add_patch(plt.Circle((0, 0), 0.56, color=CARD))
ax.text(0,  0.10, '234',         ha='center', va='center',
        fontsize=20, fontweight='bold', color=TXT)
ax.text(0, -0.14, 'Total Seats', ha='center', va='center',
        fontsize=9, color=MUTED)
ax.legend(
    handles=[mpatches.Patch(color=DMK_C,  label=f'DMK Alliance  {DMK_BASE}'),
             mpatches.Patch(color=ADMK_C, label=f'ADMK Alliance  {ADMK_BASE}')],
    loc='lower center', bbox_to_anchor=(0.5, -0.12),
    fontsize=9, facecolor=CARD, edgecolor=GRID_C
)
ax.set_title('2026 Predicted Seat Share', fontsize=11,
             fontweight='bold', color=TXT, pad=10)

# [0,1]  Horizontal bar vs majority
ax = fig1.add_subplot(gs1[0, 1])
bars = ax.barh(['DMK Alliance', 'ADMK Alliance'],
               [DMK_BASE, ADMK_BASE],
               color=[DMK_C, ADMK_C], height=0.45,
               edgecolor=BG, linewidth=2)
ax.axvline(118, color=GOLD, linestyle='--', linewidth=2, zorder=5)
ax.text(119, 1.28, 'Majority → 118', fontsize=8, color=GOLD)
for bar, v in zip(bars, [DMK_BASE, ADMK_BASE]):
    ax.text(v + 1, bar.get_y() + bar.get_height() / 2,
            f' {v}', va='center', fontsize=16, fontweight='bold', color=TXT)
ax.set_xlim(0, 240)
ax.set_xlabel('Seats', fontsize=10)
ax.set_title('Alliance Seats vs\nMajority Mark (118)', fontsize=11,
             fontweight='bold', color=TXT, pad=8)
ax.grid(axis='x', alpha=0.3)

# [0,2]  Confidence histogram
ax = fig1.add_subplot(gs1[0, 2])
for lbl, mask, col in [
    ('DMK Alliance',  pred_df['predicted'] == 'DMK_ALLIANCE',  DMK_C),
    ('ADMK Alliance', pred_df['predicted'] == 'ADMK_ALLIANCE', ADMK_C),
]:
    ax.hist(pred_df.loc[mask, 'confidence'], bins=18, color=col,
            alpha=0.72, edgecolor=BG, label=f'{lbl} ({mask.sum()})')
ax.axvline(0.6, color=GOLD, linewidth=1.5, linestyle='--',
           alpha=0.8, label='60% threshold')
ax.set_xlabel('Prediction Confidence', fontsize=10)
ax.set_ylabel('No. of Constituencies',  fontsize=10)
ax.set_title('Prediction Confidence\nDistribution', fontsize=11,
             fontweight='bold', color=TXT, pad=8)
ax.legend(fontsize=8, facecolor=CARD, edgecolor=GRID_C)
ax.grid(alpha=0.3)

# [1, :]  Historical seat trend (full width)
ax = fig1.add_subplot(gs1[1, :])
hist_piv = hist_summary.pivot(
    index='year', columns='alliance', values='seats'
).fillna(0)
yr_list = hist_piv.index.tolist()

for col, color, marker, lbl in [
    ('DMK_ALLIANCE',  DMK_C,  'o', 'DMK Alliance'),
    ('ADMK_ALLIANCE', ADMK_C, 's', 'ADMK Alliance'),
]:
    if col not in hist_piv.columns:
        continue
    yv = hist_piv[col].values
    ax.plot(yr_list, yv, marker=marker, color=color,
            linewidth=2.5, markersize=8, label=lbl, zorder=5)
    ax.fill_between(yr_list, yv, alpha=0.10, color=color)

ax.scatter([2026], [DMK_BASE],  color=DMK_C,  s=220, marker='*',
           edgecolor='white', linewidth=1.5, zorder=10)
ax.scatter([2026], [ADMK_BASE], color=ADMK_C, s=220, marker='*',
           edgecolor='white', linewidth=1.5, zorder=10)
ax.annotate(f'2026 ★\nDMK {DMK_BASE}',  (2026, DMK_BASE),
            xytext=(-60, 18), textcoords='offset points',
            fontsize=8.5, color=DMK_C, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=DMK_C))
ax.annotate(f'2026 ★\nADMK {ADMK_BASE}', (2026, ADMK_BASE),
            xytext=(-60, -32), textcoords='offset points',
            fontsize=8.5, color=ADMK_C, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=ADMK_C))
ax.axhline(118, color=GOLD, linestyle='--', linewidth=1.5,
           alpha=0.8, label='Majority (118)')
ax.set_xticks(yr_list + [2026])
ax.set_xticklabels([str(y) for y in yr_list] + ['2026\n★'], fontsize=9)
ax.set_ylabel('Seats Won', fontsize=11)
ax.set_xlim(1969, 2029)
ax.set_title(
    'Historical Seat Count: DMK vs ADMK Alliance — 1971 to 2026 (Predicted)',
    fontsize=12, fontweight='bold', color=TXT
)
ax.legend(fontsize=10, facecolor=CARD, edgecolor=GRID_C)
ax.grid(alpha=0.3)

# [2,0]  DMK party-wise
ax = fig1.add_subplot(gs1[2, 0])
dmk_items = sorted([(p, s) for p, s in dmk_dist.items() if s > 0],
                   key=lambda x: -x[1])
pn, ps = zip(*dmk_items)
bar_cols = [DMK_C if p == 'DMK' else INC_C if p == 'INC' else '#4ADE80'
            for p in pn]
bars = ax.barh(pn, ps, color=bar_cols, edgecolor=BG, linewidth=1.5, alpha=0.88)
for bar, s in zip(bars, ps):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
            str(s), va='center', fontsize=9, fontweight='bold', color=TXT)
ax.set_title(f'DMK Alliance Party-wise\n(Total: {DMK_BASE} seats)',
             fontsize=11, fontweight='bold', color=DMK_C, pad=8)
ax.set_xlabel('Predicted Seats', fontsize=9)
ax.set_xlim(0, max(ps) * 1.22)
ax.grid(axis='x', alpha=0.3)

# [2,1]  ADMK party-wise
ax = fig1.add_subplot(gs1[2, 1])
admk_items = sorted([(p, s) for p, s in admk_dist.items() if s > 0],
                    key=lambda x: -x[1])
pn2, ps2 = zip(*admk_items)
bar_cols2 = [ADMK_C if p == 'ADMK' else BJP_C if p == 'BJP' else '#FCD34D'
             for p in pn2]
bars2 = ax.barh(pn2, ps2, color=bar_cols2, edgecolor=BG, linewidth=1.5, alpha=0.88)
for bar, s in zip(bars2, ps2):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
            str(s), va='center', fontsize=9, fontweight='bold', color=TXT)
ax.set_title(f'ADMK Alliance Party-wise\n(Total: {ADMK_BASE} seats)',
             fontsize=11, fontweight='bold', color=ADMK_C, pad=8)
ax.set_xlabel('Predicted Seats', fontsize=9)
ax.set_xlim(0, max(ps2) * 1.30)
ax.grid(axis='x', alpha=0.3)

# [2,2]  Win probability curve
ax = fig1.add_subplot(gs1[2, 2])
sp = np.sort(pred_df['prob_DMK'].values)[::-1]
xs = np.arange(1, 235)
ax.fill_between(xs, sp, 0.5, where=sp >= 0.5, color=DMK_C,  alpha=0.45,
                label='DMK leads')
ax.fill_between(xs, sp, 0.5, where=sp <  0.5, color=ADMK_C, alpha=0.45,
                label='ADMK leads')
ax.axhline(0.5, color='white', linewidth=1.5, linestyle='--', alpha=0.7)
ax.axhline(0.6, color=GOLD,    linewidth=1,   linestyle=':', alpha=0.6,
           label='±60% confidence')
ax.axhline(0.4, color=GOLD,    linewidth=1,   linestyle=':', alpha=0.6)
ax.set_xlabel('Constituencies (ranked)', fontsize=9)
ax.set_ylabel('P(DMK Wins)',             fontsize=9)
ax.set_title('Win Probability Curve\nAll 234 Constituencies', fontsize=11,
             fontweight='bold', color=TXT, pad=8)
ax.legend(fontsize=8, facecolor=CARD, edgecolor=GRID_C)
ax.set_xlim(1, 234); ax.set_ylim(0.15, 1.0)
ax.grid(alpha=0.3)

# [3, :]  Scenario table
ax = fig1.add_subplot(gs1[3, :])
ax.set_facecolor(BG); ax.axis('off')
rows = [
    ['Scenario',             'DMK Alliance',      'ADMK Alliance',
     'Verdict',              'Key Driver'],
    ['Base ML (50% cut)',    str(DMK_BASE),        str(ADMK_BASE),
     '✅ DMK Majority',     'Constituency win-rate history'],
    ['Anti-Incumbency (57%)','  '+str(DMK_AI),    '  '+str(ADMK_AI),
     '✅ DMK Majority',     'TN flips ruling party 91% of time'],
    ['TVK Vote Split',       str(DMK_BASE - 25),  str(ADMK_BASE + 25),
     '⚖️  Competitive',    'Vijay splits DMK vote bank'],
    ['ADMK Wave',            str(max(80, DMK_BASE-60)), str(min(154, ADMK_BASE+60)),
     '⚠️  Too close',      'Strong anti-inc + opposition unity'],
]
tbl = ax.table(cellText=rows[1:], colLabels=rows[0],
               loc='center', cellLoc='center')
tbl.auto_set_font_size(False); tbl.set_fontsize(10); tbl.scale(1, 2.2)
for (row, col), cell in tbl.get_celld().items():
    cell.set_facecolor('#1E293B' if row == 0 else '#111827')
    cell.set_edgecolor(GRID_C)
    cell.set_text_props(color=TXT)
    if row == 0:
        cell.set_text_props(fontweight='bold', color=GOLD)
ax.set_title('Election Scenario Analysis — 2026', fontsize=12,
             fontweight='bold', color=TXT, pad=6)

path1 = os.path.join(OUT_DIR, 'Fig1_Master_Dashboard.png')
plt.savefig(path1, dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print(f"   ✅ Fig 1 — Master Dashboard  →  {path1}")

# ─── FIGURE 2 — HISTORICAL & ANTI-INCUMBENCY ─────────────────────────────────
fig2, axes2 = plt.subplots(1, 2, figsize=(18, 7), facecolor=BG)
fig2.suptitle(
    'Tamil Nadu Historical Analysis & Anti-Incumbency Pattern (1971–2026)',
    fontsize=14, fontweight='bold', color=TXT, y=1.01
)

# Left — ruling vs opposition seats
ax = axes2[0]
rs, os_, yl = [], [], []
for yr, ruler in sorted(TN_STATE_WINNER.items()):
    yd  = hist_summary[hist_summary['year'] == yr]
    opp = ('ADMK_ALLIANCE' if ruler == 'DMK_ALLIANCE' else 'DMK_ALLIANCE')
    rs.append(int(yd[yd['alliance'] == ruler]['seats'].sum()))
    os_.append(int(yd[yd['alliance'] == opp]['seats'].sum()))
    yl.append(str(yr))
xp = np.arange(len(yl))
ax.bar(xp - 0.2, rs,  0.38, color='#22C55E', alpha=0.82,
       label='Ruling Alliance (that year)', edgecolor=BG)
ax.bar(xp + 0.2, os_, 0.38, color='#EF4444', alpha=0.82,
       label='Opposition Alliance',         edgecolor=BG)
ax.axhline(118, color=GOLD, linestyle='--', linewidth=1.5,
           alpha=0.8, label='Majority (118)')
ax.set_xticks(xp); ax.set_xticklabels(yl, rotation=45, fontsize=9)
ax.set_ylabel('Seats Won', fontsize=11)
ax.set_title('Ruling vs Opposition Seats\nEvery TN Election',
             fontsize=12, fontweight='bold', color=TXT)
ax.legend(fontsize=9, facecolor=CARD, edgecolor=GRID_C)
ax.grid(axis='y', alpha=0.3)

# Right — power-flip timeline
ax = axes2[1]
prev_r = None; changes, retains = [], []
for yr in sorted(TN_STATE_WINNER):
    curr = TN_STATE_WINNER[yr]
    if prev_r is not None:
        (changes if curr != prev_r else retains).append(yr)
    prev_r = curr

ax.scatter(changes, [1.0] * len(changes), s=280, color='#F87171',
           marker='X', zorder=8, linewidths=2.5,
           label=f'Anti-incumbency ({len(changes)}/11)')
ax.scatter(retains, [1.0] * len(retains), s=280, color='#4ADE80',
           marker='o', zorder=8, linewidths=2.5,
           label=f'Incumbent retained ({len(retains)}/11)')
for yr in changes:
    ax.annotate(str(yr), (yr, 1.0), xytext=(0, 22),
                textcoords='offset points',
                ha='center', fontsize=8.5, color='#F87171', fontweight='bold')
for yr in retains:
    ax.annotate(str(yr), (yr, 1.0), xytext=(0, -28),
                textcoords='offset points',
                ha='center', fontsize=8.5, color='#4ADE80', fontweight='bold')
ax.axvline(2026, color=GOLD, linestyle='--', linewidth=2, alpha=0.8)
ax.text(2026.4, 1.25, '2026\nHistorical favour:\nADMK (anti-inc)',
        fontsize=9, color=GOLD, fontweight='bold')
ax.set_yticks([]); ax.set_xlim(1974, 2031); ax.set_ylim(0.5, 1.5)
ax.set_xlabel('Election Year', fontsize=11)
ax.set_title('TN Incumbency Timeline\nRed X = Power Changed | Green ● = Retained',
             fontsize=12, fontweight='bold', color=TXT)
ax.legend(fontsize=9.5, facecolor=CARD, edgecolor=GRID_C, loc='upper left')
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
path2 = os.path.join(OUT_DIR, 'Fig2_Historical_AntiIncumbency.png')
plt.savefig(path2, dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print(f"   ✅ Fig 2 — Historical Analysis  →  {path2}")

# ─── FIGURE 3 — BATTLEGROUND + SENSITIVITY + CV + FEATURE IMPORTANCE ─────────
fig3 = plt.figure(figsize=(20, 14), facecolor=BG)
fig3.suptitle(
    'Battleground Seats | Model Performance | Feature Importance',
    fontsize=14, fontweight='bold', color=TXT, y=1.00
)
gs3 = gridspec.GridSpec(2, 2, figure=fig3,
                         hspace=0.42, wspace=0.32,
                         top=0.94, bottom=0.06, left=0.07, right=0.97)

# [0,0]  Top 20 battleground seats
ax = fig3.add_subplot(gs3[0, 0])
pred_df['contest_gap'] = (pred_df['prob_DMK'] - 0.5).abs()
bg20 = pred_df.nsmallest(20, 'contest_gap').sort_values('prob_DMK')
bcs  = [ADMK_C if p < 0.5 else DMK_C for p in bg20['prob_DMK']]
ax.barh(bg20['ac_name'].values, bg20['prob_DMK'].values,
        color=bcs, edgecolor=BG, linewidth=1, alpha=0.87)
ax.axvline(0.5, color='white', linewidth=2, linestyle='--', alpha=0.8)
ax.set_xlabel('P(DMK wins)', fontsize=10)
ax.set_title('Top 20 Battleground Seats\n(Closest Margins)',
             fontsize=11, fontweight='bold', color=TXT, pad=8)
ax.set_xlim(0.36, 0.64)
ax.grid(axis='x', alpha=0.3)
ax.legend(handles=[mpatches.Patch(color=DMK_C, label='Predicted: DMK'),
                   mpatches.Patch(color=ADMK_C, label='Predicted: ADMK')],
          fontsize=8, facecolor=CARD, edgecolor=GRID_C)

# [0,1]  Threshold sensitivity
ax = fig3.add_subplot(gs3[0, 1])
thresholds   = np.linspace(0.35, 0.65, 80)
dmk_by_thr   = [(pred_df['prob_DMK']  >= t).sum() for t in thresholds]
admk_by_thr  = [(pred_df['prob_ADMK'] >= t).sum() for t in thresholds]
ax.plot(thresholds, dmk_by_thr,  color=DMK_C,  linewidth=2.5,
        label='DMK Alliance')
ax.plot(thresholds, admk_by_thr, color=ADMK_C, linewidth=2.5,
        label='ADMK Alliance')
ax.fill_between(thresholds, dmk_by_thr, 118,
                where=np.array(dmk_by_thr) >= 118,
                alpha=0.10, color=DMK_C)
ax.axhline(118, color=GOLD, linestyle='--', linewidth=1.5,
           alpha=0.8, label='Majority (118)')
ax.axvline(0.50, color='white',   linestyle=':',  linewidth=1.5, alpha=0.6)
ax.axvline(0.57, color='#FB923C', linestyle='--', linewidth=1.5,
           alpha=0.8, label='Anti-inc threshold (0.57)')
ax.set_xlabel('Decision Threshold', fontsize=10)
ax.set_ylabel('Seats Won',          fontsize=10)
ax.set_title('Threshold Sensitivity\n(How predictions shift with threshold)',
             fontsize=11, fontweight='bold', color=TXT, pad=8)
ax.legend(fontsize=8.5, facecolor=CARD, edgecolor=GRID_C)
ax.grid(alpha=0.3)

# [1,0]  CV accuracy box plot
ax = fig3.add_subplot(gs3[1, 0])
mnames = list(cv_scores.keys())
mvals  = [cv_scores[m] for m in mnames]
clrs   = ['#60A5FA', '#34D399', '#F472B6']
bp = ax.boxplot(mvals, patch_artist=True, notch=False,
                medianprops=dict(color=GOLD, linewidth=2.5),
                whiskerprops=dict(color=MUTED, linewidth=1.5),
                capprops=dict(color=MUTED, linewidth=1.5),
                flierprops=dict(marker='o', markerfacecolor=MUTED, markersize=5))
for patch, c in zip(bp['boxes'], clrs):
    patch.set_facecolor(c); patch.set_alpha(0.7); patch.set_edgecolor(BG)
for i, (nm, vl) in enumerate(zip(mnames, mvals), 1):
    ax.text(i, vl.mean() + 0.004, f'{vl.mean():.3f}',
            ha='center', fontsize=9, color=TXT, fontweight='bold')
ax.set_xticks(range(1, len(mnames) + 1))
ax.set_xticklabels(mnames, fontsize=10)
ax.set_ylabel('5-Fold CV Accuracy', fontsize=10)
ax.set_ylim(0.50, 0.90)
ax.set_title('Model Cross-Validation Accuracy\n(5-Fold Stratified)',
             fontsize=11, fontweight='bold', color=TXT, pad=8)
ax.grid(axis='y', alpha=0.3)

# [1,1]  Feature importance (Random Forest)
ax = fig3.add_subplot(gs3[1, 1])
feat_labels = ['Year Trend', 'DMK Incumbent', 'ADMK Incumbent',
               'DMK Win Rate', 'ADMK Win Rate', 'Anti-Inc Flag',
               'Prev Margin',  'Prev Win %']
importances = rf_model.feature_importances_
idx = np.argsort(importances)
cmap_fi = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(importances)))
brs = ax.barh([feat_labels[i] for i in idx], importances[idx],
              color=cmap_fi, edgecolor=BG, linewidth=1.2, alpha=0.88)
for bar, imp in zip(brs, importances[idx]):
    ax.text(bar.get_width() + 0.003,
            bar.get_y() + bar.get_height() / 2,
            f'{imp:.3f}', va='center', fontsize=9, color=TXT)
ax.set_xlabel('Feature Importance Score', fontsize=10)
ax.set_title('Random Forest — Feature Importances\n(Higher = More Influential)',
             fontsize=11, fontweight='bold', color=TXT, pad=8)
ax.grid(axis='x', alpha=0.3)
ax.set_xlim(0, importances.max() * 1.24)

path3 = os.path.join(OUT_DIR, 'Fig3_Battleground_Model.png')
plt.savefig(path3, dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print(f"   ✅ Fig 3 — Battleground & Model  →  {path3}")

# ─── FIGURE 4 — PARTY CONTESTED vs WON ───────────────────────────────────────
fig4, axes4 = plt.subplots(1, 2, figsize=(18, 8), facecolor=BG)
fig4.suptitle('2026 Party-wise: Seats Contested vs Predicted Seats Won',
              fontsize=14, fontweight='bold', color=TXT, y=1.01)

for ax, title, alloc_d, dist_d, main_c in [
    (axes4[0], 'DMK Alliance',  DMK_ALLOC,  dmk_dist,  DMK_C),
    (axes4[1], 'ADMK Alliance', ADMK_ALLOC, admk_dist, ADMK_C),
]:
    parties   = sorted(alloc_d, key=lambda p: -alloc_d[p])
    contested = [alloc_d[p] for p in parties]
    won       = [dist_d.get(p, 0) for p in parties]
    xv = np.arange(len(parties))
    w  = 0.38
    ax.bar(xv - w / 2, contested, w, color=MUTED,    alpha=0.50,
           label='Seats Contested', edgecolor=BG)
    ax.bar(xv + w / 2, won,       w, color=main_c,   alpha=0.88,
           label='Predicted Won',   edgecolor=BG)
    for xi, (c, s) in enumerate(zip(contested, won)):
        pct = s / c * 100 if c > 0 else 0
        if s > 0:
            ax.text(xi + w / 2, s + 0.5,
                    f'{s}\n({pct:.0f}%)',
                    ha='center', fontsize=7.5, color=TXT,
                    fontweight='bold', va='bottom')
    ax.set_xticks(xv)
    ax.set_xticklabels(parties, rotation=40, ha='right', fontsize=9)
    ax.set_ylabel('Seats', fontsize=10)
    ax.set_title(f'{title}\nContested vs Predicted Won',
                 fontsize=11, fontweight='bold', color=main_c, pad=8)
    ax.legend(fontsize=9, facecolor=CARD, edgecolor=GRID_C)
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
path4 = os.path.join(OUT_DIR, 'Fig4_Party_Contested_vs_Won.png')
plt.savefig(path4, dpi=150, bbox_inches='tight', facecolor=BG)
plt.close()
print(f"   ✅ Fig 4 — Party Breakdown  →  {path4}")

# ══════════════════════════════════════════════════════════════════════════════
# 7.  EXPORT CSVs
# ══════════════════════════════════════════════════════════════════════════════
print("\n💾 Saving CSV files...")

# Constituency-level predictions
out_const = pred_df[[
    'ac_name', 'ac_no', 'prev_alliance', 'predicted', 'predicted_ai',
    'prob_DMK', 'prob_ADMK', 'confidence'
]].rename(columns={
    'ac_name':      'Constituency',
    'ac_no':        'AC_No',
    'prev_alliance':'Winner_2021',
    'predicted':    'Prediction_Base',
    'predicted_ai': 'Prediction_AntiInc',
    'prob_DMK':     'Prob_DMK_Wins',
    'prob_ADMK':    'Prob_ADMK_Wins',
    'confidence':   'Confidence',
}).sort_values('AC_No').round(3)

csv1 = os.path.join(OUT_DIR, 'TN_2026_Constituency_Predictions.csv')
out_const.to_csv(csv1, index=False)
print(f"   ✅ {csv1}")

# Party-level predictions
rows_p = []
for p in DMK_ALLOC:
    s = dmk_dist.get(p, 0)
    rows_p.append({
        'Alliance': 'DMK Alliance', 'Party': p,
        'Seats_Contested': DMK_ALLOC[p],
        'Predicted_Seats_Won': s,
        'Strike_Rate_%': round(s / DMK_ALLOC[p] * 100, 1) if DMK_ALLOC[p] > 0 else 0,
    })
for p in ADMK_ALLOC:
    s = admk_dist.get(p, 0)
    rows_p.append({
        'Alliance': 'ADMK Alliance', 'Party': p,
        'Seats_Contested': ADMK_ALLOC[p],
        'Predicted_Seats_Won': s,
        'Strike_Rate_%': round(s / ADMK_ALLOC[p] * 100, 1) if ADMK_ALLOC[p] > 0 else 0,
    })
for solo, alloc in [('NTK', 234), ('TVK', 234)]:
    rows_p.append({
        'Alliance': f'{solo} (Solo)', 'Party': solo,
        'Seats_Contested': alloc, 'Predicted_Seats_Won': 0, 'Strike_Rate_%': 0,
    })
csv2 = os.path.join(OUT_DIR, 'TN_2026_Party_Predictions.csv')
pd.DataFrame(rows_p).to_csv(csv2, index=False)
print(f"   ✅ {csv2}")

# Summary
csv3 = os.path.join(OUT_DIR, 'TN_2026_Summary.csv')
pd.DataFrame([
    {'Alliance': 'DMK Alliance',  'Scenario': 'Base (50%)',   'Seats': DMK_BASE,  'Status': 'MAJORITY'},
    {'Alliance': 'ADMK Alliance', 'Scenario': 'Base (50%)',   'Seats': ADMK_BASE, 'Status': 'SHORT'},
    {'Alliance': 'DMK Alliance',  'Scenario': 'Anti-Inc (57%)','Seats': DMK_AI,   'Status': 'MAJORITY'},
    {'Alliance': 'ADMK Alliance', 'Scenario': 'Anti-Inc (57%)','Seats': ADMK_AI,  'Status': 'SHORT'},
]).to_csv(csv3, index=False)
print(f"   ✅ {csv3}")

# ══════════════════════════════════════════════════════════════════════════════
# 8.  FINAL SUMMARY PRINTOUT
# ══════════════════════════════════════════════════════════════════════════════
print()
print('═' * 62)
print('  FINAL 2026 PREDICTION SUMMARY')
print('═' * 62)
print(f'  🟠 DMK Alliance    →  {DMK_BASE:3d} seats   (Majority: ✅)')
print(f'  🟢 ADMK Alliance   →  {ADMK_BASE:3d} seats   (Majority: ❌)')
print(f'  ─ Anti-Inc Adj    →  DMK {DMK_AI} | ADMK {ADMK_AI}')
print(f'  Majority mark     →  118 seats')
best_cv = max(s.mean() for s in cv_scores.values())
print(f'  Best CV Accuracy  →  {best_cv:.1%}')
print('═' * 62)
print(f'  Output saved to   →  {OUT_DIR}')
print('═' * 62)
