import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import PowerTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from scipy import stats
from typing import Tuple, Dict, List
import warnings

warnings.filterwarnings('ignore')


class OutlierAnalyzer:
  def __init__(self, data: pd.DataFrame):
    self.data = data.copy()
    self.original_data = data.copy()
    self.outlier_masks = {}
    self.treated_datasets = {}

  def detect_outliers_zscore(self, threshold: float = 3.0) -> pd.DataFrame:
    numeric_cols = self.data.select_dtypes(include=[np.number]).columns
    mask = pd.DataFrame(False, index=self.data.index, columns=self.data.columns)

    if len(numeric_cols) > 0:
      z_scores = self.data[numeric_cols].apply(lambda x: np.abs(stats.zscore(x, nan_policy='omit')))
      for col in numeric_cols:
        mask[col] = z_scores[col] > threshold

    mask = mask[numeric_cols]
    self.outlier_masks['zscore'] = mask
    return mask

  def detect_outliers_iqr(self, multiplier: float = 1.5) -> pd.DataFrame:
    numeric_cols = self.data.select_dtypes(include=[np.number]).columns
    mask = pd.DataFrame(False, index=self.data.index, columns=self.data.columns)

    for col in numeric_cols:
      Q1 = self.data[col].quantile(0.25)
      Q3 = self.data[col].quantile(0.75)
      IQR = Q3 - Q1
      lower = Q1 - multiplier * IQR
      upper = Q3 + multiplier * IQR
      mask[col] = (self.data[col] < lower) | (self.data[col] > upper)

    mask = mask[numeric_cols]
    self.outlier_masks['iqr'] = mask
    return mask

  def detect_outliers_isolation_forest(self, contamination: float = 0.1,
                                       random_state: int = 42) -> pd.DataFrame:
    numeric_cols = self.data.select_dtypes(include=[np.number]).columns
    mask = pd.DataFrame(False, index=self.data.index, columns=self.data.columns)

    if len(numeric_cols) > 0:
      data_clean = self.data[numeric_cols].fillna(self.data[numeric_cols].mean())
      if not data_clean.empty:
        iso = IsolationForest(contamination=contamination, random_state=random_state)
        preds = iso.fit_predict(data_clean)

        bool_preds = (preds == -1)
        for col in numeric_cols:
          mask[col] = bool_preds

    mask = mask[numeric_cols]
    self.outlier_masks['isolation_forest'] = mask
    return mask

  def detect_outliers_lof(self, n_neighbors: int = 20,
                          contamination: float = 0.1) -> pd.DataFrame:
    numeric_cols = self.data.select_dtypes(include=[np.number]).columns
    mask = pd.DataFrame(False, index=self.data.index, columns=self.data.columns)

    if len(numeric_cols) > 0:
      data_clean = self.data[numeric_cols].fillna(self.data[numeric_cols].mean())
      if not data_clean.empty:
        if len(data_clean) <= n_neighbors:
          n_neighbors = max(1, len(data_clean) - 1)
        if n_neighbors > 0:
          lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
          preds = lof.fit_predict(data_clean)

          bool_preds = (preds == -1)
          for col in numeric_cols:
            mask[col] = bool_preds

    mask = mask[numeric_cols]
    self.outlier_masks['lof'] = mask
    return mask

  def get_outlier_summary(self) -> pd.DataFrame:
    summary = {}
    for method, mask in self.outlier_masks.items():
      summary[method] = mask.sum()
    return pd.DataFrame(summary)

  def remove_outliers(self, method: str = 'iqr') -> pd.DataFrame:
    if method not in self.outlier_masks:
      raise ValueError(f"Run {method} detection first.")

    mask = self.outlier_masks[method]
    row_mask = mask.any(axis=1)

    treated = self.data[~row_mask].copy()
    self.treated_datasets[f'removed_{method}'] = treated
    return treated

  def cap_outliers(self, method: str = 'iqr', multiplier: float = 1.5) -> pd.DataFrame:
    treated = self.data.copy()
    numeric_cols = treated.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
      Q1 = treated[col].quantile(0.25)
      Q3 = treated[col].quantile(0.75)
      IQR = Q3 - Q1
      lower = Q1 - multiplier * IQR
      upper = Q3 + multiplier * IQR
      treated[col] = treated[col].clip(lower=lower, upper=upper)

    self.treated_datasets['capped'] = treated
    return treated

  def transform_outliers_log(self) -> pd.DataFrame:
    treated = self.data.copy()
    numeric_cols = treated.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
      min_val = treated[col].min()
      if min_val < 0:
        treated[col] = np.log1p(treated[col] - min_val)
      else:
        treated[col] = np.log1p(treated[col])

    self.treated_datasets['log_transform'] = treated
    return treated

  def transform_outliers_boxcox(self) -> pd.DataFrame:
    treated = self.data.copy()
    numeric_cols = treated.select_dtypes(include=[np.number]).columns

    if len(numeric_cols) > 0:
      pt = PowerTransformer(method='yeo-johnson')
      treated[numeric_cols] = pt.fit_transform(treated[numeric_cols].fillna(treated[numeric_cols].mean()))

    self.treated_datasets['boxcox_transform'] = treated
    return treated

  def evaluate_model_performance(self, target_col: str,
                                 dataset_name: str = 'original') -> Dict[str, float]:
    if dataset_name == 'original':
      df = self.data.copy()
    elif dataset_name not in self.treated_datasets:
      raise ValueError(f"Dataset '{dataset_name}' not found")
    else:
      df = self.treated_datasets.get(dataset_name, self.data.copy())

    df = df.dropna()
    if target_col not in df.columns:
      return {'r2_score': 0.0, 'r2_std': 0.0, 'mean_cv_score': 0.0}

    X = df.drop(columns=[target_col]).select_dtypes(include=[np.number])
    y = df[target_col]

    if X.empty or len(X) < 5:
      return {'r2_score': 0.0, 'r2_std': 0.0, 'mean_cv_score': 0.0}

    lr = LinearRegression()
    scores = cross_val_score(lr, X, y, cv=min(5, len(X)), scoring='r2')

    return {'r2_score': scores.mean(), 'r2_std': scores.std(), 'mean_cv_score': scores.mean()}

  def visualize_outliers_boxplot(self, figsize: Tuple[int, int] = (15, 10)):
    numeric_cols = self.data.select_dtypes(include=[np.number]).columns[:6]
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()

    for idx, col in enumerate(numeric_cols):
      sns.boxplot(y=self.data[col], ax=axes[idx], color='skyblue')
      axes[idx].set_title(f'Box Plot: {col}', fontweight='bold')

    for idx in range(len(numeric_cols), len(axes)):
      axes[idx].axis('off')

    plt.suptitle('Box Plots for Outlier Detection', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig

  def visualize_outliers_violin(self, figsize: Tuple[int, int] = (15, 10)):
    numeric_cols = self.data.select_dtypes(include=[np.number]).columns[:6]
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()

    for idx, col in enumerate(numeric_cols):
      sns.violinplot(y=self.data[col], ax=axes[idx], color='lightgreen')
      axes[idx].set_title(f'Violin Plot: {col}', fontweight='bold')

    for idx in range(len(numeric_cols), len(axes)):
      axes[idx].axis('off')

    plt.suptitle('Violin Plots for Distribution Analysis', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig

  def visualize_outliers_scatter(self, x_col: str, y_col: str,
                                 figsize: Tuple[int, int] = (15, 10)):
    if x_col not in self.data.columns or y_col not in self.data.columns:
      print(f"Columns '{x_col}' or '{y_col}' not found")
      return None

    methods = list(self.outlier_masks.keys())[:4]
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    for idx, method in enumerate(methods):
      ax = axes[idx]
      mask = self.outlier_masks[method].any(axis=1)

      sns.scatterplot(x=self.data[x_col], y=self.data[y_col], hue=mask,
                      palette={False: 'blue', True: 'red'}, ax=ax, alpha=0.6)
      ax.set_title(f'{method.replace("_", " ").title()} Outliers', fontweight='bold')
      if ax.get_legend():
        ax.get_legend().set_title('Outlier')

    for idx in range(len(methods), len(axes)):
      axes[idx].axis('off')

    plt.suptitle(f'Outlier Scatter Plots: {x_col} vs {y_col}', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig

  def visualize_3d_outliers(self, x_col: str, y_col: str, z_col: str,
                            method: str = 'isolation_forest',
                            figsize: Tuple[int, int] = (12, 8)):
    if x_col not in self.data.columns or y_col not in self.data.columns or z_col not in self.data.columns:
      print(f"Columns not found")
      return None

    if method not in self.outlier_masks:
      print(f"Method '{method}' not found")
      return None

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    mask = self.outlier_masks.get(method).any(axis=1)

    ax.scatter(self.data.loc[~mask, x_col], self.data.loc[~mask, y_col], self.data.loc[~mask, z_col],
               c='blue', label='Inliers', alpha=0.6)
    ax.scatter(self.data.loc[mask, x_col], self.data.loc[mask, y_col], self.data.loc[mask, z_col],
               c='red', label='Outliers', s=50)

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_zlabel(z_col)
    ax.set_title(f'3D Outlier Visualization ({method.replace("_", " ").title()})', fontweight='bold')
    ax.legend()
    return fig

  def visualize_treatment_comparison(self, column: str,
                                     figsize: Tuple[int, int] = (15, 10)):
    if column not in self.data.columns:
      print(f"Column '{column}' not found")
      return

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()

    # Original data
    ax = axes[0]
    self.original_data[column].hist(bins=30, ax=ax, color='skyblue', edgecolor='black', alpha=0.7)
    ax.axvline(self.original_data[column].mean(), color='red', linestyle='--', linewidth=2, label='Mean')
    ax.set_title('Original Data', fontweight='bold')
    ax.set_xlabel(column)
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    # Treated datasets
    treatment_names = list(self.treated_datasets.keys())[:5]
    for idx, name in enumerate(treatment_names, start=1):
      ax = axes[idx]
      treated_data = self.treated_datasets[name]
      if column in treated_data.columns:
        treated_data[column].hist(bins=30, ax=ax, color='lightcoral',
                                  edgecolor='black', alpha=0.7)
        ax.axvline(treated_data[column].mean(), color='red',
                   linestyle='--', linewidth=2, label='Mean')
        ax.set_title(f'{name.replace("_", " ").title()}', fontweight='bold')
        ax.set_xlabel(column)
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

    # Hide unused subplots
    for idx in range(len(treatment_names) + 1, len(axes)):
      axes[idx].axis('off')

    plt.suptitle(f'Treatment Comparison: {column}', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig

  def visualize_qq_plots(self, figsize: Tuple[int, int] = (15, 10)):
    numeric_cols = self.data.select_dtypes(include=[np.number]).columns[:6]

    if len(numeric_cols) == 0:
      print("No numeric columns to visualize")
      return

    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()

    for idx, col in enumerate(numeric_cols):
      ax = axes[idx]

      stats.probplot(self.data[col].dropna(), dist="norm", plot=ax)
      ax.set_title(f'Q-Q Plot: {col}', fontweight='bold')
      ax.grid(alpha=0.3)

    # Hide unused subplots
    for idx in range(len(numeric_cols), len(axes)):
      axes[idx].axis('off')

    plt.suptitle('Q-Q Plots for Normality Assessment', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    return fig


def create_sample_dataset_with_outliers() -> pd.DataFrame:
  np.random.seed(42)
  n_samples = 300

  data = {
    'feature1': np.random.normal(100, 15, n_samples),
    'feature2': np.random.exponential(50, n_samples),
    'feature3': np.random.normal(500, 100, n_samples),
    'feature4': np.random.uniform(0, 100, n_samples),
    'target': np.random.normal(200, 30, n_samples)
  }

  df = pd.DataFrame(data)
  df['target'] = 0.5 * df['feature1'] + 0.3 * df['feature2'] + np.random.normal(0, 10, n_samples)

  outlier_indices = np.random.choice(df.index, size=15, replace=False)
  df.loc[outlier_indices, 'feature1'] = np.random.uniform(200, 300, len(outlier_indices))

  outlier_indices = np.random.choice(df.index, size=10, replace=False)
  df.loc[outlier_indices, 'feature2'] = np.random.uniform(200, 400, len(outlier_indices))

  outlier_indices = np.random.choice(df.index, size=20, replace=False)
  low_outliers = outlier_indices[:len(outlier_indices) // 2]
  high_outliers = outlier_indices[len(outlier_indices) // 2:]
  df.loc[low_outliers, 'feature3'] = np.random.uniform(0, 100, len(low_outliers))
  df.loc[high_outliers, 'feature3'] = np.random.uniform(1000, 1500, len(high_outliers))

  return df


def main():
  print("=" * 80)
  print("Task 2: Outlier Detection and Treatment with Advanced Visualization")
  print("=" * 80)
  print()

  print("Creating sample dataset with outliers...")
  df = create_sample_dataset_with_outliers()
  print(f"Dataset created: {df.shape[0]} rows, {df.shape[1]} columns")
  print()

  analyzer = OutlierAnalyzer(df)

  print("1. OUTLIER DETECTION")
  print("-" * 80)

  zscore_outliers = analyzer.detect_outliers_zscore(threshold=3.0)
  print(f"     Outliers detected: {zscore_outliers.sum().sum()}")

  iqr_outliers = analyzer.detect_outliers_iqr(multiplier=1.5)
  print(f"     Outliers detected: {iqr_outliers.sum().sum()}")

  iso_outliers = analyzer.detect_outliers_isolation_forest(contamination=0.1)
  print(f"     Outliers detected: {iso_outliers.sum().sum()}")

  lof_outliers = analyzer.detect_outliers_lof(n_neighbors=20, contamination=0.1)
  print(f"     Outliers detected: {lof_outliers.sum().sum()}")
  print()

  print("2. OUTLIER SUMMARY")
  print("-" * 80)
  summary = analyzer.get_outlier_summary()
  print(summary)
  print()

  print("3. OUTLIER TREATMENT")
  print("-" * 80)

  removed_data = analyzer.remove_outliers(method='iqr')
  print(f"     Rows after removal: {len(removed_data)} (removed {len(df) - len(removed_data)})")

  capped_data = analyzer.cap_outliers(multiplier=1.5)
  print(f"     Data capped at IQR bounds")

  log_data = analyzer.transform_outliers_log()
  print(f"     Log transformation applied")

  boxcox_data = analyzer.transform_outliers_boxcox()
  print(f"     Box-Cox/Yeo-Johnson transformation applied")
  print()

  print("4. MODEL PERFORMANCE COMPARISON")
  print("-" * 80)

  datasets_to_evaluate = ['original', 'removed_iqr', 'capped', 'log_transform']
  for dataset_name in datasets_to_evaluate:
    try:
      performance = analyzer.evaluate_model_performance('target', dataset_name)
      print(f"  {dataset_name.replace('_', ' ').title()}:")
      print(f"    R² Score: {performance['r2_score']:.4f} (±{performance['r2_std']:.4f})")
    except Exception as e:
      print(f"  {dataset_name}: Could not evaluate - {str(e)}")
  print()

  print("5. GENERATING VISUALIZATIONS")
  print("-" * 80)

  analyzer.visualize_outliers_boxplot()
  plt.savefig('task2_boxplots_2.png', dpi=300, bbox_inches='tight')
  print("  Saved: task2_boxplots_2.png")

  analyzer.visualize_outliers_violin()
  plt.savefig('task2_violin_plots_2.png', dpi=300, bbox_inches='tight')
  print("  Saved: task2_violin_plots_2.png")

  analyzer.visualize_outliers_scatter('feature1', 'feature2', figsize=(15, 10))
  plt.savefig('task2_scatter_outliers_2.png', dpi=300, bbox_inches='tight')
  print("  Saved: task2_scatter_outliers_2.png")

  analyzer.visualize_3d_outliers('feature1', 'feature2', 'feature3', method='isolation_forest')
  plt.savefig('task2_3d_outliers_2.png', dpi=300, bbox_inches='tight')
  print("  Saved: task2_3d_outliers_2.png")

  analyzer.visualize_treatment_comparison('feature1')
  plt.savefig('task2_treatment_comparison_2.png', dpi=300, bbox_inches='tight')
  print("  Saved: task2_treatment_comparison_2.png")

  analyzer.visualize_qq_plots()
  plt.savefig('task2_qq_plots_2.png', dpi=300, bbox_inches='tight')
  print("  Saved: task2_qq_plots_2.png")
  print()

  return analyzer


if __name__ == "__main__":
  analyzer = main()
