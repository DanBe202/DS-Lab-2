"""
Task 6: Real-World Case Study - Complete Data Preprocessing Challenge

This module implements a complete end-to-end data preprocessing workflow
for a complex synthetic dataset with multiple data quality issues.
Demonstrates best practices for EDA, preprocessing, and model evaluation.

Author: MLDS Course
Date: December 2025
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
from sklearn.utils import resample
from scipy import stats
from typing import Tuple, Dict, List, Optional, Any
from datetime import datetime, timedelta
import warnings
import json

warnings.filterwarnings('ignore')


class DataPreprocessor:
    """
    Comprehensive data preprocessor for real-world case study.
    Handles EDA, data quality assessment, and preprocessing.
    """
    
    def __init__(self, data: pd.DataFrame, target_col: str):
        self.original_data = data.copy()
        self.data = data.copy()
        self.target_col = target_col
        self.preprocessed_data = None
        self.quality_report = {}
        self.preprocessing_steps = []

    def assess_data_quality(self) -> Dict[str, Any]:
        """Perform comprehensive data quality assessment."""
        report = {
            'shape': self.data.shape,
            'total_samples': len(self.data),
            'total_features': len(self.data.columns),
            'memory_usage_mb': self.data.memory_usage(deep=True).sum() / 1024**2,
            'duplicates': int(self.data.duplicated().sum()),
            'missing_values': {},
            'data_types': {},
            'numeric_features': [],
            'categorical_features': [],
            'datetime_features': [],
            'target_distribution': {}
        }

        # Missing values analysis
        total_missing = int(self.data.isnull().sum().sum())
        missing_by_col = self.data.isnull().sum()
        missing_by_col = missing_by_col[missing_by_col > 0].to_dict()
        pct_missing = (self.data.isnull().sum() / len(self.data) * 100).to_dict()

        report['missing_values'] = {
            'total': total_missing,
            'by_column': missing_by_col,
            'percentage_by_column': pct_missing
        }

        # Data types
        report['numeric_features'] = self.data.select_dtypes(include=[np.number]).columns.tolist()
        report['categorical_features'] = self.data.select_dtypes(exclude=[np.number, 'datetime64[ns]', 'datetime64']).columns.tolist()
        report['datetime_features'] = self.data.select_dtypes(include=['datetime64[ns]', 'datetime64']).columns.tolist()

        # Target distribution
        if self.target_col in self.data.columns:
            target_counts = self.data[self.target_col].value_counts().to_dict()
            report['target_distribution'] = target_counts

            if len(target_counts) > 1:
                counts = list(target_counts.values())
                report['class_balance'] = min(counts) / max(counts)
            else:
                report['class_balance'] = 1.0

        self.quality_report = report
        return report

    def print_quality_report(self):
        """Print formatted data quality report."""
        if not self.quality_report:
            print("Run assess_data_quality() first.")
            return

        print("=" * 80)
        print("DATA QUALITY ASSESSMENT REPORT")
        print("=" * 80)
        print()

        print(f"Dataset Shape: {self.quality_report['shape']}")
        print(f"Total Samples: {self.quality_report['total_samples']:,}")
        print(f"Total Features: {self.quality_report['total_features']}")
        print(f"Memory Usage: {self.quality_report['memory_usage_mb']:.2f} MB")
        print(f"Duplicate Rows: {self.quality_report['duplicates']}")
        print()

        print("MISSING VALUES:")
        print(f"  Total Missing: {self.quality_report['missing_values']['total']:,}")
        if self.quality_report['missing_values']['by_column']:
            print("  Columns with Missing Values:")
            for col, count in self.quality_report['missing_values']['by_column'].items():
                pct = self.quality_report['missing_values']['percentage_by_column'][col]
                print(f"    {col}: {count} ({pct:.2f}%)")
        print()

        print("FEATURE TYPES:")
        print(f"  Numeric: {len(self.quality_report['numeric_features'])}")
        print(f"  Categorical: {len(self.quality_report['categorical_features'])}")
        print(f"  Datetime: {len(self.quality_report['datetime_features'])}")
        print()

        if self.quality_report['target_distribution']:
            print("TARGET DISTRIBUTION:")
            for cls, count in self.quality_report['target_distribution'].items():
                pct = (count / self.quality_report['total_samples']) * 100
                print(f"  Class {cls}: {count} ({pct:.2f}%)")
            print(f"  Class Balance Ratio: {self.quality_report['class_balance']:.3f}")
        print()

    def handle_missing_values(self, strategy: str = 'auto') -> pd.DataFrame:
        """Handle missing values with specified strategy."""
        data = self.data.copy()

        numeric_cols = [col for col in self.quality_report['numeric_features'] if col != self.target_col]
        categorical_cols = [col for col in self.quality_report['categorical_features'] if col != self.target_col]

        if strategy == 'auto' or strategy == 'simple':
            if numeric_cols:
                num_imputer = SimpleImputer(strategy='median')
                data[numeric_cols] = num_imputer.fit_transform(data[numeric_cols])

            if categorical_cols:
                cat_imputer = SimpleImputer(strategy='most_frequent')
                data[categorical_cols] = cat_imputer.fit_transform(data[categorical_cols])

        elif strategy == 'knn':
            if numeric_cols:
                num_imputer = KNNImputer(n_neighbors=5)
                data[numeric_cols] = num_imputer.fit_transform(data[numeric_cols])
            if categorical_cols:
                cat_imputer = SimpleImputer(strategy='most_frequent')
                data[categorical_cols] = cat_imputer.fit_transform(data[categorical_cols])

        elif strategy == 'iterative':
            if numeric_cols:
                num_imputer = IterativeImputer(random_state=42)
                data[numeric_cols] = num_imputer.fit_transform(data[numeric_cols])
            if categorical_cols:
                cat_imputer = SimpleImputer(strategy='most_frequent')
                data[categorical_cols] = cat_imputer.fit_transform(data[categorical_cols])

        self.data = data
        self.preprocessing_steps.append(f"Missing values handled: {strategy}")
        return data

    def detect_and_handle_outliers(self, method: str = 'iqr', action: str = 'cap') -> pd.DataFrame:
        """Detect and handle outliers."""
        data = self.data.copy()
        outliers_count = 0
        numeric_cols = [col for col in self.quality_report['numeric_features'] if col != self.target_col]

        if method == 'iqr':
            for col in numeric_cols:
                Q1 = data[col].quantile(0.25)
                Q3 = data[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                outliers = (data[col] < lower_bound) | (data[col] > upper_bound)
                outliers_count += outliers.sum()

                if action == 'cap':
                    data[col] = np.clip(data[col], lower_bound, upper_bound)
                elif action == 'drop':
                    data = data[~outliers]

        elif method == 'zscore':
            for col in numeric_cols:
                z_scores = np.abs(stats.zscore(data[col], nan_policy='omit'))
                outliers = z_scores > 3
                outliers_count += outliers.sum()

                if action == 'cap':
                    median = data[col].median()
                    std = data[col].std()
                    data[col] = np.clip(data[col], median - 3*std, median + 3*std)
                elif action == 'drop':
                    data = data[~outliers]

        self.data = data
        self.preprocessing_steps.append(
            f"Outliers handled: {method} method, {action} action, {outliers_count} outliers"
        )
        return data

    def encode_categorical_features(self, method: str = 'auto') -> pd.DataFrame:
        """Encode categorical features."""
        data = self.data.copy()

        # Handle datetimes first
        for col in self.quality_report['datetime_features']:
            if col in data.columns:
                data[f'{col}_year'] = data[col].dt.year
                data[f'{col}_month'] = data[col].dt.month
                data[f'{col}_day'] = data[col].dt.day
                data = data.drop(columns=[col])

        categorical_cols = [col for col in self.quality_report['categorical_features'] if col != self.target_col]

        if method == 'auto' or method == 'label':
            le = LabelEncoder()
            for col in categorical_cols:
                data[col] = le.fit_transform(data[col].astype(str))
        elif method == 'onehot':
            data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

        self.data = data
        self.preprocessing_steps.append(f"Categorical encoding: {method}, datetime features extracted")
        return data

    def engineer_features(self, domain_features: bool = True) -> pd.DataFrame:
        """Engineer new features."""
        data = self.data.copy()
        numeric_cols = [col for col in self.quality_report['numeric_features'] if col != self.target_col]
        new_features = []

        if len(numeric_cols) >= 2:
            data['num_features_mean'] = data[numeric_cols].mean(axis=1)
            data['num_features_std'] = data[numeric_cols].std(axis=1)
            data['num_features_max'] = data[numeric_cols].max(axis=1)
            data['num_features_min'] = data[numeric_cols].min(axis=1)
            new_features.extend(['num_features_mean', 'num_features_std', 'num_features_max', 'num_features_min'])

        # Interaction feature (safely taking the first two)
        if len(numeric_cols) >= 2:
            col1, col2 = numeric_cols[0], numeric_cols[1]
            data[f'{col1}_mult_{col2}'] = data[col1] * data[col2]
            new_features.append(f'{col1}_mult_{col2}')

        self.data = data
        self.preprocessing_steps.append(f"Feature engineering: {len(new_features)} new features created")
        return data

    def scale_features(self, method: str = 'standard') -> pd.DataFrame:
        """Scale numerical features."""
        data = self.data.copy()
        numeric_cols = [col for col in data.select_dtypes(include=[np.number]).columns if col != self.target_col]

        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            scaler = StandardScaler()

        if numeric_cols:
            data[numeric_cols] = scaler.fit_transform(data[numeric_cols])

        self.data = data
        self.preprocessing_steps.append(f"Feature scaling: {method}")
        return data

    def select_features(self, method: str = 'kbest', n_features: int = 20) -> pd.DataFrame:
        """Select most important features."""
        data = self.data.copy()

        if self.target_col not in data.columns:
            return data

        X = data.drop(columns=[self.target_col])
        y = data[self.target_col]

        n_features = min(n_features, X.shape[1])

        if method == 'kbest':
            selector = SelectKBest(score_func=f_classif, k=n_features)
        elif method == 'mutual_info':
            selector = SelectKBest(score_func=mutual_info_classif, k=n_features)
        else:
            return data

        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()].tolist()

        data = pd.DataFrame(X_selected, columns=selected_features)
        data[self.target_col] = y.values

        self.data = data
        self.preprocessing_steps.append(
            f"Feature selection: {method}, {len(selected_features)} features selected"
        )
        return data

    def balance_classes(self, method: str = 'undersample') -> pd.DataFrame:
        """Balance class distribution."""
        data = self.data.copy()

        if self.target_col not in data.columns:
            return data

        class_counts = data[self.target_col].value_counts()
        majority_class = class_counts.idxmax()
        minority_class = class_counts.idxmin()

        df_majority = data[data[self.target_col] == majority_class]
        df_minority = data[data[self.target_col] == minority_class]

        if method == 'undersample':
            df_majority_downsampled = resample(
                df_majority,
                replace=False,
                n_samples=len(df_minority),
                random_state=42
            )
            data = pd.concat([df_majority_downsampled, df_minority])

        elif method == 'oversample':
            df_minority_upsampled = resample(
                df_minority,
                replace=True,
                n_samples=len(df_majority),
                random_state=42
            )
            data = pd.concat([df_majority, df_minority_upsampled])

        self.data = data
        self.preprocessing_steps.append(f"Class balancing: {method}")
        return data

    def preprocess_pipeline(self, config: Dict[str, Any]) -> pd.DataFrame:
        """Execute complete preprocessing pipeline."""
        print("Starting preprocessing pipeline...")
        print("-" * 80)

        if config.get('handle_missing', True):
            print("Step 1: Handling missing values...")
            self.data = self.handle_missing_values(config.get('missing_strategy', 'auto'))

        if config.get('handle_outliers', True):
            print("Step 2: Handling outliers...")
            self.data = self.detect_and_handle_outliers(
                method=config.get('outlier_method', 'iqr'),
                action=config.get('outlier_action', 'cap')
            )

        if config.get('encode_categorical', True):
            print("Step 3: Encoding categorical features...")
            self.data = self.encode_categorical_features(config.get('encoding_method', 'auto'))

        if config.get('engineer_features', True):
            print("Step 4: Engineering features...")
            self.data = self.engineer_features()

        if config.get('scale_features', True):
            print("Step 5: Scaling features...")
            self.data = self.scale_features(config.get('scaling_method', 'standard'))

        if config.get('select_features', False):
            print("Step 6: Selecting features...")
            self.data = self.select_features(
                method=config.get('selection_method', 'kbest'),
                n_features=config.get('n_features', 20)
            )

        if config.get('balance_classes', False):
            print("Step 7: Balancing classes...")
            self.data = self.balance_classes(config.get('balance_method', 'undersample'))

        print("-" * 80)
        print(f"Preprocessing complete! Final shape: {self.data.shape}")
        print()

        self.preprocessed_data = self.data.copy()
        return self.data

    def visualize_eda(self, figsize: Tuple[int, int] = (20, 15)):
        """Create comprehensive EDA visualizations."""
        numeric_cols = [col for col in self.quality_report['numeric_features'] if col != self.target_col]
        categorical_cols = self.quality_report['categorical_features']

        n_rows, n_cols = 3, 3
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(n_rows, n_cols, hspace=0.3, wspace=0.3)

        for i, col in enumerate(numeric_cols[:6]):
            row, col_idx = i // 3, i % 3
            ax = fig.add_subplot(gs[row, col_idx])
            self.original_data[col].hist(bins=30, ax=ax, edgecolor='black', alpha=0.7)
            ax.set_title(f'Distribution: {col}', fontweight='bold')
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
            ax.grid(alpha=0.3)

        for i, col in enumerate(categorical_cols[:3]):
            if i + 6 < 9:
                ax = fig.add_subplot(gs[2, i])
                value_counts = self.original_data[col].value_counts().head(10)
                value_counts.plot(kind='bar', ax=ax, color='steelblue', edgecolor='black')
                ax.set_title(f'Distribution: {col}', fontweight='bold')
                ax.set_xlabel('Category')
                ax.set_ylabel('Count')
                ax.tick_params(axis='x', rotation=45)
                ax.grid(alpha=0.3, axis='y')

        plt.suptitle('Exploratory Data Analysis - Feature Distributions',
                    fontsize=16, fontweight='bold', y=0.995)
        return fig

    def visualize_correlations(self, figsize: Tuple[int, int] = (12, 10)):
        """Visualize feature correlations."""
        numeric_data = self.original_data.select_dtypes(include=[np.number])
        if numeric_data.shape[1] > 20:
            numeric_data = numeric_data.iloc[:, :20]

        fig, ax = plt.subplots(figsize=figsize)
        corr_matrix = numeric_data.corr()
        sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
                   center=0, square=True, ax=ax, cbar_kws={'label': 'Correlation'})

        ax.set_title('Feature Correlation Matrix', fontweight='bold', pad=20, fontsize=14)
        plt.tight_layout()
        return fig

    def visualize_missing_patterns(self, figsize: Tuple[int, int] = (12, 8)):
        """Visualize missing value patterns."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        missing_matrix = self.original_data.isnull()

        if len(missing_matrix) > 500:
            missing_matrix = missing_matrix.sample(500, random_state=42)

        sns.heatmap(missing_matrix, cbar=True, yticklabels=False, cmap='viridis', ax=ax1)
        ax1.set_title('Missing Values Heatmap\n(Yellow = Missing)', fontweight='bold', pad=10)
        ax1.set_xlabel('Features')
        ax1.set_ylabel('Samples')

        missing_counts = self.original_data.isnull().sum()
        missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=False)

        if len(missing_counts) > 0:
            missing_counts.plot(kind='barh', ax=ax2, color='coral', edgecolor='black')
            ax2.set_title('Missing Values by Feature', fontweight='bold', pad=10)
            ax2.set_xlabel('Count')
            ax2.set_ylabel('Feature')
            ax2.grid(alpha=0.3, axis='x')
        else:
            ax2.text(0.5, 0.5, 'No Missing Values', ha='center', va='center', fontsize=14)
            ax2.set_title('Missing Values by Feature', fontweight='bold', pad=10)

        plt.tight_layout()
        return fig

    def visualize_pca(self, n_components: int = 2, figsize: Tuple[int, int] = (10, 8)):
        """Visualize PCA dimensionality reduction."""
        if self.target_col not in self.data.columns:
            raise ValueError("Target column not found")

        X = self.data.drop(columns=[self.target_col])
        y = self.data[self.target_col]

        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X)

        fig, ax = plt.subplots(figsize=figsize)
        for class_val in y.unique():
            mask = y == class_val
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1], label=f'Class {class_val}', alpha=0.6, s=50)

        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)', fontweight='bold')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)', fontweight='bold')
        ax.set_title('PCA - Feature Space Visualization', fontweight='bold', pad=20, fontsize=14)
        ax.legend()
        ax.grid(alpha=0.3)

        plt.tight_layout()
        return fig

    def generate_report(self, filepath: str = 'preprocessing_report.json'):
        """Generate comprehensive preprocessing report."""
        report = {
            'data_quality': self.quality_report,
            'preprocessing_steps': self.preprocessing_steps,
            'final_shape': self.data.shape,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        print(f"Report saved to {filepath}")


def create_complex_synthetic_dataset(n_samples: int = 2000,
                                     n_features: int = 25,
                                     missing_rate: float = 0.15,
                                     outlier_rate: float = 0.08,
                                     imbalance_ratio: float = 0.3,
                                     random_state: int = 42) -> pd.DataFrame:
    np.random.seed(random_state)
    numeric_features = {}
    for i in range(n_features):
        if i % 3 == 0:
            numeric_features[f'feature_{i}'] = np.random.randn(n_samples) * 10 + 50
        elif i % 3 == 1:
            numeric_features[f'feature_{i}'] = np.random.exponential(scale=20, size=n_samples)
        else:
            numeric_features[f'feature_{i}'] = np.random.uniform(0, 100, n_samples)

    df = pd.DataFrame(numeric_features)

    categories = ['A', 'B', 'C', 'D', 'E']
    df['category_1'] = np.random.choice(categories, size=n_samples)
    df['category_2'] = np.random.choice(['Type1', 'Type2', 'Type3'], size=n_samples)
    df['category_3'] = np.random.choice(['Low', 'Medium', 'High', 'Very High'], size=n_samples)

    start_date = datetime(2020, 1, 1)
    df['date'] = [start_date + timedelta(days=int(x)) for x in np.random.uniform(0, 1000, n_samples)]

    target_score = (
        df['feature_0'] * 0.3 +
        df['feature_1'] * 0.2 +
        df['feature_2'] * 0.1 +
        np.random.randn(n_samples) * 10
    )

    threshold = np.percentile(target_score, imbalance_ratio * 100)
    df['target'] = (target_score > threshold).astype(int)

    n_missing = int(n_samples * len(df.columns) * missing_rate)
    missing_indices = np.random.choice(n_samples * len(df.columns), size=n_missing, replace=False)
    for idx in missing_indices:
        row = idx // len(df.columns)
        col = idx % len(df.columns)
        if col < len(numeric_features):
            df.iloc[row, col] = np.nan

    n_outliers = int(n_samples * outlier_rate)
    numeric_cols = list(numeric_features.keys())
    for _ in range(n_outliers):
        row = np.random.randint(0, n_samples)
        col = np.random.choice(numeric_cols)
        df.loc[row, col] = df[col].mean() + np.random.choice([-1, 1]) * df[col].std() * 5

    n_duplicates = int(n_samples * 0.02)
    duplicate_indices = np.random.choice(n_samples, size=n_duplicates, replace=False)
    df = pd.concat([df, df.iloc[duplicate_indices]], ignore_index=True)

    return df


def main():
    print("=" * 80)
    print("Task 6: Real-World Case Study - Complete Data Preprocessing Challenge")
    print("=" * 80)
    print()

    print("Creating complex synthetic dataset...")
    df = create_complex_synthetic_dataset(
        n_samples=2000,
        n_features=25,
        missing_rate=0.15,
        outlier_rate=0.08,
        imbalance_ratio=0.3
    )
    print(f"Dataset created: {df.shape[0]} samples, {df.shape[1]} features")
    print()

    preprocessor = DataPreprocessor(df, target_col='target')

    print("STEP 1: DATA QUALITY ASSESSMENT")
    print("=" * 80)
    preprocessor.assess_data_quality()
    preprocessor.print_quality_report()

    print("STEP 2: EXPLORATORY DATA ANALYSIS")
    print("=" * 80)
    print("Generating EDA visualizations...")

    preprocessor.visualize_eda()
    plt.savefig('task6_eda_distributions.png', dpi=300, bbox_inches='tight')
    print("  Saved: task6_eda_distributions.png")

    preprocessor.visualize_correlations()
    plt.savefig('task6_correlations.png', dpi=300, bbox_inches='tight')
    print("  Saved: task6_correlations.png")

    preprocessor.visualize_missing_patterns()
    plt.savefig('task6_missing_patterns.png', dpi=300, bbox_inches='tight')
    print("  Saved: task6_missing_patterns.png")
    print()

    print("STEP 3: PREPROCESSING PIPELINE")
    print("=" * 80)

    config = {
        'handle_missing': True,
        'missing_strategy': 'auto',
        'handle_outliers': True,
        'outlier_method': 'iqr',
        'outlier_action': 'cap',
        'encode_categorical': True,
        'encoding_method': 'auto',
        'engineer_features': True,
        'scale_features': True,
        'scaling_method': 'robust',
        'select_features': True,
        'selection_method': 'kbest',
        'n_features': 30,
        'balance_classes': True,
        'balance_method': 'undersample'
    }

    preprocessor.preprocess_pipeline(config)
    print()

    print("STEP 4: GENERATING ADDITIONAL VISUALIZATIONS")
    print("=" * 80)

    print("  Creating PCA visualization...")
    preprocessor.visualize_pca()
    plt.savefig('task6_pca_visualization.png', dpi=300, bbox_inches='tight')
    print("  Saved: task6_pca_visualization.png")
    print()

    print("STEP 5: GENERATING REPORT")
    print("=" * 80)
    preprocessor.generate_report('task6_preprocessing_report.json')
    print()

    print("=" * 80)
    print("TASK 6 COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print()
    print("SUMMARY:")
    print(f"  Original shape: {preprocessor.original_data.shape}")
    print(f"  Final shape: {preprocessor.data.shape}")
    print(f"  Preprocessing steps: {len(preprocessor.preprocessing_steps)}")
    print()
    print("FILES GENERATED:")
    print("  - task6_eda_distributions.png")
    print("  - task6_correlations.png")
    print("  - task6_missing_patterns.png")
    print("  - task6_pca_visualization.png")
    print("  - task6_preprocessing_report.json")
    print()

    return preprocessor

if __name__ == "__main__":
    preprocessor = main()
    plt.show()
