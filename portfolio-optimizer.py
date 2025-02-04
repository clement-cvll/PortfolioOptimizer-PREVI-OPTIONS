import os
import numpy as np
import pandas as pd
from typing import List
from pathlib import Path
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy.optimize import minimize
from preprocessing import process_data

# Preprocessing
class DataManager:
    """Data manager class"""
    def __init__(self, data_path: str | Path):
        self.data_path = data_path

    def process(self) -> None:
        """Process the data"""
        self.df = self._load_and_clean_data()
        self.category_data = self._prepare_category_data()
        self.returns_data = self._prepare_returns_data()

    def _load_and_clean_data(self) -> pd.DataFrame:
        """Load and clean the data"""
        df = pd.read_csv(self.data_path)
        df = df[['ISIN', 'Unit of account', 'Category', '2018', '2019', '2020', '2021', '2022', '2023', '2024', 'Volatility 3 years', 'Fees', 'Sharpe']]
        df.set_index('ISIN', inplace=True)
        return df

    def _prepare_category_data(self) -> pd.DataFrame:
        """Prepare category data by calculating average metrics per category"""
        category_data = self.df.groupby('Category').agg({
            '2018': 'mean',
            '2019': 'mean', 
            '2020': 'mean',
            '2021': 'mean',
            '2022': 'mean',
            '2023': 'mean',
            '2024': 'mean',
            'Volatility 3 years': 'mean',
        }).round(4)
        category_data['Average Return'] = category_data[['2018', '2019', '2020', '2021', '2022', '2023', '2024']].mean(axis=1)
        return category_data

    def _prepare_returns_data(self) -> pd.DataFrame:
        """Prepare returns data"""
        returns_cols = [ '2020', '2021', '2022', '2023', '2024']
        returns_data = self.df[returns_cols].copy()
        return returns_data

@dataclass
class Portfolio:
    """Portfolio class"""
    compound_return: float
    sharpe_ratio: float
    weights: np.ndarray
    assets: List[str]
    isin_codes: List[str]

# Portfolio optimizer
class PortfolioOptimizer:
    def __init__(self, data_manager: DataManager, risk_free_rate: float = 0.04, max_position_size: float = 0.4):
        """Initialize portfolio optimizer"""
        self.risk_free_rate = risk_free_rate
        self.max_position_size = max_position_size
        self.data_manager = data_manager

    def _portfolio_stats(self, weights: np.ndarray) -> tuple:
        """Calculate portfolio statistics"""
        portfolio_compound_return = np.sum(weights * (np.prod(1 + self.data_manager.returns_data, axis=1) - 1))
        portfolio_return = np.sum(weights * self.data_manager.returns_data.mean(axis=1))
        portfolio_standard_deviation = np.sqrt(np.sum(weights * self.data_manager.df['Volatility 3 years'].values))
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_standard_deviation
        return portfolio_compound_return, sharpe_ratio

    def _negative_objective(self, weights: np.ndarray) -> float:
        """Objective function to minimize"""
        _, sharpe_ratio = self._portfolio_stats(weights)
        return -sharpe_ratio

    def optimize_portfolio(self) -> Portfolio:
        """Optimize portfolio using scipy optimize"""
        n_assets = len(self.data_manager.returns_data.index)
        
        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # weights sum to 1
        ]
        
        # Bounds for each weight (0 to max_position_size)
        bounds = tuple((0, self.max_position_size) for _ in range(n_assets))
        
        # Initial guess (equal weights)
        initial_weights = np.array([1/n_assets] * n_assets)
        
        result = minimize(
            self._negative_objective,
            initial_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
            
        # Get optimal weights and round small weights to 0
        weights = result.x
        weights[weights < 0.01] = 0
        weights = weights / np.sum(weights)  # Renormalize
        
        # Calculate portfolio metrics
        compound_return, sharpe_ratio = self._portfolio_stats(weights)
        
        return Portfolio(
            compound_return=compound_return,
            sharpe_ratio=sharpe_ratio,
            weights=weights,
            assets=list(self.data_manager.df['Unit of account']),
            isin_codes=list(self.data_manager.returns_data.index)
        )

    def get_allocation_summary(self, min_weight: float = 0.01) -> pd.DataFrame:
        """Get portfolio allocation summary"""
        portfolio = self.optimize_portfolio()
        allocation = pd.DataFrame({
            'Asset': portfolio.assets,
            'Weight (%)': portfolio.weights * 100,
            'Sharpe': self.data_manager.df.loc[portfolio.isin_codes, 'Sharpe'],
            'Volatility (%)': self.data_manager.df.loc[portfolio.isin_codes, 'Volatility 3 years'] * 100,
            'Return (%)': self.data_manager.returns_data.loc[portfolio.isin_codes, ['2020', '2021', '2022', '2023', '2024']].mean(axis=1) * 100,
            'Category': self.data_manager.df.loc[portfolio.isin_codes, 'Category'],
        })
        allocation['Weight (%)'] = allocation['Weight (%)'].round(1)
        allocation['Sharpe'] = allocation['Sharpe'].round(2)
        allocation['Volatility (%)'] = allocation['Volatility (%)'].round(2)
        allocation['Return (%)'] = allocation['Return (%)'].round(2)
        allocation = allocation[allocation['Weight (%)'] > min_weight * 100]
        return allocation.sort_values('Weight (%)', ascending=False)

def plot_portfolio_allocation(allocation: pd.DataFrame, save_path: str, compound_return: float, years: int) -> None:
    """Create and save a donut chart visualization of portfolio allocation."""
    plt.figure(figsize=(12, 8))
    colors = plt.cm.Pastel1(np.linspace(0, 1, len(allocation)))
    
    # Pie chart
    wedges, texts, autotexts = plt.pie(
        allocation['Weight (%)'], 
        labels=allocation['Asset'],
        autopct='%1.1f%%',
        colors=colors,
        startangle=90,
        pctdistance=0.85,
        explode=[0.05] * len(allocation)
    )
    plt.title('Portfolio Allocation', pad=20, size=14, fontweight='bold')
    
    # Labels easier to read
    plt.setp(autotexts, size=8, weight="bold")
    plt.setp(texts, size=8)
    
    # Donut chart
    centre_circle = plt.Circle((0,0), 0.70, fc='white')
    plt.gca().add_artist(centre_circle)
    
    # Compound return and duration in center
    plt.text(0, 0.1, f"Compound return: {compound_return:.1f}%", 
             ha='center', va='center', fontsize=10, fontweight='bold')
    plt.text(0, -0.1, f"({years} years)", 
             ha='center', va='center', fontsize=9)
    
    # Legend with categories
    legend_labels = [f"{category}" 
                    for category in allocation['Category']]
    plt.legend(wedges, legend_labels, 
              title="Assets Category",
              loc="lower right",
              bbox_to_anchor=(2.0, 0))
    
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

# Main function
def main():
    """Main function to run the portfolio optimizer"""
    if not os.path.exists(os.path.join('data', 'processed', 'data.csv')):
        print("Processing data...")
        process_data(os.path.join('data', 'PREVI-OPTIONS.xls'))

    # Preprocess the data
    data_manager = DataManager(
        data_path=os.path.join('data', 'processed', 'data.csv'),
    )
    data_manager.process()

    # Optimize the portfolio
    optimizer = PortfolioOptimizer(
        data_manager=data_manager,
        max_position_size=0.1,
        risk_free_rate=0.04,
    )
    portfolio = optimizer.optimize_portfolio()

    # Get the portfolio metrics
    print("\nPortfolio Metrics:")
    print(f"Compound Return: {portfolio.compound_return*100:.2f}%  ({data_manager.returns_data.shape[1]} years)")
    print(f"Sharpe Ratio: {portfolio.sharpe_ratio:.2f}")

    # Plot and save allocation visualization
    os.makedirs('visualizations', exist_ok=True)
    allocation = optimizer.get_allocation_summary()
    plot_portfolio_allocation( 
        allocation, 
        save_path=os.path.join('visualizations', f'portfolio_allocation_{round(optimizer.max_position_size*100)}.png'), 
        compound_return=portfolio.compound_return*100, 
        years=data_manager.returns_data.shape[1]
    )
    print("\nAsset Allocation:")
    print(allocation)

if __name__ == "__main__":
    main()