import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

class PerformanceAnalytics:
    def __init__(self, results_df, trade_log_df, stats):
        self.results_df = results_df
        self.trade_log_df = trade_log_df
        self.stats = stats
        # Get absolute path of current directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Create base directories
        self.base_reports_dir = os.path.join(current_dir, '..', 'reports')
        self.base_charts_dir = os.path.join(current_dir, '..', 'charts')
        print(f"Creating base directories: {self.base_reports_dir}, {self.base_charts_dir}")
        os.makedirs(self.base_reports_dir, exist_ok=True)
        os.makedirs(self.base_charts_dir, exist_ok=True)
        
        # Create timestamp-based subdirectories
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.reports_dir = os.path.join(self.base_reports_dir, self.timestamp)
        self.charts_dir = os.path.join(self.base_charts_dir, self.timestamp)
        print(f"Creating timestamp directories: {self.reports_dir}, {self.charts_dir}")
        os.makedirs(self.reports_dir, exist_ok=True)
        os.makedirs(self.charts_dir, exist_ok=True)
        
    def calculate_metrics(self, initial_balance):
        """Calculate performance metrics"""
        try:
            final_balance = self.stats.get('final_balance', initial_balance)
            total_return = ((final_balance - initial_balance) / initial_balance) * 100
            
            total_trades = self.stats.get('total_trades', 0)
            winning_trades = self.stats.get('winning_trades', 0)
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            
            # Handle case where 'pnl' column doesn't exist or has a different name
            pnl_column = None
            for col in ['pnl', 'PnL', 'profit_loss', 'profit']:
                if col in self.trade_log_df.columns:
                    pnl_column = col
                    break
            
            if pnl_column is not None and not self.trade_log_df.empty:
                pnl_series = self.trade_log_df[pnl_column].dropna()
                total_pnl = pnl_series.sum()
                avg_pnl_per_trade = total_pnl / total_trades if total_trades > 0 else 0
                
                # Calculate profit factor
                winning_pnl = pnl_series[pnl_series > 0].sum()
                losing_pnl = pnl_series[pnl_series < 0].sum()
                profit_factor = abs(winning_pnl / losing_pnl) if losing_pnl != 0 else float('inf')
            else:
                total_pnl = final_balance - initial_balance
                avg_pnl_per_trade = total_pnl / total_trades if total_trades > 0 else 0
                profit_factor = float('inf')  # Default to infinity if no losing trades
            
            metrics = {
                'Initial Balance': f"Rs. {initial_balance:,.2f}",
                'Final Balance': f"Rs. {final_balance:,.2f}",
                'Total Return': f"{total_return:.2f}%",
                'Total Trades': total_trades,
                'Win Rate': f"{win_rate:.1f}%" if total_trades > 0 else "N/A",
                'Average P&L per trade': f"Rs. {avg_pnl_per_trade:,.2f}",
                'Total P&L': f"Rs. {total_pnl:,.2f}",
                'Profit Factor': f"{profit_factor:.2f}"
            }
            
            # Add max profit/loss if available
            if 'max_profit' in self.stats:
                metrics['Largest Profit'] = f"Rs. {self.stats['max_profit']:,.2f}"
            if 'max_loss' in self.stats:
                metrics['Largest Loss'] = f"Rs. {self.stats['max_loss']:,.2f}"
                
            return metrics
            
        except Exception as e:
            import logging
            logging.error(f"Error in calculate_metrics: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            
            # Return basic metrics even if there's an error
            return {
                'Initial Balance': f"Rs. {initial_balance:,.2f}",
                'Final Balance': f"Rs. {self.stats.get('final_balance', initial_balance):,.2f}",
                'Total Trades': self.stats.get('total_trades', 0),
                'Error': f"Error calculating metrics: {str(e)}"
            }
    
    def plot_equity_curve(self):
        """Plot equity curve"""
        plt.figure(figsize=(12, 6))
        plt.plot(self.results_df['date'], self.results_df['balance'])
        plt.title('Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Balance (Rs.)')
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
    
    def plot_and_save_charts(self, timestamp):
        """Plot and save charts"""
        # Charts directory is already created in __init__
        
        # Plot and save equity curve
        plt.figure(figsize=(12, 6))
        plt.plot(self.results_df.index, self.results_df['balance'], label='Portfolio Value')
        plt.title('Equity Curve')
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value (Rs.)')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        equity_curve_file = os.path.join(self.charts_dir, 'equity_curve.png')
        plt.savefig(equity_curve_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot and save trade distribution
        plt.figure(figsize=(10, 6))
        trade_dist = [self.stats['winning_trades'], self.stats['losing_trades'], self.stats['break_even_trades']]
        plt.bar(['Winning', 'Losing', 'Break-even'], trade_dist)
        plt.title('Trade Distribution')
        plt.ylabel('Number of Trades')
        plt.grid(True, axis='y')
        trade_dist_file = os.path.join(self.charts_dir, 'trade_distribution.png')
        plt.savefig(trade_dist_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot and save daily returns
        daily_returns = self.results_df['balance'].pct_change()
        plt.figure(figsize=(12, 6))
        plt.plot(self.results_df.index, daily_returns, label='Daily Returns')
        plt.title('Daily Returns')
        plt.xlabel('Date')
        plt.ylabel('Return (%)')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        daily_returns_file = os.path.join(self.charts_dir, 'daily_returns.png')
        plt.savefig(daily_returns_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return equity_curve_file, trade_dist_file, daily_returns_file
        
        # Plot and save trade distribution
        plt.figure(figsize=(10, 6))
        trade_dist = [self.stats['winning_trades'], self.stats['losing_trades'], self.stats['break_even_trades']]
        plt.bar(['Winning', 'Losing', 'Break-even'], trade_dist)
        plt.title('Trade Distribution')
        plt.ylabel('Number of Trades')
        plt.grid(True, axis='y')
        trade_dist_file = os.path.join(self.charts_dir, 'trade_distribution.png')
        plt.savefig(trade_dist_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot and save daily returns
        daily_returns = self.results_df['balance'].pct_change()
        plt.figure(figsize=(12, 6))
        plt.plot(self.results_df.index, daily_returns, label='Daily Returns')
        plt.title('Daily Returns')
        plt.xlabel('Date')
        plt.ylabel('Return (%)')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        daily_returns_file = os.path.join(self.charts_dir, 'daily_returns.png')
        plt.savefig(daily_returns_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        return equity_curve_file, trade_dist_file, daily_returns_file
    
    def export_to_excel(self, initial_balance):
        """Export results to Excel file with multiple sheets"""
        # Use the timestamp from initialization
        excel_file = os.path.join(self.reports_dir, 'backtesting_results.xlsx')
        
        # Create Excel writer object
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            # Export trade log
            self.trade_log_df.to_excel(writer, sheet_name='Trade Log', index=True)
            
            # Export daily results
            self.results_df.to_excel(writer, sheet_name='Daily Results', index=True)
            
            # Export strategy statistics
            metrics = self.calculate_metrics(initial_balance)
            stats_df = pd.DataFrame([
                {
                    'Metric': key,
                    'Value': value
                } for key, value in metrics.items()
            ])
            stats_df.to_excel(writer, sheet_name='Strategy Stats', index=False)
            
            # Auto-adjust column widths
            for sheet_name in writer.sheets:
                worksheet = writer.sheets[sheet_name]
                for column_cells in worksheet.columns:
                    length = max(len(str(cell.value)) for cell in column_cells)
                    worksheet.column_dimensions[column_cells[0].column_letter].width = length + 2
        
        return excel_file
    
    def print_summary(self, initial_balance):
        """Print performance summary and export results"""
        metrics = self.calculate_metrics(initial_balance)
        
        # Export to Excel and generate charts using the timestamp from initialization
        excel_file = self.export_to_excel(initial_balance)
        equity_curve_file, trade_dist_file, daily_returns_file = self.plot_and_save_charts(self.timestamp)
        
        print("\nBacktesting Results Summary:")
        for key, value in metrics.items():
            print(f"{key}: {value}")
        
        print("\nTrade Distribution:")
        print(f"Winning Trades: {self.stats['winning_trades']}")
        print(f"Losing Trades: {self.stats['losing_trades']}")
        print(f"Break-even Trades: {self.stats['break_even_trades']}")
        
        print("\nLast 5 Trades:")
        print(self.trade_log_df.tail())
        
        print("\nExported files:")
        print(f"Excel Report: {excel_file}")
        print(f"Equity Curve Chart: {equity_curve_file}")
        print(f"Trade Distribution Chart: {trade_dist_file}")
        print(f"Daily Returns Chart: {daily_returns_file}")
