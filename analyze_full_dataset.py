import pandas as pd
import numpy as np
from collections import Counter
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

def analyze_full_dataset():
    """Analyze the full dataset to understand the complete data distribution"""
    console = Console()
    
    console.print("[bold blue]Analyzing full dataset: cleaned_scraped_gmtkn_data.csv[/bold blue]")
    
    # Load the full dataset
    df = pd.read_csv("cleaned_scraped_gmtkn_data.csv")
    console.print(f"Total entries in full dataset: {len(df)}")
    
    # Count entries per functional
    functional_counts = df['functional'].value_counts()
    console.print(f"\n[bold green]Entries per functional in full dataset:[/bold green]")
    
    func_table = Table(title="Full Dataset - Entries per Functional")
    func_table.add_column("Functional", style="cyan")
    func_table.add_column("Count", justify="right", style="magenta")
    func_table.add_column("Percentage", justify="right", style="yellow")
    
    for func, count in functional_counts.items():
        percentage = (count / len(df)) * 100
        func_table.add_row(func, str(count), f"{percentage:.1f}%")
    console.print(func_table)
    
    # Focus on PBE, M06, BLYP
    target_functionals = ['PBE', 'M06', 'BLYP']
    target_df = df[df['functional'].isin(target_functionals)]
    
    console.print(f"\n[bold green]Target functionals (PBE, M06, BLYP) analysis:[/bold green]")
    console.print(f"Total entries for target functionals: {len(target_df)}")
    
    # Check if D3(BJ) column exists and has data
    if 'D3(BJ)' in df.columns:
        console.print(f"\n[bold green]D3(BJ) dispersion correction analysis:[/bold green]")
        
        # Count non-null D3(BJ) entries per functional
        d3bj_counts = {}
        for func in target_functionals:
            func_df = df[df['functional'] == func]
            d3bj_count = func_df['D3(BJ)'].notna().sum()
            d3bj_counts[func] = d3bj_count
            console.print(f"{func}: {d3bj_count} entries with D3(BJ) data")
    
    # Check subsets distribution
    subset_counts = target_df['subset'].value_counts()
    console.print(f"\n[bold green]Subset distribution for target functionals:[/bold green]")
    
    subset_table = Table(title="Subset Distribution")
    subset_table.add_column("Subset", style="cyan")
    subset_table.add_column("Count", justify="right", style="magenta")
    subset_table.add_column("Percentage", justify="right", style="yellow")
    
    for subset, count in subset_counts.head(10).items():  # Show top 10
        percentage = (count / len(target_df)) * 100
        subset_table.add_row(subset, str(count), f"{percentage:.1f}%")
    console.print(subset_table)
    
    # Summary
    console.print(Panel(
        f"[bold]Full Dataset Summary:[/bold]\n"
        f"• Total entries: {len(df)}\n"
        f"• Target functionals (PBE, M06, BLYP): {len(target_df)}\n"
        f"• Available for training: {len(target_df)} (vs current 960)\n"
        f"• Potential improvement: {len(target_df) - 960} additional examples",
        title="Dataset Comparison",
        border_style="blue"
    ))

if __name__ == "__main__":
    analyze_full_dataset() 