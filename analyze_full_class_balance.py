import pandas as pd
import numpy as np
from collections import Counter
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

def analyze_full_class_balance():
    """Analyze the class balance in the full dataset"""
    console = Console()
    
    console.print("[bold blue]Analyzing class balance in full dataset[/bold blue]")
    
    # Load the full dataset
    df = pd.read_csv("cleaned_scraped_gmtkn_data.csv")
    
    # Load the SWARM data to get density sensitivity labels
    try:
        swarm_df = pd.read_csv("all_v2_SWARM.csv")
        console.print("Loaded SWARM data for density sensitivity analysis")
    except FileNotFoundError:
        console.print("[red]Warning: all_v2_SWARM.csv not found. Cannot analyze class balance.[/red]")
        return
    
    # Focus on target functionals
    target_functionals = ['PBE', 'M06', 'BLYP']
    target_df = df[df['functional'].isin(target_functionals)].copy()
    
    console.print(f"Analyzing {len(target_df)} entries for target functionals")
    
    # Calculate density sensitivity for each entry
    threshold = 2.0
    true_count = 0
    false_count = 0
    na_count = 0
    
    functional_true_counts = Counter()
    functional_false_counts = Counter()
    
    for index, row in target_df.iterrows():
        setname = row['subset']
        calctype = row['functional']
        rxnidx = int(row['#'])
        
        # Find matching entry in SWARM data
        match = swarm_df[
            (swarm_df['setname'] == setname) &
            (swarm_df['calctype'] == calctype) &
            (swarm_df['rxnidx'] == rxnidx)
        ]
        
        if not match.empty:
            s_val = match.iloc[0]['S']
            is_sensitive = s_val > threshold
            
            if is_sensitive:
                true_count += 1
                functional_true_counts[calctype] += 1
            else:
                false_count += 1
                functional_false_counts[calctype] += 1
        else:
            na_count += 1
    
    # Display results
    console.print(f"\n[bold green]Class Balance Analysis for Full Dataset[/bold green]")
    
    balance_table = Table(title="Full Dataset - Sensitivity Distribution")
    balance_table.add_column("Class", style="cyan")
    balance_table.add_column("Count", justify="right", style="magenta")
    balance_table.add_column("Percentage", justify="right", style="yellow")
    
    total_parseable = true_count + false_count
    if total_parseable > 0:
        balance_table.add_row("True", str(true_count), f"{true_count/total_parseable*100:.1f}%")
        balance_table.add_row("False", str(false_count), f"{false_count/total_parseable*100:.1f}%")
    
    balance_table.add_row("N/A", str(na_count), f"{na_count/len(target_df)*100:.1f}%")
    console.print(balance_table)
    
    # Per-functional breakdown
    console.print(f"\n[bold green]Per-Functional Breakdown[/bold green]")
    func_table = Table(title="Per-Functional Class Distribution")
    func_table.add_column("Functional", style="cyan")
    func_table.add_column("Total", justify="right", style="magenta")
    func_table.add_column("True", justify="right", style="green")
    func_table.add_column("False", justify="right", style="red")
    func_table.add_column("True %", justify="right", style="yellow")
    
    for func in target_functionals:
        total_func = functional_true_counts[func] + functional_false_counts[func]
        true_func = functional_true_counts[func]
        false_func = functional_false_counts[func]
        
        if total_func > 0:
            true_percent = (true_func / total_func * 100)
            func_table.add_row(
                func,
                str(total_func),
                str(true_func),
                str(false_func),
                f"{true_percent:.1f}%"
            )
    
    console.print(func_table)
    
    # Comparison with current dataset
    console.print(Panel(
        f"[bold]Comparison Summary:[/bold]\n"
        f"• Full dataset (PBE, M06, BLYP): {len(target_df)} entries\n"
        f"• Current training dataset: 960 entries\n"
        f"• Full dataset True: {true_count} ({true_count/total_parseable*100:.1f}%)\n"
        f"• Full dataset False: {false_count} ({false_count/total_parseable*100:.1f}%)\n"
        f"• Current dataset True: 299 (31.1%)\n"
        f"• Current dataset False: 661 (68.9%)\n"
        f"• Class balance ratio (full): {max(true_count, false_count)/min(true_count, false_count):.2f}:1\n"
        f"• Class balance ratio (current): 2.21:1",
        title="Dataset Comparison",
        border_style="blue"
    ))

if __name__ == "__main__":
    analyze_full_class_balance() 