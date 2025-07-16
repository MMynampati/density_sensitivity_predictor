import json
import re
from collections import Counter
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

def parse_sensitivity(text: str):
    """Parse sensitivity from text, returns True, False, or None"""
    text = text.split('\n')[0].split('.')[0].strip()
    
    match = re.search(r"Density sensitive[^:]*:\s*(True|False)", text, re.IGNORECASE)
    if match:
        return match.group(1).lower() == 'true'
    
    match = re.search(r"Density sensitive[^;]*;\s*(True|False)", text, re.IGNORECASE)
    if match:
        return match.group(1).lower() == 'true'
    
    match = re.search(r"Density sensitive[^a-zA-Z]*(True|False)", text, re.IGNORECASE)
    if match:
        return match.group(1).lower() == 'true'
    
    return None

def analyze_class_balance(data_file: str):
    """Analyze class balance in the dataset"""
    console = Console()
    
    console.print(f"[bold blue]Analyzing class balance in: {data_file}[/bold blue]")
    
    # Load data
    with open(data_file, 'r') as f:
        examples = [json.loads(line) for line in f]
    
    console.print(f"Total examples: {len(examples)}")
    
    # Parse completions
    true_count = 0
    false_count = 0
    unparseable_count = 0
    functional_counts = Counter()
    functional_true_counts = Counter()
    functional_false_counts = Counter()
    
    for example in examples:
        completion = example.get('completion', '')
        sensitivity = parse_sensitivity(completion)
        
        if sensitivity is True:
            true_count += 1
        elif sensitivity is False:
            false_count += 1
        else:
            unparseable_count += 1
        
        # Count functionals and their True/False breakdown
        prompt = example.get('prompt', '')
        if 'Functional:' in prompt:
            functional = prompt.split('Functional:')[1].split(',')[0].strip()
            functional_counts[functional] += 1
            
            if sensitivity is True:
                functional_true_counts[functional] += 1
            elif sensitivity is False:
                functional_false_counts[functional] += 1
    
    # Display results
    console.print("\n[bold green]Class Balance Analysis[/bold green]")
    
    balance_table = Table(title="Sensitivity Distribution")
    balance_table.add_column("Class", style="cyan")
    balance_table.add_column("Count", justify="right", style="magenta")
    balance_table.add_column("Percentage", justify="right", style="yellow")
    
    total_parseable = true_count + false_count
    if total_parseable > 0:
        balance_table.add_row("True", str(true_count), f"{true_count/total_parseable*100:.1f}%")
        balance_table.add_row("False", str(false_count), f"{false_count/total_parseable*100:.1f}%")
    
    balance_table.add_row("Unparseable", str(unparseable_count), f"{unparseable_count/len(examples)*100:.1f}%")
    console.print(balance_table)
    
    # Functional distribution with True/False breakdown
    if functional_counts:
        console.print("\n[bold green]Functional Distribution with True/False Breakdown[/bold green]")
        func_table = Table(title="Functional Distribution")
        func_table.add_column("Functional", style="cyan")
        func_table.add_column("Total", justify="right", style="magenta")
        func_table.add_column("True", justify="right", style="green")
        func_table.add_column("False", justify="right", style="red")
        func_table.add_column("True %", justify="right", style="yellow")
        
        for func in sorted(functional_counts.keys()):
            total = functional_counts[func]
            true_count_func = functional_true_counts[func]
            false_count_func = functional_false_counts[func]
            true_percent = (true_count_func / total * 100) if total > 0 else 0
            
            func_table.add_row(
                func, 
                str(total), 
                str(true_count_func), 
                str(false_count_func),
                f"{true_percent:.1f}%"
            )
        console.print(func_table)
    
    # Analysis summary
    if total_parseable > 0:
        ratio = max(true_count, false_count) / min(true_count, false_count) if min(true_count, false_count) > 0 else float('inf')
        
        console.print(Panel(
            f"[bold]Analysis Summary:[/bold]\n"
            f"• True/False ratio: {ratio:.2f}:1\n"
            f"• {'Severe' if ratio > 3 else 'Moderate' if ratio > 2 else 'Minor'} class imbalance detected\n"
            f"• Model bias likely toward {'False' if false_count > true_count else 'True'}\n"
            f"• {unparseable_count} examples ({unparseable_count/len(examples)*100:.1f}%) could not be parsed",
            title="Class Balance Summary",
            border_style="blue"
        ))
        
        if ratio > 2:
            console.print("\n[bold yellow]Recommendations:[/bold yellow]")
            console.print("1. Consider upsampling the minority class")
            console.print("2. Use class weights in training")
            console.print("3. Add more examples of the minority class")
            console.print("4. Consider using a classification head instead of text generation")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Check class balance in training data")
    parser.add_argument(
        "--data_file",
        type=str,
        required=True,
        help="Path to the JSONL data file"
    )
    args = parser.parse_args()
    
    analyze_class_balance(args.data_file) 