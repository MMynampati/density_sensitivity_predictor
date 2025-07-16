# Density Sensitivity Prediction for DFT Functionals

A machine learning pipeline for predicting density functional sensitivity to dispersion corrections using fine-tuned GPT-2 models. This project focuses on computational chemistry applications, specifically analyzing the impact of D3(BJ) dispersion corrections on density functional theory (DFT) calculations.

## 🔬 Scientific Background

This project addresses a critical question in computational chemistry: **When does dispersion correction significantly impact DFT calculations?** 

Density functional theory (DFT) is widely used for electronic structure calculations, but standard functionals often fail to properly describe long-range van der Waals interactions. Dispersion corrections like D3(BJ) can improve accuracy, but their impact varies significantly depending on the molecular system and functional used.

Our approach uses machine learning to predict when a given functional-molecule combination will be "density sensitive" (i.e., when adding dispersion correction changes the calculated energy by more than a threshold value).

### Key Features
- **Data Source**: GMTKN55 benchmark database with comprehensive DFT functional results
- **Target Functionals**: PBE, M06, and BLYP with D3(BJ) dispersion corrections
- **Molecular Representation**: SMILES strings derived from chemical systems
- **Prediction Task**: Binary classification of density sensitivity (threshold: 2.0 energy units)
- **Model Architecture**: Fine-tuned GPT-2 for prompt-completion learning

## 📁 Project Structure

```
BurkeLab/
├── README.md                          # This file
├── requirements_cpu.txt               # Python dependencies for CPU usage
├── 
├── # Data Files
├── cleaned_scraped_gmtkn_data.csv     # Main dataset from GMTKN55
├── all_v2_SWARM.csv                   # SWARM analysis data for sensitivity labels
├── 
├── # Data Collection & Preprocessing
├── generate_dataset.py                # Web scraper for GMTKN55 data
├── scraper.py                         # Core scraping utilities
├── helpers.py                         # URL lists and helper functions
├── preprocessing_data.py              # Dataset preparation and JSONL generation
├── 
├── # Model Training
├── finetune_gpt2_cpu.py              # CPU-optimized GPT-2 fine-tuning
├── finetune_gpt2_gpu.py              # GPU-optimized GPT-2 fine-tuning
├── 
├── # Model Inference & Evaluation
├── prompt_model.py                    # Interactive model prompting
├── evaluate_density_sensitivity.py   # Model performance evaluation
├── analyze_model_errors.py           # Error analysis and pattern detection
├── 
├── # Data Analysis
├── analyze_full_dataset.py           # Dataset statistics and distribution
├── analyze_full_class_balance.py     # Class balance analysis
├── check_class_balance.py            # Training data balance verification
├── 
├── # Generated Data & Models
├── finetuning_sets/                  # Training datasets (JSONL format)
│   ├── PBE+M06+BLYP_large_finetuned_dataset.jsonl    # Large dataset (4285 samples)
│   ├── PBE+M06+BLYP_medium_finetuned_dataset.jsonl   # Medium dataset (961 samples)
│   ├── PBE+M06+BLYP_finetuned_data.jsonl            # Small dataset (301 samples)
│   └── [timestamped experiment files...]
└── results/                          # Trained models
    └── gpt2_density_finetuned_cpu_v5/ # Latest fine-tuned model
```

## 🚀 Installation

### Requirements
- Python 3.8+
- CPU-based training (GPU support available)
- ~4GB RAM for training
- ~16GB disk space for datasets and models

### Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd BurkeLab
```

2. **Install dependencies**
```bash
pip install -r requirements_cpu.txt
```

3. **Verify data files**
Ensure you have the main data files:
- `cleaned_scraped_gmtkn_data.csv` (10MB)
- `all_v2_SWARM.csv` (6.1MB)

## 📊 Workflow

### 1. Data Collection (Optional - data already provided)
```bash
# Scrape fresh data from GMTKN55 (takes ~1 hour)
python generate_dataset.py
```

### 2. Dataset Preparation
```bash
# Generate training datasets with different sizes
python preprocessing_data.py
```
This creates JSONL files in `finetuning_sets/` with prompt-completion pairs:
- **Prompt**: `"Functional: PBE, SMILES: C1=CC=CC=C1"`
- **Completion**: `"Density sensitive: True"`

### 3. Data Analysis
```bash
# Analyze the full dataset distribution
python analyze_full_dataset.py

# Check class balance in training data
python check_class_balance.py --data_file finetuning_sets/PBE+M06+BLYP_large_finetuned_dataset.jsonl

# Analyze class balance across all functionals
python analyze_full_class_balance.py
```

### 4. Model Training

**CPU Training (Recommended for most users):**
```bash
python finetune_gpt2_cpu.py
```

**GPU Training (if CUDA available):**
```bash
python finetune_gpt2_gpu.py
```

Training parameters:
- **Model**: GPT-2 base (124M parameters)
- **Epochs**: 5
- **Batch size**: 4 (CPU) / 2 (GPU)
- **Learning rate**: 5e-5
- **Train/Val/Test split**: 70%/15%/15%

### 5. Model Evaluation

**Interactive prompting:**
```bash
python prompt_model.py --model_dir results/gpt2_density_finetuned_cpu_v5
```

**Systematic evaluation:**
```bash
python evaluate_density_sensitivity.py \
    --model_dir results/gpt2_density_finetuned_cpu_v5 \
    --test_file results/gpt2_density_finetuned_cpu_v5/test_set_prompts_completions.jsonl
```

**Error analysis:**
```bash
python analyze_model_errors.py \
    --model_dir results/gpt2_density_finetuned_cpu_v5 \
    --test_file results/gpt2_density_finetuned_cpu_v5/test_set_prompts_completions.jsonl
```

## 🎯 Usage Examples

### Quick Prediction
```python
from prompt_model import generate_text

# Predict density sensitivity for benzene with PBE functional
result = generate_text(
    model_dir="results/gpt2_density_finetuned_cpu_v5",
    prompt_text="Functional: PBE, SMILES: C1=CC=CC=C1"
)
print(result)
# Expected output: "Functional: PBE, SMILES: C1=CC=CC=C1 Density sensitive: True"
```

### Batch Analysis
```python
import pandas as pd
from preprocessing_data import create_df_subdataset

# Create a custom dataset for specific functionals
df = pd.read_csv("cleaned_scraped_gmtkn_data.csv")
subset = create_df_subdataset(df, ['B3LYP', 'PBE0'], num_samples=100)
```

## 📈 Model Performance

The latest model (`gpt2_density_finetuned_cpu_v5`) shows:
- **Training on**: 4,285 examples (PBE, M06, BLYP functionals)
- **Test accuracy**: Available after running evaluation scripts
- **Primary challenge**: Class imbalance (most systems are density-sensitive)

### Supported Functionals
**Primary training data:**
- PBE (Perdew-Burke-Ernzerhof)
- M06 (Minnesota 06)
- BLYP (Becke-Lee-Yang-Parr)

**Available in full dataset:**
- PW91P86, N12, VV10, PKZB, M06L, PBE0, B3LYP, TPSS0, SCAN0, r2SCAN0, and many others

## 🔧 Configuration

### Training Parameters
Edit the main execution section in `finetune_gpt2_cpu.py`:
```python
output_model_dir = run_finetuning(
    jsonl_file_path="finetuning_sets/PBE+M06+BLYP_large_finetuned_dataset.jsonl",
    model_name="gpt2",  # or "gpt2-medium", "gpt2-large"
    num_train_epochs=5,
    per_device_train_batch_size=4,
    learning_rate=5e-5,
    # ... other parameters
)
```

### Dataset Parameters
Modify `preprocessing_data.py` for custom datasets:
```python
# Change functionals and sample size
sampled_df = create_df_subdataset(main_df, ['PBE', 'B3LYP'], 1000)

# Adjust sensitivity threshold
threshold = 2.0  # Energy units for density sensitivity
```

## 🧪 Scientific Applications

This model can be used for:

1. **DFT Method Selection**: Predict when dispersion corrections are necessary
2. **Computational Efficiency**: Skip expensive dispersion calculations when not needed
3. **Database Screening**: Rapid assessment of large molecular databases
4. **Method Development**: Understand functional behavior across chemical space

## 📚 Data Sources

- **GMTKN55**: General Main Group Thermochemistry, Kinetics, and Noncovalent Interactions Database
- **SWARM**: Sensitivity analysis data for dispersion corrections
- **Functional Results**: Comprehensive benchmark calculations for multiple DFT functionals

## 🤝 Contributing

1. **Data Enhancement**: Add new functionals or molecular systems
2. **Model Improvement**: Experiment with different architectures or training strategies
3. **Analysis Tools**: Develop new evaluation and visualization scripts
4. **Documentation**: Improve usage examples and scientific explanations

## 📝 Citation

If you use this work in your research, please cite:

```bibtex
@software{burklab_density_sensitivity,
  title={Density Sensitivity Prediction for DFT Functionals},
  author={[Your Name]},
  year={2025},
  url={[Repository URL]}
}
```
### This work is based on data + knowledge from the following paper:
    ```
    Correcting Dispersion Corrections with Density-Corrected DFT
    
    Minhyeok Lee, Byeongjae Kim, Mingyu Sim, Mihira Sogal, Youngsam Kim, Hayoung Yu, Kieron Burke, and Eunji Sim
    Journal of Chemical Theory and Computation 2024 20 (16), 7155-7167
    DOI: 10.1021/acs.jctc.4c00689
    ```

## ⚠️ Known Limitations

- **Class Imbalance**: Most systems are density-sensitive, leading to biased predictions
- **Limited Functionals**: Training focused on PBE, M06, and BLYP
- **SMILES Representation**: May not capture all relevant chemical features
- **Threshold Sensitivity**: Results depend on the 2.0 energy unit threshold choice

## 📞 Support

For questions, issues, or collaboration opportunities:
- Open an issue in the GitHub repository
- Contact: [Your Contact Information]
- Burke Lab: [Lab Website/Contact]

---

**Keywords**: DFT, Density Functional Theory, Dispersion Corrections, Machine Learning, Computational Chemistry, GPT-2, GMTKN55
