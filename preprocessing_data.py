import pandas as pd
import ast
import json
import time
from datetime import datetime
from typing import List, Optional


# Notes:
#     - functionals without D3(BJ): PW91P86 (has D3(0)), N12 (has D3(0)), VV10 (has neither, only has VV10), PKZB (has D3(0)), M06L (has D3(0)), M11L (has D3(0)), MN12L (has D3(BJ)), MN15L (has D3(0)), B3LYP-NL (has VV10), mPW1LYP (has D3(0)), PW1PW (has D3(0)), M05 (has D3(0)), M052X (has D3(0)), M06 (has D3(0)), M062X (has D3(0)), M08HX (has D3(0)), M11 (has D3(BJ)), SOGGA11X (has D3(BJ)), N12SX (has D3(BJ)), MN12SX (has D3(BJ)), MN15 (has D3(BJ)), ωB97X-D3 (has D3(0)), ωB97X-V (has ωB97X-V), APFD (has APFD), DSD-PBEP86 (has D3(BJ)), DSD-PBEB95 (has D3(BJ)), r2SCAN-3c (has r2SCAN-3c)


def make_finetuning_dataset(df: pd.DataFrame, timestamp: str, include_subset_in_prompt: bool = True) -> None:
    '''
    Make labeled finetuning set, adding colums for the SMILES string, density impact, prompt, and completion. 
    SMILES string is created by alternating values from systems, then stoichiometry.
    Density impact is calculated by setting a threshold and then checking if adding D3(BJ) dispersion corrected created an
    impact on the calculated density above this threshold.
    For the prompt+completion
    '''


    SWARM_df = pd.read_csv("all_v2_SWARM.csv")
    #for each row
    for index, row in df.iterrows():
        #drop if not involving explicit dispersion correction
        #df_filtered = df.dropna(subset=['D3(0)', 'D3(BJ)'])
        #continue
        #Add SMILES strings to each row
        df.at[index, 'smiles'] = construct_smiles_string(row['Stoichiometry'], row['Systems'])

        #Create binary label to check dispersion impact
        threshold = 2.0
        
        #check all_v2_SWARM[setname][rxnidx][calctype][S]; if its less than threshold mark df['density_sensitive'] +false, if equal or greater mark true
        setname = row['subset']
        calctype = row['functional']
        rxnidx = int(row['#'])

        match = SWARM_df[
        (SWARM_df['setname'] == setname) &
        (SWARM_df['calctype'] == calctype) &
        (SWARM_df['rxnidx'] == rxnidx)
        ]

        is_sensitive = ''
        if not match.empty:
            s_val = match.iloc[0]['S']
            is_sensitive = s_val > threshold
        else:
            is_sensitive = 'N/A'  # or np.nan, depending on your needs

        df.at[index, 'density_sensitive'] = is_sensitive

        #Create prompt + completions
        if include_subset_in_prompt:
            prompt = f"Subset: {setname}, Functional: {calctype}, SMILES: {df.at[index, 'smiles']}"
        else:
            prompt = f"Functional: {calctype}, SMILES: {df.at[index, 'smiles']}"
        df.at[index, 'prompt'] = prompt
        df.at[index, 'completion'] = f"Density sensitive: {df.at[index, 'density_sensitive']}"


    #Create JSONL 
    filename = f"finetuning_sets/finetuned_data_{timestamp}.jsonl"
    with open(filename, "w") as f:
        for _, row in df.iterrows():
            json.dump({"prompt": row['prompt'], "completion": row['completion']}, f)
            f.write("\n")


def construct_smiles_string(stoichiometry_list: str, systems_list: str) -> str:
    '''
    Construct a SMILES string based on the lists of stoichiometry + systems

    Arguments:
        stoichiometry (str): string representation of a list of the stoichiometry values
        systems_list (str): string representation of a list of the systems values

    Returns:
        str: constructed SMILES string
    '''
    # Parse string representations to lists
    try:
        # Try using ast.literal_eval for safe parsing
        if isinstance(stoichiometry_list, str):
            stoichiometry_list = ast.literal_eval(stoichiometry_list)
        if isinstance(systems_list, str):
            systems_list = ast.literal_eval(systems_list)
    except (SyntaxError, ValueError):
        # If literal_eval fails, try a simple string parsing approach
        # Remove brackets and split by commas
        if isinstance(stoichiometry_list, str):
            stoichiometry_list = stoichiometry_list.strip('[]').replace('"', '').replace("'", "").split(',')
            stoichiometry_list = [s.strip() for s in stoichiometry_list]
        if isinstance(systems_list, str):
            systems_list = systems_list.strip('[]').replace('"', '').replace("'", "").split(',')
            systems_list = [s.strip() for s in systems_list]
    
    # Ensure we have lists
    if not isinstance(stoichiometry_list, list):
        stoichiometry_list = [stoichiometry_list]
    if not isinstance(systems_list, list):
        systems_list = [systems_list]
    
    # Construct the SMILES string
    smiles_string = ''
    for i in range(len(systems_list)):
        smiles_string += str(systems_list[i])
        smiles_string += str(stoichiometry_list[i])

    return smiles_string



def create_df_subdataset(df: pd.DataFrame, functionals: List[str], num_samples: int, seed: int = 21) -> pd.DataFrame:
    '''
    Creates a sub-dataset from a given DataFrame based on a list of functionals.

     Args:
         df (pd.DataFrame): The source DataFrame to filter.
         functionals (List[str]): A list of functional names to include.
         num_samples (int): The number of samples to randomly draw.
         seed (int, optional): A random seed for reproducibility. Defaults to 21.

     Returns:
         pd.DataFrame: A new DataFrame containing the sampled sub-dataset.
    '''
    print(f"Filtering for functionals: {functionals}...")
    # Use .isin() to correctly filter for multiple values in the 'functional' column.
    filtered_df = df[df['functional'].isin(functionals)]
    print(f"Found {len(filtered_df)} entries for the specified functionals.")

    if len(filtered_df) == 0:
        print("Warning: No entries found. Returning an empty DataFrame.")
        return filtered_df

    if num_samples > len(filtered_df):
        print(f"Warning: Requested {num_samples} samples, but only {len(filtered_df)} are available. Using all available entries.")
        return filtered_df
    
    print(f"Sampling {num_samples} entries with random_state={seed}...")
    sampled_df = filtered_df.sample(n=num_samples, random_state=seed)
    return sampled_df


if __name__ == "__main__":
    try:
        main_df = pd.read_csv("cleaned_scraped_gmtkn_data.csv")
    except FileNotFoundError:
        print("Error: 'cleaned_scraped_gmtkn_data.csv' not found. Please ensure the file is in the correct directory.")
        exit()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    #valid functionals = ['PBE', 'REVPBE', 'BLYP', 'TPSS', 'SCAN', 'r2SCAN', 'M06L', 'PBE0', 'REVPBE0', 'B3LYP', 'TPSS0', 'SCAN0', 'r2SCAN0', 'M06']
    #Create a sub dataset with given number of samples from the given functionals 
    sampled_df = create_df_subdataset(main_df, ['PBE', 'M06', 'BLYP'], 4284)
    
    if not sampled_df.empty:
        # Create the finetuning JSONL file from the sampled DataFrame.
        # INCLUDES SUBSET IN PROMPT BY 
        print("subdataset: ", sampled_df[:10])
        make_finetuning_dataset(sampled_df, timestamp, False)
        pathname = f"finetuning_sets/finetuned_data_{timestamp}.jsonl"
        print(f"Successfully created finetuning dataset at: {pathname}")
    else:
        print("Skipping dataset creation because no data was found for the specified functionals.")



    #note: should be 1428 entries per functional (one entry per rxn, sum rxns of all subsets)
    
