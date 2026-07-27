# In file: analysis/00_preprocess_and_save.py

import pickle
import os
import sys

# We need to tell Python where to find our source code in the 'src' directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import load_and_align_data
from src.preprocessing import preprocess_data  # Using the clean version of this file


def run_preprocessing_and_save():
    """
    This script runs the full data loading and preprocessing pipeline ONCE
    and saves the final, clean 'epoched_data' dictionary to a file.
    This is a long process that you only need to run one time.
    """

    # --- DEFINE WHERE TO SAVE THE CLEAN DATA ---
    # We will save it in the 'data' folder for good organization.
    output_filename = os.path.join(
        os.path.dirname(__file__),
        '..',
        'data',
        'preprocessed_epoched_data.pkl'
    )

    print("--- STEP 1: LOADING AND ALIGNING RAW DATA ---")
    aligned_data = load_and_align_data()

    print("\n--- STEP 2: PREPROCESSING DATA (This will take a while...) ---")
    epoched_data = preprocess_data(aligned_data)

    print(f"\n--- STEP 3: SAVING CLEAN DATA to -> {output_filename} ---")
    with open(output_filename, 'wb') as f:
        pickle.dump(epoched_data, f)

    print("\n--- Preprocessing and saving complete! ---")
    print("You can now run your analysis scripts, which will load this clean file.")
    print(f"Total trials processed and saved: {len(epoched_data)}")


if __name__ == '__main__':
    run_preprocessing_and_save()