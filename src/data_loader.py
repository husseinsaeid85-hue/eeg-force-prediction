# In file: src/data_loader.py

import pandas as pd
import mne
import os
import re


def load_and_align_data():
    """
    Loads EEG, pre-processed EMG features, and raw Force data from their
    separate files, aligns them trial-by-trial, and returns a dictionary.
    EEG and EMG are assumed to be recorded at the same sampling rate.
    """
    # --- Configuration ---
    FORCE_DATA_PATH = r'data/pilot_20250807/Kinetics_EEG_AUXData_dataframe.pkl'
    EEG_BASE_PATH = r'data/pilot_20250807/eeg'
    EMG_DATA_PATH = r'data/pilot_20250807/KineticsRecord_Subj2_RMS_dataframe.pkl'

    # --- Sampling Rates ---
    # As per supervisor, Force data is at 2048 Hz
    ORIGINAL_FORCE_SAMPLING_RATE = 2048.0
    # EMG is also at 2048 Hz, same as EEG, so no resampling needed for EMG.

    # --- Load DataFrames ---
    print("--- Loading dataframes for Force and EMG... ---")
    force_df = pd.read_pickle(FORCE_DATA_PATH)
    emg_df = pd.read_pickle(EMG_DATA_PATH)
    print("--- Dataframes loaded successfully. ---")

    print("--- Finding and grouping all EEG files... ---")
    eeg_files_by_rec = {}

    def natural_sort_key(s):
        return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

    all_vhdr_files = sorted([f for f in os.listdir(EEG_BASE_PATH) if f.endswith('.vhdr')], key=natural_sort_key)

    name_map = {'D1D2': 'D12', 'D1D2D3': 'D123', 'D1D2D3D4D5': 'D12345'}
    base_names = sorted(list(force_df['recording'].unique()), key=len, reverse=True)

    for fname in all_vhdr_files:
        if 'flexex' in fname.lower():
            continue
        file_base = os.path.splitext(fname)[0]
        match = None
        for bn in base_names:
            mapped_bn = name_map.get(bn, bn)
            if file_base.lower().startswith(mapped_bn.lower()):
                match = bn
                break
        if match:
            if match not in eeg_files_by_rec:
                eeg_files_by_rec[match] = []
            eeg_files_by_rec[match].append(fname)

    aligned_data = {}
    print("\n--- Starting trial-by-trial alignment of EEG, Force, and EMG data ---")

    grouped_force_df = force_df.groupby('recording')
    grouped_emg_df = emg_df.groupby('recording')

    for rec_name, force_group in grouped_force_df:
        if rec_name not in eeg_files_by_rec or rec_name not in grouped_emg_df.groups:
            continue

        emg_group = grouped_emg_df.get_group(rec_name)
        eeg_filenames = eeg_files_by_rec[rec_name]
        num_eeg_files = len(eeg_filenames)
        if num_eeg_files == 0:
            continue

        samples_per_trial_force = len(force_group) // num_eeg_files
        samples_per_trial_emg = len(emg_group) // num_eeg_files

        for i in range(num_eeg_files):
            trial_num = i + 1
            eeg_file = eeg_filenames[i]

            # Load the raw EEG data for this trial
            raw_eeg = mne.io.read_raw_brainvision(os.path.join(EEG_BASE_PATH, eeg_file), preload=True, verbose=False)
            eeg_data_final = raw_eeg.get_data()
            # The EEG sampling rate is the master rate we align to
            final_sfreq = raw_eeg.info['sfreq']

            # --- Prepare Force data (still needs resampling if its original rate differs from EEG's final rate) ---
            start_index_force = i * samples_per_trial_force
            end_index_force = start_index_force + samples_per_trial_force
            force_df_trial = force_group.iloc[start_index_force:end_index_force]
            force_channel_names = [col for col in force_df.columns if col.startswith('D')]
            force_data_np = force_df_trial[force_channel_names].values.T
            force_data_resampled = mne.filter.resample(force_data_np, up=final_sfreq,
                                                       down=ORIGINAL_FORCE_SAMPLING_RATE, method='polyphase',
                                                       verbose=False)

            # --- Prepare EMG data (NO RESAMPLING NEEDED) ---
            start_index_emg = i * samples_per_trial_emg
            end_index_emg = start_index_emg + samples_per_trial_emg
            emg_df_trial = emg_group.iloc[start_index_emg:end_index_emg]
            emg_channel_names = [col for col in emg_df.columns if col.startswith('GR')]
            emg_data_np = emg_df_trial[emg_channel_names].values.T
            # Since EMG and EEG have the same sampling rate, we just use the numpy array directly.
            # This assumes the number of samples in the EMG slice matches the EEG file.
            emg_data_final = emg_data_np

            # Get final EEG data and ensure all signals have the same final length by truncating
            min_length = min(eeg_data_final.shape[1], force_data_resampled.shape[1], emg_data_final.shape[1])

            unique_trial_name = f"{rec_name}_trial_{trial_num}"
            aligned_data[unique_trial_name] = {
                'eeg': eeg_data_final[:, :min_length],
                'emg_features': emg_data_final[:, :min_length],
                'force': force_data_resampled[:, :min_length],
                'sfreq': final_sfreq,  # The final sampling rate for all signals
                'ch_names': raw_eeg.ch_names
            }
            print(f"    Successfully aligned and loaded Trial: {unique_trial_name}")

    print(f"\n--- Data loading complete. Found and aligned {len(aligned_data)} trials. ---")
    return aligned_data


# --- This is a test block to run the function directly for debugging ---
if __name__ == "__main__":
    print("--- Running data_loader.py directly for testing ---")

    # Call the main function in the script
    aligned_data = load_and_align_data()

    # --- Print a summary of the results to verify ---
    if aligned_data:
        print("\n--- Verification Summary ---")
        # Get the name of the first trial to inspect its contents
        first_trial_name = list(aligned_data.keys())[0]
        print(f"Successfully loaded {len(aligned_data)} trials.")
        print(f"Inspecting the first trial: '{first_trial_name}'")

        first_trial_data = aligned_data[first_trial_name]

        # Check that all the required data keys are present
        required_keys = ['eeg', 'emg_features', 'force', 'sfreq', 'ch_names']
        for key in required_keys:
            if key in first_trial_data:
                # Get the shape of the data arrays to check alignment
                data_shape = first_trial_data[key].shape if hasattr(first_trial_data[key], 'shape') else 'N/A'
                print(f"  - Found key '{key}'. Shape/Value: {data_shape}")
            else:
                print(f"  - !!! MISSING KEY: '{key}' !!!")
    else:
        print("\n--- Verification Failed: No data was loaded. ---")