# src/preprocessing.py

import mne
import numpy as np

# --- FUNCTION 1: The original EEG-only preprocessing ---
def preprocess_data(aligned_data):
    """
    Takes the aligned data dictionary and performs a standard EEG
    preprocessing pipeline: filtering, ICA for artifact removal, and epoching.

    Returns a dictionary of epoched data ready for feature extraction.
    """
    # Configuration for preprocessing
    LOW_CUTOFF = 0.1
    HIGH_CUTOFF = 48
    ICA_N_COMPONENTS = 15
    EPOCH_DURATION = 0.200
    EPOCH_OVERLAP = 0.150

    epoched_data = {}
    print("\n--- Starting EEG Preprocessing ---")

    for trial_name, data in aligned_data.items():
        print(f"  Processing trial: {trial_name}")

        info = mne.create_info(ch_names=data['ch_names'], sfreq=data['sfreq'], ch_types='eeg')
        raw = mne.io.RawArray(data['eeg'], info, verbose=False)


        # Set Standard Montage for BrainVision 32 Channels
        montage_set_successfully = False # Flag to track if montage was set
        try:
            montage = mne.channels.make_standard_montage('standard_1020')
            raw.set_montage(montage)
            print("    Set 'standard_1020' montage.")
            montage_set_successfully = True
        except Exception as e:
            print(f"    WARNING: Could not set standard 10-20 montage. Error: {e}")
            print("    Laplacian referencing requires channel locations. Please verify channel names.")


        # --- Re-referencing: Convert to Common Average Reference (Projection) ---
        # CHECK HERE: Use raw.montage instead of raw.info['montage']
        if montage_set_successfully: # Use the flag we set, or raw.montage is not None
            print("    Applying Common Average Reference (projection).")
            # Note: For CSD, it's often better to ensure an average reference *first*
            # if your data wasn't already referenced this way.
            raw.set_eeg_reference(ref_channels='average', projection=True, verbose=False)

        # Step 1: Band-pass Filtering
        iir_params = dict(order=4, ftype='butter')
        print(f"    Applying {LOW_CUTOFF}-{HIGH_CUTOFF} Hz Butterworth bandpass filter (4th order).")
        # CORRECTED LINE BELOW: Changed 'fir_design' to 'method'
        raw.filter(l_freq=LOW_CUTOFF, h_freq=HIGH_CUTOFF, method='iir', iir_params=iir_params, verbose=False)

        # --- Step 1.5: Re-referencing with Laplacian Montage (Current Source Density) ---
        # CHECK HERE: Use raw.montage instead of raw.info['montage']
        if montage_set_successfully: # Use the flag, or raw.montage is not None
            print("    Computing Current Source Density (Laplacian effect).")
            raw_csd = mne.preprocessing.compute_current_source_density(raw, verbose=False)
            raw = raw_csd
        else:
            print("    Skipping Current Source Density: Montage not set.")


        # Step 2: ICA for artifact removal
        ica = mne.preprocessing.ICA(n_components=ICA_N_COMPONENTS, random_state=97, max_iter='auto')
        print("    Fitting ICA...")
        ica.fit(raw)

        eog_channels = ['Fp1', 'Fp2']
        eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name=eog_channels)

        if eog_indices:
            print(f"    Found {len(eog_indices)} EOG component(s). Removing them.")
            ica.exclude = eog_indices
            ica.apply(raw, verbose=False)
        else:
            print("    No EOG components found.")

        # Step 3: Epoching (Windowing)
        print("    Epoching data...")
        epochs = mne.make_fixed_length_epochs(raw, duration=EPOCH_DURATION, overlap=EPOCH_OVERLAP, verbose=False)

        final_ch_names = epochs.ch_names

        force_epochs = []
        n_times = epochs.get_data().shape[2]
        step = int(n_times * (1 - (EPOCH_OVERLAP / EPOCH_DURATION)))

        for i in range(len(epochs)):
            start = i * step
            end = start + n_times
            force_epochs.append(data['force'][:, start:end])

        epoched_data[trial_name] = {
            'eeg_epochs': epochs.get_data(),
            'force_epochs': np.array(force_epochs),
            'sfreq': data['sfreq'],
            'ch_names': final_ch_names
        }

    print(f"\n--- Preprocessing complete. Processed {len(epoched_data)} trials. ---")
    return epoched_data

# --- FUNCTION 2: The new EEG + EMG preprocessing ---

def preprocess_data_with_emg(aligned_data):
    """
    Takes the aligned data dictionary (including EMG features) and performs a
    standard EEG preprocessing pipeline. It also epochs the EMG and Force data
    to match the EEG epochs.
    """
    # Configuration for preprocessing
    EPOCH_DURATION = 0.200  # 200 milliseconds
    EPOCH_OVERLAP = 0.150  # Overlap by 150ms (creates a new epoch every 50ms)

    # EEG-specific configuration
    LOW_CUTOFF = 0.1
    HIGH_CUTOFF = 48.0
    ICA_N_COMPONENTS = 15

    epoched_data = {}
    print("\n--- Starting EEG Preprocessing and Data Epoching (with EMG) ---")

    for trial_name, data in aligned_data.items():
        print(f"  Processing trial: {trial_name}")

        # --- Part 1: EEG-specific Preprocessing ---
        # This part remains the same, as it only operates on the 'eeg' data.
        info = mne.create_info(ch_names=data['ch_names'], sfreq=data['sfreq'], ch_types='eeg')
        raw_eeg = mne.io.RawArray(data['eeg'], info, verbose=False)

        montage = mne.channels.make_standard_montage('standard_1020')
        raw_eeg.set_montage(montage, on_missing='warn')

        iir_params = dict(order=4, ftype='butter')
        raw_eeg.filter(l_freq=LOW_CUTOFF, h_freq=HIGH_CUTOFF, method='iir', iir_params=iir_params, verbose=False)

        raw_csd = mne.preprocessing.compute_current_source_density(raw_eeg, verbose=False)

        ica = mne.preprocessing.ICA(n_components=ICA_N_COMPONENTS, random_state=97, max_iter='auto')
        ica.fit(raw_csd)  # Fit ICA on the CSD data for better source separation
        eog_indices, _ = ica.find_bads_eog(raw_csd, ch_name=['Fp1', 'Fp2'])
        if eog_indices:
            print(f"    Found and removing {len(eog_indices)} EOG components.")
            ica.exclude = eog_indices
            ica.apply(raw_csd, verbose=False)

        # The final cleaned EEG data is in raw_csd
        cleaned_eeg_raw = raw_csd

        # --- Part 2: Epoching for ALL Data Streams (EEG, EMG, Force) ---
        print("    Epoching all data streams...")

        # Create epochs for the cleaned EEG data
        eeg_epochs = mne.make_fixed_length_epochs(cleaned_eeg_raw, duration=EPOCH_DURATION,
                                                  overlap=EPOCH_OVERLAP, verbose=False,  preload=True)

        # Manually create matching epochs for EMG and Force data
        n_epochs = len(eeg_epochs)
        epoch_samples = eeg_epochs.get_data().shape[2]
        step_samples = int(epoch_samples * (1 - (EPOCH_OVERLAP / EPOCH_DURATION)))

        emg_feature_epochs = []
        force_epochs = []

        for i in range(n_epochs):
            start = i * step_samples
            end = start + epoch_samples

            # Check if the slice is within the bounds of the data arrays
            if end <= data['emg_features'].shape[1] and end <= data['force'].shape[1]:
                emg_feature_epochs.append(data['emg_features'][:, start:end])
                force_epochs.append(data['force'][:, start:end])
            else:
                # If the last epoch would go out of bounds, break the loop
                # This can happen due to small rounding differences in sample counts
                break

        # We might have created fewer EMG/Force epochs if the last one was out of bounds,
        # so we truncate the EEG epochs to match.
        final_n_epochs = len(force_epochs)

        epoched_data[trial_name] = {
            'eeg_epochs': eeg_epochs.get_data()[:final_n_epochs, :, :],
            'emg_feature_epochs': np.array(emg_feature_epochs),
            'force_epochs': np.array(force_epochs),
            'sfreq': data['sfreq'],
            'ch_names': eeg_epochs.ch_names
        }

    print(f"\n--- Preprocessing and epoching complete. Processed {len(epoched_data)} trials. ---")
    return epoched_data
