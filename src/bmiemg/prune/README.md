# Prune Dataset
This is a pipeline architecture that has the sole purpose of helping process datasets from raw data into final files that can be used directly for trainning. There can be multiple protocols used to generate different datasets.

---

## 1. Protocols
- **RawArchiveProtocolV1** - Used to transform the raw data acquired by the acquisition team into an usable dataset file.
  - RemoveSessionsStep - Remove corrupted or unuanted sessions based ona pre-defined list (kraken and corrupted/incomplete sessions)
  - SignalSplitStep - Split the signal into only EMG channels and Markers
  - ProtocolDefinerStep - Adds a tag to the XDF Data marking which protocol version was used during acquisition
  - MarkerCorrectionStep - Colpases the markers for simplified ones
  - AnalogFilterStep - Applies the analog filter that the signal would have in the device
  - TTVSplitStep - Splits into train/test/validation sets. two files are produced: client and server. The client has train and test data and markers and only test data. The server has the test markers.

## 2. Steps
Every step is a `PruneStep`: it receives a list of `DataActor` objects and returns the transformed list. Each step holds its parameters directly as its own attributes (set when the step is constructed), so the fields listed under each step are passed straight to the step's constructor.

### RemoveSessionStep (implemented)
Drops whole recordings from the dataset. It keeps only the actors whose `file_name` is absent from a removal list, which is how corrupted, incomplete, and kraken sessions get stripped out before anything else runs.

Attributes:
- `files_to_remove: list[str]`: File names, without suffix, to drop. Any actor whose `file_name` matches an entry is removed.

### SignalSplitStep (implemented)
Reduces each recording to one signal modality plus its markers. Marker streams are always kept. For every other stream it keeps only the channels whose type matches the requested signal (EMG accepts `emg` and `aux` channels, EEG accepts `eeg`) and rewrites that stream's `time_series`, type, and channel metadata to match. A stream left with no matching channel is dropped.

Attributes:
- `signal: Signal`: Target modality, `"emg"` or `"eeg"`. Checked at construction, and any other value raises `ValueError`.

### ProtocolDefinerStep (implemented)
Tags each `XdfData` with the protocol version used during acquisition by setting its `protocol_version` field, which defaults to `0` for undefined. The version is derived from the recording date: it is `1` plus the number of cutoff dates the recording falls on or after.

Attributes:
- `cuttoff_dates: list[date]`: Boundary dates between protocol versions. For each actor, `protocol_version` becomes `1 + (number of cutoffs <= actor.date)`; actors without a date keep the default.
