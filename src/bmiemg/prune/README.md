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
- RemoveSessionSteps -
