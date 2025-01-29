# Preprocessed Tern data
This data has been processed via the following steps:

1. Track IDs have been made globally unique
2. Tracks with fewer than 50 observations have been dropped
3. Columns have been renamed
    - TSECS renamed to t
    - BNGX renamed to observation_1
    - BNGY renamed to observation_2
4. All data has been concatenated into a single file