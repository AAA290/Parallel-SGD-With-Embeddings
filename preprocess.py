# preprocess.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

def preprocess_and_save(file_path,
                        save_prefix="nytaxi_processed",
                        chunksize=50000,
                        test_size=0.3,
                        random_state=42):
    """
    Preprocess CSV in chunks and produce:
      - train/test split for numeric features (scaled) and categorical id arrays (as integer indices)
      - mapping/vocab files for categorical IDs (pu/do/rate/payment)
      - scaler fitted on train numeric features
    The saved files are compressed npz + joblib for scalers/vocabs.
    """

    print(f"Loading dataset (chunked): {file_path}")
    usecols = [
        'tpep_pickup_datetime', 'tpep_dropoff_datetime', 'passenger_count',
        'trip_distance', 'RatecodeID', 'PULocationID', 'DOLocationID',
        'payment_type', 'extra', 'total_amount'
    ]

    # nullable integers for columns that may contain NA
    dtypes = {
        'VendorID': 'Int8',
        'passenger_count': 'Int8',
        'trip_distance': 'float32',
        'RatecodeID': 'Int16',
        'PULocationID': 'Int32',
        'DOLocationID': 'Int32',
        'payment_type': 'Int8',
        'fare_amount': 'float32',
        'extra': 'float32',
        'total_amount': 'float32'
    }

    # lists to collect chunked data
    numeric_chunks = []
    pu_chunks = []
    do_chunks = []
    rate_chunks = []
    pay_chunks = []
    y_chunks = []

    pu_set = set()
    do_set = set()
    rate_set = set()
    pay_set = set()

    reader = pd.read_csv(
        file_path,
        usecols=usecols,
        dtype=dtypes,
        # parse_dates=["tpep_pickup_datetime", "tpep_dropoff_datetime"],
        low_memory=False,
        chunksize=chunksize
    )

    total_rows = 0
    kept_rows = 0
    for i, chunk in enumerate(reader):
        print(f"[chunk {i+1}] original rows: {len(chunk)}")
        total_rows += len(chunk)

        # drop rows with core missing values
        chunk = chunk.dropna()
        # chunk = chunk.dropna(subset=['tpep_pickup_datetime', 'tpep_dropoff_datetime',
                                    #  'passenger_count', 'trip_distance', 'total_amount'])

        # filter positive target
        chunk = chunk[chunk['total_amount'] > 0]

        print(f"[chunk {i+1}] after drop: {len(chunk)}")

        if len(chunk) == 0:
            continue

        # ensure datetimes parsed; coerce with explicit format if the dataset follows known format:
        # try:
        #     chunk['tpep_pickup_datetime'] = pd.to_datetime(
        #         chunk['tpep_pickup_datetime'], format="%Y-%m-%d %H:%M:%S", errors='coerce'
        #     )
        #     chunk['tpep_dropoff_datetime'] = pd.to_datetime(
        #         chunk['tpep_dropoff_datetime'], format="%Y-%m-%d %H:%M:%S", errors='coerce'
        #     )
        # except Exception:
        chunk['tpep_pickup_datetime'] = pd.to_datetime(chunk['tpep_pickup_datetime'], format="mixed", errors='coerce')
        chunk['tpep_dropoff_datetime'] = pd.to_datetime(chunk['tpep_dropoff_datetime'], format="mixed", errors='coerce')

        chunk = chunk.dropna(subset=['tpep_pickup_datetime', 'tpep_dropoff_datetime'])
        if len(chunk) == 0:
            print(f"[chunk {i+1}] skipped")
            continue

        # engineered numeric features
        chunk['trip_duration'] = (chunk['tpep_dropoff_datetime'] - chunk['tpep_pickup_datetime']).dt.total_seconds() / 60.0
        chunk['pickup_hour'] = chunk['tpep_pickup_datetime'].dt.hour.fillna(0).astype(int)
        chunk['pickup_dayofweek'] = chunk['tpep_pickup_datetime'].dt.dayofweek.fillna(0).astype(int)

        # select numeric columns to feed into scaler / NN numeric input
        numeric_cols = ['passenger_count', 'trip_distance', 'extra', 'trip_duration', 'pickup_hour', 'pickup_dayofweek']

        # ensure columns exist and in correct dtype
        for c in numeric_cols:
            if c not in chunk.columns:
                chunk[c] = 0.0
        numeric_arr = chunk[numeric_cols].astype('float32').to_numpy()

        # categorical id columns (keep as integers for embedding later)
        pu_arr = chunk['PULocationID'].astype('int32').to_numpy()
        do_arr = chunk['DOLocationID'].astype('int32').to_numpy()
        rate_arr = chunk['RatecodeID'].astype('int32').to_numpy()
        pay_arr = chunk['payment_type'].astype('int32').to_numpy()

        # target: log1p of total_amount
        y_arr = np.log1p(chunk['total_amount'].astype('float32').to_numpy())

        # accumulate
        numeric_chunks.append(numeric_arr)
        pu_chunks.append(pu_arr)
        do_chunks.append(do_arr)
        rate_chunks.append(rate_arr)
        pay_chunks.append(pay_arr)
        y_chunks.append(y_arr)

        # update sets of unique categories
        pu_set.update(np.unique(pu_arr).tolist())
        do_set.update(np.unique(do_arr).tolist())
        rate_set.update(np.unique(rate_arr).tolist())
        pay_set.update(np.unique(pay_arr).tolist())

        kept_rows += len(chunk)
        print(f"[chunk {i+1}] kept rows so far: {kept_rows}")

    if kept_rows == 0:
        raise RuntimeError("No data kept after preprocessing. Check filters.")

    # concatenate all chunks
    print("Concatenating chunks into full arrays.")
    X_numeric = np.concatenate(numeric_chunks, axis=0)
    PU = np.concatenate(pu_chunks, axis=0)
    DO = np.concatenate(do_chunks, axis=0)
    RATE = np.concatenate(rate_chunks, axis=0)
    PAY = np.concatenate(pay_chunks, axis=0)
    Y = np.concatenate(y_chunks, axis=0)

    N = X_numeric.shape[0]
    print(f"Total rows kept: {N}, numeric shape: {X_numeric.shape}")

    # train/test split (indices), then build mappings and scalers
    idx = np.arange(N)
    train_idx, test_idx = train_test_split(idx, test_size=test_size, random_state=random_state, shuffle=True)

    # --- Build vocabulary / mapping for categorical IDs ---
    # NOTE: for simplicity we build mapping from the entire dataset (train-only mapping is also valid)
    pu_unique = np.array(sorted(list(pu_set)))
    do_unique = np.array(sorted(list(do_set)))
    rate_unique = np.array(sorted(list(rate_set)))
    pay_unique = np.array(sorted(list(pay_set)))

    # pu_to_idx = {int(v): i for i, v in enumerate(pu_unique)}
    # do_to_idx = {int(v): i for i, v in enumerate(do_unique)}
    # rate_to_idx = {int(v): i for i, v in enumerate(rate_unique)}
    # pay_to_idx = {int(v): i for i, v in enumerate(pay_unique)}

    # Map full arrays to continuous 0..V-1 indices (fast mapping via pandas.Categorical)
    PU_codes = pd.Categorical(PU, categories=pu_unique).codes.astype('int32')
    DO_codes = pd.Categorical(DO, categories=do_unique).codes.astype('int32')
    RATE_codes = pd.Categorical(RATE, categories=rate_unique).codes.astype('int16')
    PAY_codes = pd.Categorical(PAY, categories=pay_unique).codes.astype('int8')

    # split into train / test arrays
    X_train_num = X_numeric[train_idx]
    X_test_num = X_numeric[test_idx]

    pu_train = PU_codes[train_idx]
    pu_test = PU_codes[test_idx]

    do_train = DO_codes[train_idx]
    do_test = DO_codes[test_idx]

    rate_train = RATE_codes[train_idx]
    rate_test = RATE_codes[test_idx]

    pay_train = PAY_codes[train_idx]
    pay_test = PAY_codes[test_idx]

    y_train = Y[train_idx]
    y_test = Y[test_idx]

    print("Shapes after split:")
    print("X_train_num:", X_train_num.shape, "X_test_num:", X_test_num.shape)
    print("pu_train:", pu_train.shape, "do_train:", do_train.shape, "y_train:", y_train.shape)

    # --- Fit scaler on numeric train, transform both ---
    scaler = StandardScaler()
    scaler.fit(X_train_num)
    X_train_num_scaled = scaler.transform(X_train_num).astype('float32')
    X_test_num_scaled = scaler.transform(X_test_num).astype('float32')

    # --- Save everything ---
    out_npz = f"{save_prefix}.npz"
    print(f"Saving arrays to {out_npz} (compressed).")
    np.savez_compressed(
        out_npz,
        X_train_num=X_train_num_scaled,
        X_test_num=X_test_num_scaled,
        pu_train=pu_train, pu_test=pu_test,
        do_train=do_train, do_test=do_test,
        rate_train=rate_train, rate_test=rate_test,
        pay_train=pay_train, pay_test=pay_test,
        y_train=y_train, y_test=y_test
    )

    # save scalers and vocabs
    vocabs = {
        'pu_vocab': pu_unique.tolist(),
        'do_vocab': do_unique.tolist(),
        'rate_vocab': rate_unique.tolist(),
        'pay_vocab': pay_unique.tolist()
    }
    joblib.dump(scaler, f"{save_prefix}_scaler.pkl")
    joblib.dump(vocabs, f"{save_prefix}_vocabs.pkl")

    # save numeric feature names
    numeric_feature_names = ['passenger_count', 'trip_distance', 'extra', 'trip_duration', 'pickup_hour', 'pickup_dayofweek']
    with open(f"{save_prefix}_numeric_features.txt", "w") as f:
        f.write("\n".join(numeric_feature_names))

    # save metadata
    meta = {
        'N_total': N,
        'N_train': len(train_idx),
        'N_test': len(test_idx),
        'pu_cardinality': int(len(pu_unique)),
        'do_cardinality': int(len(do_unique)),
        'rate_cardinality': int(len(rate_unique)),
        'pay_cardinality': int(len(pay_unique))
    }
    joblib.dump(meta, f"{save_prefix}_meta.pkl")

    print("Saved scalers, vocabs, metadata.")
    print("Preprocessing completed successfully.")

if __name__ == "__main__":
    # example usage (change file path if needed)
    # assert os.path.exists("nytaxi2022.csv"), "nytaxi2022.csv not found in current dir"
    # preprocess_and_save("nytaxi2022.csv", save_prefix="nytaxi_processed")

    