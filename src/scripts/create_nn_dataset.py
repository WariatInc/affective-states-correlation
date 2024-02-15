import pandas as pd
import os
import glob

# Determine project directory
project_dir = os.path.dirname(os.path.abspath(__file__))
print(project_dir)

csv_valence_classes = os.path.join(project_dir, "../datasets/valence_classes_csv")
csv_path_list_valence_classes = glob.glob(csv_valence_classes + '/*')
csv_eda_peaks = os.path.join(project_dir, "EDA_peaks_determined_0_1_csv")
csv_path_list_eda_peaks = glob.glob(csv_eda_peaks + '/*')

y = 2


def find_shortest_sequence(series):
    current_value = None
    current_length = 0
    shortest_sequence = None
    shortest_length = float('inf')
    shortest_start_index = 0  # Default to the first index

    for i, value in enumerate(series):
        if value == current_value:
            current_length += 1
        else:
            if current_length > 0 and current_length < shortest_length:
                shortest_length = current_length
                shortest_sequence = current_value
                shortest_start_index = i - current_length  # Update start index
            current_value = value
            current_length = 1

    # Check for the last sequence
    if current_length > 0 and current_length < shortest_length:
        shortest_length = current_length
        shortest_sequence = current_value
        shortest_start_index = len(series) - current_length

    return shortest_sequence, shortest_length, shortest_start_index


def shortest_seq_among_files():
    for i, path in enumerate(csv_path_list_valence_classes):
        df_valence = pd.read_csv(path, delimiter=';')
        y = 2
        df_valence.columns = df_valence.columns.str.strip()
        classSeries = df_valence['classes']
        seqNum, seqLength, seqIndex = find_shortest_sequence(classSeries)
        print(f"Num: {seqNum}, lenght: {seqLength}, index: {seqIndex}")


def bool_to_int_is_peak():
    for i, path in enumerate(csv_path_list_eda_peaks):
        df_eda = pd.read_csv(path, delimiter=';')
        df_eda["is_peak"] = df_eda["is_peak"].astype(int)
        df_eda.to_csv(path, index=False, sep=';')


def drop_unnecessary_columns():
    for i, path in enumerate(csv_path_list_valence_classes):
        df_valence = pd.read_csv(path, delimiter=';')
        df_valence = df_valence.drop(columns=['FM1', 'FM2', 'FM3', 'FF1', 'FF2', 'FF3'])
        df_valence.to_csv(path, index=False, sep=';')


def create_svm_dataset():
    for eda_path, valance_path in zip(csv_path_list_eda_peaks, csv_path_list_valence_classes):
        df_valence = pd.read_csv(valance_path, delimiter=';')
        df_eda = pd.read_csv(eda_path, delimiter=';')
        extended_valance = df_valence.reindex(df_valence.index.repeat(40))
        extended_valance = extended_valance.reset_index(drop=True)
        extended_valance = extended_valance.drop(extended_valance.index[-40:])
        df_eda["classes"] = extended_valance["classes"]
        svm_path = valance_path.replace('valance_classes_csv', 'svm_dataset')
        df_eda.to_csv(svm_path, index=False, sep=';')


def create_nn_dataset():
    for eda_path, valance_path in zip(csv_path_list_eda_peaks, csv_path_list_valence_classes):
        df_valence = pd.read_csv(valance_path, delimiter=';')

        df_eda = pd.read_csv(eda_path, delimiter=';')

        df_valence["EDA"] = None
        df_valence["is_peak"] = None
        for index, row in df_valence.iterrows():
            eda_array = df_eda.loc[index * 40:(index + 1) * 40 - 1, 'EDA'].tolist()
            # is_peak_array = df_eda.loc[index * 40:(index + 1) * 40 - 1, 'is_peak'].tolist()
            df_valence.at[index, 'EDA'] = eda_array
            # df_valence.at[index, 'is_peak'] = is_peak_array
            try:
                is_peak_value = df_eda.loc[index * 40, 'is_peak']
                df_valence.at[index, 'is_peak'] = is_peak_value
            except Exception as e:
                pass

        nn_path = valance_path.replace('valence_classes_csv', 'nn_dataset')
        df_valence.to_csv(nn_path, index=False, sep=';')


def main():
    # bool_to_int_is_peak()
    # drop_unnecessary_columns()
    # create_svm_dataset()
    create_nn_dataset()


if __name__ == "__main__":
    main()
