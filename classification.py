import numpy as np
import os
import time


def format_time(seconds):
    """
    Formatting the time value for output
        :param seconds: seconds value
        :return: String with formatted time value
    """
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    d, h = divmod(h, 24)

    time_str = ""
    if d > 0:
        time_str += f"{int(d)} days "
    if h > 0:
        time_str += f"{int(h)} hours "
    if m > 0:
        time_str += f"{int(m)} minutes "
    if s > 0:
        time_str += f"{int(s)} sec."

    return time_str


def calculate_mcc(data: np.ndarray, data_pred: np.ndarray) -> (float, float):
    """
    Calculates the MCC
        :param data: numpy array of entropy data
        :param data_pred: data one step back
        :return: Value MCC and value Vth
    """
    max_val = max(np.max(data), np.max(data_pred))
    min_val = min(np.min(data), np.min(data_pred))

    dEn = (max_val - min_val) / 1000
    mcc_max = 0
    uth_m = 0

    uth = min_val - dEn
    for _ in range(1002):
        uth += dEn

        conf_matrix = np.zeros((2, 2), dtype=int)
        conf_matrix[1, 1] = np.sum(data >= uth)
        conf_matrix[1, 0] = np.sum(data < uth)
        conf_matrix[0, 1] = np.sum(data_pred >= uth)
        conf_matrix[0, 0] = np.sum(data_pred < uth)

        mcc_numerator = conf_matrix[0, 0] * conf_matrix[1, 1] - conf_matrix[0, 1] * conf_matrix[1, 0]
        mcc_denominator = np.sqrt(np.prod(conf_matrix.sum(axis=0)) * np.prod(conf_matrix.sum(axis=1)))

        mcc = mcc_numerator / mcc_denominator if mcc_denominator != 0 else 0

        if abs(mcc) > abs(mcc_max):
            mcc_max = mcc
            uth_m = uth

    if mcc_max < -1:
        mcc_max = 0

    return mcc_max, uth_m


def global_calculate_gefmcc(conf: dict):
    """
    General function for calculating GEFMCC from created entropy files for a given configuration
        :param conf: base configuration
        :return: GEFMCC value and file report
    """

    print('---------------------')
    print('START Calculating GEFMCC')

    start_time = time.time()

    name_map = conf['config_entropy']['use_chaotic_map']
    config_map = conf['config_gen'].get(name_map, None)

    if config_map is None:
        print(f"Invalid value to configuration ['config_entropy']['use_chaotic_map']")
        exit(0)

    h_values = np.arange(config_map['h1'],
                         config_map['h2'] + config_map['h_step'],
                         config_map['h_step'])
    h_values = np.round(h_values, decimals=6)

    len_h_values = len(h_values)
    MCC_M = np.zeros(len_h_values)
    Uth_M = np.zeros(len_h_values)

    entropy_values_pred = 0

    HVD = conf['config_entropy']['transform']
    entropy_name = conf['config_entropy']['type_entropy']

    for i in range(len_h_values):
        path = (f'{name_map}_entropy/{entropy_name}/{HVD}/'
                f'{i + 1}_{name_map}_{HVD}_{entropy_name}.txt')
        if os.path.isfile(path):
            entropy_values = np.loadtxt(path)
        else:
            print(f'File {path} not found')
            exit(0)

        if i > 0:
            MCC_M[i], Uth_M[i] = calculate_mcc(entropy_values, entropy_values_pred)
        entropy_values_pred = entropy_values

    GEFMCC = np.mean(np.abs(MCC_M[1:]))
    print(f'GEFMCC: {GEFMCC:.4f} [{name_map}_{entropy_name}_{HVD}]')

    str_time = format_time(time.time() - start_time)
    print(f'Calculation time {str_time}')

    directory_name = f'{name_map}_classifier'
    os.makedirs(directory_name, exist_ok=True)

    # Save to file
    out_file_name = f'{directory_name}/{name_map}_GEFMCC_{entropy_name}_{HVD}.txt'
    with open(out_file_name, 'w') as file:
        file.write(f'{name_map}_{entropy_name}_{HVD}\n')
        file.write(f'GEFMCC: {GEFMCC:.4f}\n')
        file.write(f'Calculation time: {str_time}\n\n')

        for i in range(1, len_h_values):
            file.write(f'{h_values[i]:.6f}\t{MCC_M[i]:.6f}\t{Uth_M[i]:.6f}\n')
    print(f'Result file {out_file_name} saved')

    return GEFMCC


if __name__ == '__main__':
    import average_gefmcc

    global_calculate_gefmcc(average_gefmcc.base_config)
