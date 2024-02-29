import numpy as np
import os
from NNetEn import NNetEn_entropy
import EntropyHub as EH
from multiprocessing import Pool
from transform import generate_hvg_series
from itertools import repeat

config = {
    'directory_name': 'sin_map',
    'fuzz_dim': 1,
    'fuzz_r1': 0.2,
    'fuzz_r2': 3,
    'fuzz_t': 1,
    'nneten_ds': 'D1',
    'nneten_mu': 1,
    'nneten_method': 3,
    'nneten_metric': 'Acc',
    'nneten_epoch': 5,
    'calc_hvg_series': False,
    'process': 20,
    'type_entropy': 'fuzzy'  # 'fuzzy' or 'nneten'
}


def calculate_fuzzy_entropy(time_series: np.ndarray,
                            fuzzy_m: float, fuzzy_r1: float,
                            fuzzy_r2: float, fuzzy_t: float) -> np.ndarray:
    """
    Function for calculate fuzzy entropy
        :param time_series: Numpy array containing time series
        :param fuzzy_m: Configuration FuzzyEn - embedding dimension
        :param fuzzy_r1: Configuration FuzzyEn - tolerance
        :param fuzzy_r2: Configuration FuzzyEn - argument exponent (pre-division)
        :param fuzzy_t: Configuration FuzzyEn - time delay
        :return: Numpy array containing fuzzy entropy value
    """
    fuzzy_list = []
    for series in time_series:
        FuzzEn, _, _ = EH.FuzzEn(series, m=fuzzy_m,
                                 r=(fuzzy_r1 * np.std(series),
                                    fuzzy_r2), tau=fuzzy_t)
        fuzzy_list.append(FuzzEn[0])

    return np.array(fuzzy_list)


def calculate_nneten_entropy(time_series: np.ndarray, nneten_ds: str,
                             nneten_mu: float, nneten_method: int,
                             nneten_metric: str, nneten_epoch: int) -> np.ndarray:
    """
    Function for calculate NNetEn entropy
        :param time_series: Numpy array containing time series
        :param nneten_ds: Configuration NNetEn - dataset
        :param nneten_mu: Configuration NNetEn - mu [0.01..1]
        :param nneten_method: Configuration NNetEn - method [1..6]
        :param nneten_metric: Configuration NNetEn - metric ['Acc', 'R2E', 'PE']
        :param nneten_epoch: Configuration NNetEn - epoch
        :return: Numpy array containing nneten entropy value
    """
    nneten_list = []
    NNetEn = NNetEn_entropy(database=nneten_ds, mu=nneten_mu)
    for series in time_series:
        nneten_list.append(NNetEn.calculation(series,
                                              epoch=nneten_epoch,
                                              method=nneten_method,
                                              metric=nneten_metric,
                                              log=False))

    return np.array(nneten_list)


def run_multiprocessing_mode(func, param, processes):
    with Pool(processes=processes) as pool:
        pool.starmap(func, param)
    pool.close()


def calculation_entropy(file: str, conf: dict):
    """
    General function for entropy calculations based on given parameters
        :param file: absolute path to the file with time series
        :param conf: basic configuration
        :return: The results are output to a file
    """
    file_name = os.path.basename(file).split('.')[0]
    data = np.loadtxt(file, delimiter=' ')
    print('Read {} file'.format(file_name))

    if conf['transform'] == 'hvg':
        data = generate_hvg_series(data)
        file_name += '_hvg'

        transform_dir = f"{conf['use_chaotic_map']}_transform/{conf['transform']}"
        os.makedirs(transform_dir, exist_ok=True)

        np.savetxt(os.path.join(transform_dir, file_name + '.txt'),
                   data, delimiter=' ', newline='\n', fmt='%g')
        print('Generate hvg series finished')

    elif conf['transform'] == 'no_hvg':
        file_name += '_no_hvg'
    else:
        print('Wrong parameter - transform')
        exit(0)

    if conf['type_entropy'] == 'fuzzy':
        result = calculate_fuzzy_entropy(data, **conf['fuzzyen_params'])
    elif conf['type_entropy'] == 'nneten':
        result = calculate_nneten_entropy(data, **conf['nneten_params'])
    else:
        print('Wrong parameter type_entropy')
        exit(0)

    file_name += f"_{conf['type_entropy']}"

    entropy_dir = (f"{conf['use_chaotic_map']}_entropy/"
                   f"{conf['type_entropy']}/{conf['transform']}")

    os.makedirs(entropy_dir, exist_ok=True)

    np.savetxt(os.path.join(entropy_dir, file_name + '.txt'), result,
               delimiter='\n', newline='\n', fmt='%g')
    print('File {} completed'.format(file_name))


def global_calculate_entropy(conf: dict):
    """
    Global function for calculating entropy based on specified parameters
        :param conf: Calculation parameters
        :return: The results are output to a file
    """
    print('-------------------------')
    print('START calculating entropy')

    conf = conf['config_entropy']
    if not os.path.exists(conf['use_chaotic_map']):
        print(f'Directory {conf["use_chaotic_map"]} not found.')
        print('Check configuration or Run generate chaotic map.')
        exit(0)

    path_to_files = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 conf['use_chaotic_map'])
    file_list = [os.path.join(path_to_files, _)
                 for _ in os.listdir(path_to_files) if _.endswith(r".txt")]

    if len(file_list) == 0:
        print(f'File in directory /{conf["use_chaotic_map"]} not found.')
        print('Check configuration or Run generate chaotic map.')
        exit(0)

    run_multiprocessing_mode(calculation_entropy,
                             zip(file_list, repeat(conf)),
                             processes=conf['process'])


if __name__ == '__main__':
    import average_gefmcc

    global_calculate_entropy(average_gefmcc.base_config)
