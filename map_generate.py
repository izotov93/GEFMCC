import numpy as np
import os
import math


def sin_map(N_ser: int, N_el: int,
            h1: float, h2: float, h_step: float,
            n_ignor: int, x0: float) -> np.ndarray:
    """
    To generate synthetic time series, sin chaotic map
        :param N_ser: Time series length
        :param N_el: Number of time series in a block
        :param h1: control parameter - min value
        :param h2: control parameter - max value
        :param h_step: control parameter - step
        :param n_ignor: Number of steps ignored
        :param x0: Initial value
        :return: Numpy array containing time series.
                Also, time series are saved as text files
    """
    map_name = 'sin_map'

    h_values = np.arange(h1, h2 + h_step, h_step)
    h_values = np.round(h_values, decimals=6)

    os.makedirs(map_name, exist_ok=True)

    result_map = np.empty((len(h_values), N_ser, N_el))

    for i, h in enumerate(h_values):
        x = x0

        for _ in range(n_ignor):
            x = h * np.sin(np.pi * x)

        map_series = np.empty((N_ser, N_el))
        x_series = np.full(N_el, x)

        for j in range(N_ser):
            for index in range(N_el):
                x = h * np.sin(np.pi * x)
                x_series[index] = x
            map_series[j] = x_series

        map_series = np.round(map_series, decimals=15)

        result_map[i] = map_series

        filename = f'{map_name}/{i + 1}_{map_name}.txt'
        with open(filename, 'w') as file:
            for row in map_series:
                file.write(' '.join(map(str, row)) + '\n')
        print(f'File {filename} saved')

    return result_map


def logictic_map(N_ser: int, N_el: int,
                 h1: float, h2: float, h_step: float,
                 n_ignor: int, x0: float) -> np.ndarray:
    """
    To generate synthetic time series, logistic chaotic map
        :param N_ser: Time series length
        :param N_el: Number of time series in a block
        :param h1: control parameter - min value
        :param h2: control parameter - max value
        :param h_step: control parameter - step
        :param n_ignor: Number of steps ignored
        :param x0: Initial value
        :return: Numpy array containing time series.
                Also, time series are saved as text files
    """
    map_name = 'log_map'

    h_values = np.arange(h1, h2 + h_step, h_step)
    h_values = np.round(h_values, decimals=6)

    os.makedirs(map_name, exist_ok=True)

    result_map = np.empty((len(h_values), N_ser, N_el))

    for i, h in enumerate(h_values):
        x = x0

        for _ in range(n_ignor):
            x = h * x * (1 - x)

        map_series = np.empty((N_ser, N_el))
        x_series = np.full(N_el, x)

        for j in range(N_ser):
            for index in range(N_el):
                x = h * x * (1 - x)
                x_series[index] = x
            map_series[j] = x_series

        map_series = np.round(map_series, decimals=15)

        result_map[i] = map_series

        filename = f'{map_name}/{i + 1}_{map_name}.txt'
        with open(filename, 'w') as file:
            for row in map_series:
                file.write(' '.join(map(str, row)) + '\n')
        print(f'File {filename} saved')

    return result_map


def plank_map(N_ser: int, N_el: int,
              h1: float, h2: float, h_step: float,
              n_ignor: int, x0: float) -> np.ndarray:
    """
    To generate synthetic time series, logistic chaotic map
        :param N_ser: Time series length
        :param N_el: Number of time series in a block
        :param h1: control parameter - min value
        :param h2: control parameter - max value
        :param h_step: control parameter - step
        :param n_ignor: Number of steps ignored
        :param x0: Initial value
        :return: Numpy array containing time series.
                Also, time series are saved as text files
    """
    map_name = 'plank_map'

    h_values = np.arange(h1, h2 + h_step, h_step)
    h_values = np.round(h_values, decimals=6)

    os.makedirs(map_name, exist_ok=True)

    result_map = np.empty((len(h_values), N_ser, N_el))

    for i, h in enumerate(h_values):
        x = x0

        for _ in range(n_ignor):
            x = h * x ** 3 / (1 + math.exp(x))

        map_series = np.empty((N_ser, N_el))
        x_series = np.full(N_el, x)

        for j in range(N_ser):
            for index in range(N_el):
                x = h * x ** 3 / (1 + math.exp(x))
                x_series[index] = x
            map_series[j] = x_series

        map_series = np.round(map_series, decimals=15)

        result_map[i] = map_series

        filename = f'{map_name}/{i + 1}_{map_name}.txt'
        with open(filename, 'w') as file:
            for row in map_series:
                file.write(' '.join(map(str, row)) + '\n')
        print(f'File {filename} saved')

    return result_map


def tmbm_map(N_ser: int, N_el: int, h1: float, h2: float,
             h_step: float, n_ignor: int, x0: float, y0: float,
             z0: float, a2: float, b: float, c: float) -> np.ndarray:
    """
        To generate synthetic time series,
        Two-memristor based map (TMBM)
            :param N_ser: Time series length
            :param N_el: Number of time series in a block
            :param h1: control parameter - min value
            :param h2: control parameter - max value
            :param h_step: control parameter - step
            :param n_ignor: Number of steps ignored
            :param x0: Initial value
            :return: Numpy array containing time series.
                    Also, time series are saved as text files
        """
    map_name = 'tmbm_map'

    h_values = np.arange(h1, h2 + h_step, h_step)
    h_values = np.round(h_values, decimals=6)

    os.makedirs(map_name, exist_ok=True)

    result_map = np.empty((len(h_values), N_ser, N_el))

    for i, h in enumerate(h_values):
        x = x0
        y = y0
        z = z0

        for _ in range(n_ignor):
            x1 = h * a2 * (b * abs(y) - 1) * (z**2 - 1) * x + c
            y1 = y + x
            z1 = z + h * (b * abs(y) - 1) * x

            x, y, z = x1, y1, z1

        map_series = np.empty((N_ser, N_el))
        x_series = np.full(N_el, x)

        for j in range(N_ser):
            for index in range(N_el):
                x1 = h * a2 * (b * abs(y) - 1) * (z ** 2 - 1) * x + c
                y1 = y + x
                z1 = z + h * (b * abs(y) - 1) * x

                x, y, z = x1, y1, z1

                x_series[index] = x
            map_series[j] = x_series

        map_series = np.round(map_series, decimals=15)

        result_map[i] = map_series

        filename = f'{map_name}/{i + 1}_{map_name}.txt'
        with open(filename, 'w') as file:
            for row in map_series:
                file.write(' '.join(map(str, row)) + '\n')
        print(f'File {filename} saved')

    return result_map


def global_map_generator(conf: dict):
    print('---------------------')
    print('START Generation of chaotic map')

    name_map = conf['config_entropy']['use_chaotic_map']

    if name_map == 'sin_map':
        sin_map(**conf['config_gen'][name_map])
    elif name_map == 'log_map':
        logictic_map(**conf['config_gen'][name_map])
    elif name_map == 'plank_map':
        plank_map(**conf['config_gen'][name_map])
    elif name_map == 'tmbm_map':
        tmbm_map(**conf['config_gen'][name_map])


if __name__ == '__main__':
    import average_gefmcc

    global_map_generator(average_gefmcc.base_config)
