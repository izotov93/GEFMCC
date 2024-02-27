import map_generate
import entropy
import classification
from collections import defaultdict
import numpy as np

base_config = {
    'config_gen': {
        'log_map': {
            'N_ser': 100,
            'N_el': 300,
            'h1': 3.4,
            'h2': 4,
            'h_step': 0.002,
            'n_ignor': 1000,
            'x0': 0.1
        },
        'sin_map': {
            'N_ser': 100,
            'N_el': 300,
            'h1': 0.7,
            'h2': 2,
            'h_step': 0.005,
            'n_ignor': 1000,
            'x0': 0.1
        },
        'plank_map': {
            'N_ser': 100,
            'N_el': 300,
            'h1': 3,
            'h2': 7,
            'h_step': 0.01,
            'n_ignor': 1000,
            'x0': 4
        },
        'tmbm_map': {
            'N_ser': 100,
            'N_el': 300,
            'h1': -1.7,
            'h2': -1.5,
            'h_step': 0.0005,
            'n_ignor': 1000,
            'x0': 0.01,
            'y0': 0.01,
            'z0': 0.01,
            'a2': 1.15,
            'b': 0.35,
            'c': 0.02
        }
    },
    'config_entropy': {
        'use_chaotic_map': 'tmbm_map',
        'type_entropy': 'fuzzy',
        'process': 20,
        'transform': 'hvg',
        'fuzzyen_params': {
            'fuzzy_m': 1,
            'fuzzy_r1': 0.2,
            'fuzzy_r2': 3,
            'fuzzy_t': 1
        },
        'nneten_params': {
            'nneten_ds': 'D1',
            'nneten_mu': 1,
            'nneten_method': 3,
            'nneten_metric': 'Acc',
            'nneten_epoch': 5,
        },
    }
}


def general_calculation_all_parameters(conf: dict):
    result_dict = dict()
    intermediate_values = defaultdict(list)

    for chaotic_map in ['sin_map', 'log_map', 'plank_map', 'tmbm_map']:
        conf['config_entropy']['use_chaotic_map'] = chaotic_map
        map_generate.global_map_generator(conf)

        # ['fuzzy', 'nneten']
        for type_ent in ['fuzzy']:
            conf['config_entropy']['type_entropy'] = type_ent

            for trans in ['no_hvg', 'hvg']:
                conf['config_entropy']['transform'] = trans
                entropy.global_calculate_entropy(conf)

                key = f"{type_ent}_{chaotic_map}_{trans}"
                value = classification.global_calculate_gefmcc(conf)
                result_dict[key] = value

                intermediate_key = f"average_{type_ent}_{trans}"
                intermediate_values[intermediate_key].append(value)

    for key, values in intermediate_values.items():
        average_value = np.mean(values)
        result_dict[key] = average_value

    with open('average_GEFMCC.txt', 'w') as file:
        file.write(f'GEFMCC Value\n\n')
        for key, value in result_dict.items():
            file.write(f'{key}\t{value:.6f}\n')


def single_calc_gefmcc(conf: dict):
    map_generate.global_map_generator(conf)
    entropy.global_calculate_entropy(conf)
    classification.global_calculate_gefmcc(conf)


if __name__ == '__main__':
    general_calculation_all_parameters(base_config)
    # or
    # single_calc_gefmcc(base_config)
