import map_generate
import entropy
import classification
import time

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
    start_time = time.time()

    for chaotic_map in ['sin_map', 'log_map', 'plank_map', 'tmbm_map']:
        conf['config_entropy']['use_chaotic_map'] = chaotic_map
        map_generate.global_map_generator(conf)

        # ['fuzzy', 'nneten']
        for type_ent in ['fuzzy']:
            conf['config_entropy']['type_entropy'] = type_ent

            for trans in ['no_hvg', 'hvg']:
                conf['config_entropy']['transform'] = trans
                entropy.global_calculate_entropy(conf)

                key_prim = f"{type_ent}_{trans}"
                key_second = f"{type_ent}_{chaotic_map}_{trans}"
                value = classification.global_calculate_gefmcc(conf)

                if result_dict.get(key_prim, None) is not None:
                    result_dict[key_prim][key_second] = value
                else:
                    result_dict[key_prim] = {key_second: value}

    final_file_name = 'average_GEFMCC.txt'
    final_str_time = classification.format_time(time.time() - start_time)

    with open(final_file_name, 'w') as file:
        file.write('Global Efficiency of entropy calculated using '
                   'Matthews Correlation Coefficient (GEFMCC)\n')
        file.write(f'Calculation time: {final_str_time}\n\n')

        for item in result_dict.keys():
            for key, value in result_dict[item].items():
                file.write(f'{key}\t{value:.6f}\n')
            avg_value = sum(result_dict[item].values()) / len(result_dict[item])
            file.write(f'average_{item}\t{avg_value:.6f}\n\n')

    print('---------------------')
    print(f'Final file {final_file_name} saved')
    print(f'Total calculation time {final_str_time}')


def single_calc_gefmcc(conf: dict):
    map_generate.global_map_generator(conf)
    entropy.global_calculate_entropy(conf)
    classification.global_calculate_gefmcc(conf)


if __name__ == '__main__':
    general_calculation_all_parameters(base_config)
    # or
    #single_calc_gefmcc(base_config)
