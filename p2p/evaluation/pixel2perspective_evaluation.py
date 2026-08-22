
# This script combines the per-dataset results into the Pixel2Perspective-Benchmark tables.
# Run from the root of the project with:
# python p2p/evaluation/pixel2perspective_evaluation.py --model_outputs /path/to/model_outputs
# e.g. python p2p/evaluation/pixel2perspective_evaluation.py --model_outputs output/CAT-Seg output/SAN

import argparse
import torch
import pandas as pd
from pathlib import Path

# Registered dataset name -> display name and viewpoint cluster
dataset_info = {
    'isprs_potsdam_sem_seg_test_rgb':  ('ISPRS (RGB)',        'BEV'),
    'isprs_potsdam_sem_seg_test_irrg': ('ISPRS (IRRG)',       'BEV'),
    'worldfloods_sem_seg_test_rgb':    ('WorldFloods (RGB)',  'BEV'),
    'worldfloods_sem_seg_test_irrg':   ('WorldFloods (IRRG)', 'BEV'),
    'loveda_sem_seg_test':             ('LoveDA',             'BEV'),
    'openearthmap_sem_seg_test_rgb':   ('OpenEarthMap (RGB)', 'BEV'),
    'openearthmap_sem_seg_test_sar':   ('OpenEarthMap (SAR)', 'BEV'),
    'uavid_sem_seg_val':               ('UAVid',              'Slant-angle'),
    'indraeye_sem_seg_test_rgb':       ('IndraEye (RGB)',     'Slant-angle'),
    'indraeye_sem_seg_test_ir':        ('IndraEye (IR)',      'Slant-angle'),
    'cart_sem_seg_test_rgb':           ('CART (RGB)',         'Slant-angle'),
    'cart_sem_seg_test_ir':            ('CART (IR)',          'Slant-angle'),
    'acdc_sem_seg_val':                ('ACDC',               'Slant-angle'),
    'msrs_sem_seg_test_rgb':           ('MSRS (RGB)',         'Ground-level'),
    'msrs_sem_seg_test_ir':            ('MSRS (IR)',          'Ground-level'),
    'mvseg_sem_seg_test_rgb':          ('MVSeg (RGB)',        'Ground-level'),
    'mvseg_sem_seg_test_ir':           ('MVSeg (IR)',         'Ground-level'),
    'semantic_rt_sem_seg_test_rgb':    ('SemanticRT (RGB)',   'Ground-level'),
    'semantic_rt_sem_seg_test_ir':     ('SemanticRT (IR)',    'Ground-level'),
    'mfnet_sem_seg_test_rgb':          ('MFNet (RGB)',        'Ground-level'),
    'mfnet_sem_seg_test_ir':           ('MFNet (IR)',         'Ground-level'),
    'infraparis_sem_seg_test_rgb':     ('Infraparis (RGB)',   'Ground-level'),
    'infraparis_sem_seg_test_ir':      ('Infraparis (IR)',    'Ground-level'),
}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Combine Pixel2Perspective-Benchmark results')
    parser.add_argument('--model_outputs', nargs='+', type=str,
                         help='Directories of model outputs. Multiple models can be specified, '
                              'separated by a space.')
    parser.add_argument('--metrics', nargs='+', default=['mIoU'],
                         help='A list of the evaluation metrics to report')
    parser.add_argument('--results_dir', type=str, default='results', help='Results directory')
    args = parser.parse_args()

    results = pd.DataFrame(columns=['Model', 'Dataset', 'Viewpoint'] + args.metrics)

    for model_output in args.model_outputs:
        output_dir = Path(model_output)
        for dataset_id, (dataset_name, viewpoint) in dataset_info.items():
            dataset_results = pd.DataFrame({
                'Model': [output_dir.name],
                'Dataset': [dataset_name],
                'Viewpoint': [viewpoint],
            })

            metrics_file = list(output_dir.glob(f'{dataset_id}/**/sem_seg_evaluation.pth'))
            if len(metrics_file) == 0:
                print(f'No evaluation file found for {dataset_id} in {model_output}')
                results = pd.concat([results, dataset_results], ignore_index=True)
                continue

            assert len(metrics_file) == 1, \
                f'Found {len(metrics_file)} evaluation files for {dataset_name} in {model_output}'

            metrics = torch.load(metrics_file[0])
            for metric in args.metrics:
                if metric in metrics:
                    dataset_results[metric] = [metrics[metric]]
            results = pd.concat([results, dataset_results], ignore_index=True)

        mean_results = pd.DataFrame({
            'Model': [output_dir.name],
            'Dataset': ['Mean'],
            'Viewpoint': ['Mean'],
        })
        for metric in args.metrics:
            mean_results[metric] = [results[results['Model'] == output_dir.name][metric].mean()]
        results = pd.concat([results, mean_results], ignore_index=True)

    datasets_order = [name for name, _ in dataset_info.values() if name in results['Dataset'].values] + ['Mean']

    results.set_index(['Dataset', 'Model'], inplace=True, drop=True)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    results.round(2).fillna('-').to_csv(results_dir / 'results.csv')
    for metric in args.metrics:
        metric_results = results.fillna('-').unstack(1)[metric].T
        metric_results = metric_results[datasets_order]
        metric_results.round(2).to_csv(results_dir / f'results_{metric}.csv')

    print(f'\nResults saved to {results_dir}')
