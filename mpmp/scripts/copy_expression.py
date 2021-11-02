from pathlib import Path
import shutil

import mpmp.config as cfg

results_dir = Path(cfg.results_dirs['mutation'],
                   'methylation_results',
                   'gene')

dest_dir = Path(cfg.results_dirs['mutation'],
                'methylation_results_shuffle_cancer_type',
                'gene')

if __name__ == '__main__':
    for identifier in results_dir.iterdir():
        id_string = str(identifier.stem)
        identifier_dir = Path(results_dir, identifier)
        if identifier_dir.is_file(): continue
        for results_file in identifier_dir.iterdir():
            if not results_file.is_file(): continue
            results_filename = str(results_file.name)
            if 'expression' not in results_filename: continue
            if 'signal' not in results_filename: continue
            if 'n100' in results_filename: continue
            if 'n1000' in results_filename: continue
            if 'n5000' in results_filename: continue
            # skip compressed files here, use load_compressed* functions
            # to load that data separately
            if results_filename[0] == '.': continue
            print(results_filename)
            source_filename = Path(results_dir, id_string, results_filename)
            dest_filename = Path(dest_dir, id_string, results_filename)
            print('Source: ', source_filename)
            print('Destination: ', dest_filename)
            shutil.copy(str(source_filename), str(dest_filename))

