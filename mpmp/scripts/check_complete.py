"""
Check if cluster jobs completed successfully.

Runs for the current working directory by default. Prints the filename of
jobs that completed with errors to stdout, newline separated.
"""
from pathlib import Path

def check_file(fname, check_str='Job complete'):
    with open(fname, 'r') as f:
        for line in f:
            if check_str in line:
                return True
    return False

if __name__ == '__main__':
    err_files = []
    for fname in Path.cwd().iterdir():
        # skip error logging files
        if fname.suffix != '.out': continue
        # search file for job complete string, if the script
        # terminated early this should not exist
        if not check_file(fname):
            err_files.append(fname)
    print('\n'.join([str(f) for f in err_files]))
