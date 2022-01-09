import json
import os
import sys
import time
import subprocess
from pathlib import Path
import datetime
import random

DEEP_JOB_MAX = 15
PRINT_LIMIT = 4

READY = 'ready'
PENDING = 'pending'
RUNNING = 'running'
DONE = 'done'
FAILED = 'failed'
TIMEDOUT = 'timedout'
LAUNCHED = 'launched'

FINISHED = 'finished'
NOT_LAUNCHED = 'not_launched'

CUDA_TIMEOUT = 180      # Timeout for job not starting after a few minutes


class Job:
    def __init__(self, name, script, arguments, additional, parent):
        self.name = name
        self.script = script
        self.arguments = arguments
        self.additional = additional
        self.parent = parent

        self.status = READY
        self.sbatch_id = None
        self.start_time = None

    def print_description(self, long=True):
        line = f'({self.status}, {"" if self.sbatch_id is None else str(self.sbatch_id)}) '

        if self.script is not None:
            line += '/'.join(self.script.split('/')[-7:])
        else:
            line += f'({self.name}) placeholder'

        if not long:
            line = line.replace('sbatch-', '')
            line = line.replace('.sh', '')

            line = line.replace('moco-', 'm')
            line = line.replace('baseline-', 'b')
            line = line.replace('wt-', 'w')
            line = line.replace('wo-', 'o')
            line = line.replace('full-', 'f-')
            line = line.replace('last-', 'l-')

        print(line)

    def check_squeue(self):
        '''
        Invalid, running or pending due to priority
        '''
        # print(self.sbatch_id)
        if self.sbatch_id is not None:
            check_line = f'squeue -j {self.sbatch_id}'
            try:
                output = str(subprocess.check_output(check_line, shell=True))
                if 'deep' in output and 'priority' in output:
                    return PENDING
                elif 'deep' in output:
                    return RUNNING
                elif 'error' in output:
                    # raise ValueError(f'Invalid sbatch ID: {self.sbatch_id}')
                    return FINISHED
                else:
                    return FINISHED
            except:
                return FINISHED
        else:
            return NOT_LAUNCHED

    # TODO: JBY: make this not workload dependent, okay for now
    def check_status(self):
        if self.script is None:
            self.status = DONE
            return
        
        if self.sbatch_id == 999:
            rnd = random.random()
            if rnd < 0.35:
                self.status = FAILED
            elif rnd < 0.6:
                self.status = TIMEDOUT
            else:
                self.status = DONE
            return

        if self.start_time is None:
            elapsed = 0
        else:
            elapsed = time.time() - self.start_time

        if self.status != DONE:
            
            squeue_status = self.check_squeue()
            if squeue_status == PENDING:
                self.status = PENDING
            elif squeue_status == RUNNING:
                if 'timeout' in self.additional and self.status is not DONE:
                    if elapsed > self.additional['timeout']:
                        self.status = TIMEDOUT
                        return
                self.status = RUNNING
            elif squeue_status == FINISHED or squeue_status == NOT_LAUNCHED:
                # train/selection/test have different checks
                out_folder = Path(self.additional['folder'])
                if 'train' in self.script:
                    log_file = out_folder / 'train_log.txt'
                    if os.path.exists(log_file):
                        done = False
                        with open(log_file, 'r') as f:
                            lines = f.readlines()
                            for l in lines:
                                if l.strip() == '=== Training Complete ===':
                                    done = True
                        self.status = DONE if done else FAILED
                    else:
                        self.status = FAILED
                elif 'sel.' in self.script:
                    config = out_folder / 'final.json'
                    has_config = os.path.exists(config)
                    self.status = DONE if has_config else FAILED
                elif 'test' in self.script:
                    score_csv_path = out_folder / 'test_save' /\
                                'final' / 'results' / 'test' / 'scores.csv'
                    has_test = os.path.exists(score_csv_path)
                    self.status = DONE if has_test else FAILED
                elif 'conf' in self.script:
                    conf_csv_path = out_folder / 'test_save' /\
                                'final' / 'results' / 'test' / 'confidence.csv'
                    has_conf = os.path.exists(conf_csv_path)
                    multi_conf_csv_path = out_folder / 'test_save' /\
                                'final' / 'results' / 'test' / 'confidence_multi.csv'
                    has_multi_conf = os.path.exists(multi_conf_csv_path)
                    self.status = DONE if (has_conf and has_multi_conf) else FAILED
                else:
                    raise ValueError(f'Script {self.script} is not supported')
            else:
                raise ValueError(f'Unrecognized squeue status {squeue_status}')

    def kill(self):
        cancel_line = f'scancel {self.sbatch_id}'
        output = subprocess.check_output(cancel_line, shell=True)
        print(output)

    # Only support launching sbatch jobs
    def launch(self, actual=True):

        if self.script is not None:
            arg_space = ' '.join(self.arguments)
            job_line = f'sbatch {self.script} {arg_space}'

            # print(f'Launching job {self.name}')
            # print(job_line)
            # os.system(job_line)
            if actual:
                output = str(subprocess.check_output(job_line, shell=True))
                # print(output)
                self.sbatch_id = int(output.split('\\n')[-2].split(' ')[-1])
            else:
                self.sbatch_id = 999
            # print(f'SBATCH: {self.sbatch_id}')
        else:
            print(f'Job {self.name} is a placeholder, marking as done')

        self.status = LAUNCHED
        self.start_time = time.time()

    def reset(self):
        self.start_time = None
        self.status = READY
        self.sbatch_id = None


# A BFS parser of description file tree
def generate_list(description_file):
    
    file_list = []
    job_list = []

    file_list.append((description_file, None))

    while len(file_list) > 0:
        # cur_file, parent = file_list.pop(0)
        cur_file, parent = file_list.pop(-1)
        print(cur_file)
        with open(cur_file) as f:
            data = json.load(f)
            job = Job(data['name'], data['script'], data['arguments'], data['additional'], parent)
            job_list.append(job)
            children = [(d, job) for d in data['children']]

            file_list.extend(children)
    
    return job_list


def limited_print_list(header, long_list, limit=PRINT_LIMIT, print_func=print):
    print(f'===== {header} ({len(long_list)}) =====')
    num_lines = 1

    first = min(limit, len(long_list))
    for i in range(first):
        # print(long_list[i])
        print_func(long_list[i])
        num_lines += 1

    if len(long_list) > limit:
        print(f'.... ')
        num_lines += 1
    elif len(long_list) == 0:
        print('--> Empty list')
        num_lines += 1

    last = max(first, len(long_list) - limit // 2)
    for i in range(last, len(long_list)):
        # print(long_list[i])
        print_func(long_list[i])
        num_lines += 1
    
    return num_lines


def check_deep():
    output = str(subprocess.check_output('squeue -p deep | wc -l', shell=True))
    num_jobs = int(output.split('\\n')[0][2:])

    full_potential = int((20 * 2 - num_jobs) * 0.5)
    potential = int(max(min((20 * 2 - num_jobs) * 0.5, DEEP_JOB_MAX), 5))
    return num_jobs, potential, full_potential


def check_all_jobs(root_description):
    job_list = generate_list(root_description)

    num_done = 0
    for jb in job_list:
        jb.check_status()
        jb.print_description()
        if jb.status == DONE:
            num_done += 1
    
    print(f'There is a total of {len(job_list)} jobs. {num_done} jobs already completed.')

    cur, potential, full_potential = check_deep()
    print(f'Deep has {cur} jobs, can launch another {potential} jobs. Full deep potential is {full_potential}')



def clear(): 
    # check and make call for specific operating system 
    _ = subprocess.call('clear' if os.name =='posix' else 'cls') 



def run_all_jobs(root_description, refresh_time, actual):
    
    job_list = generate_list(root_description)
    current_jobs = []
    completed_jobs = []
    
    for jb in job_list:
        jb.check_status()
        if jb.status == DONE:
            completed_jobs.append(jb)
    
    for cj in completed_jobs:
        job_list.remove(cj)
    prev_completed = completed_jobs
    
    for jb in job_list:
        jb.reset()

    num_already_done = len(completed_jobs)
    completed_jobs = []     # Re-initialize completed jobs list to only include newly completed ones

    start_date = datetime.datetime.now()

    entered = False
    num_lines_printed = 0
    while len(job_list) > 0 and len(current_jobs) >= 0:
        entered = True

        done_jobs= []
        for cj in current_jobs:
            cj.check_status()
            if cj.status == DONE:
                done_jobs.append(cj)
                completed_jobs.append(cj)
            elif cj.status == TIMEDOUT or cj.status == FAILED:
                done_jobs.append(cj)

        for dj in done_jobs:
            current_jobs.remove(dj)
            if dj.status == FAILED:
                job_list.append(dj)
            elif dj.status == TIMEDOUT:
                job_list.append(dj)
                dj.kill()

        cur, potential, full_potential = check_deep()

        num_launched = 0
        num_tries = 0
        missing = potential - len(current_jobs)
        while num_launched < missing and len(job_list) > 0 and num_tries < 500:
            nj = job_list.pop(0)
            nj.check_status()
            if nj.status != DONE and (nj.parent is None or nj.parent.status == DONE):
                nj.reset()
                nj.launch(actual=actual)
                current_jobs.append(nj)
                num_launched += 1
            elif nj.status != DONE:
                job_list.append(nj)
                nj.reset()
            else:
                completed_jobs.append(nj)
            
            num_tries += 1

        clear()

        # Handle job printing
        current_date = datetime.datetime.now()
        elapsed_time = current_date - start_date
        if len(completed_jobs):
            estimated = elapsed_time / len(completed_jobs) * (len(current_jobs) + len(job_list))
        else:
            estimated = -1
        # print('\r' * num_lines_printed)

        num_lines_printed = 0
        print(f'Started: {start_date}')
        print(f'Refresh: {refresh_time}s. Current date: {current_date}')
        print(f'Elapsed time: {elapsed_time}\t Estimated time to completion: {estimated}')
        print(f'Deep Jobs: {cur}s. Full Potential: {full_potential}')
        print(f'Prev: {num_already_done}\tCurrent: {len(current_jobs)}\tWaiting: {len(job_list)}\tCompleted: {len(completed_jobs)}')
        num_lines_printed = 2

        num_lines_printed += limited_print_list('Current', current_jobs, print_func=lambda x: x.print_description(long=False))
        num_lines_printed += limited_print_list('Completed', completed_jobs + prev_completed, print_func=lambda x: x.print_description(long=False))
        num_lines_printed += limited_print_list('Waiting', job_list, print_func=lambda x: x.print_description(long=False))

        time.sleep(refresh_time)

    if not entered:
        print('All jobs are already completed. Check again?')

if __name__ == '__main__':
    # assert os.environ['CONDA_DEFAULT_ENV'] == 'chexpert-baseline', f'conda deactivate\nconda activate chexpert-baseline'
    assert os.environ['CONDA_DEFAULT_ENV'] == 'clean', f'conda deactivate\nconda activate clean'

    if len(sys.argv) == 2:
        description_file = sys.argv[1]
        check_all_jobs(description_file)
    else:
        description_file = sys.argv[1]
        refresh_time = int(sys.argv[2])
        actual_launch = True if 't' in sys.argv[3].lower()  else False

        run_all_jobs(description_file, refresh_time, actual_launch)
    