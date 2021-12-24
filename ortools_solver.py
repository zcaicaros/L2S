import collections
# Import Python wrapper for or-tools CP-SAT solver.
from ortools.sat.python import cp_model


def MinimalJobshopSat(data):
    """Minimal jobshop problem."""
    # Create the model.
    model = cp_model.CpModel()

    jobs_data = data

    machines_count = 1 + max(task[0] for job in jobs_data for task in job)
    all_machines = range(machines_count)

    # Computes horizon dynamically as the sum of all durations.
    horizon = sum(task[1] for job in jobs_data for task in job)

    # Named tuple to store information about created variables.
    task_type = collections.namedtuple('task_type', 'start end interval')
    # Named tuple to manipulate solution information.
    assigned_task_type = collections.namedtuple('assigned_task_type',
                                                'start job index duration')

    # Creates job intervals and add to the corresponding machine lists.
    all_tasks = {}
    machine_to_intervals = collections.defaultdict(list)

    for job_id, job in enumerate(jobs_data):
        for task_id, task in enumerate(job):
            machine = task[0]
            duration = task[1]
            suffix = '_%i_%i' % (job_id, task_id)
            start_var = model.NewIntVar(0, horizon, 'start' + suffix)
            end_var = model.NewIntVar(0, horizon, 'end' + suffix)
            interval_var = model.NewIntervalVar(start_var, duration, end_var,
                                                'interval' + suffix)
            all_tasks[job_id, task_id] = task_type(
                start=start_var, end=end_var, interval=interval_var)
            machine_to_intervals[machine].append(interval_var)

    # Create and add disjunctive constraints.
    for machine in all_machines:
        model.AddNoOverlap(machine_to_intervals[machine])

    # Precedences inside a job.
    for job_id, job in enumerate(jobs_data):
        for task_id in range(len(job) - 1):
            model.Add(all_tasks[job_id, task_id +
                                1].start >= all_tasks[job_id, task_id].end)

    # Makespan objective.
    obj_var = model.NewIntVar(0, horizon, 'makespan')
    model.AddMaxEquality(obj_var, [
        all_tasks[job_id, len(job) - 1].end
        for job_id, job in enumerate(jobs_data)
    ])
    model.Minimize(obj_var)

    # Solve model.
    solver = cp_model.CpSolver()
    # Sets a time limit of 3600 seconds.
    solver.parameters.max_time_in_seconds = 3600.0
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL:
        return [0, solver.ObjectiveValue()]
    elif status == cp_model.FEASIBLE:
        return [1, solver.ObjectiveValue()]
    else:
        print('Not found any Sol. Return [-1, -1]')
        return [-1, -1]


if __name__ == '__main__':

    import numpy as np
    import time

    l = 1
    h = 99
    init_type = ['fdd-divide-mwkr']  # ['fdd-divide-mwkr', 'spt']
    testing_type = ['validation']  # ['tai', 'abz', 'orb', 'yn', 'swv', 'la', 'ft', 'syn', 'validation']
    syn_problem_j = [10, 15, 15, 20, 20, 100, 150]  # [10, 15, 15, 20, 20, 100, 200]
    syn_problem_m = [10, 10, 15, 10, 15, 20, 25]  # [10, 10, 15, 10, 15, 20, 50]
    validation_j = [10, 15, 15, 20, 20]  # [10, 15, 15, 20, 20]
    validation_m = [10, 10, 15, 10, 15]  # [10, 10, 15, 10, 15]
    tai_problem_j = [15, 20, 20, 30, 30, 50, 50, 100]  # [15, 20, 20, 30, 30, 50, 50, 100]
    tai_problem_m = [15, 15, 20, 15, 20, 15, 20, 20]  # [15, 15, 20, 15, 20, 15, 20, 20]
    abz_problem_j = [10, 20]  # [10, 20]
    abz_problem_m = [10, 15]  # [10, 15]
    orb_problem_j = [10]  # [10]
    orb_problem_m = [10]  # [10]
    yn_problem_j = [20]  # [20]
    yn_problem_m = [20]  # [20]
    swv_problem_j = [20, 20, 50]  # [20, 20, 50]
    swv_problem_m = [10, 15, 10]  # [10, 15, 10]
    la_problem_j = [10, 15, 20, 10, 15, 20, 30, 15]  # [10, 15, 20, 10, 15, 20, 30, 15]
    la_problem_m = [5, 5, 5, 10, 10, 10, 10, 15]  # [5, 5, 5, 10, 10, 10, 10, 15]
    ft_problem_j = [6, 10, 20]  # [6, 10, 20]
    ft_problem_m = [6, 10, 5]  # [6, 10, 5]


    for test_t in testing_type:  # select benchmark
        if test_t == 'syn':
            problem_j, problem_m = syn_problem_j, syn_problem_m
        elif test_t == 'validation':
            problem_j, problem_m = validation_j, validation_m
        elif test_t == 'tai':
            problem_j, problem_m = tai_problem_j, tai_problem_m
        elif test_t == 'abz':
            problem_j, problem_m = abz_problem_j, abz_problem_m
        elif test_t == 'orb':
            problem_j, problem_m = orb_problem_j, orb_problem_m
        elif test_t == 'yn':
            problem_j, problem_m = yn_problem_j, yn_problem_m
        elif test_t == 'swv':
            problem_j, problem_m = swv_problem_j, swv_problem_m
        elif test_t == 'la':
            problem_j, problem_m = la_problem_j, la_problem_m
        elif test_t == 'ft':
            problem_j, problem_m = ft_problem_j, ft_problem_m
        else:
            raise Exception(
                'Problem type must be in testing_type = ["tai", "abz", "orb", "yn", "swv", "la", "ft", "syn", "validation"].')

        for p_j, p_m in zip(problem_j, problem_m):  # select problem size

            if test_t == 'validation':
                inst = np.load('./validation_data/validation_instance_{}x{}[1,99].npy'.format(p_j, p_m))
            else:
                inst = np.load('./test_data/{}{}x{}.npy'.format(test_t, p_j, p_m))
            print('\nStart solving {}{}x{} using OR-Tools...\n'.format(test_t, p_j, p_m))

            # read saved gap_against or use ortools to solve it.
            if test_t in ['tai', 'abz', 'orb', 'yn', 'swv', 'la', 'ft']:
                from pathlib import Path
                ortools_path = Path('./ortools_result/ortools_{}{}x{}_result.npy'.format(test_t, p_j, p_m))
                if not ortools_path.is_file():
                    gap_against = np.load('./test_data/{}{}x{}_result.npy'.format(test_t, p_j, p_m))
                    results = []
                    time_log = []
                    for i, data in enumerate(inst):
                        time_start = time.time()
                        times_rearrange = np.expand_dims(data[0], axis=-1)
                        machines_rearrange = np.expand_dims(data[1], axis=-1)
                        data = np.concatenate((machines_rearrange, times_rearrange), axis=-1)
                        result = MinimalJobshopSat(data.tolist())
                        print('Instance' + str(i + 1) + ' makespan:', result)
                        results.append(result)
                        time_end = time.time()
                        time_log.append(time_end - time_start)
                    results = np.array(results)
                    time_log = np.array(time_log)
                    ortools_obj = results[:, 1]
                    ortools_gap = (ortools_obj - gap_against)/gap_against
                    ortools_gap_mean = ortools_gap.mean()
                    np.save('./ortools_result/ortools_{}{}x{}_result.npy'.format(test_t, p_j, p_m), results)
                    np.save('./ortools_result/ortools_{}{}x{}_time.npy'.format(test_t, p_j, p_m), time_log.reshape(-1, 1))
                    print('Or-Tools mean gap:', ortools_gap_mean)
                    print('Or-Tools mean time:', time_log.mean())
            elif test_t == 'validation':
                # ortools solver
                from pathlib import Path
                ortools_path = Path('./validation_data/{}{}x{}_ortools_result.npy'.format(test_t, p_j, p_m))
                if not ortools_path.is_file():
                    results = []
                    time_log = []
                    print('Starting Ortools...')
                    for i, data in enumerate(inst):
                        time_start = time.time()
                        times_rearrange = np.expand_dims(data[0], axis=-1)
                        machines_rearrange = np.expand_dims(data[1], axis=-1)
                        data = np.concatenate((machines_rearrange, times_rearrange), axis=-1)
                        result = MinimalJobshopSat(data.tolist())
                        print('Instance-' + str(i + 1) + ' Ortools makespan:', result)
                        results.append(result[1])
                        time_end = time.time()
                        time_log.append(time_end - time_start)
                    results = np.array(results)
                    time_log = np.array(time_log)
                    np.save('./validation_data/{}{}x{}_ortools_result.npy'.format(test_t, p_j, p_m), results)
                    np.save('./validation_data/{}{}x{}_ortools_time.npy'.format(test_t, p_j, p_m), time_log.reshape(-1, 1))
                    print('Or-Tools mean gap:', 0)
                    print('Or-Tools mean time:', time_log.mean())
            else:
                # ortools solver
                from pathlib import Path
                ortools_path = Path('./test_data/{}{}x{}_result.npy'.format(test_t, p_j, p_m))
                if not ortools_path.is_file():
                    results = []
                    time_log = []
                    print('Starting Ortools...')
                    for i, data in enumerate(inst):
                        time_start = time.time()
                        times_rearrange = np.expand_dims(data[0], axis=-1)
                        machines_rearrange = np.expand_dims(data[1], axis=-1)
                        data = np.concatenate((machines_rearrange, times_rearrange), axis=-1)
                        result = MinimalJobshopSat(data.tolist())
                        print('Instance-' + str(i + 1) + ' Ortools makespan:', result)
                        results.append(result[1])
                        time_end = time.time()
                        time_log.append(time_end - time_start)
                    results = np.array(results)
                    time_log = np.array(time_log)
                    np.save('./test_data/{}{}x{}_result.npy'.format(test_t, p_j, p_m), results)
                    np.save('./test_data/{}{}x{}_time.npy'.format(test_t, p_j, p_m), time_log.reshape(-1, 1))
                    print('Or-Tools mean gap:', 0)
                    print('Or-Tools mean time:', time_log.mean())