import collections
# Import Python wrapper for or-tools CP-SAT solver.
from ortools.sat.python import cp_model
from env.generateJSP import uni_instance_gen


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
    datas = np.array([uni_instance_gen(n_j=10, n_m=10, low=1, high=99) for _ in range(10)])
    results = []
    for i, data in enumerate(datas):
        times_rearrange = np.expand_dims(data[0], axis=-1)
        machines_rearrange = np.expand_dims(data[1], axis=-1)
        data = np.concatenate((machines_rearrange, times_rearrange), axis=-1)
        result = MinimalJobshopSat(data.tolist())
        print('Instance' + str(i + 1) + ' makespan:', result)
        results.append(result)