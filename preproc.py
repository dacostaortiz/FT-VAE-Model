import re
import os
import numpy as np
import pandas as pd
from collections import Counter

path = './data/logs'

def read_log_files(path):
    files = os.listdir(path)
    files.sort()
    logs = []
    for file in files:
        with open(path + '/' + file, 'r') as file:
            logs.append(file.readlines())
    return logs

logs = read_log_files(path)

# Append all logs into a single list
all_logs = []
for log in logs:
    all_logs += log

logs = all_logs

patterns = {
    "Restoring Original State":    r"\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3})\] restoring original state of nodes",
    "Warning":                     r"\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3})\] warning: (.*)",
    "Select Part":                 r"\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3})\] select/cons_tres: part_data_create_array: (.*)",
    "Select Reconf":               r"\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3})\] select/cons_tres: select_p_reconfigure",
    "Plugin Initialization":       r"\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3})\] select/cons_tres: job_res_rm_job: plugin still initializing",
    "No Parameter for mcs plugin": r"\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3})\] No parameter for mcs plugin, default values set",
    "mcs: MCSParameters":          r"\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3})\] mcs: MCSParameters = (.*)",
    "reconfigure_slurm":           r"\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3})\] reconfigure_slurm: (.*)",
    "SchedulerParameters":         r"\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3})\] SchedulerParameters=(.*)",
    "Error":                       r"\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3})\] error: (.*)",
    "Error Invalid Part Spec":     r"\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3})\] _get_job_parts: invalid partition specified:",
    "Error Invalid Job":           r"\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3})\] job_str_signal\(3\): invalid JobId=(\d+)",
    "Error Invalid Job Slurm":     r"\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3})\] _slurm_rpc_kill_job: job_str_signal\(\) uid=(\d+) JobId=(\d+) sig=(\d+) returned: Invalid job id specified",
    "Error Batch Invalid Part":    r"\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3})\] _slurm_rpc_submit_batch_job: Invalid partition name specified",
    "Error Batch Node Not Ava":    r"\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3})\] _slurm_rpc_submit_batch_job: Requested node configuration is not available",
    "Error Batch Dependency":      r"\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3})\] _slurm_rpc_submit_batch_job: Job dependency problem",
    "Error Batch Unspecified":     r"\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3})\] _slurm_rpc_submit_batch_job: Unspecified error",
    "Error Batch Limit":           r"\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3})\] _slurm_rpc_submit_batch_job: Job violates accounting/QOS policy (job submit limit, user's size and/or time limits)",
    "Error Slurm scriptd":         r"\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3})\] slurmscriptd: error: (.*)",
    "Job Submissions":             r"\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3})\] _slurm_rpc_submit_batch_job: JobId=(\d+) InitPrio=(\d+) usec=(\d+)",
    "Error Requeue":               r"\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3})\] _slurm_rpc_requeue: Requeue of JobId=(\d+)(?:_\d+)?(?:_\[.*?\])?(?:_\(\d+\))?|(\d+) returned an error: (.*)",
    "Reconfiguration Request":     r"\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3})\] Processing Reconfiguration Request",
    "Job Started":                 r"\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3})\] sched/backfill: _start_job: Started JobId=(\d+)(?:_\d+)?(?:_\[.*?\])?(?:_\(\d+\))?|(\d+) in (\w+) on (\w+)",
    "Schedule Allocation":         r"\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3})\] sched: Allocate JobId=(\d+)(?:_\d+)?(?:_\[.*?\])?(?:_\(\d+\))?|(\d+) NodeList=([\w\s]+\[[^\]]*\]|\w+\d+) #CPUs=(\d+) Partition=(\w+)",
    "Schedule Allocation Slurm":   r"\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3})\] sched: _slurm_rpc_allocate_resources JobId=(\d+)(?:_\d+)?(?:_\[.*?\])?(?:_\(\d+\))?|(\d+) NodeList=([\w\s]+\[[^\]]*\]|\w+\d+) usec=(\d+)",
    "Job Starts":                  r"\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3})\] sched/backfill: _start_job: Started JobId=(\d+)(?:_\d+)?(?:_\[.*?\])?(?:_\(\d+\))?|(\d+) in (.*) on (.*)",
    "Job Terminated":              r"\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3})\] Resending TERMINATE_JOB request JobId=(\d+)(?:_\d+)?(?:_\[.*?\])?(?:_\(\d+\))?|(\d+) Nodelist=(\w+(?:\[\d+(?:-\d+)?(?:,\d+)*\])?)",
    "Time Limit Exceeded":         r"\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3})\] Time limit exhausted for JobId=(\d+(?:_\d+\(\d+\))?)",
    "Job Completion":              r"\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3})\] _sync_nodes_to_comp_job: JobId=(\d+_\d+\(\d+\)) in completing state",
    "Completing Jobs":             r"\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3})\] _sync_nodes_to_comp_job: completing (\d+) jobs",
    "Job Completion Time":         r"\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3})\] cleanup_completing: JobId=(\d+)(?:_\d+)?(?:_\[.*?\])?(?:_\(\d+\))?|(\d+) completion process took (\d+) seconds",
    "Job Complete Status":         r"\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3})\] _job_complete: JobId=(\d+(?:_\d+\(\d+\))?) WEXITSTATUS (\d+)",
    "Job Complete TermSig":        r"\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3})\] _job_complete: JobId=(\d+(?:_\d+\(\d+\))?) WTERMSIG (\d+)",
    "Job Complete":                r"\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3})\] _job_complete: JobId=(\d+(?:_\d+\(\d+\))?) done",
    "Job Complete Fail":           r"\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3})\] _job_complete: JobId=(\d+(?:_\d+\(\d+\))?) OOM failure",
    "Job Complete Requeue":        r"\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3})\] _job_complete: requeue JobId=(\d+)(?:_\d+)?(?:_\[.*?\])?(?:_\(\d+\))?|(\d+) per user/system request",
    "Kill Job Slurm":              r"\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3})\] _slurm_rpc_kill_job: REQUEST_KILL_JOB JobId=(\d+)(?:_\d+)?(?:_\[.*?\])?(?:_\(\d+\))?|(\d+) uid (\d+)",
    "Kill Job":                    r"\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3})\] Killing JobId=(\d+)(?:_\d+)?(?:_\[.*?\])?(?:_\(\d+\))?|(\d+) on failed node (\w+(?:\[\d+(?:-\d+)?(?:,\d+)*\])?)",
    "Kill Dependent":              r"\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3})\] _kill_dependent: Job dependency can't be satisfied, cancelling JobId=(\d+(?:_\d+\(\d+\))?)",
    "Ping":                        r"\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3})\] Node (\w+(?:\[\d+(?:-\d+)?(?:,\d+)*\])?) now responding",
    "Slurm Job Submit":            r"\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3})\] lua: slurm_job_submit: Job from uid (\d+), (.*)",
    "Retry List":                  r"\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3})\]\s+retry_list retry_list_size:(\d+) (.*)",
    "Agent":                       r"\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3})\] agent msg_type=(.*)",
    "Update Node State":           r"\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3})\] update_node: node (\w+(?:\[\d+(?:-\d+)?(?:,\d+)*\])?) reason set to: (.*)",
    "Update Node Reason":          r"\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3})\] update_node: node (\w+(?:\[\d+(?:-\d+)?(?:,\d+)*\])?) state set to (.*)",
    "Update Job Setting":          r"\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3})\] _update_job: setting QOS to 2112-30-std for JobId=(\d+(?:_\d+\(\d+\))?)",
    "Update Job Accounting":       r"\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3})\] _update_job: updating accounting",
    "Update Job Slurm":            r"\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3})\] _slurm_rpc_update_job: complete JobId=(\d+(?:_\d+\(\d+\))?) uid=(\d+) usec=(\d+)",
    "Error Slurm Build Node":      r"\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3})\] _build_node_list: No nodes satisfy JobId=(\d+)(?:_\d+)?(?:_\[.*?\])?(?:_\(\d+\))?|(\d+) requirements in partition (\w+)",
    "Node Return":                 r"\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3})\] node (\w+(?:\[\d+(?:-\d+)?(?:,\d+)*\])?) returned to service",
    "Node Return Resp":            r"\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3})\] node_did_resp: node (\w+(?:\[\d+(?:-\d+)?(?:,\d+)*\])?) returned to service",
    "Sync Nodes Job":              r"\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3})\] _sync_nodes_to_jobs updated state of (\d+) nodes",
    "Sync Nodes Comp Job":         r"\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3})\] _sync_nodes_to_comp_job: JobId=(\d+(?:_\d+\(\d+\))?) in completing state",
    "Step Partial Comp":           r"\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3})\] step_partial_comp: JobId=(\d+)(?:_\d+)?(?:_\[.*?\])?(?:_\(\d+\))?|(\d+) pending",
    "Storage":                     r"\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3})\] accounting_storage/slurmdbd: (.*)"
}

def process_log_data(log_data, patterns=patterns):
    key_value_pairs = []
    
    for line in log_data:
        line = line.strip()
        matched = False
        for pattern_name, pattern in patterns.items():
            match = re.search(pattern, line)
            if match:
                data = {
                    "Event": pattern_name,
                    "Timestamp": match.group(1),
                    "Line": line
                }
                if pattern_name in ("Job Submissions", "Error Requeue", "Job Started", "Schedule Allocation", "Schedule Allocation Slurm", "Job Starts", "Job Terminated",
                "Time Limit Exceeded", "Job Completion", "Job Completion Time", "Job Complete Status", "Job Complete TermSig", "Job Complete", "Job Complete Fail",
                "Job Complete Requeue", "Kill Job", "Kill Job Slurm", "Kill Dependent",  "Error Slurm Build Node",
                "Update Job Setting", "Update Job Slurm","Sync Nodes Comp Job", "Error Invalid Job", "Step Partial Comp"):
                    data["JobId"] = match.group(2)
                if pattern_name == "Error Invalid Job Slurm":
                    data["JobId"] = match.group(3)
                if pattern_name in ("Job Terminated", "Schedule Allocation"):
                    data["Node"] = match.group(3)
                if pattern_name == "Job Started":
                    data["Partition"] = match.group(3)
                    data["Node"] = match.group(4)
                if pattern_name == "Error Slurm Build Node":
                    data["Partition"] = match.group(3)
                if pattern_name in ("Warning", "Select Part", "mcs: MCSParameters", "reconfigure_slurm", "SchedulerParameters", "Error", "Agent", "Error Slurm scriptd"):
                    data["Message"] = match.group(2)
                if pattern_name in ("Update Node State", "Update Node Reason","Sync Nodes Job"):
                    data["Node"] = match.group(2)
                if pattern_name in ("Retry List", "Slurm Job Submit", "Error Requeue","Update Node State", "Update Node Reason"):
                    data["Message"] = match.group(3)
                if pattern_name == "Completing Jobs":
                    data["JobCount"] = match.group(2)
                if pattern_name == "Job Completion Time":
                    data["Completion Time (seconds)"] = match.group(3)
                if pattern_name == "Job Submissions":
                    data["InitPrio"] = match.group(3)
                    data["usec"] = match.group(4)
                if pattern_name == "Job Completions Status":
                    data["WEXITSTATUS"] = match.group(3)
                if pattern_name == "Job Complete TermSig":
                    data["WTERMSIG"] = match.group(3)
                if pattern_name == "Job Starts":
                    data["Partition"] = match.group(3)
                    data["Node"] = match.group(4)
                if pattern_name == "Kill Job Slurm":
                    data["uid"] = match.group(3)
                if pattern_name == "Kill Job":
                    data["Node"] = match.group(3)
                if pattern_name == "Schedule Allocation":
                    data["NodeList"] = match.group(3)
                    data["#CPUs"] = match.group(4)
                    data["Partition"] = match.group(5)
                if pattern_name == "Slurm Job Submit":
                    data["uid"] = match.group(2)
                if pattern_name == "Retry List":
                    data["retry_list_size"] = match.group(2)
                if pattern_name == "Schedule Allocation Slurm":
                    data["NodeList"] = match.group(3)
                    data["usec"] = match.group(4)
                if pattern_name == "Update Job Slurm":
                    data["uid"] = match.group(3)
                    data["usec"] = match.group(4)

                key_value_pairs.append(data)
                matched = True
                break
        if not matched:
            data = {
                "Event": "Unknown",
                "Message": line
            }
            key_value_pairs.append(data)

    return key_value_pairs

data = process_log_data(logs)

def build_df(data):
    df = pd.DataFrame(data)
    df = df[['Event', 'Timestamp', 'Line', 'JobId', 'Message']]
    df['Event'] = df['Event'].astype('category')
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df['JobId'] = df['JobId'].fillna('').astype(str)
    return df

df = build_df(data)
df['Global'] = df['JobId'].str.extract(r"(\d+)")
df

patterns_error = ["Error", "Error Invalid Job", "Error Invalid Job Slurm", "Error Requeue", "Time Limit Exceeded",
    "Job Complete Fail", "Kill Job", "Kill Dependent", "Error Slurm Build Node"]
# Count each event type in the dataframe
event_counts = df['Event'].value_counts()

# In column 'Event' replace the values in patterns_error with 'Error'
df['Event'] = df['Event'].apply(lambda x: 'Error' if x in patterns_error else x)

job_ids = df['Global'].unique()
print(len(job_ids))

# Get the job ids asoociated with at least one error
error_job_ids = df[df['Event'] == 'Error']['Global'].unique()
print(len(error_job_ids))

# Convert job_ids and error_job_ids to lists
job_ids = job_ids.tolist()
error_job_ids = error_job_ids.tolist()

# Save in a new list the job ids that are not in error_job_ids
non_error_job_ids = [job_id for job_id in job_ids if job_id not in error_job_ids]
print(len(non_error_job_ids))

def get_global_events(job_id, df):
    return df[df['Global']==job_id]['Event'].tolist()

sample_job_events = get_global_events(job_ids[479], df)
print(sample_job_events)

#Given a list of events, count the number of times each event appears in the list
def count_events(events):
    return Counter(events)

#Count the number of times each event appears in the list of events
event_counts = count_events(sample_job_events)
print(event_counts)

# Divide the dataframe into Event with "Error" in the Event column and any other Event in the Event column
error_df = df[df['Event'] == 'Error']
non_error_df = df[df['Event'] != 'Error']

print(error_df.shape)
print(non_error_df.shape)

non_error_sample = non_error_df.sample(n=10000, random_state=1)

#Create a fixed list of event names in the order they appear in the patterns dictionary
fixed_event_names = list(patterns.keys())

#Remove the elements that appears in patterns_error from the fixed_event_names list
for event in patterns_error:
    fixed_event_names.remove(event)


from tqdm import tqdm
#Create a matrix where each row corresponds to a JobId and each column corresponds to the number of times an event appears in the list of events for that JobId
def create_event_count_matrix(job_ids, df, fixed_event_names):
    event_matrix = []
    for job_id in tqdm(job_ids, desc="Processing Jobs", unit="job"):
        job_events = get_global_events(job_id, df)
        event_counts = count_events(job_events)
        row = [event_counts[event] for event in fixed_event_names]
        event_matrix.append(row)
    return event_matrix

#Create the event matrix
count_1 = create_event_count_matrix(error_job_ids, df, fixed_event_names)
count_0 = create_event_count_matrix(non_error_job_ids, df, fixed_event_names)

count_0 = np.array(count_0)
count_1 = np.array(count_1)

print(count_0.shape, count_1.shape)

# Create a matrix where each row corresponds to a JobId and each column corresponds to the order of appearance of the event in the patterns dictionary,
# if the event appears first, put a 1 in the cell if another event appears next, put a 2, and so on, if the event appears more than once, ignore the rest of the appearances
def create_event_order_matrix(job_ids, df, fixed_event_names):
    event_order_matrix = []
    for job_id in tqdm(job_ids, desc="Processing Jobs", unit="job"):
        job_events = get_global_events(job_id, df)
        row = []
        for event in fixed_event_names:
            if event in job_events:
                row.append(job_events.index(event) + 1)
            else:
                row.append(0)
        event_order_matrix.append(row)
    return event_order_matrix

#Create the event order matrix
order_0 = create_event_order_matrix(non_error_job_ids, df, fixed_event_names)
order_1 = create_event_order_matrix(error_job_ids, df, fixed_event_names)

order_0 = np.array(order_0)
order_1 = np.array(order_1)

print(order_0.shape, order_1.shape)

def replace_non_zero_with_order(vector):
    non_zero_values = [value for value in vector if value != 0]
    sorted_indices = sorted(range(len(non_zero_values)), key=lambda k: non_zero_values[k])

    sequential_order = 0
    result_vector = []

    for value in vector:
        if value != 0:
            result_vector.append(sorted_indices.index(sequential_order) + 1)
            sequential_order += 1
        else:
            result_vector.append(0)

    return result_vector

#Replace the non zero values in each row of the event order matrix with the order of appearance of the event in the patterns dictionary
order_0 = np.apply_along_axis(replace_non_zero_with_order, 1, order_0)
order_1 = np.apply_along_axis(replace_non_zero_with_order, 1, order_1)

#Convert count_0, count_1, order_0, and order_1 to int and save them in text files
np.savetxt('data/count_0.txt', count_0, fmt='%d')
np.savetxt('data/count_1.txt', count_1, fmt='%d')
np.savetxt('data/order_0.txt', order_0, fmt='%d')
np.savetxt('data/order_1.txt', order_1, fmt='%d')