import pandas as pd
import sys
import os

if __name__ == '__main__':
    data = pd.read_csv(sys.argv[1], header = 0)

    commit_id_list = list(data['cycle_id'])
    # #print(cycle_id_list)
    commit_id_list = list(dict.fromkeys(commit_id_list))

    output_data = pd.DataFrame()
    for commit_id in commit_id_list:
        data_subset = data.loc[data['cycle_id'] == commit_id]
        d_mean = data_subset.mean()

        output_data_temp = pd.DataFrame({"cycle_id":[commit_id], "num_testsuite":[d_mean[0]], "NORMALIZED_RPA":[d_mean[1]], "total_failures_in_cycle":[d_mean[2]], "exec_time":[d_mean[3]], "optimal_failures_25%":[d_mean[4]], "failures_in_25%_ordered":[d_mean[5]], "optimal_exec_time_25%":[d_mean[6]], "exec_time_25%":[d_mean[7]], "optimal_failures_50%":[d_mean[8]], "failures_in_50%_ordered":[d_mean[9]], "optimal_exec_time_50%":[d_mean[10]],"exec_time_50%":[d_mean[11]], "optimal_failures_75%":[d_mean[12]], "failures_in_75%_ordered":[d_mean[13]], "optimal_exec_time_75%":[d_mean[14]], "exec_time_75%":[d_mean[15]]})
        output_data = output_data.append(output_data_temp)
        
    if not os.path.isfile('mean'+sys.argv[1]):
        output_data.to_csv('mean'+sys.argv[1], index = False, header = True)
    else: # else it exists so append without writing the header
        output_data.to_csv('mean'+sys.argv[1],index = False, mode = 'a', header = False)