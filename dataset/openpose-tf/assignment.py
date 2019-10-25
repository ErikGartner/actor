connection = []
used_idx1, used_idx2 = [], []
# sort possible connections by score, from maximum to minimum
for conn_candidate in sorted(connection_temp, key=lambda x: x['score'], reverse=True):
    # check not connected
    if conn_candidate['idx'][0] in used_idx1 or conn_candidate['idx'][1] in used_idx2:
        continue
    connection.append(conn_candidate)
    used_idx1.append(conn_candidate['idx'][0])
    used_idx2.append(conn_candidate['idx'][1])
