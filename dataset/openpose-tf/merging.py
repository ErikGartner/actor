from collections import defaultdict
import itertools

no_merge_cache = defaultdict(list)
empty_set = set()

while True:
    is_merged = False

    for h1, h2 in itertools.combinations(connections_by_human.keys(), 2):
        for c1, c2 in itertools.product(connections_by_human[h1], connections_by_human[h2]):
            # if two humans share a part (same part idx and coordinates), merge those humans
            if set(c1['partCoordsAndIdx']) & set(c2['partCoordsAndIdx']) != empty_set:
                is_merged = True
                # extend human1 connections with human2 connections
                connections_by_human[h1].extend(connections_by_human[h2])
                connections_by_human.pop(h2) # delete human2
                break

    if not is_merged: # if no more mergings are possible, then break
        break

# describe humans as a set of parts, not as a set of connections
humans = [human_conns_to_human_parts(human_conns) for human_conns in connections_by_human.values()]