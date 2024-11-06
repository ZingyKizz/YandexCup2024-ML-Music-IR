# import math
# import networkx as nx
# from collections import defaultdict, Counter
# from tqdm import tqdm
# import walker
#
#
# def recalculate_neighbors(df, n_walks=1000, walk_len=4):
#     track2sourceneighbors = {}
#     for row in df.itertuples(index=False):
#         track, *original_neighbors = map(int, row[0].split())
#         track2sourceneighbors[track] = original_neighbors
#
#     G = nx.DiGraph()
#     track_to_node = {}
#     node_to_track = {}
#     node_id = 0
#
#     for track, track_neighbors in track2sourceneighbors.items():
#         if track not in track_to_node:
#             track_to_node[track] = node_id
#             node_to_track[node_id] = track
#             node_id += 1
#         for idx, neighbor in enumerate(track_neighbors, start=1):
#             if neighbor not in track_to_node:
#                 track_to_node[neighbor] = node_id
#                 node_to_track[node_id] = neighbor
#                 node_id += 1
#             weight = 1 / math.sqrt(idx)
#             G.add_edge(track_to_node[track], track_to_node[neighbor], weight=weight)
#
#     walks = walker.random_walks(G, n_walks=n_walks, walk_len=walk_len)
#     co_occurrence = defaultdict(Counter)
#
#     for walk in tqdm(walks, desc="Processing random walks"):
#         current_node, *walked_nodes = walk
#         for walked_node in walked_nodes:
#             co_occurrence[node_to_track[current_node]][node_to_track[walked_node]] += 1
#
#     def find_original_index(array, value):
#         try:
#             return array.index(value)
#         except ValueError:
#             return float('inf')
#
#     reordered_neighbors = {}
#     for track, original_neighbors in tqdm(track2sourceneighbors.items(), desc="Reordering neighbors"):
#         new_neighbors = [
#             neighbor
#             for neighbor, _ in sorted(
#                 co_occurrence[track].items(),
#                 key=lambda x: (-x[1], find_original_index(original_neighbors, x[0]))
#             )[:100]
#             if neighbor != track
#         ]
#         reordered_neighbors[track] = new_neighbors or original_neighbors
#
#     return reordered_neighbors
