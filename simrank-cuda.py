import random

from numba import cuda
from tqdm import tqdm
import pandas as pd
import numpy as np
from networkx.algorithms import bipartite


def create_random_bipartite_graph(P=50, C=50, p=0.1):
    # Create a random bipartite graph
    B = bipartite.random_graph(P, C, p)

    for u, v in B.edges():
        B[u][v]["weight"] = round(
            random.random(), 2
        )  # Assigns a random float from 0 to 1

    # The nodes are labeled 0 to P-1 for the first set, and n1 to P+C-1 for the second set
    p_nodes = {n for n, d in B.nodes(data=True) if d["bipartite"] == 0}
    c_nodes = set(B) - p_nodes

    data_lst = []
    for u, v in B.edges():
        #     if u in c_nodes:
        data_lst.append((u, v, B[u][v]["weight"]))

    data = pd.DataFrame(data_lst, columns=["P", "C", "weight"])
    return data


@cuda.jit()
def update_bipartite_similarities(dampening_factor, N, W, N_offsets, S2, S1_out):
    i_start, j_start = cuda.grid(2)
    stride_i, stride_j = cuda.gridsize(2)
    for i in range(i_start, S1_out.shape[0], stride_i):
        for j in range(j_start, S1_out.shape[1], stride_j):
            if i == j:
                S1_out[i, j] = 1.0
            else:
                sum_ = 0.0
                num_neighbors = 0
                for ni_index in range(N_offsets[i], N_offsets[i + 1]):
                    ni = N[ni_index]
                    wi = W[ni_index]
                    for nj_index in range(N_offsets[j], N_offsets[j + 1]):
                        nj = N[nj_index]
                        wj = W[nj_index]
                        sum_ += wi * wj * S2[ni, nj]
                        # sum_ += S2[ni, nj]
                        num_neighbors += 1
                S1_out[i, j] = dampening_factor / num_neighbors * sum_


@cuda.jit("(float32[:,:],float32[:,:],float64[:])")
def cuda_sum_abs_diff(A, B, out):
    for i in range(cuda.grid(1), A.shape[0], cuda.gridsize(1)):
        thread_sum = 0.0
        for j in range(A.shape[1]):
            thread_sum += abs(A[i, j] - B[i, j])
        out[i] = thread_sum


def sum_abs_diff(A, B) -> float:
    assert (
        A.shape == B.shape
    ), "A and B must have the same shape and have shapes {}, {}".format(
        A.shape, B.shape
    )
    reductions_threads_per_block = 16 * 16
    reductions_blocks_per_grid = int(np.ceil(A.shape[0] / reductions_threads_per_block))
    _sum = cuda.to_device(np.zeros((A.shape[0],), dtype=np.float64))
    cuda_sum_abs_diff[reductions_blocks_per_grid, reductions_threads_per_block](
        A, B, _sum
    )
    return np.sum(_sum) / A.size


def main():
    P = 50
    C = 50
    p = 0.1

    data = create_random_bipartite_graph(P=P, C=C, p=p)

    p_id_to_index = {p_id: i for i, p_id in enumerate(data.P.unique())}
    c_id_to_index = {c_id: i for i, c_id in enumerate(data.C.unique())}
    data.loc[:, "P"] = data.P.map(p_id_to_index).astype(np.int32)
    data.loc[:, "C"] = data.C.map(c_id_to_index).astype(np.int32)

    # We store the graph as a list of neighbors and a list of weights
    # The lists are flattened, but we keep offsets to know where each node's neighbors start and end

    N1 = []  # Neighbors of nodes V1 in V2
    W1 = []  # Weights of V1->V2 edges
    N1_offsets = [0]
    for _, group in data.groupby("P"):
        for t in group.itertuples():
            N1.append(t.C)
            W1.append(t.weight)
        N1_offsets.append(len(N1))

    N2 = []
    W2 = []
    N2_offsets = [0]
    for _, group in data.groupby("C"):
        for t in group.itertuples():
            N2.append(t.P)
            W2.append(t.weight)
        N2_offsets.append(len(N2))

    dN1 = cuda.to_device(np.array(N1, dtype=np.int32))
    dN1_offsets = cuda.to_device(np.array(N1_offsets, dtype=np.int32))
    dW1 = cuda.to_device(np.array(W1, dtype=np.float32))
    dN2 = cuda.to_device(np.array(N2, dtype=np.int32))
    dN2_offsets = cuda.to_device(np.array(N2_offsets, dtype=np.int32))
    dW2 = cuda.to_device(np.array(W2, dtype=np.float32))
    S1 = cuda.to_device(np.eye(P, dtype=np.float32))
    S2 = cuda.to_device(np.eye(C, dtype=np.float32))
    S1_next = cuda.to_device(np.eye(P, dtype=np.float32))
    S2_next = cuda.to_device(np.eye(C, dtype=np.float32))

    threadsperblock = (16, 16)
    blockspergrid1 = (
        int(np.ceil(S1.shape[0] / threadsperblock[0])),
        int(np.ceil(S1.shape[1] / threadsperblock[1])),
    )
    blockspergrid2 = (
        int(np.ceil(S2.shape[0] / threadsperblock[0])),
        int(np.ceil(S2.shape[1] / threadsperblock[1])),
    )

    damping_factor = 0.95

    for it in tqdm(range(10)):
        update_bipartite_similarities[blockspergrid1, threadsperblock](
            damping_factor, dN1, dW1, dN1_offsets, S2, S1_next
        )
        update_bipartite_similarities[blockspergrid2, threadsperblock](
            damping_factor, dN2, dW2, dN2_offsets, S1, S2_next
        )
        cuda.synchronize()
        S1_diff = sum_abs_diff(S1, S1_next)
        S2_diff = sum_abs_diff(S2, S2_next)
        cuda.synchronize()
        print(f"S1 Diff = {S1_diff:.6f}, S2 Diff = {S2_diff:.6f}")

        if S1_diff < 1e-4 and S2_diff < 1e-4:
            print("Converged!")
            break

        # Swap buffers for next iteration
        buff = S1
        S1 = S1_next
        S1_next = buff
        buff = S2
        S2 = S2_next
        S2_next = buff

    print("done")

    S1_host = S1.copy_to_host() - np.eye(S1.shape[0], dtype=np.float32)
    S2_host = S2.copy_to_host() - np.eye(S2.shape[0], dtype=np.float32)

    # For each P and C, find the 5 most similar ones
    sorted_S1_idx = np.argsort(S1_host, axis=1)
    sorted_S2_idx = np.argsort(S2_host, axis=1)

    S1_top5_indices = sorted_S1_idx[:, -5:]
    S2_top5_indices = sorted_S2_idx[:, -5:]

    top5_S1 = np.take_along_axis(S1_host, S1_top5_indices, axis=1)
    top5_S2 = np.take_along_axis(S2_host, S2_top5_indices, axis=1)

    # we have the most similar indices, but not their original ids -> find them
    p_index_to_id = {i: p_id for p_id, i in p_id_to_index.items()}
    c_index_to_id = {i: c_id for c_id, i in c_id_to_index.items()}

    assert (
        len(S1_host) == P
    ), f"Unexpected number of rows in S1_host: {len(S1_host)}, expected {P}"
    assert (
        len(S2_host) == C
    ), f"Unexpected number of rows in S2_host: {len(S2_host)}, expected {C}"

    p_similarities = {}
    for p_id, p_idx in tqdm(p_id_to_index.items()):
        p_similarities[p_id] = {}
        for i in range(S1_top5_indices.shape[1]):
            similar_p_idx = S1_top5_indices[p_idx, i]
            similar_p_id = p_index_to_id[similar_p_idx]
            p_similarities[p_id][similar_p_id] = float(top5_S1[p_idx, i])

    c_similarities = {}
    for c_id, c_idx in tqdm(c_id_to_index.items()):
        c_similarities[c_id] = {}
        for i in range(S2_top5_indices.shape[1]):
            similar_c_idx = S2_top5_indices[c_idx, i]
            similar_c_id = c_index_to_id[similar_c_idx]
            c_similarities[c_id][similar_c_id] = float(top5_S2[c_idx, i])


if __name__ == "__main__":
    main()
