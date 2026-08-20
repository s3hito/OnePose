import numpy as np
import scipy.spatial.distance as distance
from src.utils import path_utils


def get_pairswise_distances(pose_files):
    Rs = []
    ts = []

    seqs_ids = {}
    for i in range(len(pose_files)):
        pose_file = pose_files[i]
        seq_name = pose_file.split('/')[-3]
        if seq_name not in seqs_ids.keys():
            seqs_ids[seq_name] = [i]     
        else:
            seqs_ids[seq_name].append(i)
         
    for pose_file in pose_files:
        pose = np.loadtxt(pose_file)
        R = pose[:3, :3]
        t = pose[:3, 3:]
        Rs.append(R)
        ts.append(t)
    
    Rs = np.stack(Rs, axis=0)
    ts = np.stack(ts, axis=0)

    Rs = Rs.transpose(0, 2, 1) # [n, 3, 3]
    ts = -(Rs @ ts)[:, :, 0] # [n, 3, 3] @ [n, 3, 1]

    dist = distance.squareform(distance.pdist(ts))
    trace = np.einsum('nji,mji->mn', Rs, Rs, optimize=True)
    dR = np.clip((trace - 1) / 2, -1., 1.)
    dR = np.rad2deg(np.abs(np.arccos(dR)))

    return dist, dR, seqs_ids



def pairs_from_retrieval(img_lists, k):
    import torch.nn.functional as F, torch
    import numpy as np
    from src.sfm.dino_extractor import extract_embeddings

    embeddings = extract_embeddings(img_lists, batch_size=16)

    global_feats = F.normalize(embeddings, dim=-1)
    sim_matrix = torch.mm(global_feats, global_feats.T).cpu().numpy()
    np.fill_diagonal(sim_matrix, 0)
    pairs = []

    for i in range(len(sim_matrix)):
        top_k_matches = np.argsort(sim_matrix[i])[::-1][:k]
        for j in top_k_matches:
            if i < j:
                pairs.append((img_lists[i].split('/')[-1], img_lists[int(j)].split('/')[-1]))
    return pairs


def covis_from_pose(img_lists, covis_pairs_out, cfg, max_rotation, do_ba=False):
    pose_lists = [path_utils.get_gt_pose_path_by_color(color_path) for color_path in img_lists]
    if not cfg.sfm.use_sam3_masks:
        print("Computing co-visibility on spacial distance ")
        dist, dR, seqs_ids = get_pairswise_distances(pose_lists)

        min_rotation = 10
        valid = dR > min_rotation
        np.fill_diagonal(valid, False)
        dist = np.where(valid, dist, np.inf)

        pairs = []
        num_matched_per_seq = cfg.sfm.covis_num // len(seqs_ids.keys())
        for i in range(len(img_lists)):
            dist_i = dist[i]
            for seq_id in seqs_ids:
                ids = np.array(seqs_ids[seq_id])
                try:
                    idx = np.argpartition(dist_i[ids], num_matched_per_seq * 2)[: num_matched_per_seq:2]
                except:
                    idx = np.argpartition(dist_i[ids], dist_i.shape[0] - 1)
                idx = ids[idx]
                idx = idx[np.argsort(dist_i[idx])]
                idx = idx[valid[i][idx]]

                for j in idx:
                    name0 = img_lists[i]
                    name1 = img_lists[j]

                    pairs.append((name0, name1))
    else:
        print("Computing co-visibility on embeddings ")
        pairs = pairs_from_retrieval(img_lists, cfg.sfm.covis_num)

    with open(covis_pairs_out, 'w') as f:
        f.write('\n'.join(' '.join([i, j]) for i, j in pairs))
