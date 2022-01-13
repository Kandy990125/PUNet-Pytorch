import torch
from pointnet2_ops import pointnet2_utils
from knn_cuda import KNN
from PUNet_Pytorch.emd.emd import earth_mover_distance


def knn_point(group_size, point_cloud, query_cloud, transpose_mode=False):
    knn_obj = KNN(k=group_size, transpose_mode=transpose_mode)
    dist, idx = knn_obj(point_cloud, query_cloud)
    return dist, idx


def get_repulsion_loss(pred):
    _, idx = knn_point(5, pred, pred, transpose_mode=True)
    idx = idx[:, :, 1:].to(torch.int32)  # remove first one
    idx = idx.contiguous()  # B, N, nn

    pred = pred.transpose(1, 2).contiguous()  # B, 3, N
    grouped_points = pointnet2_utils.grouping_operation(pred, idx)  # (B, 3, N), (B, N, nn) => (B, 3, N, nn)

    grouped_points = grouped_points - pred.unsqueeze(-1)
    dist2 = torch.sum(grouped_points ** 2, dim=1)
    dist2 = torch.max(dist2, torch.tensor(1e-12).cuda())
    dist = torch.sqrt(dist2)
    weight = torch.exp(- dist2 / 0.03 ** 2)

    uniform_loss = torch.mean((0.07 - dist) * weight)
    # uniform_loss = torch.mean(self.radius - dist * weight) # punet
    return uniform_loss


def get_emd_loss(pred, gt, pcd_radius=1.0):
    return torch.mean(earth_mover_distance(pred, gt))


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = torch.rand(size=(2, 4096, 3)).to(device)
    data2 = torch.rand(size=(2, 2048, 3)).to(device)
    print(get_emd_loss(data.transpose(1, 2).contiguous(), data2.transpose(1, 2).contiguous()))

