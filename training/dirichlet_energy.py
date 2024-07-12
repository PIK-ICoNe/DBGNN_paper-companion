from gnn_models import *
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix
from torch_geometric.utils import *

def compute_dirichlet_energy(model,data):
    model.eval()

    x, edge_index, e = data.x, data.edge_index, data.edge_attr
    L_tmp = get_laplacian(edge_index, e)[0]
    L = to_torch_csr_tensor(L_tmp)

    if type(model) == DBGNNModel:
        return compute_DE_DBGNN(model, L, edge_index, x, e)
    elif type(model) == TAGModel:
        return compute_DE_TAG(model, L, data)


def compute_DE_DBGNN(model, L, edge_index, x, e):
    dbgnn = model.dbgnn_conv
    num_layers = model.dbgnn_conv.num_layers
    num_steps = model.dbgnn_conv.num_steps

    x_n = dbgnn.dense_n_in(x)
    x_e = dbgnn.dense_e_in(e)
    energies = torch.Tensor(0)
    for lay in range(num_layers):
        for step in range(num_steps):
            energy = comp_dirEnergy(x_n, L)
            energies = torch.cat([energies, energy.unsqueeze(-1)],0)
            x_n, x_e = dbgnn.linDB[lay](x_n, edge_index, x_e)
    
    return energies

def compute_DE_TAG(model, L, data):
    convlist = model.convlist
    num_layers = len(convlist)
    x = data.x
    energies = torch.Tensor(0)
    for i in range(num_layers):
        x = convlist[i](data, x)
        energy = comp_dirEnergy(x, L)
        energies = torch.cat([energies, energy.unsqueeze(-1)],0)
    return energies


def comp_dirEnergy(x, L):
    return (torch.trace(torch.t(x) @ L @ x) / torch.trace( x @ torch.t(x))) 