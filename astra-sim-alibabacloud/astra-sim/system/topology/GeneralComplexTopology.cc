/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "GeneralComplexTopology.hh"
#include "DoubleBinaryTreeTopology.hh"
#include "RingTopology.hh"

namespace AstraSim
{
BasicLogicalTopology *GeneralComplexTopology::get_basic_topology_at_dimension(int dimension, ComType type)
{
    return dimension_topology[dimension]->get_basic_topology_at_dimension(0, type);
}
int GeneralComplexTopology::get_num_of_nodes_in_dimension(int dimension)
{
    if (dimension >= dimension_topology.size())
    {
        std::cout << "dim: " << dimension << " requested! but max dim is: " << dimension_topology.size() - 1 << std::endl;
    }
    assert(dimension < dimension_topology.size());
    return dimension_topology[dimension]->get_num_of_nodes_in_dimension(0);
}
int GeneralComplexTopology::get_num_of_dimensions() { return dimension_topology.size(); }
GeneralComplexTopology::~GeneralComplexTopology()
{
    for (int i = 0; i < dimension_topology.size(); i++)
    {
        delete dimension_topology[i];
    }
}
// dimension_size：每一维的规模，比如 [4, 4] 表示 2D 4x4，这里为physical_dims
// collective_implementation：每个维度采用什么集体通信实现如[CollectiveImplementation(Ring), CollectiveImplementation(Direct,
// 5)]，这里为[NcclFlowModel]
GeneralComplexTopology::GeneralComplexTopology(int id, std::vector<int> dimension_size,
                                               std::vector<CollectiveImplementation *> collective_implementation)
{
    int offset = 1;
    int last_dim = collective_implementation.size() - 1;
    for (int dim = 0; dim < collective_implementation.size(); dim++)
    {
        if (collective_implementation[dim]->type == CollectiveImplementationType::Ring ||
            collective_implementation[dim]->type == CollectiveImplementationType::Direct ||
            collective_implementation[dim]->type == CollectiveImplementationType::HalvingDoubling ||
            collective_implementation[dim]->type == CollectiveImplementationType::NcclFlowModel ||
            collective_implementation[dim]->type == CollectiveImplementationType::NcclTreeFlowModel)
        // 每一维都建一个拓扑，为每个节点在某一维度上分配其在 Ring 拓扑中的“逻辑环位置”，从而实现类似 NCCL Ring AllReduce 里的环通信流调度
        {
            RingTopology *ring =
                new RingTopology(RingTopology::Dimension::NA,                    // 指定本 ring 拓扑的维度类型（如 X, Y, Z，或者未区分时用 NA）
                                 id,                                             // 本节点的全局唯一编号，如第0~N-1号节点。
                                 dimension_size[dim],                            // 本维度上的节点数量。决定了环的“长度”，也就是每个环多少节点。
                                 (id % (offset * dimension_size[dim])) / offset, // 本节点在本维度环中的索引位置。决定在本维上的通信邻居是谁
                                 offset);                                        // 本维度的“步长”，例如在第2维，offset等于第1维长度
            dimension_topology.push_back(ring);
        }
        else if (collective_implementation[dim]->type == CollectiveImplementationType::OneRing ||
                 collective_implementation[dim]->type == CollectiveImplementationType::OneDirect ||
                 collective_implementation[dim]->type == CollectiveImplementationType::OneHalvingDoubling)
        // 整个多维空间只建一个总拓扑
        {
            int total_npus = 1;
            for (int d : dimension_size)
            {
                total_npus *= d;
            }
            RingTopology *ring = new RingTopology(RingTopology::Dimension::NA, id, total_npus, id % total_npus, 1);
            dimension_topology.push_back(ring);
            return;
        }
        else if (collective_implementation[dim]->type == CollectiveImplementationType::DoubleBinaryTree)
        // 双二叉树拓扑
        {
            if (dim == last_dim)
            {
                DoubleBinaryTreeTopology *DBT = new DoubleBinaryTreeTopology(id, dimension_size[dim], id % offset, offset);
                dimension_topology.push_back(DBT);
            }
            else
            {
                DoubleBinaryTreeTopology *DBT =
                    new DoubleBinaryTreeTopology(id, dimension_size[dim], (id - (id % (offset * dimension_size[dim]))) + (id % offset), offset);
                dimension_topology.push_back(DBT);
            }
        }
        offset *= dimension_size[dim];
    }
}
} // namespace AstraSim
