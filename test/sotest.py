import json
import ssg4wann as sw
from ssg4wann.core.sogroup import coset_decomposition

with open('../examples/Fe/ssg_symm.json', 'r', encoding='utf-8') as f:
    ops_list = json.load(f)
    ops_list = ops_list["ssg"]["ops"]  

is_real_matrix, G_SO, G_NS = coset_decomposition(ops_list)
print(f"is_real_matrix: {is_real_matrix}")
print(f"G len={len(ops_list)}")
print(f"G_SO len={len(G_SO)}")
print(f"G_NS len={len(G_NS)}")