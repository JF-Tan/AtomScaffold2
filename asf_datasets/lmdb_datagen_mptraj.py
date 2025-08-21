# import os 
# os.environ["HF_ENDPOINT"] = 'https://hf-mirror.com'


from datasets import load_dataset,load_from_disk

ds = load_dataset("/root/.cache/huggingface/hub/datasets--nimashoghi--mptrj/snapshots/f88fbe46e16524223210654bad9e1b05a15c2adb/data/")

import lmdb
import pickle
from tqdm import tqdm
def save_lmdb(lmdb_path = '',atoms_list=None):
    
    env = lmdb.open(
        lmdb_path,
        map_size=1 << 40,  # 1 TB
        subdir=False,
        meminit=False,
        map_async=True
    )

    with env.begin(write=True) as txn:
        for idx, atoms in tqdm(enumerate(atoms_list), total=len(atoms_list)):
            txn.put(
                key=f"{idx}".encode("ascii"),
                value=pickle.dumps(atoms)
            )
        txn.put(b'length', pickle.dumps(len(atoms_list)))
        print(f"✅ 已保存 {len(atoms_list)} 个结构到 {lmdb_path}")
    env.sync()
    env.close()

save_lmdb(lmdb_path='/ve/junfu/AtomScaffold2/data/mptraj/train/data',atoms_list=ds['train'])
save_lmdb(lmdb_path='/ve/junfu/AtomScaffold2/data/mptraj/test/data',atoms_list=ds['test'])
save_lmdb(lmdb_path='/ve/junfu/AtomScaffold2/data/mptraj/valid/data',atoms_list=ds['valid'])
