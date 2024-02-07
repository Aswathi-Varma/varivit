from dataset.brats_dataset import brats
from dataset.glioma_dataset import glioma

def get_dataset(dataset_name, mode, args, sizes = None, transforms=None, use_z_score=False, split='idh'):
    assert dataset_name in ['brats', 'glioma'], f"Unsupported dataset {dataset_name}"
    
    if dataset_name == 'brats':
        if sizes is None:
            return brats.build_dataset(mode=mode, local_seed=None, args=args, transforms=transforms, use_z_score=use_z_score, split=split)
        else:
            return brats.build_multi_dataset(mode=mode, sizes=sizes, args=args , transforms=transforms , use_z_score=use_z_score)
        
    if dataset_name == 'glioma':
        if sizes is None:
            return glioma.build_dataset(mode=mode, local_seed=None, args=args, transforms=transforms, use_z_score=use_z_score, split=split)
        else:
            return glioma.build_multi_dataset(mode=mode, sizes=sizes, args=args , transforms=transforms , use_z_score=use_z_score)
    else:
        raise NotImplementedError(f"Unsupported dataset {dataset_name}")