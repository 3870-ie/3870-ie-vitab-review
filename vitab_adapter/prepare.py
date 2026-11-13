"""Paper-version MulTaBench splits; no changes to source data or historical runs."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import time
import types
import urllib.request
import zipfile

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

DATASETS = ['mt_mango_mass','vt_skin_cancer','vt_breast_cancer','vt_infarction',
 'mt_amazon_bestseller','mt_mammography_cmmd','mt_hateful_meme','vt_pawpularity',
 'mt_justin_instagram','mt_glaucoma_smdg','mt_mkphoto_bots','vt_adoption','vt_anime','mt_chexpert']


def official_splits(vendor):
    vendor.mkdir(parents=True,exist_ok=True)
    target=vendor/'tabstar_1_1_15_splits.py'
    if not target.exists():
        meta=json.load(urllib.request.urlopen('https://pypi.org/pypi/tabstar/1.1.15/json',timeout=60))
        wheel=next(v for v in meta['urls'] if v['packagetype']=='bdist_wheel')
        blob=urllib.request.urlopen(wheel['url'],timeout=120).read()
        assert hashlib.sha256(blob).hexdigest()==wheel['digests']['sha256']
        with zipfile.ZipFile(io.BytesIO(blob)) as archive:
            raw=archive.read('tabstar/preprocessing/splits.py').decode()
            constants=archive.read('tabstar/constants.py').decode()
        assert 'SEED = 42' in constants
        # Only replace the constant import; all official splitting functions unchanged.
        target.write_text(raw.replace('from tabstar.constants import SEED','SEED = 42'),encoding='utf-8')
        (vendor/'provenance.json').write_text(json.dumps(dict(version='1.1.15',wheel_url=wheel['url'],
            wheel_sha256=wheel['digests']['sha256'],original_source_sha256=hashlib.sha256(raw.encode()).hexdigest()),indent=2))
    module=types.ModuleType('paper_tabstar_splits')
    exec(compile(target.read_text(),str(target),'exec'),module.__dict__)
    return module


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset', action='append', choices=DATASETS)
    p.add_argument('--data-base',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    from mmpfn.benchmarking.registry import get_dataset_spec
    from mmpfn.benchmarking.data import load_benchmark_splits
    from mmpfn.prepare_multabench import _fit_transform_features,_resolve_image
    split=official_splits(a.output/'vendor');manifest=[]
    for dataset in a.dataset or DATASETS:
        spec=get_dataset_spec(dataset)
        if not dataset.startswith('mt_'):
            tag=spec.legacy_vtbench_name
            source=a.data_base/'vt'/('breast' if tag=='breast' else tag)/'features'
            out=a.output/dataset/'fold_0';out.mkdir(parents=True,exist_ok=True)
            if (out/'metadata.json').exists():
                manifest.append(json.loads((out/'metadata.json').read_text()));continue
            raw=load_benchmark_splits(spec,source)
            for name,part in raw.items():
                paths=[]
                for v in part.image_paths:
                    paths.append([str(w) for w in v] if isinstance(v,(list,tuple)) else [str(v)])
                np.savez_compressed(out/f'{name}.npz',x=part.x,y=part.y,image_paths=np.array(paths),source_row_ids=np.arange(len(part.y)))
            lengths=raw['train'].field_lengths
            meta=dict(dataset=dataset,fold=0,source_root=str(source),image_base_dir=str(source),
                image_encoding=spec.image_encoding,categorical_indices=list(raw['train'].categorical_features),
                field_lengths=[int(v) for v in lengths] if lengths is not None else None,
                counts={k:len(v.y) for k,v in raw.items()},protocol='Unchanged VT official export; fixed split',
                row_hashes={k:hashlib.sha256((source/k).as_posix().encode()+v.y.tobytes()+v.x.tobytes()).hexdigest() for k,v in raw.items()})
            (out/'metadata.json').write_text(json.dumps(meta,indent=2));manifest.append(meta)
            print(json.dumps(dict(dataset=dataset,counts=meta['counts'])),flush=True);continue
        source=a.data_base/'multabench/_downloads'/spec.kaggle_slug
        sm=json.loads((source/'metadata.json').read_text());frame=pd.read_csv(source/'data.csv')
        image_col=sm['image_col'];target=sm['target'];features=[c for c in frame if c not in (image_col,target)]
        paths=[_resolve_image(source,v) for v in frame[image_col]]
        valid=frame[target].notna().to_numpy() & np.array([v is not None and v.is_file() for v in paths])
        if not valid.all():raise ValueError(f'{dataset}: {int((~valid).sum())} missing images/targets; review required, no silent dropping')
        label=LabelEncoder();is_cls=spec.task=='classification'
        y=pd.Series(label.fit_transform(frame[target].astype(str)) if is_cls else pd.to_numeric(frame[target]).to_numpy(),index=frame.index)
        for fold in range(3):
            out=a.output/dataset/f'fold_{fold}';out.mkdir(parents=True,exist_ok=True)
            if (out/'metadata.json').exists():
                manifest.append(json.loads((out/'metadata.json').read_text()));continue
            started=time.perf_counter();x=frame.copy();yy=y.copy()
            # Exact paper_version subsample_dataset contract.
            test_examples=min(int(len(yy)*split.TEST_RATIO),split.MAX_TEST_SIZE)
            if len(yy)>10000+test_examples:
                x,_,yy,_=split.do_split(x,yy,len(yy)-(10000+test_examples),is_cls,fold)
            outer,test,yo,yt=split.split_to_test(x,yy,is_cls,fold=fold,train_examples=10000)
            train,val,ytrain,yval=split.split_to_val(outer,yo,is_cls) # official default fold=-1
            rows={name:part.index.to_numpy(dtype=np.int64) for name,part in [('train',train),('val',val),('test',test)]}
            sets=[set(v.tolist()) for v in rows.values()]
            assert not (sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
            encoded,cats,lengths,processing=_fit_transform_features(frame,features,rows['train'],sm)
            for name,idx in rows.items():
                np.savez_compressed(out/f'{name}.npz',x=encoded[idx],y=y.iloc[idx].to_numpy(dtype=np.int64 if is_cls else np.float32),
                    image_paths=np.array([str(paths[i]) for i in idx]),source_row_ids=idx)
            meta=dict(dataset=dataset,fold=fold,training_seed=42+fold,outer_random_state=42+fold,
                inner_random_state=41,source_root=str(source),source_csv_sha256=hashlib.sha256((source/'data.csv').read_bytes()).hexdigest(),
                image_base_dir=str(source),image_encoding='image_file',categorical_indices=cats,field_lengths=lengths,
                feature_names=features,preprocessing=processing,label_classes=label.classes_.tolist() if is_cls else None,
                counts={k:len(v) for k,v in rows.items()},row_hashes={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in rows.items()},
                outer_train_budget=10000,shared_validation_for_all_methods=True,
                official_binary_split_count_quirk_preserved=True,preparation_wall_seconds=time.perf_counter()-started,
                protocol='MulTaBench paper_version + tabstar1.1.15; shared inner validation for all methods')
            (out/'metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8');manifest.append(meta)
            print(json.dumps(dict(dataset=dataset,fold=fold,counts=meta['counts'])),flush=True)
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2));print('SHARED_SPLITS_READY',flush=True)

if __name__=='__main__':main()
