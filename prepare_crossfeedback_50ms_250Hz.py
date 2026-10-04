"""从长EEGLAB epochs重新提取50ms功率；不从200ms平均特征插值。
Spyder先运行本文件，再运行EEG_within_subject_crossfeedback_v6_50ms_250Hz.py。
采用已有5-cycle Morlet、250Hz重采样、-300至-100ms单试次dB基线。
"""
from pathlib import Path
import json, hashlib, os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import resample_poly, fftconvolve

BASE = Path(r'D:/LOOP脑电数据')
SOURCE = BASE / 'EEG_main_analysis/results/feedback_patterns_20261001'
OUTPUT = BASE / 'EEG_main_analysis/results/feedback_patterns_50ms_250Hz_20261004'
BANDS = dict(delta=[2], theta=[4,6], alpha=[8,10,12],
             beta=list(range(14,31,2)), lowbeta=[12,14,16,18], highbeta=list(range(20,31,2)))
FS_OUT = 250
STARTS = np.arange(0,1000,50)
SUBJECTS = None  # 正式保持None；如只验证可指定[101]并更换OUTPUT。

def extract(g):
    sid = int(g.SubjectID.iloc[0])
    path = BASE / f'Exp2aLOOP_EEG/EEG/8_ICs/LoopEEG{sid}_8ICs.set'
    e = loadmat(path, simplify_cells=True)
    e = e.get('EEG',e)
    fs, nt = float(e['srate']), int(e['pnts'])
    assert fs == 500 and int(e['nbchan']) == 60
    labels = [str(c['labels']).lower() for c in e['chanlocs']]
    data = e['data']
    if isinstance(data, str):
        data = np.memmap(path.parent/data, dtype='<f4', mode='r').reshape(
            (60,nt,int(e['trials'])), order='F')
    mapping = {}
    for ev in e['event']:
        u = np.asarray(ev.get('urevent',[])).ravel()
        if len(u) == 1:
            local = float(ev['latency'])-1-(int(ev['epoch'])-1)*nt
            if abs(float(e['xmin'])+local/fs) < .0021:
                key = int(u[0])
                epoch = int(ev['epoch'])-1
                if key in mapping and mapping[key] != epoch:
                    raise ValueError('不唯一的epoch映射')
                mapping[key] = epoch
    ep = np.array([mapping[int(u)] for u in g.zero_urevent])
    assert len(set(ep)) == len(ep)
    # 500 -> 250 Hz真实抗混叠重采样，随后重新计算Morlet功率。
    times = float(e['xmin']) + np.arange((nt+1)//2)/FS_OUT
    ms = times*1000
    baseline = np.flatnonzero((ms >= -300-1e-7) & (ms <= -100+1e-7))
    sample = np.flatnonzero((ms >= -1e-7) & (ms < 1000-1e-7))
    windows = [np.flatnonzero((ms[sample] >= st-1e-7) & (ms[sample] < st+50-1e-7))
               for st in STARTS]
    assert sorted(set(map(len,windows))) == [12,13]
    assert sum(map(len,windows)) == 250
    out = np.empty((len(g),6,20,60), np.float32)
    freqs = sorted(set(f for v in BANDS.values() for f in v))
    for ch in range(60):
        raw = np.asarray(data[ch][:,ep],float).T
        raw -= raw.mean(1,keepdims=True)
        raw = resample_poly(raw,1,2,axis=1)
        byfreq = {}
        for f in freqs:
            sigma = 5/(2*np.pi*f)
            half = int(np.ceil(3.5*sigma*FS_OUT))
            assert baseline.min()-half >= 0 and sample.max()+half < raw.shape[1]
            wt = np.arange(-half,half+1)/FS_OUT
            w = (np.exp(2j*np.pi*f*wt)-np.exp(-25/2))*np.exp(-wt**2/(2*sigma*sigma))
            w /= np.linalg.norm(w)
            z = fftconvolve(raw,w[None,:],mode='same',axes=1)
            bp = (abs(z[:,baseline])**2).mean(1)
            pw = abs(z[:,sample])**2
            assert (bp>0).all() and (pw>0).all()
            byfreq[f] = 10*np.log10(pw/bp[:,None])
        for bi,(band,ff) in enumerate(BANDS.items()):
            db = sum(byfreq[f] for f in ff)/len(ff)
            out[:,bi,:,ch] = np.stack([db[:,idx].mean(1) for idx in windows],axis=1)
    assert np.isfinite(out).all()
    return out, labels, dict(subject=sid, trials=len(g), source_hz=fs, sampling_hz=FS_OUT,
                            samples_per_window=[len(w) for w in windows],
                            baseline_samples=len(baseline), epoch_mapping_unique=True)

def main():
    d = pd.read_csv(SOURCE/'trials.csv')
    if SUBJECTS is not None:
        d = d[d.SubjectID.isin(SUBJECTS)].reset_index(drop=True)
    OUTPUT.mkdir(parents=True,exist_ok=True)
    if (OUTPUT/'neural.npy').exists():
        raise FileExistsError('输出已存在；保留旧数据，请更换OUTPUT或直接使用已完成特征。')
    d.to_csv(OUTPUT/'trials.csv',index=False)
    x = np.lib.format.open_memmap(OUTPUT/'neural.npy', mode='w+', dtype='<f4', shape=(len(d),6,20,60))
    audits, expected = [], None
    for sid, g in d.groupby('SubjectID',sort=True):
        power, labels, audit = extract(g)
        if expected is None:
            expected = labels
        assert labels == expected
        x[g.index.to_numpy()] = power
        x.flush()
        audits.append(audit)
        print('完成50ms提取',sid,flush=True)
    pd.DataFrame(audits).to_csv(OUTPUT/'extraction_audit.csv',index=False)
    spec = dict(bands=list(BANDS), frequencies=BANDS, labels=expected,
                windows_ms=[[int(s),int(s+50)] for s in STARTS],
                sampling_hz=FS_OUT, cycles=5, baseline_ms=[-300,-100],
                feature='mean single-trial dB power within each half-open 50ms window',
                subjects=int(d.SubjectID.nunique()), trials=len(d),
                trials_hash=hashlib.sha256((OUTPUT/'trials.csv').read_bytes()).hexdigest(),
                script_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (OUTPUT/'feature_spec.json').write_text(json.dumps(spec,ensure_ascii=False,indent=2),encoding='utf-8')
    print('特征准备完成',x.shape,flush=True)

if __name__ == '__main__':
    main()


