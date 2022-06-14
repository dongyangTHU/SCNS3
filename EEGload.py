# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import mne
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score
from mne import Epochs, pick_types, events_from_annotations
from mne.channels import make_standard_montage
from mne.io import concatenate_raws, read_raw_edf
from mne.time_frequency import tfr_morlet, psd_multitaper, psd_welch
from mne.datasets import eegbci
from mne.decoding import CSP
import json
from mne.time_frequency import psd_welch
from mne.minimum_norm import read_inverse_operator, compute_source_psd_epochs,source_band_induced_power

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

event_id = dict(move=1, rest=2)
#加载数据
subj = 'S1'
f_name = f'{subj}_mi1.cnt'
data_path = os.path.join('../data/eeg/S1/')
raw = mne.io.read_raw_cnt(os.path.join(data_path, f_name), preload=True)
# drop bad channels
raw.drop_channels(['left chn', 'rigth chn', 'M1', 'M2', 'CB1', 'CB2'])
#raw.plot()
#plt.show()


events, _ = events_from_annotations(raw)
picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                   exclude='bads')
epochs = Epochs(raw, events=events, event_id=event_id, tmin=-1, tmax=2, proj=True, picks=picks,
                baseline=None, preload=True)


#V1 DEBUG
print(epochs.info['dig'])

#V2


#V3(地形图）
#'''
freqs = np.logspace(*np.log10([6, 35]), num=8)
n_cycles = freqs / 2.  # different number of cycle per frequency
power, itc = tfr_morlet(epochs, freqs=freqs, n_cycles=n_cycles, use_fft=True,
                        return_itc=True, decim=3, n_jobs=1)
fig, axis = plt.subplots(1, 2, figsize=(7, 4))
power.plot_topomap(ch_type='eeg', tmin=0.5, tmax=1.5, fmin=8, fmax=12,
                   baseline=(-0.5, 0), mode='zscore', axes=axis[0],
                   title='alpha', show=False)
power.plot_topomap(ch_type='eeg', tmin=0.5, tmax=1.5, fmin=13, fmax=25,
                   baseline=(-0.5, 0), mode='zscore', axes=axis[1],
                   title='beta', show=False)
mne.viz.tight_layout()
plt.show()
#'''
'''
# V4（散点图和折线图）
print(events.shape)
events_new = events[1:]
print(events_new.shape)

fig = mne.viz.plot_events(events_new, event_id=event_id, sfreq=raw.info['sfreq'],
                         first_samp=raw.first_samp)
raw.filter(10., 30., fir_design='firwin', skip_by_annotation='edge')
raw, ref_data = mne.set_eeg_reference(raw) # average rereference
epochs = mne.Epochs(raw, events_new, event_id=event_id, tmin=0, tmax=10, proj=True, baseline=None, preload=True)

fig, ax = plt.subplots(1, 1)
epochs['move'].plot_psd(picks=['C3', 'C4'], dB=True, tmin=1, tmax=9, fmin=5, fmax=20, show=False, ax=ax)
ax.set_title('Move')
ax.set_ylim([15, 35])
fig, ax = plt.subplots(1, 1)
epochs['rest'].plot_psd(picks=['C3', 'C4'], dB=True, tmin=1, tmax=9, fmin=5, fmax=20, show=False, ax=ax)

ax.set_title('Rest')
ax.set_ylim([15, 35])

plt.show()
#'''
'''
#V5
raw.plot_psd(fmax=50)
raw.plot(duration=5, n_channels=30)
#'''

#V6  ICA分析
'''
ica = mne.preprocessing.ICA(n_components=20, random_state=97, max_iter=800)
ica.fit(raw)
ica.exclude = [1, 2]  # details on how we picked these are omitted here
ica.plot_properties(raw, picks=ica.exclude)
#'''

#V7
#'''

#'''