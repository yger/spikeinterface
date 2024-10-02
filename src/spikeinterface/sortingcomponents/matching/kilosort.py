"""Sorting components: template matching."""

from __future__ import annotations


import numpy as np
import scipy

from .base import BaseTemplateMatching, _base_matching_dtype

class KiloSortPeeler(BaseTemplateMatching):

    def __init__(self, recording, return_output=True, parents=None,
        templates=None,
        temporal_components=None,
        spatial_components=None,
        Th=8,
        random_chunk_kwargs={},
        ):

        BaseTemplateMatching.__init__(self, recording, templates, return_output=True, parents=None)
        self.templates_array = self.templates.get_dense_templates()
        self.spatial_components = spatial_components
        self.temporal_components = temporal_components

        n_components = len(self.temporal_components)
        n_templates = len(self.templates_array)
        n_channels = recording.get_num_channels()
        n_t = self.templates_array.shape[1]

        U = np.zeros((n_templates, n_channels, n_components), dtype=np.float32)
        for i in range(n_templates):
            U[i] = np.dot(spatial_components, self.templates_array[i]).T

        Uex = np.einsum('xyz, zt -> xty', U, self.spatial_components)
        X = Uex.reshape(-1, n_channels).T
        X = scipy.signal.oaconvolve(X[:, None, :], self.temporal_components[None, :, ::-1], mode='full', axes=2)
        X = X[:, :, n_t//2:n_t//2+n_t*n_templates]

        Xmax = np.abs(X).max(0).max(0).reshape(-1, n_t)
        imax = np.argmax(Xmax, 1)

        Unew = Uex.copy()
        for j in range(n_t):
            ix = imax == j
            Unew[ix] = np.roll(Unew[ix], n_t//2 - j, -2)
        Unew = np.einsum('xty, zt -> xzy', Unew, spatial_components)

        self.U = Unew
        self.W = self.spatial_components
        WtW = scipy.signal.oaconvolve(self.W[None, :, ::-1], self.W[:, None, :], mode='full', axes=2)

        self.WtW = np.flip(WtW, 2)
        UtU = np.einsum('ikl, jml -> ijkm',  self.U, self.U)
        self.ctc = np.einsum('ijkm, kml -> ijl', UtU, WtW)

        self.nbefore = self.templates.nbefore
        self.nafter = self.templates.nafter
        self.margin = max(self.nbefore, self.nafter)
        self.nm = (U**2).sum(-1).sum(-1)

    def get_trace_margin(self):
        return self.margin

    def compute_matching(self, traces, start_frame, end_frame, segment_index):
        
        B = scipy.signal.oaconvolve(traces.T[np.newaxis, :, :], self.W[:, None, :], mode='full', axes=2)
        B = np.einsum('ijk, kjl -> il', self.U, B)

        # trange = torch.arange(-nt, nt+1, device=device) 
        # tiwave = torch.arange(-(nt//2), nt//2+1, device=device) 

        # st = torch.zeros((100000,2), dtype = torch.int64, device = device)
        # amps = torch.zeros((100000,1), dtype = torch.float, device = device)
        # k = 0

        # Xres = X.clone()
        # lam = 20

        for t in range(100):
            # Cf = 2 * B - nm.unsqueeze(-1) 
            Cf = torch.relu(B)**2 /nm.unsqueeze(-1)
            Cf[:, :nt] = 0
            Cf[:, -nt:] = 0

            Cfmax, imax = np.max(Cf, 0)
            Cmax  = max_pool1d(Cfmax.unsqueeze(0).unsqueeze(0), (2*nt+1), stride = 1, padding = (nt))

            #print(Cfmax.shape)
            #import pdb; pdb.set_trace()
            cnd1 = Cmax[0,0] > Th**2
            cnd2 = torch.abs(Cmax[0,0] - Cfmax) < 1e-9
            xs = torch.nonzero(cnd1 * cnd2)

            if len(xs)==0:
                #print('iter %d'%t)
                break

            iX = xs[:,:1]
            iY = imax[iX]

            #isort = torch.sort(iX)

            nsp = len(iX)
            st[k:k+nsp, 0] = iX[:,0]
            st[k:k+nsp, 1] = iY[:,0]
            amps[k:k+nsp] = B[iY,iX] / nm[iY]
            amp = amps[k:k+nsp]

            k+= nsp

            #amp = B[iY,iX] 

            n = 2
            for j in range(n):
                Xres[:, iX[j::n] + tiwave]  -= amp[j::n] * torch.einsum('ijk, jl -> kil', U[iY[j::n,0]], W)
                B[   :, iX[j::n] + trange]  -= amp[j::n] * ctc[:,iY[j::n,0],:]

        st = st[:k]
        amps = amps[:k]

        return spikes
    