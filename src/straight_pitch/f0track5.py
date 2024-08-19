from __future__ import annotations

import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

from numpy.typing import NDArray

from math import sqrt
import numpy as np


def f0track5(
    f0v: NDArray,
    vrv: NDArray,
    htr: NDArray,
    thf0j: float = 0.04,  #  f0 change wrt next/prev frame, f0_current/|f0_current-f0_neighbor| < thf0j => voiced
    hth: float = 0.164, # high-conf threshold on vrv < hth => voiced
    lth: float = 0.9,  # low-conf threshold on vrv < lth => voiced
    htrth: float = 0.02,  # threshold on htr  < htrth => voiced
    bklm: int = 100,  # back track length for voicing decision
    lalm: int = 10,  # look ahead length for silence decision
) -> NDArray:
    """F0 trajectory tracker

    :param f0v: fixed point frequency vector in normalized frequency (nb_frames x nb_candidates 2D)
    :param vrv: relative interfering energy vector (nb_frames x nb_candidates 2D)
    :param dfv: fixed point slope vector (nb_frames x nb_candidates 2D)
    :param htr: (power in higher frequency range)/(total power) in dB (nb_frames long 1D)
    :param aav: amplitude list for fixed points (nb_frames x nb_candidates 2D)
    :return: sequence of candidate indices per frame (nb_frames-long 1D int array)
    """
    # 	coded by Hideki Kawahara
    # 	copyright(c) Wakayama University/CREST/ATR
    # 	10/April/1999 first version
    # 	17/May/1999 relative fq jump thresholding
    # 	01/August/1999 parameter tweeking
    #   07/Dec./2002 waitbar was added
    #   13/Jan./2005 bug fix on lines 58, 97 (Thanx Ishikasa-san)
    # 	30/April/2005 modification for Matlab v7.0 compatibility
    # 	10/Aug./2005 modified  by Takahashi on waitbar
    # 	10/Sept./2005 modified by Kawahara on waitbar

    # indices of candidates with the lowest noise level
    ixx = np.argmin(vrv, axis=1)

    # conditions for voiced frame:
    # 1. gomi < thf0j # f0 change wrt a previous/next frame
    # 2. if a candidate has low relative noise energy (high confidence heuristic)
    # 3. frame has its energy concentration in low-frequency region (low confidence heuristic)
    #
    # 2 & 3 are independent of f0
    cn_ok_cands = vrv < hth
    cn_ok = (cn_ok_cands).any(axis=1)
    pwr_ok = htr < htrth

    # if no candidate, 100% not voiced
    no_f0 = np.isnan(f0v).all(axis=1)
    unvoiced = no_f0 | ~(cn_ok | pwr_ok)  # true if 100% unvoiced
    voiced = ~no_f0 & cn_ok & pwr_ok  # true if 100% voiced

    j_unknown = np.where(~(unvoiced | voiced))[0]

    jsel = np.empty_like(htr)

    is_voiced = False
    f0ref = 0

    # if imgi==1 hpg=waitbar(0,'F0 tracking') end # 07/Dec./2002 by H.K.#10/Aug./2005


#     ii = 0
#     while ii < mm:

#         ixxi = ixx[ii]
#         f0vi = f0v[ii]
#         vrvi = vrv[ii]
#         htri = htr[ii]

#         if (not is_voiced) and (vrvi[ixxi] < hth) and (htri < htrth):
#             # if new voiced frame is found, backtrack and overwrite
#             # previous decisions until definitive unvoiced/silent frame
#             is_voiced = True
#             f0ref = f0vi[ixx[0]]  # start value for search
#             idx = slice(ii, ii - bkls, -1)
#             for k, j in _iter_backtrack_voiced(
#                 f0ref, f0v[idx], vrv[idx], htr[idx], lth, thf0j, htrth
#             ):
#                 jsel[ii - k] = j

#             f0ref = f0vi[ixxi]

#         if np.isnan(f0ref):
#             gomi = 10
#         else:
#             if np.isnan(f0vi).all():
#                 is_voiced = False
#                 jsel[ii] = -1
#                 continue

#             gom = np.abs((f0vi - f0ref) / f0ref)
#             jxx = np.argmin(gom)
#             gomi = gom[jxx]

#         if is_voiced and (vrvi[ixxi] > hth):
#             # high C/N, strong presenc ef voice
#             idx = slice(ii, ii + lals)
#             k = 0
#             for k, j in _iter_lookforward_silence(
#                 f0ref, f0v[idx], vrv[idx], htr[idx], lth, thf0j, htrth
#             ):
#                 jsel[ii + k] = j
#             ii = ii + k
#             if k < lals:
#                 is_voiced = False
#                 jsel[ii + k] = -1

#         elif is_voiced and (gomi < thf0j) and (htri < htrth):
#             # voice present
#             jsel[ii] = jxx
#             f0ref = f0vi[jxx]
#         else:
#             is_voiced = False
#             jsel[ii] = -1

#         # if imgi==1 waitbar(ii/mm) end #,hpg) # 07/Dec./2002 by H.K.#10/Aug./2005
#         ii += 1

#     return jsel


# def _iter_backtrack_voiced(f0ref, f0v, vrv, htr, lth, thf0j, htrth):

#     def process_frame(
#         f0ref: float, f0vj: NDArray, vrvj: NDArray, htrj: NDArray
#     ) -> tuple[int, float] | None:

#         if np.isnan(f0vj).all():
#             # no valid f0 candidate
#             return None

#         gom = np.abs((f0vj - f0ref) / f0ref)
#         jxx = np.nanargmin(gom)
#         gomi = gom[jxx]
#         # gomi = gomi + (f0ref > 10000) + (f0vj[jxx] > 10000)

#         # if (((gomi > thf0j) or (vrvj[jxx] > lth) or (htrj > htrth))) and htrj > -18:
#         #     # print(['break pt1 at ' num2str[jj]])
#         #     return None

#         if gomi > thf0j:
#             # print(['break pt2 at ' num2str[jj]])
#             return None

#         return jxx, f0vj[jxx]

#     for i, (f0vj, vrvj, htrj) in enumerate(zip(f0v, vrv, htr)):
#         res = process_frame(f0ref, f0vj, vrvj, htrj)
#         if res is None:
#             # end of voiced block
#             break
#         f0ref = res[1]
#         yield i, res[0]  # i, jxx


# def _iter_lookforward_silence(f0ref, f0v, vrv, htr, lth, thf0j, htrth):
#     # iterate forward until the end of the block or finding an unvoiced frame

#     def process_frame(
#         f0ref: float, f0vj: NDArray, vrvj: NDArray, htrj: NDArray
#     ) -> tuple[bool, int, float] | None:

#         if np.isnan(f0vj).all():
#             # no valid f0 candidate
#             return None

#         gom = np.abs((f0vj - f0ref) / f0ref)
#         jxx = np.nanargmin(gom)
#         gomi = gom[jxx]

#         if (gomi < thf0j) and (htrj < htrth):
#             return jxx, f0vj[jxx]
#         # elif vrvj[jxx] > lth:
#         else:
#             return None

#     for i, (f0vj, vrvj, htrj) in enumerate(zip(f0v, vrv, htr)):
#         res = process_frame(f0ref, f0vj, vrvj, htrj)

#         if res is None:
#             break

#         f0ref = res[1]
#         yield i, res[0]  # i, jxx


# # for jj=ii:min(mm,ii+lals)
# # 	ii=jj;
# # 	[gomi,jxx]=min(abs((f0v(:,ii)-f0ref)/f0ref));
# # 	gomi=gomi+(f0ref>10000)+(f0v(jxx,ii)>10000);
# # 	if (gomi< thf0j) && ((htr(ii)<htrth)||(f0v(jxx,ii)>=1000))
# # 		f0(ii)=f0v(jxx,ii);
# # 		irms(ii)=vrv(jxx,ii);
# # 		df(ii)=dfv(jxx,ii);
# # 		amp(ii)=aav(jxx,ii);
# # 		f0ref=f0(ii);
# # 	end;
# # 	if (gomi>thf0j) || (vrv(jxx,ii)>lth) || ((htr(ii)>htrth)&&(f0v(jxx,ii)<1000))
# # 		von = 0;f0ref=0;
# # 		break
# # 	end;
# # end;
