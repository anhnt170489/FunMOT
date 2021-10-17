# -*- coding: utf-8 -*-
import os

import argparse

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import numpy as np
import pandas as pd

import json


class COCOeval2(COCOeval):
    def __init__(self, cocoGt=None, cocoDt=None, iouType='bbox'):
        super().__init__(cocoGt=cocoGt, cocoDt=cocoDt, iouType=iouType)

    def _load_classes(self):
        # return [
        #     "no_mask",
        #     "mask",
        #     "All"
        # ]

        return [
            "hs",
            "All"
        ]

    def summarize(self):
        '''
        Compute and display summary metrics for evaluation results.
        Note this functin can *only* be applied on the default parameter setting
        '''

        def _summarize(ap=1, iouThr=None, areaRng='all', maxDets=100):
            p = self.params
            iStr = ' {:<18} {} @[ IoU={:<9} | area={:>6s} | maxDets={:>3d} ] = {:0.3f}'
            titleStr = 'Average Precision' if ap == 1 else 'Average Recall'
            typeStr = '(AP)' if ap == 1 else '(AR)'
            iouStr = '{:0.2f}:{:0.2f}'.format(p.iouThrs[0], p.iouThrs[-1]) \
                if iouThr is None else '{:0.2f}'.format(iouThr)

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]
            if ap == 1:
                # dimension of precision: [TxRxKxAxM]
                s = self.eval['precision']
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, :, aind, mind]
            else:
                # dimension of recall: [TxKxAxM]
                s = self.eval['recall']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, aind, mind]
            if len(s[s > -1]) == 0:
                mean_s = -1
            else:
                # mean_s = np.mean(s[s>-1])
                mean_s = []

                # caculate AP(average precision) for each category
                num_classes = len(self.params.catIds)
                avg_ap = 0.0
                avg_ar = 0.0

                if ap == 1:
                    for i in range(0, num_classes):
                        # print('category : {0} : {1}'.format(i, np.mean(s[:,:,i,:])))
                        # avg_ap += np.mean(s[:, :, i, :])
                        mean_s.append(np.mean(s[:, :, i, :]))
                    # print('(all categories) mAP : {}'.format(avg_ap / num_classes))
                else:
                    # print(s.shape)
                    for i in range(0, num_classes):
                        # print('category : {0} : {1}'.format(i, np.mean(s[:,i,:])))
                        # avg_ar += np.mean(s[:, i, :])
                        mean_s.append(np.mean(s[:, i, :]))
                    # print('(all categories) mAR : {}'.format(avg_ar / num_classes))

            # print(iStr.format(titleStr, typeStr, iouStr, areaRng, maxDets, mean_s))
            return mean_s

        def _summarizeDets():
            stats = np.zeros((8, 2))
            stats[0] = list(range(2))
            stats[1] = np.full((2,), len(self.params.imgIds))
            stats[2] = _summarize(1)
            stats[3] = _summarize(1, iouThr=.5, maxDets=self.params.maxDets[2])
            # stats[2] = _summarize(1, iouThr=.75, maxDets=self.params.maxDets[2])
            stats[4] = _summarize(1, iouThr=.5, areaRng='small', maxDets=self.params.maxDets[2])
            stats[5] = _summarize(1, iouThr=.5, areaRng='medium', maxDets=self.params.maxDets[2])
            stats[6] = _summarize(1, iouThr=.5, areaRng='large', maxDets=self.params.maxDets[2])
            # stats[6] = _summarize(0, maxDets=self.params.maxDets[0])
            # stats[7] = _summarize(0, maxDets=self.params.maxDets[1])
            stats[7] = _summarize(0, iouThr=.5, maxDets=self.params.maxDets[2])
            # stats[9] = _summarize(0, areaRng='small', maxDets=self.params.maxDets[2])
            # stats[10] = _summarize(0, areaRng='medium', maxDets=self.params.maxDets[2])
            # stats[11] = _summarize(0, areaRng='large', maxDets=self.params.maxDets[2])
            return stats

        def _summarizeKps():
            stats = np.zeros((10,))
            stats[0] = _summarize(1, maxDets=20)
            stats[1] = _summarize(1, maxDets=20, iouThr=.5)
            stats[2] = _summarize(1, maxDets=20, iouThr=.75)
            stats[3] = _summarize(1, maxDets=20, areaRng='medium')
            stats[4] = _summarize(1, maxDets=20, areaRng='large')
            stats[5] = _summarize(0, maxDets=20)
            stats[6] = _summarize(0, maxDets=20, iouThr=.5)
            stats[7] = _summarize(0, maxDets=20, iouThr=.75)
            stats[8] = _summarize(0, maxDets=20, areaRng='medium')
            stats[9] = _summarize(0, maxDets=20, areaRng='large')
            return stats

        if not self.eval:
            raise Exception('Please run accumulate() first')
        iouType = self.params.iouType
        if iouType == 'segm' or iouType == 'bbox':
            summarize = _summarizeDets
        elif iouType == 'keypoints':
            summarize = _summarizeKps
        self.stats = summarize()
        print(self.params)

        self.stats = np.array(self.stats)
        self.stats = np.transpose(self.stats)
        self.stats = np.concatenate((self.stats, np.expand_dims(np.mean(self.stats, axis=0), axis=0)), axis=0)
        pd_stat = pd.DataFrame(self.stats)
        pd_stat.columns = ["classes", "images", "mAP@.5:@.95", "mAP@.5", "mAP@.5(small)", "mAP@.5(medium)",
                           "mAP@.5(large)", "mAR@.5:.95"]
        print(self.stats)
        # print(pd_stat["classes"])
        pd_stat["classes"] = pd_stat["classes"].apply(lambda x: (self._load_classes()[int(x)] if x != 0.5 else "All"))
        # print(pd_stat)
        return pd_stat['mAP@.5'].values[-1]
