import unittest
import numpy as np
from brno_heat_history import period_pair


class HistoricalPairTests(unittest.TestCase):
    def test_difference_is_paired_not_difference_of_spatial_medians(self):
        early=np.array([[0.,0.,100.]])
        recent=np.array([[0.,100.,100.]])
        values=np.concatenate([np.repeat(early[None],5,axis=0),np.repeat(recent[None],5,axis=0)])
        a,b,d=period_pair(values,np.full_like(values,2),np.ones((1,3),bool))
        np.testing.assert_array_equal(d,[[0.,100.,0.]])
        self.assertEqual(float(np.median(d)),0.)
        self.assertEqual(float(np.median(b)-np.median(a)),100.)

    def test_both_periods_use_identical_observed_support(self):
        values=np.full((10,1,3),30.);values[5:]+=2
        counts=np.full_like(values,2)
        # Cell 1 lacks three recent years, despite many observations in two years.
        counts[5:8,0,1]=0;counts[8:,0,1]=10
        # Counts cannot rescue missing retrieval temperatures in cell 2.
        values[:3,0,2]=np.nan
        a,b,d=period_pair(values,counts,np.ones((1,3),bool))
        np.testing.assert_array_equal(np.isfinite(a),[[True,False,False]])
        np.testing.assert_array_equal(np.isfinite(a),np.isfinite(b))
        np.testing.assert_array_equal(np.isfinite(a),np.isfinite(d))
        self.assertEqual(d[0,0],2.)


if __name__=='__main__':unittest.main()
