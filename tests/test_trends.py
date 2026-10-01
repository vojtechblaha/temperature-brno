import unittest
import numpy as np
import brno_heat_trends as t

class TrendEstimatorTests(unittest.TestCase):
    def test_missing_years_do_not_create_composition_trend(self):
        years=np.arange(2001,2026)
        values=np.broadcast_to(np.array([10.,35.]),(25,2)).copy()
        values[[0,2,5,7],1]=np.nan;values[[17,19,22,24],0]=np.nan
        np.testing.assert_allclose(t.local_slopes(years,values),0,atol=1e-12)
        self.assertTrue(np.all(t.eligible_years(values)))

    def test_recovers_known_slopes_without_filling_missing_years(self):
        years=np.arange(2001,2026)
        values=np.array([10.,30.])[None,:]+(years-2001)[:,None]*np.array([.1,.2])[None,:]
        values[[0,3,10,13],0]=np.nan
        values[[2,8,15,22],1]=np.nan
        np.testing.assert_allclose(t.local_slopes(years,values),[1,2],atol=1e-12)
        self.assertAlmostEqual(t.weighted_mean(t.local_slopes(years,values),np.array([3.,1.])),1.25)
        self.assertEqual(np.isnan(values).sum(),8)

    def test_support_requires_early_and_late_observations(self):
        values=np.ones((25,3))
        values[:5,0]=np.nan
        values[[0,3,10,13],1]=np.nan
        values[10:16,2]=np.nan
        np.testing.assert_array_equal(t.eligible_years(values),[0,1,0])

    def test_spatial_duplication_does_not_shrink_uncertainty(self):
        years=np.arange(2001,2026)
        y=(years-2001)*.1+np.sin(np.arange(25))
        values=y[:,None];weights=np.ones(1)
        one=t.bootstrap_trends(years,values,weights,replicates=300)
        repeated=t.bootstrap_trends(years,np.repeat(values,20,axis=1),np.ones(20),replicates=300)
        np.testing.assert_allclose(one['CI95_C_decade'],repeated['CI95_C_decade'],atol=1e-10)
        hac1=t.joint_hac_interval(years,values,weights)
        hac2=t.joint_hac_interval(years,np.repeat(values,20,axis=1),np.ones(20))
        np.testing.assert_allclose(hac1['CI95_C_decade'],hac2['CI95_C_decade'],atol=1e-10)

    def test_joint_hac_matches_independent_single_series_calculation(self):
        import brno_heat_landsat as b
        years=np.arange(2001,2026)
        values=.1*(years-2001)+np.sin(np.arange(25))
        ref=b.trend(years,values)['hac_lag2_95CI']
        got=t.joint_hac_interval(years,values[:,None],np.ones(1))['CI95_C_decade']
        np.testing.assert_allclose(got,ref,atol=1e-10)

if __name__=='__main__':unittest.main()
