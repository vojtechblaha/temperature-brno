import unittest
import warnings
import tempfile
from pathlib import Path
import numpy as np
import rasterio
from rasterio.transform import from_origin
import brno_heat_landsat as b

class ScientificRegressionTests(unittest.TestCase):
    def test_quality_flags_that_previously_erased_nights(self):
        q=np.array([0,17,65,81,145,2,3,4,49,np.nan])
        np.testing.assert_array_equal(b.qa_good(q),[1,1,1,1,0,0,0,0,0,0])
        self.assertFalse(b.qa_good(np.array([65]),0)[0])

    def test_months_equal_weight_and_missing_month_not_imputed(self):
        months=np.array([6]*5+[7]*20+[8]*5)
        v=np.array([10.]*5+[30.]*20+[20.]*5)[:,None]
        self.assertEqual(b.monthly_summer(v,months)[0],20.)
        v[-5:]=np.nan
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.assertTrue(np.isnan(b.monthly_summer(v,months)[0]))

    def test_recover_only_quality_zero_not_missing_temperature(self):
        q=np.array([np.nan,np.nan,np.nan,65.])
        raw=np.array([15000.,0.,np.nan,15000.])
        recovered=b.recover_qa(q,raw)
        self.assertEqual(recovered[0],0.)
        self.assertTrue(np.isnan(recovered[1]))
        self.assertTrue(np.isnan(recovered[2]))
        self.assertEqual(recovered[3],65.)
        np.testing.assert_array_equal(b.qa_good(recovered),[1,0,0,1])

    def test_native_grid_contains_brno(self):
        cached=b.CACHE/'modis_daily_2025.tif'
        if not cached.exists(): self.skipTest('Downloaded integration fixture absent')
        _,profile=b.read(cached)
        city=b.boundary().to_crs(profile['crs']).geometry.union_all()
        weights=b.area_weights(profile)
        self.assertAlmostEqual(weights.sum()/city.area,1.,places=5)

    def test_weighted_spatial_summary(self):
        self.assertEqual(b.weighted_median(np.array([1.,2.,99.]),np.array([1.,3.,0.])),2.)

    def test_trend_units(self):
        x=np.arange(2001,2026)
        y=.1*(x-2001)+np.sin(np.arange(25))*.001
        t=b.trend(x,y)
        self.assertAlmostEqual(t['ols_C_decade'],1.,places=2)
        self.assertTrue(t['ols_95CI'][0]<1.<t['ols_95CI'][1])

    def test_zero_is_valid_and_nodata_is_declared(self):
        with tempfile.TemporaryDirectory() as folder:
            original=b.RASTER
            try:
                b.RASTER=Path(folder)
                profile=dict(driver='GTiff',width=2,height=1,count=1,dtype='float32',
                             crs='EPSG:32633',transform=from_origin(600000,5400000,30,30))
                b.raster_write('test.tif',np.array([[0.,np.nan]]),profile)
                a,p=b.read(Path(folder)/'test.tif')
                self.assertEqual(a[0,0,0],0.)
                self.assertTrue(np.isnan(a[0,0,1]))
                self.assertEqual(p['nodata'],-9999.)
            finally: b.RASTER=original

    def test_change_classes_preserve_missing_and_sensitive_cases(self):
        inside=np.array([True]*7+[False])
        strict=np.array([np.nan,np.nan,0.,3.,-4.,3.,-3.,4.])
        broad=np.array([np.nan,3.,0.,2.,-2.,-3.,-.5,4.])
        classes=b.change_classes(strict,broad,inside)
        np.testing.assert_array_equal(classes[:7],[0,1,2,4,3,2,2])
        self.assertTrue(np.isnan(classes[-1]))
        self.assertTrue(np.isfinite(classes[inside]).all())

    def test_comparison_requires_both_periods_and_distributed_years(self):
        counts=np.array([[5,1,1,1],[0,1,1,1],[0,3,3,1],[0,0,0,1],[0,0,0,1],
                         [5,1,1,1],[0,1,1,1],[0,3,2,1],[0,0,0,1],[0,0,0,1]])
        np.testing.assert_array_equal(b.comparison_mask(counts,np.ones(4,dtype=bool)),[0,1,0,1])

    def test_published_evidence_map_classifies_entire_city(self):
        path=b.RASTER/'07_landsat_change_evidence_class.tif'
        if not path.exists(): self.skipTest('Downloaded integration fixture absent')
        a,p=b.read(path);inside=b.inside_mask(p)
        self.assertTrue(np.isfinite(a[0,inside]).all())
        self.assertTrue(np.isnan(a[0,~inside]).all())
        self.assertEqual(set(np.unique(a[0,inside])),{0.,1.,2.,3.,4.})
        strict=b.read(b.RASTER/'03_landsat_relative_heat_change_C.tif')[0][0]
        broad=b.read(b.RASTER/'06_landsat_broad_relative_change_C.tif')[0][0]
        self.assertTrue(np.all(np.isfinite(broad[np.isfinite(strict)])))
        self.assertTrue(np.all(strict[a[0]==4]>=2))
        self.assertTrue(np.all(broad[a[0]==4]>=2))
        self.assertTrue(np.all(strict[a[0]==3]<=-2))
        self.assertTrue(np.all(broad[a[0]==3]<=-2))

if __name__=='__main__': unittest.main()

