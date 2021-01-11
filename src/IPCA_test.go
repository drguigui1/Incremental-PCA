package main

import (
    "testing"
    "reflect"
)

// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
// Ref taken from sklearn IncrementalPCA results
// !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

func CompareResRef(t *testing.T, testName string, ref, res interface{}) {
    if !reflect.DeepEqual(ref, res) {
        t.Errorf("Error test1: %v", testName)
        t.Errorf("res: %v", res)
        t.Errorf("ref: %v", ref)
    }
}

func TestIPCA1(t *testing.T) {
    batch1 := [][]float64{
        {2.3, 4.5, 6.6, 1.0, 5.5},
        {5.5, 1.2, 0.54, 0.1, 1.1},
        {3.3, 7.4, 9.9, 0.12, 7.7},
    }

    ipca := InitIncrementalPCA(2, 5)
    ipca.PartialFit(&batch1)

    refnSampleSeen := 3
    refComponents := [][]float64{
        {-0.19326570526249473,0.458197383492821,0.7080308813692526,0.014925350719010435,0.5011716570827975},
        {0.8325576624820022,0.3583885541574162,0.03112896296236444,-0.41950679019804216,-0.03808442639826538},
    }
    refExplainedVariance := []float64{44.954026178180825,1.4439738218191682}
    refExplainedVarianceRatio := []float64{0.9688785330872196,0.031121466912780033}
    refSingularValues := []float64{9.481985675815043,1.6993962585690061}
    refMean := []float64{3.6999999999999997,4.366666666666667,5.68,0.40666666666666673,4.766666666666667}
    refVar := []float64{1.786666666666667,6.415555555555557,15.0248,0.17608888888888893,7.528888888888889}

    CompareResRef(t, "nSampleSeen", refnSampleSeen, ipca.nSampleSeen_)
    CompareResRef(t, "components", refComponents, ipca.components_)
    CompareResRef(t, "explainedVariance", refExplainedVariance, ipca.explainedVariance_)
    CompareResRef(t, "explainedVarianceRatio", refExplainedVarianceRatio, ipca.explainedVarianceRatio_)
    CompareResRef(t, "singularValues", refSingularValues, ipca.singularValues_)
    CompareResRef(t, "mean", refMean, ipca.mean_)
    CompareResRef(t, "variance", refVar, ipca.var_)
}

func TestIPCA2(t *testing.T) {
    batch1 := [][]float64{
        {0.3, 0.5, 0.6, 0},
        {0.5, 0.2, 0.54, 0.1},
    }
    batch2 := [][]float64{
        {0,0,0,1},
        {1,0,0,1},
    }

    ipca := InitIncrementalPCA(2, 4)
    ipca.PartialFit(&batch1)
    ipca.PartialFit(&batch2)

    refnSampleSeen := 4
    refComponents := [][]float64{
        {0.15672482247480857,-0.317704704062687,-0.48112947564375946,0.8018824594062655},
        {0.9865579377194105,0.0060025512336807085,0.08401796798612515,-0.1400299466435418},
    }
    refExplainedVariance := []float64{0.46413361847219964,0.16977859605555667}
    refExplainedVarianceRatio := []float64{0.7208163045072211,0.26367230323894497}
    refSingularValues := []float64{1.180000362464605,0.7136776500400374}
    refMean := []float64{0.45,0.175,0.28500000000000003,0.525}
    refVar := []float64{0.1325,0.041874999999999996,0.08167500000000003,0.226875}

    CompareResRef(t, "nSampleSeen", refnSampleSeen, ipca.nSampleSeen_)
    CompareResRef(t, "components", refComponents, ipca.components_)
    CompareResRef(t, "explainedVariance", refExplainedVariance, ipca.explainedVariance_)
    CompareResRef(t, "explainedVarianceRatio", refExplainedVarianceRatio, ipca.explainedVarianceRatio_)
    CompareResRef(t, "singularValues", refSingularValues, ipca.singularValues_)
    CompareResRef(t, "mean", refMean, ipca.mean_)
    CompareResRef(t, "variance", refVar, ipca.var_)
}

func TestIPCA3(t *testing.T) {
    batch1 := [][]float64{
        {2.3, 4.5, 6.6, 1.0, 5.5},
        {5.5, 1.2, 0.54, 0.1, 1.1},
        {3.3, 7.4, 9.9, 0.12, 7.7},
    }

    batch2 := [][]float64{
        {23.3, 0.5, 1.6, 1.4, 8.5},
        {1.5, 2.2, 0.54, 0.1, 0},
        {0, 7.4, 0, 0, 0},
    }

    batch3 := [][]float64{
        {0,0,0,0,0},
        {1,0,0,0,1},
    }

    batch4 := [][]float64{
        {0,0,0.01,0,0},
        {0,0,0,0.1,0},
    }

    data := [][]float64{
        {1,2,3,4,5},
    }

    ref := [][]float64{
        {-1.186098879912303, 2.3696648273979384},
    }

    ipca := InitIncrementalPCA(2, 5)
    ipca.PartialFit(&batch1)
    ipca.PartialFit(&batch2)
    ipca.PartialFit(&batch3)
    ipca.PartialFit(&batch4)
    res := ipca.Transform(&data)

    CompareResRef(t, "", ref, *res)
}

func TestIPCA4(t *testing.T) {
    batch1 := [][]float64{
        {2.3, 4.5, 6.6, 1.0, 5.5},
        {5.5, 1.2, 0.54, 0.1, 1.1},
        {3.3, 7.4, 9.9, 0.12, 7.7},
    }

    batch2 := [][]float64{
        {23.3, 0.5, 1.6, 1.4, 8.5},
        {1.5, 2.2, 0.54, 0.1, 0},
        {0, 7.4, 0, 0, 0},
    }

    batch3 := [][]float64{
        {0,0,0,0,0},
        {1,0,0,0,1},
    }

    batch4 := [][]float64{
        {0,0,0.01,0,0},
        {0,0,0,0.1,0},
    }

    data := [][]float64{
        {2.3, 4.5, 6.6, 1.0, 5.5},
    }

    ref := [][]float64{
        {0.37024882576964135,6.0769938122759095},
    }

    ipca := InitIncrementalPCA(2, 5)
    ipca.PartialFit(&batch1)
    ipca.PartialFit(&batch2)
    ipca.PartialFit(&batch3)
    ipca.PartialFit(&batch4)
    res := ipca.Transform(&data)

    CompareResRef(t, "", ref, *res)
}