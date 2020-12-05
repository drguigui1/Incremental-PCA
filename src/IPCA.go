package main

import (
	"gonum.org/v1/gonum/mat"
)

/*
** Incremental PCA structure
**
** 'nFeatures_': number of features of the dataset (number of dimensions)
**
** 'nComponents_': number of components of the PCA (final dimension we want to have)
**
** 'nSampleSeen_': The number of samples processed by the estimator
**
** 'mean_': Per-feature empirical mean, aggregate over calls to ``partial_fit``
**  shape=(nFeatures,)
**
** 'var_': Per-feature empirical variance, aggregate over calls to ``partial_fit``.
**
** 'components_': Components with maximum variance.
** 	shape=(nComponents_, nFeatures_)
**
*/
type IPCA struct {
	nFeatures_ uint
	nComponents_ uint
	nSampleSeen_ uint
	mean_ []float64
	var_ []float64
	components_ mat.Dense
	singularValues_ []float64
	explainedVariance_ []float64
	explainedVarianceRatio_ []float64

}

func InitIncrementalPCA(nComponents int) *IPCA {
	// TODO
	return nil
}

func (ipca *IPCA) PartialFit(data *[][]float64) {
    // TODO
}

func (ipca *IPCA) Transform(data *[][]float64) {
    // TODO
}