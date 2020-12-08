package main

import (
	//"gonum.org/v1/gonum/mat"
	"math"
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
	nFeatures_ int
	nComponents_ int
	nSampleSeen_ int
	mean_ []float64
	var_ []float64
	components_ [][]float64
	singularValues_ []float64
	explainedVariance_ []float64
	explainedVarianceRatio_ []float64

}

func InitIncrementalPCA(nComponents, nFeatures int) *IPCA {
	// set mean and var to slice of 0
	mean := make([]float64, nFeatures)
	variance := make([]float64, nFeatures)
	components := InitSlicesFloat64(nComponents, nFeatures)
	singularValues := make([]float64, nComponents)
	explainedVariance := make([]float64, nComponents)
	explainedVarianceRatio := make([]float64, nComponents)

	newIPCA := IPCA{nFeatures, nComponents, 0,
					mean, variance, *components,
					singularValues, explainedVariance,
					explainedVarianceRatio}
	return &newIPCA
}

func (ipca *IPCA) PartialFit(data *[][]float64) {
	nSamples := len(*data)

	// compute mean and var incrementally
	colMean, colVar, nTotalSamples := IncrementalMeanAndVar(*data, ipca.mean_, ipca.var_, ipca.nSampleSeen_)

	// first path
	if ipca.nSampleSeen_ == 0 {
		SubVecToMatInPlace(data, colMean, 0)
	} else {
		// compute col_batch_mean
		colBatchMean := MeanMat(*data, 0)
		SubVecToMatInPlace(data, colBatchMean, 0)

		// mean correction
		sqrtTmp := math.Sqrt((float64(ipca.nSampleSeen_) / float64(nTotalSamples)) * float64(nSamples))
		meanCorrection := MultSliceByConst(Sub2Slices(ipca.mean_, colBatchMean), sqrtTmp)

		// vstack
	}

	// TODO
	// SVD
	// explained variance / explained variance ratio
	// set vars
}

func (ipca *IPCA) Transform(data *[][]float64) {
    // TODO
}