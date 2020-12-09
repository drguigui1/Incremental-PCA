package main

import (
	"gonum.org/v1/gonum/mat"
    "math"
    "log"
    "fmt"
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

/*
** Initialization of the Incremental PCA model
*/
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

/*
** Pretty print of the model
*/
func (ipca *IPCA) Print() {
    fmt.Println("nbSampleSeen: ", ipca.nSampleSeen_)
    fmt.Println("components: ", ipca.components_)
    fmt.Println("explained variance: ", ipca.explainedVariance_)
    fmt.Println("explained variance ratio: ", ipca.explainedVarianceRatio_)
    fmt.Println("singular values: ", ipca.singularValues_)
    fmt.Println("mean: ", ipca.mean_)
    fmt.Println("variance: ", ipca.var_)
}

/*
** Partial Fit on the batch of data
** Update parameters of the incremental PCA
*/
func (ipca *IPCA) PartialFit(data *[][]float64) {
	nSamples := len(*data)

	// compute mean and var incrementally
    colMean, colVar, nTotalSamples := IncrementalMeanAndVar(*data, ipca.mean_, ipca.var_, ipca.nSampleSeen_)

	// first path
	if ipca.nSampleSeen_ == 0 {
        // data is directly modified in place
		SubVecToMatInPlace(data, colMean, 0)
	} else {
		// compute col_batch_mean
        colBatchMean := MeanMat(*data, 0)
        // data is directly modified in place
		SubVecToMatInPlace(data, colBatchMean, 0)

		// mean correction
        sqrtTmp := math.Sqrt((float64(ipca.nSampleSeen_) / float64(nTotalSamples)) * float64(nSamples))
        //fmt.Println(Sub2Slices(ipca.mean_, colBatchMean))
        meanCorrectionTmp := MultSliceByConst(Sub2Slices(ipca.mean_, colBatchMean), sqrtTmp)
        //fmt.Println(meanCorrectionTmp)

		// vstack
        multCompSingValues := MultMatByVec(ipca.components_, ipca.singularValues_, 1)
        meanCorrection := [][]float64{ meanCorrectionTmp }
        data = Vstack(*multCompSingValues, *data, meanCorrection)
	}

    // convert data into Dense
    dataDense := FromSliceFloatToMat(data)

    // SVD
    var svd mat.SVD
	ok := svd.Factorize(dataDense, mat.SVDThin)

	if !ok {
    	log.Fatal("SVD Decomposition failed")
	}
	var u, v mat.Dense
	svd.UTo(&u)
    svd.VTo(&v)
    vt := mat.DenseCopyOf(v.T())
    values := svd.Values(nil)
    SvdFlip(&u, vt)

    // explained variance / explained variance ratio
    squareValues := Mult2Slices(values, values)
    explainedVariance := DivSliceByConst(squareValues, float64(nTotalSamples - 1))
    explainedVarianceRatio := DivSliceByConst(squareValues, MultVecAndSum(colVar, float64(nTotalSamples)))

    // set vars
    ipca.nSampleSeen_ = nTotalSamples
    ipca.components_ = *ExtractComponents(vt, ipca.nComponents_)
    //ipca.components_ = Vt[:ipca.nComponents_]
    ipca.singularValues_ = values[:ipca.nComponents_]
    ipca.mean_ = colMean
    ipca.var_ = colVar
    ipca.explainedVariance_ = explainedVariance[:ipca.nComponents_]
    ipca.explainedVarianceRatio_ = explainedVarianceRatio[:ipca.nComponents_]
}

func (ipca *IPCA) Transform(data *[][]float64) *[][]float64 {
    // substract data by the mean
    SubVecToMatInPlace(data, ipca.mean_, 0)

    // dot product between data and ipca.components_.T
    return DotProduct(data, MatTranspose(&ipca.components_))
}