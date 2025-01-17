package main

import (
    "io/ioutil"
    "encoding/csv"
    "os"
    "strings"
    "strconv"
    "gonum.org/v1/gonum/mat"
    "math"
    "log"
)

/*
** Read csv
**
** param:
** 'csvPath': path of the csv
**
** return:
** Data structure containing the csv data
*/
func FromCSV(csvPath string) [][]string {
    // read the file
    data, err := ioutil.ReadFile(csvPath)
    if err != nil {
        panic(err)
    }

    r := csv.NewReader(strings.NewReader(string(data)))
    records, err := r.ReadAll()
    if err != nil {
        panic(err)
    }

    return records
}

/*
** Save the data into csv
**
** params:
** 'data': data structure containing the data to save
** 'csv_path': path to the csv
*/
func ToCSV(data [][]string, csv_path string) {
    // create a new file
    file, err := os.Create(csv_path)
    if err != nil {
        panic(err)
    }

    // create a new writer
    w := csv.NewWriter(file)

    // write data
    w.WriteAll(data)
    file.Close()
}

/*
** Init Slice of slice
*/
func InitSlicesFloat32(dim1, dim2 int) *[][]float32 {
    newSlices := make([][]float32, dim1)
    for i := 0; i < dim1; i++ {
        newSlices[i] = make([]float32, dim2)
    }
    return &newSlices
}

/*
** Init Slice of slice
*/
func InitSliceFloat64(dim1, dim2 int) *[]float64 {
    newSlice := make([]float64, dim1 * dim2)
    return &newSlice
}

/*
** Init Slice of slice
*/
func InitSlicesFloat64(dim1, dim2 int) *[][]float64 {
    newSlices := make([][]float64, dim1)
    for i := 0; i < dim1; i++ {
        newSlices[i] = make([]float64, dim2)
    }
    return &newSlices
}

/*
** From string slices to float32
*/
func ToFloat32Slices(slice *[][]string) *[][]float32 {
    n_samples := len(*slice)
    dim := len((*slice)[0])
    res := InitSlicesFloat32(n_samples, dim)

    for i := 0; i < n_samples; i++ {
        for j := 0; j < dim; j++ {
            tmp, err := strconv.ParseFloat((*slice)[i][j], 32)

            if err != nil {
                panic(err)
            }

            (*res)[i][j] = float32(tmp)
        }
    }
    return res
}

/*
** From string slices to float64
*/
func ToFloat64Slices(slice *[][]string) *[][]float64 {
    n_samples := len(*slice)
    dim := len((*slice)[0])
    res := InitSlicesFloat64(n_samples, dim)

    for i := 0; i < n_samples; i++ {
        for j := 0; j < dim; j++ {
            tmp, err := strconv.ParseFloat((*slice)[i][j], 64)

            if err != nil {
                panic(err)
            }

            (*res)[i][j] = float64(tmp)
        }
    }
    return res
}

/*
** NOT WORKING
*/
func FromSliceToMat(slice *[][]string) *mat.Dense {
    n_samples := len((*slice))
    n_dims := len((*slice)[0])

    // flatten and convert the data
    data := InitSliceFloat64(n_samples, n_dims)

    for i := 0; i < n_samples; i++ {
        for j := 0; j < n_dims; j++ {
            tmp, err := strconv.ParseFloat((*slice)[i][j], 64)

            if err != nil {
                panic(err)
            }

            (*data)[i * n_dims + j] = tmp
        }
    }

    return mat.NewDense(n_samples, n_dims, *data)
}

/*
**
*/
func FromSliceFloatToMat(slice *[][]float64) *mat.Dense {
    n_samples := len((*slice))
    n_dims := len((*slice)[0])

    // flatten and convert the data
    data := make([]float64, n_samples * n_dims)

    for i := 0; i < n_samples; i++ {
        for j := 0; j < n_dims; j++ {
            data[i * n_dims + j] = (*slice)[i][j]
        }
    }

    return mat.NewDense(n_samples, n_dims, data)
}

/*
** Get the absolute maximum over the specified axis
**
** params:
** 'mat': dense matrix
** 'axis': specific axis (can be 0 / 1)
**
** return
** The absolute maximum vector over the specified axis
*/
func GetAbsoluteMax(m *mat.Dense, axis int) []float64 {
    nSamples, nFeatures := m.Dims()

    var result []float64
    var currMax []*float64
    var idx int
    if axis == 1 {
        result = make([]float64, nSamples)
        currMax = make([]*float64, nSamples)
    } else {
        result = make([]float64, nFeatures)
        currMax = make([]*float64, nFeatures)
    }

    for i := 0; i < nSamples; i++ {
        for j := 0; j < nFeatures; j++ {
            valueAbs := math.Abs(m.At(i, j))
            value := m.At(i, j)

            if axis == 1 {
                idx = i
            } else {
                idx = j
            }

            // not already initialized
            if currMax[idx] == nil {
                currMax[idx] = &valueAbs
                result[idx] = value
            } else {
                if valueAbs > *currMax[idx] {
                    currMax[idx] = &valueAbs
                    result[idx] = value
                }
            }
        }
    }
    return result
}

/*
** Return the sign of each element
** -1 if negativ
**  1 if positiv
**  0 else
*/
func SignSlice(s []float64) []float64 {
    n := len(s)
    res := make([]float64, n)

    for idx, elm := range s {
        if elm > 0 {
            res[idx] = 1
        } else if elm < 0 {
            res[idx] = -1
        } else {
            res[idx] = 0
        }
    }
    return res
}

/*
** Multiply 'm' by the vector 'v' along specific axis in place
**
** ex (axis = 1):
** [               [           [
**  1, 2, 3,   *     2,   =      2, 4, 6,
**  4, 5, 6          3           12, 15, 18
**  ]                ]          ]
**
** ex (axis = 0):
** [                              [
**  1, 2, 3,   *   [ 1, 2, 3 ] =    1, 4, 9,
**  4, 5, 6                         4, 10, 18
**  ]                              ]
**
** Operations are done in place
*/
func MultiplyVec(m *mat.Dense, v []float64, axis int) {
    nSamples, nFeatures := m.Dims()
    var idx int

    for i := 0; i < nSamples; i++ {
        for j := 0; j < nFeatures; j++ {
            if axis == 1 {
                idx = i
            } else {
                idx = j
            }

            m.Set(i, j, m.At(i, j) * v[idx])
        }
    }
}

/*
** Multiply 'm' by the vector 'v' along specific axis in place
**
** ex (axis = 1):
** [               [           [
**  1, 2, 3,   *     2,   =      2, 4, 6,
**  4, 5, 6          3           12, 15, 18
**  ]                ]          ]
**
** ex (axis = 0):
** [                              [
**  1, 2, 3,   *   [ 1, 2, 3 ] =    1, 4, 9,
**  4, 5, 6                         4, 10, 18
**  ]                              ]
**
** Operations are done in place
*/
func MultMatByVec(m [][]float64, v []float64, axis int) *[][]float64 {
    nSamples, nFeatures := len(m), len(m[0])
    res := InitSlicesFloat64(nSamples, nFeatures)
    var idx int

    for i := 0; i < nSamples; i++ {
        for j := 0; j < nFeatures; j++ {
            if axis == 1 {
                idx = i
            } else {
                idx = j
            }

            (*res)[i][j] = m[i][j] * v[idx]
        }
    }

    return res
}

/*
** Multiply slice by const flaot64
*/
func MultSliceByConst(slice []float64, val float64) []float64 {
    m := len(slice)
    res := make([]float64, m)

    for i := 0; i < m; i++ {
        res[i] = slice[i] * val
    }

    return res
}

/*
** Sum two slice
*/
func Sum2Slices(slice1 []float64, slice2 []float64) []float64 {
    m := len(slice1)
    res := make([]float64, m)

    for i := 0; i < m; i++ {
        res[i] = slice1[i] + slice2[i]
    }

    return res
}

/*
** Mult two slice
*/
func Mult2Slices(slice1 []float64, slice2 []float64) []float64 {
    m := len(slice1)
    res := make([]float64, m)

    for i := 0; i < m; i++ {
        res[i] = slice1[i] * slice2[i]
    }

    return res
}

/*
** Sub two slice
*/
func Sub2Slices(slice1 []float64, slice2 []float64) []float64 {
    m := len(slice1)
    res := make([]float64, m)

    for i := 0; i < m; i++ {
        res[i] = slice1[i] - slice2[i]
    }

    return res
}

/*
** Add a constant value to a slice
*/
func AddConstToSlice(slice []float64, val float64) []float64 {
    m := len(slice)
    res := make([]float64, m)

    for i := 0; i < m; i++ {
        res[i] = slice[i] + val
    }

    return res
}

/*
** Divide slice by a constant
*/
func DivSliceByConst(slice []float64, val float64) []float64 {
    if val == 0. {
        log.Fatalf("Error cannot divide by 0")
    }

    m := len(slice)
    res := make([]float64, m)

    for i := 0; i < m; i++ {
        res[i] = slice[i] / val
    }

    return res
}

/*
** Sum elements of the dense matrix over a specific axis
** axis = 0 -> sum for each column (len(result) == n_features)
** axis = 1 -> sum for each rows
*/
func SumMat(m [][]float64, axis int) []float64 {
    nSamples, nFeatures := len(m), len(m[0])

    var res []float64
    var idx int
    if axis == 1 {
        res = make([]float64, nSamples)
    } else {
        res = make([]float64, nFeatures)
    }

    for i := 0; i < nSamples; i++ {
        for j := 0; j < nFeatures; j++ {
            if axis == 1 {
                idx = i
            } else {
                idx = j
            }

            res[idx] += m[i][j]
        }
    }
    return res
}

/*
** Compute mean over specific axis
** axis = 1 -> compute variance for each rows (len(res) = nSamples)
** axis = 0 -> compute variance for each columns (len(res) = nFeatures)
*/
func MeanMat(m [][]float64, axis int) []float64 {
    nSamples, nFeatures := len(m), len(m[0])
    var mean []float64
    var idx int
    var idxOther int
    var cpt int = 1

    if axis == 0 {
        mean = make([]float64, nFeatures)
    } else {
        mean = make([]float64, nSamples)
    }

    for i := 0; i < nSamples; i++ {
        if axis == 1 {
            cpt = 1
        }
        for j := 0; j < nFeatures; j++ {
            elm := m[i][j]

            if axis == 0 {
                idx = j
                idxOther = i
            } else {
                idx = i
                idxOther = j
            }

            if idxOther == 0 {
                // First elm case
                mean[idx] = elm
            } else {
                // Incrementaly update mean
                newMean := mean[idx] + (elm - mean[idx]) / float64(cpt)
                mean[idx] = newMean
            }
            if axis == 1 {
                cpt++
            }
        }
        if axis == 0 {
            cpt++
        }
    }
    return mean

}

/*
** Compute variance over specific axis
** axis = 1 -> compute variance for each rows (len(res) = nSamples)
** axis = 0 -> compute variance for each columns (len(res) = nFeatures)
*/
func VarMat(m [][]float64, axis int) []float64 {
    nSamples, nFeatures := len(m), len(m[0])

    var res []float64
    var mean []float64
    var sumSquareDiff []float64
    var cpt int = 1
    var idx int
    var idxOther int

    if axis == 0 {
        res = make([]float64, nFeatures)
        mean = make([]float64, nFeatures)
        sumSquareDiff = make([]float64, nFeatures)
    } else {
        res = make([]float64, nSamples)
        mean = make([]float64, nSamples)
        sumSquareDiff = make([]float64, nSamples)
    }

    for i := 0; i < nSamples; i++ {
        if axis == 1 {
            cpt = 1
        }
        for j := 0; j < nFeatures; j++ {
            elm := m[i][j]

            if axis == 0 {
                idx = j
                idxOther = i
            } else {
                idx = i
                idxOther = j
            }

            if idxOther == 0 {
                // First elm case
                mean[idx] = elm
                sumSquareDiff[idx] = 0.
            } else {
                // Incrementaly update mean
                newMean := mean[idx] + (elm - mean[idx]) / float64(cpt)

                // Incrementaly update sum square diff
                newSumSquare := sumSquareDiff[idx] + (elm - mean[idx]) * (elm - newMean)

                mean[idx] = newMean
                sumSquareDiff[idx] = newSumSquare

                res[idx] = newSumSquare / float64(cpt)
            }
            if axis == 1 {
                cpt++
            }
        }
        if axis == 0 {
            cpt++
        }
    }
    return res
}

/*
** Substract a vector to a matrix over a specified axis in place
** axis = 0 -> vector substraction for each row
** axis = 1 -> vector substraction for each column
*/
func SubVecToMatInPlace(m *[][]float64, v[]float64, axis int) {
    nSamples, nFeatures := len(*m), len((*m)[0])
    var idx int

    for i := 0; i < nSamples; i++ {
        for j := 0; j < nFeatures; j++ {
            if axis == 1 {
                idx = i
            } else {
                idx = j
            }

            (*m)[i][j] -= v[idx]
        }
    }
}

/*
** vstack (vertical stack) of 3 matrix
**
*/
func Vstack(m1 [][]float64, m2 [][]float64, m3 [][]float64) *[][]float64 {
    nFeatures := len(m1[0])
    nSamples1 := len(m1)
    nSamples2 := len(m2)
    nSamples3 := len(m3)
    nTotal := nSamples1 + nSamples2 + nSamples3

    res := InitSlicesFloat64(nTotal, nFeatures)


    for i := 0; i < nTotal; i++ {
        for j := 0; j < nFeatures; j++ {
            if i >= nSamples1 + nSamples2 {
                (*res)[i][j] = m3[i - nSamples1 - nSamples2][j]
            } else if i >= nSamples1 {
                (*res)[i][j] = m2[i - nSamples1][j]
            } else {
                (*res)[i][j] = m1[i][j]
            }
        }
    }

    return res
}

/*
** Mult vec by const and sum all the values
*/
func MultVecAndSum(vec []float64, val float64) float64 {
    var res float64 = 0
    for _, elm := range vec {
        res += elm * val
    }

    return res
}

/*
** Extract components from 'vt'
** vt -> shape = (nSamples, nFeatures)
** We want to extract the first 'nComponents'
*/
func ExtractComponents(vt *mat.Dense, nComponents int) *[][]float64 {
    _, nFeatures := vt.Dims()
    res := InitSlicesFloat64(nComponents, nFeatures)

    for i := 0; i < nComponents; i++ {
        for j := 0; j < nFeatures; j++ {
            (*res)[i][j] = vt.At(i, j)
        }
    }
    return res
}

/*
** Transpose matrice
*/
func MatTranspose(m *[][]float64) *[][]float64 {
    dim1, dim2 := len(*m), len((*m)[0])
    res := InitSlicesFloat64(dim2, dim1)

    for i := 0; i < dim1; i++ {
        for j := 0; j < dim2; j++ {
            (*res)[j][i] = (*m)[i][j]
        }
    }
    return res
}

/*
** Dot product between two matrices (slice of slice)
*/
func DotProduct(m1 *[][]float64, m2 *[][]float64) *[][]float64 {
    nbRowM1 := len(*m1)
    nbRowM2 := len(*m2)
    nbColM2 := len((*m2)[0])

    res := InitSlicesFloat64(nbRowM1, nbColM2)

    for i := 0; i < nbRowM1; i++ {
        for j := 0; j < nbColM2; j++ {
            for k := 0; k < nbRowM2; k++ {
                (*res)[i][j] += (*m1)[i][k] * (*m2)[k][j]
            }
        }
    }
    return res
}

/*
** Sign correction to ensure deterministic output from SVD.
** Adjusts the columns of u and the rows of v such that the loadings in the
** columns in u that are largest in absolute value are always positive.
**
** params:
** 'u': output of SVD decomposition -> shape=(n_samples, n_samples)
** 'v': output of SVD decomposition -> shape=(n_samples, n_features)
**
** No return because 'SvdFlip' is making operations in place
*/
func SvdFlip(u, vt *mat.Dense) {
    maxVec := GetAbsoluteMax(vt, 1)
    signedMaxVec := SignSlice(maxVec)
    MultiplyVec(u, signedMaxVec, 0)
    MultiplyVec(vt, signedMaxVec, 1)
}

/*
** Calculate mean update and a Youngs and Cramer variance update.
** last_mean and last_variance are statistics computed at the last step by the
** function. Both must be initialized to 0.0. In case no scaling is required
** last_variance can be None. The mean is always required and returned because
** necessary for the calculation of the variance. last_n_samples_seen is the
** number of samples encountered until now.
** From the paper "Algorithms for computing the sample variance: analysis and
** recommendations", by Chan, Golub, and LeVeque.
*/
func IncrementalMeanAndVar(data [][]float64,
                           lastMean, lastVar []float64,
                           lastSampleCount int) ([]float64, []float64, int) {
    newSampleCount := len(data)

    // get the last sum
    lastSum := MultSliceByConst(lastMean, float64(lastSampleCount))

    // sum over the axis 0
    newSum := SumMat(data, 0)

    // update the sample count
    updatedSampleCount := lastSampleCount + newSampleCount

    // update mean
    updatedMean := DivSliceByConst(Sum2Slices(lastSum, newSum), float64(updatedSampleCount))

    // compute variance over the axis 0
    newUnnormalizedVariance := MultSliceByConst(VarMat(data, 0), float64(newSampleCount))

    var updatedUnormalizedVariance []float64
    if lastSampleCount == 0 {
        updatedUnormalizedVariance = newUnnormalizedVariance
    } else {
        lastUnnormalizedVariance := MultSliceByConst(lastVar, float64(lastSampleCount))
        lastOverNewCount := float64(lastSampleCount) / float64(newSampleCount)


        // updated normalized variance
        sumUnormalized := Sum2Slices(lastUnnormalizedVariance, newUnnormalizedVariance)
        divCount := float64(lastOverNewCount) / float64(updatedSampleCount)
        divSum := DivSliceByConst(lastSum, lastOverNewCount)
        subSum := Sub2Slices(divSum, newSum)
        squareSum := Mult2Slices(subSum, subSum)
        tmp := MultSliceByConst(squareSum, divCount)
        updatedUnormalizedVariance = Sum2Slices(sumUnormalized, tmp)
    }
    // update the variance
    updatedVariance := DivSliceByConst(updatedUnormalizedVariance, float64(updatedSampleCount))
    return updatedMean, updatedVariance, updatedSampleCount
}
