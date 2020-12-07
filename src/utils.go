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
**
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
** Multiply 'm' by the vector 'v' along specific axis
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
func SumMat(m *mat.Dense, axis int) []float64 {
    nSamples, nFeatures := m.Dims()

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

            res[idx] += m.At(i, j)
        }
    }
    return res
}

/*
** Compute variance over specific axis
** axis = 0 -> variance for each column (len(result) == n_features)
** axis = 1 -> variance for each rows
*/
func VarMat(m *mat.Dense, axis int) []float64 {
    // TODO
    return nil
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
func IncrementalMeanAndVar(data [][]float64) {
    // TODO
}
