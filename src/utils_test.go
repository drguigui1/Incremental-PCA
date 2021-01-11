package main

import (
    "testing"
    "gonum.org/v1/gonum/mat"
    "reflect"
)

func fromPointersToFloatSlice(s []*float64) []float64 {
    n := len(s)
    res := make([]float64, n)

    for idx, elm := range s {
        res[idx] = *elm
    }

    return res
}

func TestGetAbsoluteMax1(t *testing.T) {
    test := mat.NewDense(4, 3, []float64{
        -3, 4, 5,
        11, 6, 7,
        6, -9, 7,
        0, 6, 7,
    })

    ref := []float64{5, 11, -9, 7}
    axis := 1

    res := GetAbsoluteMax(test, axis)

    if !reflect.DeepEqual(ref, res) {
        t.Errorf("Error test1\n")
        t.Errorf("res %v\n", res)
        t.Errorf("ref %v\n", ref)
    }
}

func TestGetAbsoluteMax2(t *testing.T) {
    test := mat.NewDense(4, 3, []float64{
        -3, 4, 5,
        11, 6, 7,
        6, -9, 7,
        0, 6, 7,
    })

    ref := []float64{11, -9, 7}
    axis := 0

    res := GetAbsoluteMax(test, axis)

    if !reflect.DeepEqual(ref, res) {
        t.Errorf("Error test2\n")
        t.Errorf("res %v\n", res)
        t.Errorf("ref %v\n", ref)
    }
}

func TestGetAbsoluteMax3(t *testing.T) {
    test := mat.NewDense(3, 3, []float64{
        -3, 4, -5,
        15, 6, -7,
        6, -9, 7,
    })

    ref := []float64{-5, 15, -9}
    axis := 1

    res := GetAbsoluteMax(test, axis)

    if !reflect.DeepEqual(ref, res) {
        t.Errorf("Error test3\n")
        t.Errorf("res %v\n", res)
        t.Errorf("ref %v\n", ref)
    }
}

func TestGetAbsoluteMax4(t *testing.T) {
    test := mat.NewDense(3, 3, []float64{
        -3, 4, -5,
        15, 6, -7,
        6, -9, 7,
    })

    ref := []float64{15, -9, -7}
    axis := 0

    res := GetAbsoluteMax(test, axis)

    if !reflect.DeepEqual(ref, res) {
        t.Errorf("Error test4\n")
        t.Errorf("res %v\n", res)
        t.Errorf("ref %v\n", ref)
    }
}

func TestSignSlice(t *testing.T) {
    test := []float64{1, -5, 6, -2, 0}

    ref := []float64{1, -1, 1, -1, 0}

    res := SignSlice(test)

    if !reflect.DeepEqual(ref, res) {
        t.Errorf("Error test1\n")
        t.Errorf("res %v\n", res)
        t.Errorf("ref %v\n", ref)
    }
}

func TestMultiplyVec1(t *testing.T) {
    test := mat.NewDense(4, 3, []float64{
        -3, 4, 5,
        11, 6, 7,
        6, -9, 7,
        0, 6, 7,
    })

    v := []float64{0, 1, 2}
    axis := 0

    ref := mat.NewDense(4, 3, []float64{
        0, 4, 10,
        0, 6, 14,
        0, -9, 14,
        0, 6, 14,
    })

    MultiplyVec(test, v, axis)

    if !reflect.DeepEqual(ref, test) {
        t.Errorf("Error test1\n")
        t.Errorf("res %v\n", test)
        t.Errorf("ref %v\n", ref)
    }
}

func TestMultiplyVec2(t *testing.T) {
    test := mat.NewDense(4, 3, []float64{
        -3, 4, 5,
        11, 6, 7,
        6, -9, 7,
        0, 6, 7,
    })

    v := []float64{0, 1, 2, 0}
    axis := 1

    ref := mat.NewDense(4, 3, []float64{
        0, 0, 0,
        11, 6, 7,
        12, -18, 14,
        0, 0, 0,
    })

    MultiplyVec(test, v, axis)

    if !reflect.DeepEqual(ref, test) {
        t.Errorf("Error test2\n")
        t.Errorf("res %v\n", test)
        t.Errorf("ref %v\n", ref)
    }
}

func TestSumMat1(t *testing.T) {
    test := [][]float64{
        { -3, 4, 5 },
        { 11, 6, 7 },
        { 6, -9, 7 },
        { 0, 6, 7  },
    }

    axis := 1

    ref := []float64{6, 24, 4, 13}

    res := SumMat(test, axis)

    if !reflect.DeepEqual(ref, res) {
        t.Errorf("Error test1\n")
        t.Errorf("res %v\n", res)
        t.Errorf("ref %v\n", ref)
    }
}

func TestSumMat2(t *testing.T) {
    test := [][]float64{
        { -3, 4, 5 },
        { 11, 6, 7 },
        { 6, -9, 7 },
        { 0, 6, 7  },
    }

    axis := 0

    ref := []float64{14, 7, 26}

    res := SumMat(test, axis)

    if !reflect.DeepEqual(ref, res) {
        t.Errorf("Error test2\n")
        t.Errorf("res %v\n", res)
        t.Errorf("ref %v\n", ref)
    }
}

func TestVarMat1(t *testing.T) {
    test := [][]float64{
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9},
        {1, 1, 1},
    }

    axis := 0
    ref := []float64{6.1875, 7.5, 9.1875}

    res := VarMat(test, axis)

    if !reflect.DeepEqual(ref, res) {
        t.Errorf("Error test1\n")
        t.Errorf("res %v\n", res)
        t.Errorf("ref %v\n", ref)
    }
}

func TestVarMat2(t *testing.T) {
    test := [][]float64{
        {1, 2, 3},
        {4, 5, 6},
        {7, 8, 9},
        {1, 1, 1},
    }

    axis := 1
    ref := []float64{0.6666666666666666, 0.6666666666666666, 0.6666666666666666, 0.}

    res := VarMat(test, axis)

    if !reflect.DeepEqual(ref, res) {
        t.Errorf("Error test2\n")
        t.Errorf("res %v\n", res)
        t.Errorf("ref %v\n", ref)
    }
}

func TestVarMat3(t *testing.T) {
    test := [][]float64{
        { 55, 0, 150 },
        {  0, 12, 99 },
        { -5, 15, 0  },
    }

    axis := 0
    ref := []float64{738.8888888888888, 42., 3878.}

    res := VarMat(test, axis)

    if !reflect.DeepEqual(ref, res) {
        t.Errorf("Error test3\n")
        t.Errorf("res %v\n", res)
        t.Errorf("ref %v\n", ref)
    }
}

func TestMeanMat1(t *testing.T) {
    test := [][]float64{
        { 1,2,3 },
        { 5,1,4 },
        { 8,6,4 },
    }

    axis := 0
    ref := []float64{4.666666666666667, 3, 3.6666666666666665}

    res := MeanMat(test, axis)

    if !reflect.DeepEqual(ref, res) {
        t.Errorf("Error test1\n")
        t.Errorf("res %v\n", res)
        t.Errorf("ref %v\n", ref)
    }
}

func TestMeanMat2(t *testing.T) {
    test := [][]float64{
        { 1,2,3 },
        { 5,1,4 },
        { 8,6,4 },
    }

    axis := 1
    ref := []float64{2., 3.3333333333333335, 6.}

    res := MeanMat(test, axis)

    if !reflect.DeepEqual(ref, res) {
        t.Errorf("Error test1\n")
        t.Errorf("res %v\n", res)
        t.Errorf("ref %v\n", ref)
    }
}

func TestMultMatByVec1(t *testing.T) {
    test := [][]float64{
        { 1,2,3 },
        { 5,1,4 },
        { 8,6,4 },
        { 8,6,4 },
    }

    vec := []float64{1, 2, 3}

    axis := 0

    ref := [][]float64{
        { 1,4,9 },
        { 5,2,12 },
        { 8,12,12 },
        { 8,12,12 },
    }


    res := MultMatByVec(test, vec, axis)

    if !reflect.DeepEqual(ref, *res) {
        t.Errorf("Error test1\n")
        t.Errorf("res %v\n", *res)
        t.Errorf("ref %v\n", ref)
    }
}

func TestMultMatByVec2(t *testing.T) {
    test := [][]float64{
        { 1,2,3 },
        { 5,1,4 },
        { 8,6,4 },
        { 8,6,4 },
    }

    vec := []float64{1, 2, 3, 0.5}

    axis := 1

    ref := [][]float64{
        { 1,2,3 },
        { 10,2,8 },
        { 24,18,12 },
        { 4,3,2 },
    }


    res := MultMatByVec(test, vec, axis)

    if !reflect.DeepEqual(ref, *res) {
        t.Errorf("Error test2\n")
        t.Errorf("res %v\n", *res)
        t.Errorf("ref %v\n", ref)
    }
}

func TestVstack1(t *testing.T) {
    m1 := [][]float64{
        { 1,2,3 },
        { 5,1,4 },
    }

    m2 := [][]float64{
        { 1,2,3 },
        { 5,1,4 },
        { 5,8,4 },
    }

    m3 := [][]float64{
        { 0,2,3 },
    }

    ref := [][]float64{
        { 1,2,3 },
        { 5,1,4 },
        { 1,2,3 },
        { 5,1,4 },
        { 5,8,4 },
        { 0,2,3 },
    }


    res := Vstack(m1, m2, m3)

    if !reflect.DeepEqual(ref, *res) {
        t.Errorf("Error test1\n")
        t.Errorf("res %v\n", *res)
        t.Errorf("ref %v\n", ref)
    }
}

func TestVstack2(t *testing.T) {
    m1 := [][]float64{
        { 1,2,3 },
    }

    m2 := [][]float64{
        { 5,8,4 },
    }

    m3 := [][]float64{
        { 0,2,3 },
    }

    ref := [][]float64{
        { 1,2,3 },
        { 5,8,4 },
        { 0,2,3 },
    }


    res := Vstack(m1, m2, m3)

    if !reflect.DeepEqual(ref, *res) {
        t.Errorf("Error test2\n")
        t.Errorf("res %v\n", *res)
        t.Errorf("ref %v\n", ref)
    }
}

func TestMultVecAndSum1(t *testing.T) {
    test := []float64{1,2,3,4,5,6}
    var val float64 = 2

    var ref float64 = 42

    res := MultVecAndSum(test, val)

    if ref != res {
        t.Errorf("Error test1\n")
        t.Errorf("res %v\n", res)
        t.Errorf("ref %v\n", ref)
    }
}

func TestExtractComponents1(t *testing.T) {
    test := mat.NewDense(6, 3, []float64{
        3, 4, 5,
        6, 0, 7,
        1, 6, 12,
        6, 4, 15,
        6, 4, 15,
        6, 4, 15,
    })

    nComponents := 3
    ref := [][]float64{
        {3, 4, 5},
        {6, 0, 7},
        {1, 6, 12},
    }

    res := ExtractComponents(test, nComponents)

    if !reflect.DeepEqual(ref, *res) {
        t.Errorf("Error test1\n")
        t.Errorf("res %v\n", *res)
        t.Errorf("ref %v\n", ref)
    }
}

func TestExtractComponents2(t *testing.T) {
    test := mat.NewDense(2, 7, []float64{
        3, 4, 5, 6, 8, 32, 2,
        6, 0, 7, 4.3, 6.6, 1., 6,
    })

    nComponents := 1
    ref := [][]float64{
        {3, 4, 5, 6, 8, 32, 2},
    }

    res := ExtractComponents(test, nComponents)

    if !reflect.DeepEqual(ref, *res) {
        t.Errorf("Error test2\n")
        t.Errorf("res %v\n", *res)
        t.Errorf("ref %v\n", ref)
    }
}

func TestMatTranspose1(t *testing.T) {
    test := [][]float64{
        {1,2,3,4,5},
        {6,7,8,9,9},
    }
    ref := [][]float64{
        {1, 6},
        {2, 7},
        {3, 8},
        {4, 9},
        {5, 9},
    }

    res := MatTranspose(&test)

    if !reflect.DeepEqual(ref, *res) {
        t.Errorf("Error test1\n")
        t.Errorf("res %v\n", *res)
        t.Errorf("ref %v\n", ref)
    }
}

func TestMatTranspose2(t *testing.T) {
    test := [][]float64{
        {1,2,3,4,5},
        {6,7,8,9,9},
        {1,0,4,0,11},
    }
    ref := [][]float64{
        {1,6,1},
        {2,7,0},
        {3,8,4},
        {4,9,0},
        {5,9,11},
    }

    res := MatTranspose(&test)

    if !reflect.DeepEqual(ref, *res) {
        t.Errorf("Error test2\n")
        t.Errorf("res %v\n", *res)
        t.Errorf("ref %v\n", ref)
    }
}

func TestMatTranspose3(t *testing.T) {
    test := [][]float64{
        {1,2},
    }
    ref := [][]float64{
        {1},
        {2},
    }

    res := MatTranspose(&test)

    if !reflect.DeepEqual(ref, *res) {
        t.Errorf("Error test3\n")
        t.Errorf("res %v\n", *res)
        t.Errorf("ref %v\n", ref)
    }
}

func TestDotProduct1(t *testing.T) {
    m1 := [][]float64{
        {1,2},
    }
    m2 := [][]float64{
        {1},
        {2},
    }

    ref := [][]float64{
        {5},
    }

    res := DotProduct(&m1, &m2)

    if !reflect.DeepEqual(ref, *res) {
        t.Errorf("Error test1\n")
        t.Errorf("res %v\n", *res)
        t.Errorf("ref %v\n", ref)
    }
}

func TestDotProduct2(t *testing.T) {
    m1 := [][]float64{
        {1,2,3,4,5},
        {6,7,8,9,9},
        {1,0,4,0,11},
    }
    m2 := [][]float64{
        {3,8},
        {4,2},
        {5,1},
        {6,2},
        {8,3},
    }

    ref := [][]float64{
        {90,38},
           {212,115},
        {111,45},
    }

    res := DotProduct(&m1, &m2)

    if !reflect.DeepEqual(ref, *res) {
        t.Errorf("Error test2\n")
        t.Errorf("res %v\n", *res)
        t.Errorf("ref %v\n", ref)
    }
}