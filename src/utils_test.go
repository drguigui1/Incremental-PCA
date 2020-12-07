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