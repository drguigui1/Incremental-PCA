package main

import (
	//"gonum.org/v1/gonum/mat"
	//"log"
)

func main() {
	// load data
	//fmt.Println("Loading data")
	/*dataStr := FromCSV("./dataset_final_batch_nb_0_300")
	// convert data
	//fmt.Println("Convert loaded data")
	data := FromSliceToMat(&dataStr)

	//fmt.Println("SVD Factorization")
	var svd mat.SVD
	ok := svd.Factorize(data, mat.SVDThin)

	if !ok {
		log.Fatal("SVD Decomposition failed")
	}

	//fmt.Println("Ready to get the data")
	var u, v mat.Dense
	svd.UTo(&u)
	svd.VTo(&v)
	values := svd.Values(nil)

	//fmt.Println(u)
	//fmt.Println(v)
	fmt.Println(values)
*/
	/*a := mat.NewDense(4, 3, []float64{
		3, 4, 5,
		6, 6, 7,
		6, 6, 7,
		6, 6, 7,
	})

	fmt.Printf("A = %v\n\n", a)

	var svd mat.SVD
	ok := svd.Factorize(a, mat.SVDThin)
	//var eig mat.Eigen
	//var eigenVec mat.CDense

	if !ok {
    	log.Fatal("SVD Decomposition failed")
	}
	var u, v mat.Dense
	svd.UTo(&u)
	svd.VTo(&v)
	//values := svd.Values(nil)

	//tmp := mat.DenseCopyOf(v.T())
    fmt.Println(u)
    fmt.Println(u.RowView(0))
    fmt.Println(u.RowView(1))
    fmt.Println()
	//fmt.Println(v)
	//fmt.Println(values)
    //fmt.Println(tmp)
    */
	/*n, m := tmp.Dims()
	for i := 0; i < n; i++ {
		for j := 0; j < m; j++ {
			fmt.Printf("%v ", tmp.At(i, j))
		}
		fmt.Printf("\n")
	}*/

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

	ipca := InitIncrementalPCA(2, 5)
	ipca.PartialFit(&batch1)
	ipca.PartialFit(&batch2)
	ipca.PartialFit(&batch3)
	ipca.PartialFit(&batch4)
	ipca.Print()
}