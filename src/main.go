package main

import (
    "fmt"
)

func main() {
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

    ipca := InitIncrementalPCA(2, 5)
    ipca.PartialFit(&batch1)
    ipca.PartialFit(&batch2)
    ipca.PartialFit(&batch3)
    ipca.PartialFit(&batch4)
    //ipca.Print()
    fmt.Println(ipca.Transform(&data))
}