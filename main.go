package main

import (
	"fmt"
    "strconv"
	"os"



    "encoding/csv"
)

//has to be mupen64plus 64 bit linux, with default input plugin
func main() {
	args := os.Args[1:]
	mapPath := "./statemap.json"
	env := NewEnvironment(args, mapPath)

	env.Init()

	episodeLength := 1200
	episodeProgress := float32(0.0)


	agent := NewAgent()

//	agent.LoadNN()

	agent.SetTau(0.01)

	stateArr := []float32{0.01, 0.01, 0.01, 0.01, 0.01, 0.01}
	var stateArray [3]float32
	env.GetState(stateArr)

	epoch := 0
	action := uint64(0x00)
	//actionP := uint64(0x00)

//	state := mat.NewDense(1, len(stateArr), nil)
//	state := []float64

	//actionMem := mat.NewDense(1000, 1, nil)
	//rewardMem := mat.NewDense(1000, 1, nil)

	for epoch < 100 {
		fmt.Println("Epoch ", epoch)

		env.LoadGame()

		endstate := false
		step := 1

	        epochMem := [][]float32{}

		for step < episodeLength && endstate != true {
			episodeProgress = float32(step) / float32(episodeLength + 1)

			//action
			//state.SetRow(0, stateArr)

			//greedy
			//action = agent.GetActionGreedy(state)

			//e greedy exploration
			//action = agent.GetActionEGreedy(state)

			//boltzmann
			//action = agent.GetActionBoltzmann(state)

			//python
			//action = getWebAction(stateArr)

			//tf go
			copy(stateArray[:], stateArr[:3])
			//action = agent.Predict(stateArray)

			//tf epsilon 
			//copy(stateArray[:], stateArr[:3])
                        action = agent.GetActionEGreedy(stateArray)

			//training
			//_ = env.GameStepTrain()
			//actionP = env.GameStepTrain()
			//fmt.Println(action)


			env.GameStep(action)

			//observation
			env.GetState(stateArr)

			epochMem = append(epochMem,append(stateArr,float32(action)))
			fmt.Println(stateArr)


			stateArr[5] = episodeProgress


			step += 1
		}
		//decrease temp
		curTemp := agent.GetTau()
		if curTemp > 0.002 {
			agent.SetTau(curTemp * 0.8)
		}

		fmt.Println(agent.Q)
		fmt.Println(curTemp)
//		agent.SaveNN()

		epoch += 1
        fmt.Println(len(epochMem))
        file, _ := os.Create("epochMeme.csv")
        defer file.Close()
        writer := csv.NewWriter(file)
        defer writer.Flush()
        for _, value := range epochMem {
            strArr := []string{}
            for _, fval := range value {
                strArr = append(strArr, strconv.FormatFloat(float64(fval),'f', 6, 64))
            }
            _ = writer.Write(strArr)
        }

	}
	fmt.Println("done :D")
}


