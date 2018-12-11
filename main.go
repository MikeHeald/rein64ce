package main

import (
	"fmt"
	"strconv"
	"os"
	"encoding/csv"
	"io/ioutil"
)

//has to be mupen64plus 64 bit linux, with default input plugin
func main() {
	args := os.Args[1:]
	mapPath := "./statemap.json"
	agent := NewAgent()
	env := NewEnvironment(args, mapPath)

	env.Init()

	episodeLength := 900
	episodeProgress := float32(0.0)


	agent.SetEpsilon(0.01)

	stateArr := []float32{0.01, 0.01, 0.01, 0.01, 0.01, 0.01}
	var stateArray [3]float32
	env.GetState(stateArr)

	epoch := 0
	action := uint64(0x00)


	for epoch < 100 {
		fmt.Println("Epoch ", epoch)

		endstate := false
		step := 1

	        epochMem := [][]float32{}

		for step < episodeLength && endstate != true {
			episodeProgress = float32(step)

			copy(stateArray[:], stateArr[:3])

//                        action = agent.GetActionEGreedy(stateArray)
			action = agent.Predict(stateArray)

			//supervised training
			//_ = env.GameStepTrain()
			//actionP = env.GameStepTrain()

			env.GameStep(action)

			//observation
			env.GetState(stateArr)

			epochMem = append(epochMem,append(stateArr,float32(action)))
			//fmt.Println(stateArr)

			stateArr[5] = episodeProgress


			step += 1
		}

		//detach
		env.Disconnect()


		epoch += 1

		memFiles, err := ioutil.ReadDir("./memz")
		if err != nil{
			panic(err)
		}

	        file, _ := os.Create(fmt.Sprintf("memz/epochMeme%d.csv",len(memFiles)))
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

		env.LoadGame()

		env.Reconnect()

	}
	fmt.Println("done :D")
}


