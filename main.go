package main

import (
	"fmt"
    "strconv"
	"os"



    "encoding/csv"
    "log"
    "net/http"
    "encoding/json"
    "bytes"
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

	reward := float32(0.0)
	epoch := 0
	action := uint64(0x00)
	//actionP := uint64(0x00)
	mapPositionVec(stateArr[0:3])

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

			mapPositionVec(stateArr[0:3])

			stateArr[3] = stateArr[3] / 65536.0
			stateArr[4] = stateArr[4] * 0.01

			stateArr[5] = episodeProgress

			//reward
			//reward, endstate = getReward(stateArr, epoch, step)


			//scale reward
			reward = reward * 0.5

//			agent.GiveReward(state, stateP, reward)
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

func getWebAction(stateArr []float32) uint64 {
	message := map[string]interface{}{
		"obs" : stateArr,
	}

	bytesRepresentation, err := json.Marshal(message)
	if err != nil {
		log.Fatalln(err)
	}

	resp, err := http.Post("http://localhost:5000/predict", "application/json", bytes.NewBuffer(bytesRepresentation))
	if err != nil {
		log.Fatalln(err)
	}

	var result map[string]interface{}

	json.NewDecoder(resp.Body).Decode(&result)

	fmt.Println(result)
	defer resp.Body.Close()

	return 32

}

func getWebActionMock(stateArr []float32) uint64 {
	return 32
}

func getReward(stateArr []float32, epoch int, step int) (float32, bool) {
	reward := 0.0
	endstate := false
/*
//stairs
	//left the room
	if stateArr[2] > 0.7{
		reward = -1.0
		fmt.Println("left the room")
		endstate = true
	}	
	//made it
	if stateArr[1] > 0.8{
		reward = 2.0
		fmt.Println("OMG")
		endstate = true
	}
	//more height -> more reward
	reward += stateArr[1] - 1.0
*/
//slide

	reward = -0.3
	if stateArr[1] > 0.27055 && stateArr[1] < 0.28270 {
		if stateArr[0] > 0.12314 && stateArr[0] < 0.24100 {
			if stateArr[2] < 0.82216 && stateArr[2] > 0.70840 {
				reward = 5.0
				fmt.Println("WIN")
				fmt.Println(stateArr)
				endstate = true
			}
		}
	}

	//fell off the level
	if stateArr[1] < 0.26 {
		reward = -2.0
		endstate = true
	}

	//reward = stateArr[1]*-0.8 + stateArr[3]*0.2
	//avoid start area
	if stateArr[0] > 0.64 && stateArr[1] > 0.76{
		reward += -0.5
	} else {
		reward += -0.05
	}
	//reward = reward * 0.3 - 0.2

	//slight reward for moving down
	reward += 0.05 * (1.0 - float64(stateArr[1]))

	return float32(reward), endstate

}

func mapPositionVec(fArr []float32) {
	for i, v := range fArr {
		fArr[i] = mapPositionVal(v)
	}
}

//normalize - max value is 10000, min -10000 (maybe not true for all levels)
func mapPositionVal(x float32) float32 {
	return -1.0 + (x - -10000.0) * (1.0 - -1.0) / (10000.0 - -10000.0)
}
