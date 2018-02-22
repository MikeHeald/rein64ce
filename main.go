package main

import (
    "os"
    "fmt"

    "gonum.org/v1/gonum/mat"

    "github.com/SageFocusLLC/gophernet"
)

//has to be mupen64plus 64 bit linux, with default input plugin
func main(){
    args := os.Args[1:]
    mapPath := "./statemap.json"
    env := NewEnvironment(args, mapPath)

    env.Init()

    //create neural net
    nnConfig := gonet.NeuralNetConfig{
        InputNeurons: 4,
        OutputNeurons: 8,
        HiddenNeurons: 10,
        NumEpochs: 1,
        LearningRate: 0.1,
    }

    agent := NewAgent(nnConfig)

    agent.LoadNN()

    stateArr := []float64{0.01, 0.01, 0.01, 0.01}
    env.GetState(stateArr)

    reward := 0.0
    epoch := 0
    action := uint64(0x00)
    mapPositionVec(stateArr[0:3])

    stateP := mat.NewDense(1, len(stateArr), nil)
    state  := mat.NewDense(1, len(stateArr), nil)

    for epoch < 100{
        fmt.Println("Epoch ", epoch)

        env.LoadGame()

        endstate := false
        step := 0

        for step < 2000 && endstate != true{
            //action
            state.SetRow(0, stateArr)

            action = agent.GetAction(state)

            env.GameStep(action)

            //observation
            env.GetState(stateArr)

            mapPositionVec(stateArr[0:3])

            stateArr[3] = stateArr[3] * 0.01

            //reward
            reward, endstate = getReward(stateArr, epoch, step)

            stateP.SetRow(0, stateArr)

            agent.GiveReward(state,stateP,reward)

            step += 1
        }
        agent.SaveNN()

        epoch += 1
    }
    fmt.Println("done :D")
}

func getReward(stateArr []float64, epoch int, step int) (float64, bool) {
    reward := 0.0
    endstate := false

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
    if stateArr[1] < 0.27 {
        reward = -10.0
        fmt.Println("Fell")
        endstate = true
    }

    //avoid start area
    if stateArr[1] > 0.791 {
        reward = -1.0
    } else {
        reward = -0.5
    }

    reward += stateArr[2] * 0.5

    return reward, endstate

}


func mapPositionVec(fArr []float64){
    for i, v := range fArr{
        fArr[i] = mapPositionVal(v)
    }
}

func mapPositionVal(x float64) float64{
    return (x / 20000.0) + 0.5
}
