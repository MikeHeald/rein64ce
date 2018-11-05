package main

import (
	"fmt"
	"math/rand"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

//Agent - has some way of predicting Q
type Agent struct {
	ActionSpace []uint64
	nn          *tf.SavedModel
	maxQ        float64
	maxQInd     int
	Q           []float64
	tau         float64
}

//NewAgent - Agent constructor
func NewAgent() *Agent {
	aspace := []uint64{
		0x0000,     //idle
		0x0020,     //Z
		0x0040,     //B
		0x0080,     //A
		0x50000000, //up
		0xb0000000, //down
		0xb00000,   //left
		0x500000,   //right
	}
	model, err := tf.LoadSavedModel("myModel", []string{"myTag"}, nil)
	if err != nil {
		fmt.Printf("err")
	}


	agent := &Agent{
		ActionSpace: aspace,
		nn:          model,
		Q:           make([]float64, len(aspace)),
		maxQ:        0.0,
		maxQInd:     0,
		tau:         0.0,
	}

	return agent
}


func getMaxFloat(fArr []float32) (int, float32) {
	curMax := fArr[0]
	curMaxInd := 0
	for i, v := range fArr {
		if v > curMax {
			curMax = v
			curMaxInd = i
		}
	}
	return curMaxInd, curMax
}

func (agent *Agent) GetTau() float64 {
	return agent.tau
}

func (agent *Agent) SetTau(tau float64) {
	agent.tau = tau
}


//adding tensorflow
func (agent *Agent) GetActionEGreedy(state [3]float32) uint64 {
        eVal := rand.Float64()
        if eVal < agent.tau {
                return agent.GetRandAction()
        } else {
                return agent.Predict(state)
        }
}

//GetRandAction - select random action
func (agent *Agent) GetRandAction() uint64 {
        return agent.ActionSpace[rand.Intn(len(agent.ActionSpace))]
}



func (agent *Agent) Predict(state [3]float32) uint64 {
	tensor, _ := tf.NewTensor([1][3]float32{state,})

	result, err := agent.nn.Session.Run(
                map[tf.Output]*tf.Tensor{
                        agent.nn.Graph.Operation("inputLayer_input").Output(0): tensor,
                },
                []tf.Output{
                        agent.nn.Graph.Operation("outputLayer/Softmax").Output(0),
                },
                nil,
        )

	if err != nil {
		fmt.Println(err)
	}

        maxQ1Ind, _ := getMaxFloat(result[0].Value().([][]float32)[0])

        return agent.ActionSpace[maxQ1Ind]
}
