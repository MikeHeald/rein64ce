package main

import (
	"github.com/SageFocusLLC/gophernet"
	"gonum.org/v1/gonum/mat"
	"io/ioutil"
	"math/rand"
	"math"
)

//Agent - has some way of predicting Q
type Agent struct {
	ActionSpace []uint64
	nn          *gonet.NeuralNet
	lRate       float64
	nnOut       *mat.Dense
	maxQ        float64
	maxQInd     int
	Q           []float64
	tau         float64
}

//NewAgent - Agent constructor
func NewAgent(nnConf gonet.NeuralNetConfig) *Agent {
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
	agent := &Agent{
		ActionSpace: aspace,
		nn:          gonet.NewNetwork(nnConf),
		lRate:       nnConf.LearningRate,
		nnOut:       mat.NewDense(1, len(aspace), nil),
		Q:           make([]float64, len(aspace)),
		maxQ:        0.0,
		maxQInd:     0,
		tau:         0.0,
	}

	return agent
}

//LoadNN - Load the weights and biases from binary file
func (agent *Agent) LoadNN() {
	wHid, er1 := ioutil.ReadFile("wHid.nn")
	bHid, er2 := ioutil.ReadFile("bHid.nn")
	wOut, er3 := ioutil.ReadFile("wOut.nn")
	bOut, er4 := ioutil.ReadFile("bOut.nn")
	if er1 != nil || er2 != nil || er3 != nil || er4 != nil {
		panic("error reading nn files")
	}

	mNN := [][]byte{wHid, bHid, wOut, bOut}

	agent.nn.UnmarshalNN(mNN)

	agent.lRate = 0.0
}

//SaveNN - Save the current weights and biases to a binary file
func (agent *Agent) SaveNN() {
	binnn := agent.nn.MarshalNN()

	_ = ioutil.WriteFile("wHid.nn", binnn[0], 0755)
	_ = ioutil.WriteFile("bHid.nn", binnn[1], 0755)
	_ = ioutil.WriteFile("wOut.nn", binnn[2], 0755)
	_ = ioutil.WriteFile("bOut.nn", binnn[3], 0755)
}

//GetActionBoltzmann
func (agent *Agent) GetActionBoltzmann(state *mat.Dense) uint64 {
	agent.nn.Feedforward(state)

	actInd := getBoltzFloat(agent.nn.Output.RawRowView(0), agent.tau)

	return agent.ActionSpace[actInd]
}

func getBoltzFloat(fArr []float64, tau float64) int {
	//calc boltzmann for each qval
	var pqArr []float64
	for _, v := range fArr {
		pqArr = append(pqArr, math.Exp(v / tau))
	}

	//sum the weights
	pqSum := 0.0
	for _, v := range pqArr {
		pqSum += v
	}

	//divide each boltz by sum
	for i, v := range pqArr {
		pqArr[i] = v / pqSum
	}

	//select max?
	maxBup, _ := getMaxFloat(pqArr)
	return maxBup


}

//GetActionBayesian

//GetActionEGreedy - agent.tau is used as epsilon
func (agent *Agent) GetActionEGreedy(state *mat.Dense) uint64 {
	eVal := rand.Float64()
	if eVal < agent.tau {
		return agent.GetRandAction()
	} else {
		return agent.GetActionGreedy(state)
	}
}

//GetRandAction - select random action
func (agent *Agent) GetRandAction() uint64 {
	return agent.ActionSpace[rand.Intn(len(agent.ActionSpace))]
}

//GetActionGreedy - given the current state, predict Q and select an action
func (agent *Agent) GetActionGreedy(state *mat.Dense) uint64 {
	agent.nn.Feedforward(state)

	maxQ1Ind, _ := getMaxFloat(agent.nn.Output.RawRowView(0))

	return agent.ActionSpace[maxQ1Ind]
}

//GiveReward - Bellman Eqn
func (agent *Agent) GiveReward(state *mat.Dense, statePrime *mat.Dense, reward float64) {
	agent.nn.Feedforward(statePrime)

	copy(agent.Q, agent.nn.Output.RawRowView(0))
	agent.maxQInd, agent.maxQ = getMaxFloat(agent.Q)
	agent.Q[agent.maxQInd] = reward + (1.0-agent.lRate)*agent.maxQ

	agent.nn.Feedforward(state)
	agent.nnOut.SetRow(0, agent.Q)
	_ = agent.nn.Backpropagate(state, agent.nnOut)
}

func getMaxFloat(fArr []float64) (int, float64) {
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