package main

type Environment struct {
	emuCtrlr *Controller
}

func NewEnvironment(cmdArr []string, mapPath string) *Environment {
	env := &Environment{
		emuCtrlr: NewController(cmdArr, mapPath),
	}
	return env
}

func (env *Environment) Init() {
	env.emuCtrlr.Init()
}

func (env *Environment) GetState(state []float64) {
	env.emuCtrlr.GetState(state)
}

func (env *Environment) GameStep(action uint64) {
	env.emuCtrlr.GameStep(action)
}

func (env *Environment) LoadGame() {
	env.emuCtrlr.LoadGame()
}
