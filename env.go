package main

//Environment - the intention is to wrap the Controller class
//              should someone want to implement a controller for another emulator
type Environment struct {
	emuCtrlr *Controller
}

//NewEnvironment - Environment constructor
func NewEnvironment(cmdArr []string, mapPath string) *Environment {
	env := &Environment{
		emuCtrlr: NewController(cmdArr, mapPath),
	}
	return env
}

//Init - init the env
func (env *Environment) Init() {
	env.emuCtrlr.Init()
}

//GetState - get game state
func (env *Environment) GetState(state []float64) {
	env.emuCtrlr.GetState(state)
}

//GameStep - step the game a single input frame
func (env *Environment) GameStep(action uint64) {
	env.emuCtrlr.GameStep(action)
}

//LoadGame - load saved state
func (env *Environment) LoadGame() {
	env.emuCtrlr.LoadGame()
}
