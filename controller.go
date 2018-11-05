package main

import (
	"bufio"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"math"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"syscall"
	"time"

	"github.com/micmonay/keybd_event"
)

//Controller - launches and interacts with the emulator as a debugger
type Controller struct {
	cmd            *exec.Cmd
	inPlugAddr     uint64
	libmupenAddr   uint64
	bpOffset       uint64
	bpAddr         uintptr
	stateAddrSlice []string
	StateAddrMap   map[string]StateValue
	origByte       []byte
	KeyBond        keybd_event.KeyBonding
}

//StateValue - contains the memory address and the data type of a state var
type StateValue struct {
	Addr string
	Type string
}

//JsonMessage - used for unpacking state info from json file
type JsonMessage struct {
	Name string `json:"name"`
	Addr string `json:"address"`
	Type string `json:"type"`
}

//NewController - controller constructor
func NewController(cmdArr []string, mapPath string) *Controller {
	jsonfile, err := ioutil.ReadFile(mapPath)
	if err != nil {
		panic(err)
	}

	jsonmsgs := []JsonMessage{}
	json.Unmarshal(jsonfile, &jsonmsgs)

	var addrslice []string
	addrmap := make(map[string]StateValue)

	for _, v := range jsonmsgs {
		addrslice = append(addrslice, v.Name)
		addrmap[v.Name] = StateValue{v.Addr, v.Type}
	}

	kb, err := keybd_event.NewKeyBonding()
	if err != nil {
		panic(err)
	}

	emuCtrlr := &Controller{
		cmd:            exec.Command(cmdArr[0], cmdArr[1:]...),
		inPlugAddr:     uint64(0x00),
		libmupenAddr:   uint64(0x00),
		bpOffset:       uint64(0x35cd),
		bpAddr:         uintptr(0x00),
		stateAddrSlice: addrslice,
		StateAddrMap:   addrmap,
		KeyBond:        kb,
		origByte:       make([]byte, 1, 1),
	}

	return emuCtrlr
}

//Init - initialize the emulator and key bindings
func (cont *Controller) Init() {
	cont.cmd.Start()
	time.Sleep(2 * time.Second)
	pid := cont.cmd.Process.Pid
	pAttach(pid)
	sysWait(pid)

	cont.inPlugAddr = getInputAddr(pid)
	cont.libmupenAddr = getLibmupenAddr(pid)
	cont.bpAddr = uintptr(cont.bpOffset + cont.inPlugAddr)
	setBreakpoint(pid, cont.bpAddr, cont.origByte)

	kb, err := keybd_event.NewKeyBonding()
	if err != nil {
		panic(err)
	}

	cont.KeyBond = kb

	time.Sleep(2 * time.Second)

	cont.KeyBond.SetKeys(keybd_event.VK_1, keybd_event.VK_F7)
	err = cont.KeyBond.Launching()
	if err != nil {
		panic(err)
	}
}

//GetState - get the values in memory for all state vars
func (cont *Controller) GetState(state []float32) {
	for i, v := range cont.stateAddrSlice {
		state[i] = cont.ReadStateVal(cont.cmd.Process.Pid, cont.StateAddrMap[v])
	}
}

//ReadStateVal - read the memory for a given state var
func (cont *Controller) ReadStateVal(pid int, v StateValue) float32 {
	addrSlice, err := hex.DecodeString(v.Addr[2:])
	addr := uint64(binary.BigEndian.Uint32(addrSlice)) + cont.libmupenAddr
	if err != nil {
		panic(err)
	}
	if v.Type == "float32" {
		return float32(peekFloat32(pid, uintptr(addr)))
	} else if v.Type == "uint16" {
		return float32(peekuInt16(pid, uintptr(addr)))
	}
	return 0.0
}

//GameStep - Set the action and run the game for one frame
func (cont *Controller) GameStep(action uint64) {
	//run one frame
	pCont(cont.cmd.Process.Pid)

	//wait for breakpoint
	sysWait(cont.cmd.Process.Pid)

	//replace int3 instruction with mov
	clearBreakpoint(cont.cmd.Process.Pid, cont.bpAddr, cont.origByte)

	//set rax register with controller input
	pSetRax(cont.cmd.Process.Pid, action)

	//reset pc
	pSetPC(cont.cmd.Process.Pid, uint64(cont.bpAddr))

	//single step
	pStep(cont.cmd.Process.Pid)

	//wait for step to finish
	sysWait(cont.cmd.Process.Pid)

	//replace interrupt
	setBreakpoint(cont.cmd.Process.Pid, cont.bpAddr, cont.origByte)
}

//GameStep - Run the game one frame accepting user input
func (cont *Controller) GameStepTrain() uint64 {
	//run one frame
	pCont(cont.cmd.Process.Pid)

	//wait for breakpoint
	sysWait(cont.cmd.Process.Pid)

	//replace int3 instruction with mov
	clearBreakpoint(cont.cmd.Process.Pid, cont.bpAddr, cont.origByte)

	//get rax register for user input
	action := pGetRax(cont.cmd.Process.Pid)

	//reset pc
	pSetPC(cont.cmd.Process.Pid, uint64(cont.bpAddr))

	//single step
	pStep(cont.cmd.Process.Pid)

	//wait for step to finish
	sysWait(cont.cmd.Process.Pid)

	//replace interrupt
	setBreakpoint(cont.cmd.Process.Pid, cont.bpAddr, cont.origByte)

	return action
}

//LoadGame - Load the save-state
func (cont *Controller) LoadGame() {
	err := cont.KeyBond.Launching()
	if err != nil {
		panic(err)
	}
}

func setBreakpoint(pid int, breakpoint uintptr, original []byte) {
	_, err := syscall.PtracePeekData(pid, breakpoint, original)
	if err != nil {
		panic(err)
	}
	_, err = syscall.PtracePokeData(pid, breakpoint, []byte{0xCC})
	if err != nil {
		panic(err)
	}
}

func getInputAddr(pid int) uint64 {
	f, err := os.Open(fmt.Sprintf("/proc/%d/maps", pid))
	if err != nil {
		panic(err)
	}

	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		if strings.Contains(scanner.Text(), "input") {
			strInLine := strings.Split(scanner.Text(), "-")
			addr, err := strconv.ParseInt(strInLine[0], 16, 64)
			if err != nil {
				panic(err)
			}
			return uint64(addr)
		}
	}
	return uint64(0)
}

func getLibmupenAddr(pid int) uint64 {
        f, err := os.Open(fmt.Sprintf("/proc/%d/maps", pid))
        if err != nil {
                panic(err)
        }

        scanner := bufio.NewScanner(f)
        for scanner.Scan() {
                if strings.Contains(scanner.Text(), "libmupen") {
                        strInLine := strings.Split(scanner.Text(), "-")
                        addr, err := strconv.ParseInt(strInLine[0], 16, 64)
                        if err != nil {
                                panic (err)
                        }
                        return uint64(addr)
                }
        }
        return uint64(0)
}

func pStep(pid int) {
	err := syscall.PtraceSingleStep(pid)
	if err != nil {
		panic(err)
	}
}

func pGetRax(pid int) uint64 {
	var regs syscall.PtraceRegs
	err := syscall.PtraceGetRegs(pid, &regs)
	if err != nil {
		panic(err)
	}
	return regs.Rax
}

func pSetRax(pid int, cInput uint64) {
	var regs syscall.PtraceRegs
	err := syscall.PtraceGetRegs(pid, &regs)
	if err != nil {
		panic(err)
	}
	regs.Rax = cInput
	err = syscall.PtraceSetRegs(pid, &regs)
	if err != nil {
		panic(err)
	}
}

func pSetPC(pid int, pc uint64) {
	var regs syscall.PtraceRegs
	err := syscall.PtraceGetRegs(pid, &regs)
	if err != nil {
		panic(err)
	}
	regs.SetPC(pc)
	err = syscall.PtraceSetRegs(pid, &regs)
	if err != nil {
		panic(err)
	}
}

func pAttach(pid int) {
	err := syscall.PtraceAttach(pid)
	if err != nil {
		panic(err)
	}
}

func pDetach(pid int) {
	err := syscall.PtraceDetach(pid)
	if err != nil {
		panic(err)
	}
}

func sysWait(pid int) {
	_, err := syscall.Wait4(-1, nil, syscall.WALL, nil)
	if err != nil {
		panic(err)
	}
}

func clearBreakpoint(pid int, breakpoint uintptr, original []byte) {
	_, err := syscall.PtracePokeData(pid, breakpoint, original)
	if err != nil {
		panic(err)
	}
}

func pCont(pid int) {
	err := syscall.PtraceCont(pid, 0)
	if err != nil {
		panic(err)
	}
}

func peekuInt16(pid int, addr uintptr) uint16 {
	d := make([]byte, 2)
	_, err := syscall.PtracePeekData(pid, addr, d)
	if err != nil {
		panic(err)
	}
	return bytes2uint16(d)
}

func bytes2uint16(bytes []byte) uint16 {
	return binary.LittleEndian.Uint16(bytes)
}

func peekFloat32(pid int, addr uintptr) float32 {
	d := make([]byte, 4)
	_, err := syscall.PtracePeekData(pid, addr, d)
	if err != nil {
		panic(err)
	}
	return bytes2Float32(d)
}

func bytes2Float32(bytes []byte) float32 {
	bits := binary.LittleEndian.Uint32(bytes)
	float := math.Float32frombits(bits)
	return float
}
