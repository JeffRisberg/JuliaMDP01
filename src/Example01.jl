#=
MDP:
- Julia version: 1.6.0
- Author: jeff
- Date: 2021-05-06
=#
include("common_defs.jl")
include("common_checks.jl")

const MinX = 0
const MaxX = 100
const StepX = 20

# mdp definition
mdp = MDP()

# state space
statevariable!(mdp, "x", MinX, MaxX)  # continuous
statevariable!(mdp, "goal", ["no", "yes"])  # discrete

# action space
actionvariable!(mdp, "move", ["W", "E", "stop"])  # discrete
