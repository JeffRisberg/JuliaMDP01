#=
MDP:
- Julia version: 1.6.0
- Author: jeff
- Date: 2021-05-06
=#
include("common_defs.jl")
include("common_checks.jl")
include("valueiteration.jl")

# mdp definition
mdp = MDP()

# state space
statevariable!(mdp, "x", 1.0, 4.0)
statevariable!(mdp, "y", 1.0, 3.0)

# action space
actionvariable!(mdp, "move", ["U", "D", "L", "R"])  # discrete

function is_valid_state(state::Tuple{Vector{Float64}, Float64})
    if state[1][1] == 2.0 && state[1][2] == 2.0
        return false
    end
    return state[1][1] > 0 && state[1][1] < 5 && state[1][2] > 0 && state[1][2] < 4
end

function mytransition(x::Float64, y::Float64, move::AbstractString)
  if x == 4.0 && y == 2.0
    result = [([x, y], 0.0)]
  elseif x == 4.0 && y == 3.0
    result = [([x, y], 0.0)]
  elseif move == "L"
    if x > 1.0
      result = [([x - 1.0, y], 0.8), ([x, y - 1.0], 0.1), ([x, y + 1.0], 0.1)]
    else
      result = [([x, y], 1.0)]
    end
  elseif move == "R"
    if x < 4.0
      result = [([x + 1.0, y], 0.8), ([x, y - 1.0], 0.1), ([x, y + 1.0], 0.1)]
    else
      result = [([x, y], 1.0)]
    end
  elseif move == "U"
    if y < 3.0
      result = [([x, y + 1.0], 0.8), ([x - 1.0, y], 0.1), ([x + 1.0, y], 0.1)]
    else
      result = [([x, y], 1.0)]
    end
  elseif move == "D"
    if y > 1.0
      result = [([x, y - 1.0], 0.8), ([x - 1.0, y], 0.1), ([x + 1.0, y], 0.1)]
    else
      result = [([x, y], 1.0)]
    end
  end
  filtered_result = filter(is_valid_state, result)
  return filtered_result
end

transition!(mdp, ["x", "y", "move"], mytransition)

function myreward(x::Float64, y::Float64, move::AbstractString)
  if x == 4.0 && y == 3.0
    return 1.0
  elseif x == 4.0 && y == 2.0
    return -1.0
  else
    return 0.0
  end
end

reward!(mdp, ["x", "y", "move"], myreward)

solver = SerialValueIteration()

discretize_statevariable!(solver, "x", 1.0)
discretize_statevariable!(solver, "y", 1.0)

lazyCheck(mdp, solver)

statespace, actionspace = getspaces(mdp, solver)
solver.stategrid = RectangleGrid(statespace...)
println(solver.stategrid)
solver.actiongrid = RectangleGrid(actionspace...)
println(solver.actiongrid)

println(length(mdp.transition.argnames) == length(mdp.statemap) + length(mdp.actionmap))

function solveset2(mdp::MDP, svi::SerialValueIteration)

  statedim = length(mdp.statemap)
  stateargs = mdp.reward.argnames[1:statedim]
  actionargs = mdp.reward.argnames[1 + statedim:end]

  nstates = length(svi.stategrid)
  nactions = length(svi.actiongrid)

  vold = zeros(nstates)
  vnew = zeros(nstates)

  qval = zeros(nactions, nstates)
  resid = 0.0

  iter = 0
  itertime = 0.0
  cputime = 0.0

  for i in 1:svi.maxiter
    vnew = zeros(nstates)
    #tic()
    resid = -Inf

    for istate in 1:nstates

      state = getvar(svi.stategrid, mdp.statemap, stateargs, istate)
      vmax = -Inf

      for iaction in 1:nactions

        action = getvar(svi.actiongrid, mdp.actionmap, actionargs, iaction)

        statepIdxs, probs = transition(mdp, svi, state, action, stateargs)
        v = 0.0

        for istatep in 1:length(statepIdxs)
          v += probs[istatep] * vold[statepIdxs[istatep]]
        end

        if v > vmax
          vmax = v
        end
      end

      vmax *= svi.discount
      vmax += reward(mdp, state, action)

      vnew[istate] = vmax

      # use infinity-norm
      newresid = (vold[istate] - vnew[istate])^2
      newresid > resid ? resid = newresid : nothing
    end

    #itertime = toq()
    #cputime += itertime

    if svi.verbose
      println(string("iter $iter, resid: $resid"))
    end

    resid < svi.tol ? break : nothing

    vold = copy(vnew)
    iter = i
  end

  if iter == svi.maxiter
    @warn(string(
      "maximum number of iterations reached; check accuracy of solutions"))
  end

  @info(string(
    "value iteration solution generated\n",
    "cputime [s] = ", cputime, "\n",
    "number of iterations = ", iter, "\n",
    "residual = ", resid))

  return ValueIterationSolution(
    qval,
    vnew,
    svi.stategrid,
    svi.actiongrid,
    cputime,
    iter,
    resid)
end

function getpolicy2(mdp::MDP, svi::SerialValueIteration, solution::ValueIterationSolution)
  statedim = length(mdp.statemap)
  stateargs = mdp.reward.argnames[1:statedim]
  actionargs = mdp.reward.argnames[1 + statedim:end]

  nstates = length(svi.stategrid)
  nactions = length(svi.actiongrid)

  local pi::Dict = Dict();

  for istate in 1:nstates

      state = getvar(svi.stategrid, mdp.statemap, stateargs, istate)
      vmax = -Inf

      for iaction in 1:nactions
        action = getvar(svi.actiongrid, mdp.actionmap, actionargs, iaction)
        statepIdxs, probs = transition(mdp, svi, state, action, stateargs)

        v = 0.0

        for istatep in 1:length(statepIdxs)
          v += probs[istatep] * solution.v[statepIdxs[istatep]]
        end

        if v > vmax
          vmax = v
          pi[istate] = action
        end
      end
  end
  return pi;
end

statedim = length(mdp.statemap)
stateargs = mdp.reward.argnames[1:statedim]
actionargs = mdp.reward.argnames[1 + statedim:end]

istate = 1
state = getvar(solver.stategrid, mdp.statemap, stateargs, istate)

iaction = 4
action = getvar(solver.actiongrid, mdp.actionmap, actionargs, iaction)

statepIdxs, probs = transition(mdp, solver, state, action, stateargs)
println(statepIdxs)
println(probs)


istate = 8
state = getvar(solver.stategrid, mdp.statemap, stateargs, istate)
iaction = 4
action = getvar(solver.actiongrid, mdp.actionmap, actionargs, iaction)
@info ("reward for state 8:", reward(mdp, state, action))

istate = 12
state = getvar(solver.stategrid, mdp.statemap, stateargs, istate)
iaction = 4
action = getvar(solver.actiongrid, mdp.actionmap, actionargs, iaction)
@info ("reward for state 12:", reward(mdp, state, action))

solver.discount = 0.9
solver.maxiter = 100

solution = solveset2(mdp, solver)
println(solution.v)

policy = getpolicy2(mdp, solver, solution)
println(policy)
