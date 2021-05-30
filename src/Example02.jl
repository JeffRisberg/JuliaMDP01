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

function mytransition(x::Float64, y::Float64, move::AbstractString)
  if move == "L"
    if x > 1.0
      return [([x - 1.0, y], 1.0)]
    else
      return [([x, y], 1.0)]
    end
  elseif move == "R"
    if x < 4.0
      return [([x + 1.0, y], 1.0)]
    else
      return [([x, y], 1.0)]
    end
  elseif move == "U"
    if y > 1.0
      return [([x, y - 1.0], 1.0)]
    else
      return [([x, y], 1.0)]
    end
  elseif move == "D"
    if y < 3.0
      return [([x, y + 1.0], 1.0)]
    else
      return [([x, y], 1.0)]
    end
  end
end

transition!(mdp, ["x", "y", "move"], mytransition)

function myreward(x::Float64, y::Float64, move::AbstractString)
  if x == 4 && y == 3
    return 1
  elseif x == 4 && y == 2
    return -1
  else
    return 0
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

function solveset(mdp::MDP, svi::SerialValueIteration)

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

    #tic()
    resid = -Inf

    for istate in 1:nstates

      state = getvar(svi.stategrid, mdp.statemap, stateargs, istate)
      qhi = -Inf

      for iaction in 1:nactions

        action = getvar(svi.actiongrid, mdp.actionmap, actionargs, iaction)

        statepIdxs, probs = transition(mdp, svi, state, action, stateargs)
        qnow = 0.0

        for istatep in 1:length(statepIdxs)
          qnow += probs[istatep] * vold[statepIdxs[istatep]]
        end

        qnow *= svi.discount
        qnow += reward(mdp, state, action)

        qval[iaction, istate] = qnow

        if qnow > qhi
          qhi = qnow
          vnew[istate] = qhi
        end
      end

      # use infinity-norm
      newresid = (vold[istate] - vnew[istate])^2
      newresid > resid ? resid = newresid : nothing

    end

    #itertime = toq()
    #cputime += itertime

    if svi.verbose
      println(string("iter $iter, resid: $resid"))
      println()
      println()
    end

    resid < svi.tol ? break : nothing

    vtmp = vold
    vold = vnew
    vnew = vtmp
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
    svi.stategrid,
    svi.actiongrid,
    cputime,
    iter,
    resid)
end

solution = solveset(mdp, solver)
println(solution)

policy = getpolicy(mdp, solution)
println(policy(1))
