#=
MDP:
- Julia version: 1.6.0
- Author: jeff
- Date: 2021-05-06
=#
using LightGraphs
import Cairo, Fontconfig

g = SimpleGraph(7);
println(g)

my_nodes = ["Bob", "John", "Bill", "Helen", "Jack", "Thomas", "Sally"]
println(typeof(my_nodes))
my_colors = ["#8080FF", "yellow", "goldenrod", "thistle", "wheat", "#80FF80", "#FF8080"]

b = Dict("Bob" => 4, "Jack" => 5)
println(typeof(b))

add_edge!(g, 1, 2);

add_edge!(g, 2, 3);

add_edge!(g, 3, 4);

add_edge!(g, 4, 5);

add_edge!(g, 5, 1);

add_edge!(g, 5, 6)

add_edge!(g, 3, 7)

h = SimpleGraphFromIterator(edges(g));

using GraphPlot, Compose, Colors

nodelabel = my_nodes

nodecolor = [parse(Colorant, x) for x in my_colors]

# nodesize = [(3 * LightGraphs.outdegree(g, v)) for v in LightGraphs.vertices(g)]
nodesize = [140,200,280,240,180,260,300]

# The LightGraph code will attempt to place the nodes of the network
# but there will be random changes each time.
draw(PNG("lightgraph.png", 10cm, 10cm), gplot(g, nodelabel=nodelabel, nodefillc=nodecolor, nodesize=nodesize))

