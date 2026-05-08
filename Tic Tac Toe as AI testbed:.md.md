Tic Tac Toe as AI testbed:

Wanting to learn more about AI/ML concepts and linear algebra I chose Tic-Tac-Toe as a simple testbed for understanding how it works under the hood.

In this first installment I'll discuss the initial game buildout in python and two 'hand hacked' models.

Tic-Tac-Toe is a good platform for this kind of learning because the problem space is quite small and it is easy to reason about.

Consider the empty board:
[Pic of empty playfield goes here]

Since the goal is to make 3 of your marks in a row, a couple of things become obvious:

1. The middle square is the most valuable on the board because it can contribute to 4 different winning lines 
[Pic of playfield with arrows from center showing winning lines. Center square should be dark blue all others white]
2. The corner spaces each contribute to 3 different winning lines 
[Pic showing the 3 winning lines from a single corner the corner we are showing the lines from should be dark blue, the other 3 corners should be light blue all other squares white]
4. The remaining squares are the least valuable on a blank board as they can only contribute to two different winning lines. 
[Pic showing the 2 winning lines from one of the non center/non corner squares - our square of interest should be dark blue, all other '2 line' squares should be light blue, the rest should be white]

This shows the value of each square as a first move - but the question is - what about a game that is underway?

The heuristic model I put together has three phases:
1. Each square gets a base value equal to the number of winning lines it contributes to.
[Picture of field with base values printed on them; center = 4, corners = 3, all others = 2]

2. Next we look to for lines that if we play there we get an instant win. If any such squares are found, their value is incremented by 1000

3. Finally we look for lines that will result in an opponent win if we don't block by playing this square first- if any such squares are found their value is incremented by 900. 

This heuristic model wins handily against a totally random model where square values are randomly generated and it just plays the unoccupied square with the highest score - but there are concepts like forks that the model doesn't understand since it only looks 1 move ahead. This is likely why the random model manages to win once in a blue moon.

[Pic - Example of a fork]

A playable game with two 'CPU' models forms the basis for further exploration of AI/ML concepts.

Part 2. Training Day - matching the heuristic model