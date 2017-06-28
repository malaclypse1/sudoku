# sudoku
# this seems to work well on 5 star puzzles
# sample3.txt is an 11 star puzzle from
# http://www.telegraph.co.uk/news/science/science-news/9359579/Worlds-hardest-sudoku-can-you-crack-it.html
# to solve it we will need to implement guessing
#	the plan:
#	when no more changes, but puzzle not solved:
#		make a guess, recursively solve puzzle with guess in place
#			if solved, print solution and exit
#			otherwise, return unsolved and make a different guess
