#!/usr/bin/env python
"""Solves sudoku puzzles"""

#terminology:
# spot or cell - a single digit containing grid location
# square - a 3x3 section of the sudoku

import numpy as np
import itertools as it
import sys
from io import BytesIO, open

__author__ = "Troy Routley"
__copyright__ = "Copyright 2017, Troy Routley"
__licence__ = ""
__version__ = "0.1"

def square(grid, row, col):
    return grid[(row//3)*3:(row//3)*3+3, (col//3)*3:(col//3)*3+3]

def matchPair(array):
    #find doubles in row/col/sqr
    # array can be square(valid,r,c), valid[r,0:9] or valid[0:9,c]
    for (a,b) in it.combinations(array.flatten(),2):
        if len(a|b) == 2:
            double = set(a|b)
            for val in np.nditer(array,flags=['refs_ok'],op_flags=['readwrite']):
                if not(val <= double):
                    val -= double
                    changed = True
                    #print("found double {}".format(double))

def matchTriple(array):
    #find triples in row/col/sqr
    # array can be square(valid,r,c), valid[r,0:9] or valid[0:9,c]
    for (a,b,c) in it.combinations(array.flatten(),3):
        if len(a|b|c) == 3:
            triple = set(a|b|c)
            for val in np.nditer(array,flags=['refs_ok'],op_flags=['readwrite']):
                if not(val <= triple):
                    val -= triple
                    changed = True
                    #print("found triple {}".format(triple))

def matchQuad(array):
    #find Quads in row/col/sqr
    # array can be square(valid,r,c), valid[r,0:9] or valid[0:9,c]
    for (a,b,c,d) in it.combinations(array.flatten(),4):
        if len(a|b|c|d) == 4:
            quad = set(a|b|c|d)
            for val in np.nditer(array,flags=['refs_ok'],op_flags=['readwrite']):
                if not(val <= quad):
                    val -= quad
                    changed = True
                    #print("found quad {}".format(quad))
                    
def printPuzzle(puzzle):
    squareR = 0
    while squareR < 9:
        for row in range(squareR,squareR+3):
            squareC = 0
            #print("row:{} ".format(row), end="")
            while squareC < 9:
                for col in range(squareC,squareC+3):
                    if puzzle[row][col] > 0:
                        #print("[{}][{}]:{}".format(row,col,puzzle[row][col]), end="")
                        print(puzzle[row][col], end="")
                    else:
                        print(" ", end="")
                #print("sc{}sc".format(squareC), end="")
                if squareC < 6:
                    print("|", end="")
                else:
                    print("")
                squareC += 3
        if squareR < 6:
            print("---+---+---")
        squareR += 3
        
#input
puzzle = np.zeros(shape=(9,9), dtype = np.int8)
for rowNo, row in enumerate(open(0, "r", encoding="utf-8")):
    for colNo, col in enumerate(row):
        if col == "\n":
            break
        if col.isdigit():
            puzzle[rowNo,colNo] = col
printPuzzle(puzzle)

#setup to solve
valid = np.empty((9,9), dtype = object)
#using np.full() to initialize as a set links all values to the same set, so we
#need to loop to create new sets for each cell
for (row,col),_ in np.ndenumerate(valid):
    valid[row,col]=set()

#initialize known values
for (row,col),val in np.ndenumerate(puzzle):
    if val > 0:
        #print("{}, {}".format(row,col))
        valid[row,col].add(val)
    else:
        for n in range(1,10):
            #if n not in square, row, or column add to guess set
            if (n not in square(puzzle, row, col)
                and n not in puzzle[row,:] and n not in puzzle[:,col]):
                valid[row,col].add(n)
#print(valid)
#check for known values

changed = True
while changed:
    changed = False
    #part 1: single square of single value
    for row,col in np.ndindex(9,9):
        if puzzle[row,col]==0:
            if len(valid[row,col]) == 1:
                #remove value from rows,cols,squares
                n=valid[row,col]
                valid[row,0:9] -= n
                valid[0:9,col] -= n
                square(valid, row, col)[:,:] -= n
                #after removing guess from rows,cols,squares put it back in to cell
                valid[row,col] |= n
                puzzle[row,col] = list(n)[0]
                changed = True
                #print("\nchanged {},{} ss/sv".format(row,col))
                #printPuzzle(puzzle)
#part 2: unique value in square, row, or column
            #reminder: only looking at unsolved cells
            uRow = valid[row,col].copy()
            uCol = valid[row,col].copy()
            uSqr = valid[row,col].copy()
            for c in range(9):
                if col != c:
                    uRow -= valid[row,c]
            for r in range(9):
                if row != r:
                    uCol -= valid[r,col]
            for r in range(row//3*3,row//3*3+3):
                for c in range(col//3*3,col//3*3+3):
                    if (row != r) or (col != c):
                        uSqr -= valid[r,c]
            #unique in row
            if len(uRow) == 1:
                puzzle[row,col]=list(uRow)[0]
                valid[row,col]=uRow.copy()
                #remove from column and square
                for r in range(9):
                    if row != r: #don't remove from cell
                        valid[r,col] -= uRow
                for r in range(row//3*3,row//3*3+3):
                    if row != r: #don't remove from column in square - already did that
                        for c in range(col//3*3,col//3*3+3):
                            if col != c: #don't remove from row in square - known absent
                                valid[r,c] -= uRow
                changed = True
            #unique in column
            if len(uCol) == 1:
                puzzle[row,col]=list(uCol)[0]
                valid[row,col]=uCol.copy()
                #remove from column and square
                for c in range(9):
                    if col != c: #don't remove from cell
                        valid[row,c] -= uCol
                for r in range(row//3*3,row//3*3+3):
                    if row != r: #don't remove from column in square - known absent
                        for c in range(col//3*3,col//3*3+3):
                            if col != c: #don't remove from row in square - already did
                                valid[r,c] -= uCol
                changed = True
            #unique in square
            if len(uSqr) == 1:
                puzzle[row,col]=list(uSqr)[0]
                valid[row,col]=uSqr.copy()
                #remove from column and row
                for r in range(9):
                    if row != r: #don't remove from cell
                        valid[r,col] -= uCol
                for c in range(9):
                    if col != c: #don't remove from cell
                        valid[row,c] -= uCol
                changed = True

#part 3: if a col/row/sqr has 2 cells that have 2 guesses, those guesses can't appear
#   elsewhere in the col/row/sqr
# 27   27   273 -> 27   27  3
#   similarly, for 3 cells with 3 guesses
# 273  27   37  31 -> 273  27  37  1
#   if 2 cells are the only places those guesses appear, those cells can't have other
#   guesses
# 237  247  3  1  45 89 13 56 13 -> 27 27  3  1  45 89 13 56 13

    for n in range(9):
        matchPair(valid[n,0:9])
        matchPair(valid[0:9,n])
    for r in range(0,9,3):
        for c in range(0,9,3):
            matchPair(square(valid,r,c))  

#part 4: triples
#for triples, we need to look for combinations that can include [1,3] [2,3] [1,2]
#so find spots with 2-3 guesses, match with spots with 2-3 guesses,
#see if set has 3...
    for n in range(9):
        matchTriple(valid[n,0:9])
        matchTriple(valid[0:9,n])
    for r in range(0,9,3):
        for c in range(0,9,3):
            matchTriple(square(valid,r,c))
            
    for n in range(9):
        matchQuad(valid[n,0:9])
        matchQuad(valid[0:9,n])
    for r in range(0,9,3):
        for c in range(0,9,3):
            matchQuad(square(valid,r,c))


print("") 
printPuzzle(puzzle)

#dump unknowns
for row in range(9):
    for col in range(9):
        if puzzle[row,col]==0:
            print("guesses for ({},{}):{}".format(row+1,col+1,
                 valid[row,col]))
