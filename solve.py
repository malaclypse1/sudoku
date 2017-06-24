#!/usr/bin/env python
"""Solves sudoku puzzles"""

#terminology:
# spot - a single digit containing grid location
# square - a 3x3 section of the sudoku

import numpy as np
import sys
from io import BytesIO, open

__author__ = "Troy Routley"
__copyright__ = "Copyright 2017, Troy Routley"
__licence__ = ""
__version__ = "0.1"

def square(grid, row, col):
    return grid[(row//3)*3:(row//3)*3+3, (col//3)*3:(col//3)*3+3]
    
def listGuesses(grid, row, col):
    return (np.argwhere(grid[row,col])+1).flatten().tolist()

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
for rowNo, row in enumerate(open("sample.txt", "r", encoding="utf-8")):
    for colNo, col in enumerate(row):
        if col == "\n":
            break
        if col.isdigit():
            puzzle[rowNo,colNo] = col
printPuzzle(puzzle)

#setup to solve
valid = np.zeros(shape=(9,9,9), dtype = np.bool_)
#initialize known values
for row in range(9):
    for col in range(9):
        val = puzzle[row][col]
        if val > 0:
            #print("{}, {}".format(row,col))
            valid[row][col][val-1] = True
        else:
            for n in range(9):
                #if n in square, row, or column valid = false, else valid = true
                if (n+1 not in square(puzzle, row, col)
                    and n+1 not in puzzle[row,:] and n+1 not in puzzle[:,col]):
                    valid[row][col][n]=True
#print(valid)
#check for known values
doubles = set()
triples = set()
changed = True
while changed:
    changed = False
    #part 1: single square of single value
    for row in range(9):
        for col in range(9):
            if (row==1 and col==1):
                print("guesses for (82) {}:{}".format((row+1,col+1),listGuesses(valid,row,col)))
                printPuzzle(puzzle)
            if puzzle[row,col]==0:
                if np.sum(valid[row,col]) == 1:
                    #print("{},{}:{}".format(row,col, np.argmax(valid[row,col])))
                    #remove value from rows,cols,squares
                    n=np.argmax(valid[row,col])
                    valid[row,0:9,n] = False
                    valid[0:9,col,n] = False
                    square(valid, row, col)[:,:,n] = False
                    valid[row,col,n] = True
                    puzzle[row,col] = n+1
                    changed = True
                    #print("\nchanged {},{} ss/sv".format(row,col))
                    #printPuzzle(puzzle)
    #part 2: unique value in square, row, or column
                for n,val in enumerate(valid[row,col]):
                    if val:
                        valid[row,col,n] = False
                        if (val not in valid[row,0:9,n]
                            or val not in valid[0:9,col,n]
                            or val not in square(valid, row, col)[:,:,n]):
                            #then val is unique
                            #remove value from rows,cols,squares
                            valid[row,0:9,n] = False
                            valid[0:9,col,n] = False
                            square(valid, row, col)[:,:,n] = False
                            valid[row,col] = False
                            #mark puzzle
                            puzzle[row,col]=n+1
                            changed = True
                            if (row == 1 and col == 1):
                                print("unique (line114) {}:{}".format((row+1,col+1,n+1),listGuesses(valid,row,col)))
                                print("{}:{}".format((8,col+1),listGuesses(valid,7,col)))
                                print("{}:{}".format((9,col+1),listGuesses(valid,8,col)))
                            #input()
                            #print("\nchanged {},{} uv".format(row,col))
                            #printPuzzle(puzzle)
                        valid[row,col,n] = True

                            
    #part 3: twin values in two spots (infinite loop...)
                if (row,col) not in doubles and np.sum(valid[row,col]) == 2:
                    #find matching 2 true spots in row
                    #print("twin check {},{}: {}".format(row,col, np.argwhere(valid[row,col]).flatten().tolist()))
                    for c in range(9):
                        if c==col:
                            continue
                        if False not in (valid[row,col] == valid[row,c]):
                            # col and c have two matching guesses, so no other
                            # spots in that row can have those
                            changed = True
                            #print("found double ({},{}) and ({},{})".format(row,col,row,c))
                            doubles.add((row,col))
                            for cf in range(9):
                                if (c==cf or col==cf):
                                    continue
                                #a guess in row,col/c -> not a guess elsewhere
                                for nf in range(9):
                                    if valid[row,col,nf]:
                                        valid[row,cf,nf] = False
                    #find matching 2 true spots in col
                    for r in range(9):
                        if r==row:
                            continue
                        if False not in (valid[row,col] == valid[r,col]):
                            # row and r have two matching guesses, so no other
                            # spots in that col can have those
                            changed = True
                            #print("found double ({},{}) and ({},{})".format(row,col,r,col))
                            doubles.add((row,col))
                            for rf in range(9):
                                if (r==rf or row==rf):
                                    continue
                                #a guess in row/r,col -> not a guess elsewhere
                                for nf in range(9):
                                    if valid[row,col,nf]:
                                        valid[rf,col,nf] = False
                    #find matching 2 true spots in square
                    for r,rval in enumerate(square(valid,row,col)):
                        for c,_ in enumerate(rval):
                            if (r==row and c==col):
                                continue
                            if False not in (valid[row,col] == valid[r,c]):
                                # square has two matching guesses, so no other
                                # spots in that square can have those
                                changed = True
                                #print("found double ({},{}) and ({},{})".format(row,col,r,c))
                                doubles.add((row,col))
                                for rf in range(row//3*3,row//3*3+3):
                                    for cf in range(col//3*3,col//3*3+3):
                                        if ((r==rf and c==cf) or (row==rf) and (col==cf)):
                                            continue
                                        #a guess in row/r,col -> not a guess elsewhere
                                        for nf in range(9):
                                            if valid[row,col,nf]:
                                                valid[rf,cf,nf] = False
    #part 4: triples
    #for triples, we need to look for combinations that can include [1,3] [2,3] [1,2]
    #so find spots with 2-3 guesses, match with spots with 2-3 guesses,
    #see if set has 3...
                if (row,col) not in triples:
                    trio = listGuesses(valid,row,col)
                    if (len(trio) > 1 and len(trio) < 4):
                        #look for row matches
                        if (row == 1 and col == 1):
                            print("new triple (line 188) {}:{}".format((row+1,col+1),listGuesses(valid,row,col)))
#                             print("{}:{}".format((8,col+1),listGuesses(valid,7,col)))
#                             print("{}:{}".format((9,col+1),listGuesses(valid,8,col)))
                        for c in range(9):
                            if c==col:
                                continue
                            guesses2 = set(listGuesses(valid,row,c))
                            if len(guesses2) < 2:
                                continue
                            guesses = set(trio) | guesses2
                            if len(guesses) < 4:
                                for c2 in range(9):
                                    if (c2==col or c2==c):
                                        continue
                                    guesses3 = set(listGuesses(valid,row,c2))
                                    if len(guesses3) < 2:
                                        continue
                                    guesses = guesses | guesses3
                                    if len(guesses) < 4:
                                        #col, c, c2 all share 3 guesses
                                        print("")
                                        printPuzzle(puzzle)
                                        print("{} common to {}:{}, {}:{}, {}:{}".format(
                                            guesses,(row+1,col+1),listGuesses(valid,row,col),
                                            (row+1,c+1),listGuesses(valid,row,c),
                                            (row+1,c2+1),listGuesses(valid,row,c2)))
                                        triples.add((row,col))
                                        changed=True
                                        for c3 in range(9):
                                            if not (c==c3 or c2==c3 or col==c3):
                                                for f in guesses:
                                                    valid[row,c3,f-1] = False
                
                        #look for col matches
                        for r in range(9):
                            if r==row:
                                continue
                            guesses2 = set(listGuesses(valid,r,col))
                            if len(guesses2) < 2:
                                continue
                            guesses = set(trio) | guesses2
                            if len(guesses) < 4:
                                for r2 in range(9):
                                    if (r2==row or r2==r):
                                        continue
                                    guesses3 = set(listGuesses(valid,r2,col))
                                    if len(guesses3) < 2:
                                        continue
                                    guesses = guesses | guesses3
                                    if len(guesses) < 4:
                                        #row, r, r2 all share 3 guesses
                                        triples.add((row,col))
                                        changed=True
                                        for r3 in range(9):
                                            if not (r==r3 or r2==r3 or row==r3):
                                                for f in guesses:
                                                    valid[r3,col,f-1] = False
    
                        #look for square matches
                        for r,rval in enumerate(square(valid,row,col)):
                            for c,val in enumerate(rval):
                                if (r==row and c==col):
                                    continue
                                guesses2 = set(listGuesses(valid,r,c))
                                if len(guesses2) < 2:
                                    continue
                                guesses = set(trio) | guesses2
                                if len(guesses) < 4:
                                    for r2,r2val in enumerate(square(valid,row,col)):
                                        for c2,_ in enumerate(r2val):
                                            if ((c2==col and r2==row) or (c2==c and r2==r)):
                                                continue
                                            guesses3 = set(listGuesses(valid,c2,c2))
                                            if len(guesses3) < 2:
                                                continue
                                            guesses = guesses | guesses3
                                            if len(guesses) < 4:
                                                #(row,col), (r,c), (r2,c2) all share 3 guesses
                                                triples.add((row,col))
                                                changed=True
                                                for r3,r3val in enumerate(square(valid,row,col)):
                                                    for c3,_ in enumerate(r3val):
                                                        if not ((c3==col and r3==row)
                                                                or (c3==c2 and r3==r2)
                                                                or (c3==c and r3==r)):
                                                            for f in guesses:
                                                                valid[r3,c3,f-1] = False

print("") 
printPuzzle(puzzle)

#dump unknowns
for row in range(9):
    for col in range(9):
        if puzzle[row,col]==0:
            print("guesses for ({},{}):{}".format(row+1,col+1,
                 listGuesses(valid,row,col)))
