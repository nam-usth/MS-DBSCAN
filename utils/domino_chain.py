#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Author       : HUYNH Vinh-Nam
# Email        : huynh-vinh.nam@usth.edu.vn
# Created Date : 11-January-2024
# Description  : 
"""
    This script demonstrates how to form large polylines from subsets of 
    smaller lines.
    
    Given that each smaller line is denoted by 2 vertices ID.
"""
#----------------------------------------------------------------------------

def merge_dominoes(dominoes):
    def can_extend(chain, domino):
        return chain[-1] in domino

    def extend_chain(chain, domino):
        if chain[-1] != domino[0]:
            domino.reverse()
        chain.extend(domino)

    merged = []

    while dominoes:
        current_chain = dominoes.pop(0)

        for domino in dominoes[:]:
            if can_extend(current_chain, domino):
                extend_chain(current_chain, domino)
                dominoes.remove(domino)

            elif can_extend(list(reversed(current_chain)), domino):
                current_chain.reverse()
                extend_chain(current_chain, domino)
                dominoes.remove(domino)

        # Remove duplicates from the current chain
        current_chain = list(dict.fromkeys(current_chain))

        merged.append(current_chain)

    return merged


# %% Main function

if __name__ == "__main__":
    # Given list of lists
    domino_list = [[28, 38], [23, 38], [58, 101], [58, 70], [89, 143], [89, 93], [81, 93], [143, 165]]

    # Call the function to merge and remove duplicates from the dominoes
    result = merge_dominoes(domino_list)

    # Print the result
    print(result)
