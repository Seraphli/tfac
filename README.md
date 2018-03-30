# TF AC

Package for accelerate tensorflow input pipeline using tensorflow queue

Example are provide [here](example/mlp.py)

## Why build this package

Because feed_dict in tensorflow is too slow,
and for reinforcement learning research,
you have to feed new data while running.
So it's hard to avoid using feed_dict.

Using queue will improve performance significantly.

## Installation

Clone this repository and use `pip install .`
