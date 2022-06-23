from threading import Thread
from time import sleep
from datetime import datetime

import numpy as np

# position constant
LONG = 0
SHORT = 1
FLAT = 2
POSITIONS = ["LONG", "SHORT", "FLAT"]

# Possible signals
SIGNAL = ['HOLD', 'BUY', 'SELL']

# action constant
HOLD = 0
BUY = 1
SELL = 2

# Latency in milliseconds
LATENCY = 100

# Position params
TRADING_FEES = 1  # in bps

# Max retries count
MAX_PATIENCE = 3


def write_to_out(mess, logger):
    logger.write_info(mess)
    print(f'{datetime.now().strftime("%d/%m/%Y %H:%M:%S.%f")} | {mess}')


class OrderProcessor(Thread):
    def __init__(self, logger, order_queue, position_globals, action, depth=None, pos_size=None, observation=None):
        Thread.__init__(self)
        self.logger = logger
        self.order_queue = order_queue
        self.position_globals = position_globals
        self.action = action
        self.pos_size = pos_size
        self.observation = observation
        self.fee = TRADING_FEES / 10000
        self.reward = 0
        self.patience = 0
        self.depth = depth
        self.position_globals.order_counter += 1

        # Make a copy of position globals
        self.tmp_position = self.position_globals.position
        self.tmp_prev_position = self.position_globals.prev_position
        self.tmp_entry_price = self.position_globals.entry_price
        self.tmp_cum_return = self.position_globals.cum_return
        self.tmp_n_long = self.position_globals.n_long
        self.tmp_n_short = self.position_globals.n_short
        self.tmp_order_counter = self.position_globals.order_counter

        # halt_exec indicates if there has been a state update in the main thread. Had a new state arrived
        # update process should not be done to the global position because order will be placed with outdate quantities
        # and prices.
        self.halt_exec = False

    def run(self):
        self.evaluate_position(self.action, self.observation, self.pos_size)

    def getting_impatient(self):
        self.patience += 1
        if self.patience > MAX_PATIENCE:
            self.halt_exec = True
            write_to_out(
                f'Order ID: {self.position_globals.order_counter} 'f' SIGNAL: {SIGNAL[self.action]}. Discarded.',
                self.logger
            )

    def evaluate_position(self, action, observation, pos_size):
        # action comes from the agent
        # 0 buy, 1 sell, 2 hold
        # New 0 hold, 1 Buy, 2 Sell
        # single position can be opened per trade
        # valid action sequence would be
        # LONG : buy - hold - hold - sell
        # SHORT : sell - hold - hold - buy
        # invalid action sequence is just considered hold
        # (e.g.) "buy - buy" would be considered "buy - hold"
        self.action = HOLD  # hold
        residual = 0

        if action == BUY:  # buy
            if self.tmp_position == FLAT:  # if previous position was flat
                self.tmp_position = LONG  # update position to long
                residual, prices, quantities = self.match_position(pos_size, observation)
                self.execute_trade(prices, quantities)
                self.tmp_entry_price = np.average(prices, weights=quantities)
                self.reward = 0
            elif self.tmp_position == SHORT:  # if previous position was short
                self.tmp_position = FLAT  # update position to flat
                residual, prices, quantities = self.match_position(pos_size, observation)
                self.execute_trade(prices, quantities)
                self.exit_price = np.average(prices, weights=quantities)
                self.reward = (
                        ((self.tmp_entry_price - self.exit_price) / self.exit_price + 1)
                        * (1 - self.fee) ** 2
                        - 1
                )
                self.tmp_cum_return += self.reward
                self.tmp_entry_price = 0  # clear entry price
                self.tmp_n_short += 1  # record number of short"""
        elif action == SELL:
            # vice versa for short trade
            if self.tmp_position == FLAT:
                self.tmp_position = SHORT
                residual, prices, quantities = self.match_position(pos_size, observation, order_type=SELL)
                self.execute_trade(prices, quantities)
                self.tmp_entry_price = np.average(prices, weights=quantities)
                self.reward = 0
            elif self.tmp_position == LONG:
                self.tmp_position = FLAT
                residual, prices, quantities = self.match_position(pos_size, observation, order_type=SELL)
                self.execute_trade(prices, quantities)
                self.exit_price = np.average(prices, weights=quantities)
                self.reward = (
                    (self.exit_price - self.tmp_entry_price) / self.tmp_entry_price + 1
                ) * (1 - self.fee) ** 2 - 1
                self.tmp_cum_return += self.reward
                self.tmp_entry_price = 0
                self.tmp_n_long += 1

        if not self.halt_exec and action != HOLD:
            if residual != 0:
                # Enqueue not fully executed orders
                write_to_out('Order could not be fully matched. Target volume higher than total available.', self.logger)
                write_to_out(f'Order ID: {self.position_globals.order_counter} RESIDUAL: {residual} enqueued', self.logger)
                self.order_queue.put({'ACTION': action,
                                      'POSITION_SIZE': residual,
                                      'DEPTH': self.depth})

            # Update global state
            self.position_globals.update_state(
                self.tmp_prev_position,
                self.tmp_position,
                self.tmp_cum_return,
                self.tmp_n_long,
                self.tmp_n_short,
                self.tmp_entry_price,
                self.tmp_order_counter,
                self.action
            )
            write_to_out(
                f'                                                            '
                f'CUMULATIVE REWARD: {self.position_globals.cum_return}.',
                self.logger
            )

    def match_position(self, pos_size, lob_state, order_type=BUY):
        target_volume = pos_size
        step_size = 4
        quantities = []
        prices = []
        # Padding parameter indicates the way we will move inside the state loop.
        # lob_state = [BID_PRICE1, BID_Q1, ASK_PRICE1, ASK_Q1, ... , BID_PRICE10, BID_Q10, ASK_PRICE10, ASK_Q10]
        # A padding of 2 is used whenever a BUY order is placed. We will retrieve only ASK elements of the list.
        # We want to immediately execute the orders, hence, bid offers will be placed, matching the best available
        # ask prices to achieve an immediate execution.
        padding = 2
        # padding param indicates the slide that will be carried out at index level depending of bid or ask.
        if order_type != BUY:
            padding = 0
        for index in range(0, len(lob_state), step_size):
            quant = float(lob_state[index + padding + 1])
            price = float(lob_state[index + padding])
            if target_volume <= quant:
                quantities += [quant]
                prices += [price]
                return 0, prices, quantities
            else:
                quantities += [quant]
                prices += [price]
                target_volume -= quant
        return target_volume, prices, quantities

    def execute_trade(self, prices, quantities):
        # We don't really execute any trade here. For further implementations order would have to be executed
        sleep(LATENCY / 1000)
        if not self.halt_exec:
            write_to_out(
                f'Order ID: {self.position_globals.order_counter} executed',
                self.logger
            )
            write_to_out(f'Entered into {POSITIONS[self.tmp_position]} position.', self.logger)
