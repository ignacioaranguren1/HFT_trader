import pandas as pd
import numpy as np
import pickle
import os
import CNN_keras_specification
import datetime
import logging

from time import time
from logger.trader_logger import TraderLogger
from queue import Queue
from utils import prepare_x_y, load_df, create_queue_from_delta_series
from order_processor import OrderProcessor
from position_globals import PosistionGlobals

# Path to test working files
TEST_PATH = '../data/sets/test_set_k_20.csv'
TEST_PATH_UNNORMALIZED = '../data/sets/unnormalized/test_set_k_20.csv'
MODEL_PATH = '../models/cnn/model_CNN_refitted_k_20.pkl'

# Reduced df ratio
REDUCTION_FACTOR = 0.7

# Number of inputs of the CNN
N_INPUTS = 20

# Time unit
TIME_UNIT = 1000

# Set a time scaling factor to reduce the duration of the test
SCALING_FACTOR = 1

# Possible signals
SIGNAL = ['HOLD', 'BUY', 'SELL']

# Position params
POSITION_SIZE = 1  # we assume a constant trade of POSTION_SIZE whenever a trade is executed

#Acceptable slippafe
MAX_SLIPPAGE = 3

####################
# Class Definition #
####################

class TraderEnv(object):
    def __init__(self, logger):
        # Init logger interface
        self.logger = logger
        # Order execution queue
        self.order_queue = Queue()
        # Set working dir
        self.BASE_DIR = os.path.dirname(os.path.realpath(__file__))
        # Load normalized test
        self.test_df = load_df(TEST_PATH)
        # Load unnormalized test
        self.test_df_un = load_df(TEST_PATH_UNNORMALIZED)
        # Get X and Y
        self.X_test, self.y_test = prepare_x_y(self.test_df, N_INPUTS)
        # Retrieve model
        with open(MODEL_PATH, 'rb') as f:
            self.model = pickle.load(f)
        # Init time parameters
        self.time_delta_series, self.total_runtime = self.init_time_params()
        # Create queue with time_delta_series elements
        self.delta_queue = create_queue_from_delta_series(self.time_delta_series[N_INPUTS - 1:])
        # Create simulation
        self.init_simulation()

    def init_time_params(self):
        # Format receiving times. The first 20 observations are omitted since we need 20 previous observations to get
        # the first input
        tmp_series = pd.to_datetime(self.test_df.iloc[:, 0], format='%Y-%m-%dT%H:%M:%S.%fZ')
        # Compute time delta in milliseconds
        time_delta_series = (tmp_series - tmp_series[N_INPUTS - 1]).dt.total_seconds() * TIME_UNIT
        # Get last time delta in millisenconds
        total_runtime = time_delta_series[-1]
        return time_delta_series, total_runtime

    def init_simulation(self):
        # Get the total time of the simulation
        # Retrieve next state time
        next_lob_state = self.delta_queue.get()
        # Init end and start timestamp
        finish_time = time() * TIME_UNIT + self.total_runtime
        start_time = time() * TIME_UNIT
        next_lob_state_time = start_time + next_lob_state[1]
        # Counter starts at N_INPUTS - 1 index.
        next_lob_state_index = next_lob_state[0] + N_INPUTS - 1

        # Init position globals object. This object acts as a position buffer for threads .
        position_globals = PosistionGlobals()
        # Init processor thread
        processor = OrderProcessor(self.logger, self.order_queue, position_globals, next_lob_state_index)

        # Simulation main loop
        write_to_out(f">>>>>> SIMULATION STARTED <<<<<<<<<<\n")
        write_to_out(f"------ FINISHING AT {datetime.datetime.fromtimestamp(finish_time / 1000)} -------\n")
        while time() * TIME_UNIT < finish_time * SCALING_FACTOR:
            if time() * TIME_UNIT >= next_lob_state_time:
                # Get lob current state
                current_lob_state_index = next_lob_state_index
                curr_lob = list(self.test_df_un.iloc[current_lob_state_index, 2:-2])
                curr_signal = np.argmax(
                    self.model.predict(np.expand_dims(self.X_test[current_lob_state_index - (N_INPUTS - 1)], axis=0), verbose=0)
                )
                if processor.is_alive():
                    processor.getting_impatient()
                elif self.order_queue.qsize() > 0:
                    element_in_queue = self.order_queue.get()
                    signal, pos_size, depth = element_in_queue['ACTION'], element_in_queue['POSITION_SIZE'], element_in_queue['DEPTH']
                    if signal == curr_signal and current_lob_state_index - depth <= MAX_SLIPPAGE:
                        # If signal is outdated, it is discarded
                        processor = OrderProcessor(
                            self.logger,
                            self.order_queue,
                            position_globals,
                            signal,
                            pos_size=pos_size,
                            observation=curr_lob,
                            depth=current_lob_state_index
                        )
                        processor.start()
                        write_to_out(f'Executing order ID: {position_globals.order_counter} ACTION: {SIGNAL[curr_signal]} in queue.')
                    else:
                        write_to_out(f'Order ID: {position_globals.order_counter} ACTION: {SIGNAL[curr_signal]} in queue discarded.')
                else:
                    processor = OrderProcessor(
                        self.logger,
                        self.order_queue,
                        position_globals,
                        curr_signal,
                        pos_size=POSITION_SIZE,
                        observation=curr_lob,
                        depth=current_lob_state_index
                    )
                    processor.start()
                    write_to_out(f'Executing order ID: {position_globals.order_counter} ACTION {SIGNAL[curr_signal]}.')
                next_lob_state = self.delta_queue.get()
                next_lob_state_delta = next_lob_state[1]
                next_lob_state_index = next_lob_state[0] + N_INPUTS - 1
                next_lob_state_time = start_time + next_lob_state_delta


def write_to_out(mess):
    logger.write_info(mess)
    print(f'{datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S.%f")} | {mess}')


if __name__ == '__main__':
    logger = TraderLogger(logging.INFO)
    # Launch simulation
    simulation = TraderEnv(logger)
