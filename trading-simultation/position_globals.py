# position constant
LONG = 0
SHORT = 1
FLAT = 2


class PosistionGlobals():
    def __init__(self):
        self.order_counter = 0
        # Init class variables
        self.prev_position = FLAT
        self.position = FLAT
        self.history = []
        self.cum_return = 0
        self.n_long = 0
        self.n_short = 0
        self.entry_price = 0
        self.order_counter = 0

    def update_state(self, prev_position, position, cum_return, n_long, n_short, entry_price, order_counter, action):
        self.prev_position = prev_position
        self.position = position
        self.cum_return = cum_return
        self.n_short = n_short
        self.n_long = n_long
        self.entry_price = entry_price
        self.order_counter = order_counter
        self.history.append((
            self.prev_position,
            self.position,
            self.cum_return,
            self.n_short,
            self.n_long,
            self.entry_price,
            self.order_counter,
            action,
        ))
